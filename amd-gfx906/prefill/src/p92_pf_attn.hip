// Chunked MLA attention for openPangu prefill on gfx906.
//
// NOT mla_g16 widened. That kernel splits one query's key range across 16
// partial-softmax slots and writes pacc[48][16][512]; at T=512 the same
// structure would materialise 805 MB of partials. The decomposition is wrong
// for prefill, not merely large.
//
// INVERTED REUSE. Decode sweeps one query across every key, so keys stream and
// the query sits in registers. Prefill has TQ queries sharing a key range, so
// the KEY is loaded once and swept against all TQ - the attention analogue of
// holding an expert's weight row resident across its routed tokens in
// k_pf_gateup. Key traffic becomes independent of TQ.
//
// STATE IS BOUNDED, which is what kills pacc. Each block carries TQ running
// (m, l) pairs and a TQ x 512 latent accumulator distributed over 64 lanes at
// 8 dims each: TQ=4 gives 32 accumulator floats a lane, against the 48 mla_g16
// already holds. No partials reach memory; the online softmax closes inside the
// block and writes the final latent once.
//
// CAUSALITY. Query q attends to the 128 sinks plus every position <= its own
// absolute index. The key loop is bounded per query tile rather than masked
// per element, so the causal edge costs one comparison at the tile boundary.
#include <hip/hip_runtime.h>
#include <cstdint>

#ifndef PF_ATTN_H
#define PF_ATTN_H
#define PF_NH        48
#define PF_KVLORA   512
#define PF_PE        64
#define PF_SINKS    128
#define PF_QK_DIM   192
// TQ is the reuse/occupancy trade and is NOT settled. Each query costs
// PF_DPL accumulator floats plus PF_DPL query floats a lane, so TQ=4 gives
// VGPR 122 and 2 waves/SIMD - lower than mla_g16, which at 3 turned out
// latency-bound. Higher TQ amortises each key load over more queries; lower TQ
// buys occupancy back. Templated so the sweep is a launch parameter rather than
// an edit, and measured on the composed pipeline, never on the register count.
#define PF_TQ_DEFAULT 4
#define PF_LANES     64
#define PF_DPL      (PF_KVLORA / PF_LANES)   // 8 latent dims a lane
#define PF_PEPL     (PF_PE / PF_LANES)       // 1 rope dim a lane
typedef uint16_t pf_bf;
#endif

static __device__ __forceinline__ float pf_b2f(pf_bf b) {
    return __uint_as_float((uint32_t)b << 16);
}

// grid (T/PF_TQ, PF_NH), block 64.
//   qlat  [T][NH][KVLORA]   absorbed queries, fp32
//   qpe   [T][NH][PE]       roped query, fp32
//   lat   [P][KVLORA]       latent cache, bf16, P = chunk_base + T
//   kpe   [P][PE]           roped key cache, bf16
//   out   [T][NH][KVLORA]   latent context, fp32
template <int TQ>
__global__ void __launch_bounds__(PF_LANES)
k_pf_attn(const float *__restrict__ qlat, const float *__restrict__ qpe,
          const pf_bf *__restrict__ lat, const pf_bf *__restrict__ kpe,
          const pf_bf *__restrict__ sink_lat, const pf_bf *__restrict__ sink_pe,
          int chunk_base, int T, int win, float *__restrict__ out)
{
    const int h  = blockIdx.y;
    const int q0 = blockIdx.x * TQ;
    const int l  = threadIdx.x;                 // 0..63
    const float scale = rsqrtf((float)PF_QK_DIM);

    // ---- hoist the TQ queries into registers, once ------------------------
    float qv[TQ][PF_DPL], qp[TQ][PF_PEPL], acc[TQ][PF_DPL];
    float m_run[TQ], l_run[TQ];
    #pragma unroll
    for (int t = 0; t < TQ; t++) {
        const int qi = q0 + t;
        const bool live = (qi < T);
        const float *ql = qlat + ((size_t)qi * PF_NH + h) * PF_KVLORA;
        const float *qr = qpe  + ((size_t)qi * PF_NH + h) * PF_PE;
        // LANE-CONTIGUOUS LATENT. Lane l owns dims [l*PF_DPL, l*PF_DPL+8), so
        // one 16-byte load a lane covers the wave's 1024-byte slice in a single
        // instruction. The strided form (d*64+l) fetched the same bytes as
        // eight separate 128-byte transactions - correct, coalesced, and eight
        // times the instruction count on the hottest loop in the engine.
        if (live) {
            const float4 *q4 = (const float4 *)(ql + l * PF_DPL);
            const float4 a = q4[0], b = q4[1];
            qv[t][0]=a.x; qv[t][1]=a.y; qv[t][2]=a.z; qv[t][3]=a.w;
            qv[t][4]=b.x; qv[t][5]=b.y; qv[t][6]=b.z; qv[t][7]=b.w;
        } else {
            #pragma unroll
            for (int d = 0; d < PF_DPL; d++) qv[t][d] = 0.f;
        }
        #pragma unroll
        for (int d = 0; d < PF_DPL; d++) acc[t][d] = 0.f;
        #pragma unroll
        for (int d = 0; d < PF_PEPL; d++) qp[t][d] = live ? qr[d * PF_LANES + l] : 0.f;
        m_run[t] = -1e30f; l_run[t] = 0.f;
    }

    // one entry: load the key ONCE, score it against all TQ queries
    #define PF_ENTRY(CLAT, CPE, LIVE_MASK)                                        \
    {                                                                             \
        float kv[PF_DPL], kp[PF_PEPL];                                            \
        { const uint4 kr = *(const uint4 *)((CLAT) + l * PF_DPL);                 \
          kv[0]=__uint_as_float(kr.x<<16); kv[1]=__uint_as_float(kr.x&0xFFFF0000u);\
          kv[2]=__uint_as_float(kr.y<<16); kv[3]=__uint_as_float(kr.y&0xFFFF0000u);\
          kv[4]=__uint_as_float(kr.z<<16); kv[5]=__uint_as_float(kr.z&0xFFFF0000u);\
          kv[6]=__uint_as_float(kr.w<<16); kv[7]=__uint_as_float(kr.w&0xFFFF0000u);}\
        _Pragma("unroll")                                                         \
        for (int d = 0; d < PF_PEPL; d++) kp[d] = pf_b2f((CPE)[d * PF_LANES + l]); \
        _Pragma("unroll")                                                         \
        for (int t = 0; t < TQ; t++) {                                         \
            float s = 0.f;                                                        \
            _Pragma("unroll")                                                     \
            for (int d = 0; d < PF_DPL; d++)  s = fmaf(qv[t][d], kv[d], s);       \
            _Pragma("unroll")                                                     \
            for (int d = 0; d < PF_PEPL; d++) s = fmaf(qp[t][d], kp[d], s);       \
            _Pragma("unroll")                                                     \
            for (int o = 32; o; o >>= 1) s += __shfl_xor(s, o, 64);               \
            s = ((LIVE_MASK) & (1 << t)) ? s * scale : -1e30f;                    \
            const float mn = fmaxf(m_run[t], s);                                  \
            const float al = __expf(m_run[t] - mn);                               \
            const float p  = __expf(s - mn);                                      \
            l_run[t] = l_run[t] * al + p;                                         \
            _Pragma("unroll")                                                     \
            for (int d = 0; d < PF_DPL; d++) acc[t][d] = acc[t][d] * al + p * kv[d]; \
            m_run[t] = mn;                                                        \
        }                                                                         \
    }

    // sinks: visible to every query in the tile
    for (int sIdx = 0; sIdx < PF_SINKS; sIdx++)
        PF_ENTRY(sink_lat + (size_t)sIdx * PF_KVLORA,
                 sink_pe  + (size_t)sIdx * PF_PE, (1 << TQ) - 1)

    // body: causal. Keys strictly below the tile are visible to all TQ; inside
    // the tile the mask narrows one query at a time, so only PF_TQ entries need
    // per-query masking rather than every element.
    const int qabs0 = chunk_base + q0;
    // Sliding-window layers (index % 3 != 0) attend to the sinks plus the last
    // `win` positions; DSA layers pass win = 0 and see the whole prefix. The
    // bound is per QUERY, not per tile: a tile shares one key sweep, so without
    // the lower mask the first query of a tile would see up to TQ-1 keys past
    // its own window edge - which is exactly the kind of off-by-a-tile the
    // causal negative control is there to catch.
    const int kstart = win ? (qabs0 - win + 1 > 0 ? qabs0 - win + 1 : 0) : 0;
    for (int k = kstart; k < qabs0; k++) {
        const int rel = k - qabs0;                  // negative: causal for all t
        int hi = win ? rel + win - 1 : TQ - 1;
        if (hi > TQ - 1) hi = TQ - 1;
        const int mask = (hi >= 0) ? ((1 << (hi + 1)) - 1) : 0;
        PF_ENTRY(lat + (size_t)k * PF_KVLORA, kpe + (size_t)k * PF_PE, mask)
    }
    #pragma unroll
    for (int j = 0; j < TQ; j++) {
        const int k = qabs0 + j;
        int hi = win ? j + win - 1 : TQ - 1;
        if (hi > TQ - 1) hi = TQ - 1;
        const int mask = (hi >= j) ? (((1 << (hi - j + 1)) - 1) << j) : 0;
        PF_ENTRY(lat + (size_t)k * PF_KVLORA, kpe + (size_t)k * PF_PE, mask)
    }
    #undef PF_ENTRY

    #pragma unroll
    for (int t = 0; t < TQ; t++) {
        const int qi = q0 + t;
        if (qi >= T) continue;
        const float inv = l_run[t] > 0.f ? 1.f / l_run[t] : 0.f;
        float *o = out + ((size_t)qi * PF_NH + h) * PF_KVLORA;
        float4 *o4 = (float4 *)(o + l * PF_DPL);
        o4[0] = make_float4(acc[t][0]*inv, acc[t][1]*inv, acc[t][2]*inv, acc[t][3]*inv);
        o4[1] = make_float4(acc[t][4]*inv, acc[t][5]*inv, acc[t][6]*inv, acc[t][7]*inv);
    }
}

// Explicit instantiations for the sweep. The composed pipeline picks one.
template __global__ void k_pf_attn<2>(const float*, const float*, const pf_bf*,
    const pf_bf*, const pf_bf*, const pf_bf*, int, int, int, float*);
template __global__ void k_pf_attn<4>(const float*, const float*, const pf_bf*,
    const pf_bf*, const pf_bf*, const pf_bf*, int, int, int, float*);
template __global__ void k_pf_attn<8>(const float*, const float*, const pf_bf*,
    const pf_bf*, const pf_bf*, const pf_bf*, int, int, int, float*);

// ===========================================================================
// k_pf_attn2 - a different inner loop, not a different parameterization.
//
// Built against the measured ISA of k_pf_attn's body loop (.LBB1_23, 258
// instructions to consume TWO loads):
//     s_waitcnt 26 | vmcnt(0) 1 | ds_bpermute 24 | v_mul 44 | v_fma 72 | v_exp 8
//
// Cooperative multi-wave was considered and REJECTED on arithmetic, not taste.
// waves/SIMD = floor(256/VGPR); this kernel measures 118 VGPRs, so 2 waves, and
// 3 would need <=84. The named state is only acc 32 + qv 32 + m/l 8 = 72; the
// other ~46 are addresses, staging and temporaries. Splitting the accumulator
// across W waves takes acc 32->8, landing at ~94 - still 2 waves/SIMD. Even a
// free, infinitely-split accumulator stays at 2. The accumulator is not what
// holds occupancy here, so the mechanism cannot reach its own goal. Splitting
// the SCORE contraction to cut qv would, but it breaks the rule that makes the
// load work at all: 512 bf16 = 1024 B = exactly 64 lanes x 16 B, and halving
// the per-lane width is the difference between two measured points, 615 GB/s at
// 16 B and 327 GB/s at 8 B.
//
// So this attacks the serial chain instead. Four changes, each aimed at one of
// the four numbers above:
//
// 1. DPP BUTTERFLY. __shfl_xor lowers to ds_bpermute - the LDS path - and this
//    kernel issues 24 per key in four dependent chains of six. Three of the six
//    steps have exact DPP equivalents, which run on the VALU path with no
//    lgkmcnt: xor 2 is quad_perm 0x4E, xor 1 is quad_perm 0xB1, and xor 8 is
//    row_mirror THEN row_half_mirror (xor 15 then xor 7 composes to xor 8).
//    xor 4 has NO DPP substitute and xor 16/32 cross the row, so those three
//    stay on ds. 24 ds_bpermute a key becomes 12. Lane mappings are the ones
//    this project already verified and shipped in p92_mla_g16.hip.
//
// 2. EXP2 WITH A PRESCALED QUERY. v_exp_f32 IS base-2 on gfx906, so every
//    __expf pays an extra multiply by log2(e). Folding scale*log2(e) into the
//    query once at load time makes the score arrive in log2 units and lets the
//    softmax call exp2f directly. Deletes 8 multiplies a key.
//
// 3. GATED RESCALE. The score is wave-uniform after the butterfly, so m_run and
//    alpha are uniform and "did the running max advance" is a SCALAR branch.
//    When it did not - most keys, once the distribution settles - alpha is
//    exactly 1.0 and the rescale is 8 multiplies a query multiplying by one.
//    Skipping it removes up to 36 of the 44 multiplies a key.
//
// 4. TWO KEYS IN FLIGHT. Both loads are issued before either is consumed, so
//    the pair costs one vmcnt drain instead of two, and the second key's dot
//    has independent work to overlap the first key's butterfly latency. Costs
//    ~6 VGPRs, which the budget has: 118 against the 128 ceiling for 2 waves.
//
// Semantics are identical to k_pf_attn. Reduction ORDER is not: the DPP
// butterfly sums the same lanes in a different association, which this engine
// is explicitly allowed to do.
// ===========================================================================
#define PF_DPPF(x, c) __int_as_float(__builtin_amdgcn_update_dpp(          \
    __float_as_int(x), __float_as_int(x), (c), 0xF, 0xF, false))

// 64-lane sum: 3 ds steps (xor 32, 16, 4) + 3 DPP steps (xor 8, 2, 1).
static __device__ __forceinline__ float pf_wave_sum(float s) {
    s += __shfl_xor(s, 32, 64);
    s += __shfl_xor(s, 16, 64);
    { const float t = PF_DPPF(s, 0x140u); s += PF_DPPF(t, 0x141u); }  // xor 8
    s += __shfl_xor(s, 4, 64);                                        // no DPP
    s += PF_DPPF(s, 0x4Eu);                                           // xor 2
    s += PF_DPPF(s, 0xB1u);                                           // xor 1
    return s;
}

template <int TQ>
__global__ void __launch_bounds__(PF_LANES)
k_pf_attn2(const float *__restrict__ qlat, const float *__restrict__ qpe,
           const pf_bf *__restrict__ lat, const pf_bf *__restrict__ kpe,
           const pf_bf *__restrict__ sink_lat, const pf_bf *__restrict__ sink_pe,
           int chunk_base, int T, int win, float *__restrict__ out)
{
    const int h  = blockIdx.y;
    const int q0 = blockIdx.x * TQ;
    const int l  = threadIdx.x;
    // scale AND log2(e) folded in once, so the score arrives in log2 units
    const float qs = rsqrtf((float)PF_QK_DIM) * 1.44269504088896340736f;

    float qv[TQ][PF_DPL], qp[TQ][PF_PEPL], acc[TQ][PF_DPL];
    float m_run[TQ], l_run[TQ];
    #pragma unroll
    for (int t = 0; t < TQ; t++) {
        const int qi = q0 + t;
        const bool live = (qi < T);
        const float *ql = qlat + ((size_t)qi * PF_NH + h) * PF_KVLORA;
        const float *qr = qpe  + ((size_t)qi * PF_NH + h) * PF_PE;
        if (live) {
            const float4 *q4 = (const float4 *)(ql + l * PF_DPL);
            const float4 a = q4[0], b = q4[1];
            qv[t][0]=a.x*qs; qv[t][1]=a.y*qs; qv[t][2]=a.z*qs; qv[t][3]=a.w*qs;
            qv[t][4]=b.x*qs; qv[t][5]=b.y*qs; qv[t][6]=b.z*qs; qv[t][7]=b.w*qs;
        } else {
            #pragma unroll
            for (int d = 0; d < PF_DPL; d++) qv[t][d] = 0.f;
        }
        #pragma unroll
        for (int d = 0; d < PF_PEPL; d++) qp[t][d] = live ? qr[d*PF_LANES+l]*qs : 0.f;
        #pragma unroll
        for (int d = 0; d < PF_DPL; d++) acc[t][d] = 0.f;
        m_run[t] = -1e30f; l_run[t] = 0.f;
    }

    // unpack one key's lane slice from a single 16-byte load
#define PF2_LOAD(CLAT, CPE, KV, KP)                                            \
    { const uint4 kr = *(const uint4 *)((CLAT) + l * PF_DPL);                  \
      KV[0]=__uint_as_float(kr.x<<16); KV[1]=__uint_as_float(kr.x&0xFFFF0000u);\
      KV[2]=__uint_as_float(kr.y<<16); KV[3]=__uint_as_float(kr.y&0xFFFF0000u);\
      KV[4]=__uint_as_float(kr.z<<16); KV[5]=__uint_as_float(kr.z&0xFFFF0000u);\
      KV[6]=__uint_as_float(kr.w<<16); KV[7]=__uint_as_float(kr.w&0xFFFF0000u);\
      _Pragma("unroll")                                                        \
      for (int d = 0; d < PF_PEPL; d++) KP[d] = pf_b2f((CPE)[d*PF_LANES + l]); }

    // score + online-softmax update for one already-loaded key
#define PF2_STEP(KV, KP, LIVE_MASK)                                            \
    { _Pragma("unroll")                                                        \
      for (int t = 0; t < TQ; t++) {                                           \
        float s = 0.f;                                                         \
        _Pragma("unroll")                                                      \
        for (int d = 0; d < PF_DPL; d++)  s = fmaf(qv[t][d], KV[d], s);        \
        _Pragma("unroll")                                                      \
        for (int d = 0; d < PF_PEPL; d++) s = fmaf(qp[t][d], KP[d], s);        \
        s = pf_wave_sum(s);                                                    \
        s = (((LIVE_MASK) >> t) & 1) ? s : -1e30f;                             \
        const float mn = fmaxf(m_run[t], s);                                   \
        /* wave-uniform: a scalar branch, and alpha is exactly 1 when not taken */\
        if (mn > m_run[t]) {                                                   \
            const float al = exp2f(m_run[t] - mn);                             \
            l_run[t] *= al;                                                    \
            _Pragma("unroll")                                                  \
            for (int d = 0; d < PF_DPL; d++) acc[t][d] *= al;                  \
            m_run[t] = mn;                                                     \
        }                                                                      \
        const float p = exp2f(s - mn);                                         \
        l_run[t] += p;                                                         \
        _Pragma("unroll")                                                      \
        for (int d = 0; d < PF_DPL; d++) acc[t][d] = fmaf(p, KV[d], acc[t][d]);\
      } }

    // ---- sinks, two in flight ------------------------------------------
    {
        const int FULL = (1 << TQ) - 1;
        float kv0[PF_DPL], kp0[PF_PEPL], kv1[PF_DPL], kp1[PF_PEPL];
        int i = 0;
        for (; i + 1 < PF_SINKS; i += 2) {
            PF2_LOAD(sink_lat + (size_t)i * PF_KVLORA,     sink_pe + (size_t)i * PF_PE,     kv0, kp0)
            PF2_LOAD(sink_lat + (size_t)(i+1) * PF_KVLORA, sink_pe + (size_t)(i+1) * PF_PE, kv1, kp1)
            PF2_STEP(kv0, kp0, FULL)
            PF2_STEP(kv1, kp1, FULL)
        }
        for (; i < PF_SINKS; i++) {
            PF2_LOAD(sink_lat + (size_t)i * PF_KVLORA, sink_pe + (size_t)i * PF_PE, kv0, kp0)
            PF2_STEP(kv0, kp0, FULL)
        }
    }

    // ---- causal body, two keys in flight -------------------------------
    const int qabs0 = chunk_base + q0;
    const int kstart = win ? (qabs0 - win + 1 > 0 ? qabs0 - win + 1 : 0) : 0;
    {
        float kv0[PF_DPL], kp0[PF_PEPL], kv1[PF_DPL], kp1[PF_PEPL];
        int k = kstart;
        for (; k + 1 < qabs0; k += 2) {
            PF2_LOAD(lat + (size_t)k * PF_KVLORA,     kpe + (size_t)k * PF_PE,     kv0, kp0)
            PF2_LOAD(lat + (size_t)(k+1) * PF_KVLORA, kpe + (size_t)(k+1) * PF_PE, kv1, kp1)
            int h0 = win ? (k - qabs0) + win - 1 : TQ - 1; if (h0 > TQ-1) h0 = TQ-1;
            int h1 = win ? (k+1 - qabs0) + win - 1 : TQ - 1; if (h1 > TQ-1) h1 = TQ-1;
            PF2_STEP(kv0, kp0, (h0 >= 0) ? ((1 << (h0+1)) - 1) : 0)
            PF2_STEP(kv1, kp1, (h1 >= 0) ? ((1 << (h1+1)) - 1) : 0)
        }
        for (; k < qabs0; k++) {
            PF2_LOAD(lat + (size_t)k * PF_KVLORA, kpe + (size_t)k * PF_PE, kv0, kp0)
            int h0 = win ? (k - qabs0) + win - 1 : TQ - 1; if (h0 > TQ-1) h0 = TQ-1;
            PF2_STEP(kv0, kp0, (h0 >= 0) ? ((1 << (h0+1)) - 1) : 0)
        }
        // the tile's own TQ keys: per-query causal edge
        #pragma unroll
        for (int j = 0; j < TQ; j++) {
            const int kk = qabs0 + j;
            PF2_LOAD(lat + (size_t)kk * PF_KVLORA, kpe + (size_t)kk * PF_PE, kv0, kp0)
            int hi = win ? j + win - 1 : TQ - 1; if (hi > TQ-1) hi = TQ-1;
            PF2_STEP(kv0, kp0, (hi >= j) ? (((1 << (hi-j+1)) - 1) << j) : 0)
        }
    }
#undef PF2_LOAD
#undef PF2_STEP

    #pragma unroll
    for (int t = 0; t < TQ; t++) {
        const int qi = q0 + t;
        if (qi >= T) continue;
        const float inv = l_run[t] > 0.f ? 1.f / l_run[t] : 0.f;
        float *o = out + ((size_t)qi * PF_NH + h) * PF_KVLORA;
        float4 *o4 = (float4 *)(o + l * PF_DPL);
        o4[0] = make_float4(acc[t][0]*inv, acc[t][1]*inv, acc[t][2]*inv, acc[t][3]*inv);
        o4[1] = make_float4(acc[t][4]*inv, acc[t][5]*inv, acc[t][6]*inv, acc[t][7]*inv);
    }
}

template __global__ void k_pf_attn2<2>(const float*, const float*, const pf_bf*,
    const pf_bf*, const pf_bf*, const pf_bf*, int, int, int, float*);
template __global__ void k_pf_attn2<4>(const float*, const float*, const pf_bf*,
    const pf_bf*, const pf_bf*, const pf_bf*, int, int, int, float*);
template __global__ void k_pf_attn2<8>(const float*, const float*, const pf_bf*,
    const pf_bf*, const pf_bf*, const pf_bf*, int, int, int, float*);

// ===========================================================================
// UNION-TILED SPARSE ATTENTION
//
// A per-query index list destroys the thing that makes the dense kernel work: a
// TQ=4 tile sweeps ONE key range shared by four queries, so key traffic is
// npos/4 a query. Give each query its own 2048 positions and the sharing is
// gone - TQ=1 sparse reads 2048 a query against dense's 1024 at 4096 context,
// which is twice as much work, not half.
//
// The way back is the UNION. Measured on real selections, four adjacent queries
// select 4 x 2048 = 8192 slots that collapse to 3794 unique positions, and eight
// queries collapse 16384 to 4020. Adjacent queries want very nearly the same
// keys. So a tile sweeps the union ONCE and each query masks to its own subset,
// which is exactly the model's semantics - every query still attends to its own
// top-2048 and nothing else - at a quarter of the key traffic.
//
// k_pf_union builds that: an LDS occupancy byte per position holding a TQ-bit
// membership mask, then an ordered ballot compaction into (position, mask)
// pairs. Ordered, because the sweep's locality is worth more than the few
// instructions it costs.
// ===========================================================================
template <int TQ>
__global__ void __launch_bounds__(256)
k_pf_union(const int32_t *__restrict__ sel, const int32_t *__restrict__ nsel,
           int T, int want, int maxpos,
           int32_t *__restrict__ uni, uint8_t *__restrict__ umask,
           int32_t *__restrict__ ucnt, int umax)
{
    const int tile = blockIdx.x;
    const int q0   = tile * TQ;
    if (q0 >= T) return;
    extern __shared__ uint8_t occ[];              // maxpos bytes, one per position
    for (int i = threadIdx.x; i < maxpos; i += 256) occ[i] = 0;
    __syncthreads();
    #pragma unroll
    for (int t = 0; t < TQ; t++) {
        const int q = q0 + t;
        if (q >= T) continue;
        const int n = nsel[q];
        const int32_t *sq = sel + (size_t)q * want;
        for (int i = threadIdx.x; i < n; i += 256) {
            const int p = sq[i];
            if (p >= 0 && p < maxpos) atomicOr((unsigned int *)&occ[p & ~3],
                                               ((unsigned int)(1u << t)) << ((p & 3) * 8));
        }
    }
    __syncthreads();

    __shared__ int cur;
    __shared__ uint32_t wv[4];
    if (threadIdx.x == 0) cur = 0;
    __syncthreads();
    const int lane = threadIdx.x & 63, w = threadIdx.x >> 6;
    for (int b = 0; b < maxpos; b += 256) {
        const int p = b + threadIdx.x;
        const bool on = (p < maxpos) && (occ[p] != 0);
        const uint64_t bal = __ballot(on);
        const uint32_t pre = (uint32_t)__popcll(bal & ((1ull << lane) - 1ull));
        if (lane == 0) wv[w] = (uint32_t)__popcll(bal);
        __syncthreads();
        uint32_t before = 0, total = 0;
        for (int k = 0; k < 4; k++) { if (k < w) before += wv[k]; total += wv[k]; }
        if (on) {
            const int slot = cur + (int)(before + pre);
            if (slot < umax) { uni[(size_t)tile * umax + slot] = p;
                               umask[(size_t)tile * umax + slot] = occ[p]; }
        }
        __syncthreads();
        if (threadIdx.x == 0) cur += (int)total;
        __syncthreads();
    }
    if (threadIdx.x == 0) ucnt[tile] = cur < umax ? cur : umax;
}

// Sparse attention over the tile's union. Identical inner mathematics to
// k_pf_attn2 - DPP butterfly, exp2 on a prescaled query, gated rescale - with
// the causal range replaced by the union list and the causal mask replaced by
// the per-slot membership mask. Causality is already inside the selection:
// a query's top-2048 is drawn only from positions at or before it.
template <int TQ>
__global__ void __launch_bounds__(PF_LANES)
k_pf_attn_sp(const float *__restrict__ qlat, const float *__restrict__ qpe,
             const pf_bf *__restrict__ lat, const pf_bf *__restrict__ kpe,
             const pf_bf *__restrict__ sink_lat, const pf_bf *__restrict__ sink_pe,
             const int32_t *__restrict__ uni, const uint8_t *__restrict__ umask,
             const int32_t *__restrict__ ucnt, int umax,
             int T, float *__restrict__ out)
{
    const int h  = blockIdx.y;
    const int q0 = blockIdx.x * TQ;
    const int l  = threadIdx.x;
    const float qs = rsqrtf((float)PF_QK_DIM) * 1.44269504088896340736f;

    float qv[TQ][PF_DPL], qp[TQ][PF_PEPL], acc[TQ][PF_DPL];
    float m_run[TQ], l_run[TQ];
    #pragma unroll
    for (int t = 0; t < TQ; t++) {
        const int qi = q0 + t;
        const bool live = (qi < T);
        const float *ql = qlat + ((size_t)qi * PF_NH + h) * PF_KVLORA;
        const float *qr = qpe  + ((size_t)qi * PF_NH + h) * PF_PE;
        if (live) {
            const float4 *q4 = (const float4 *)(ql + l * PF_DPL);
            const float4 a = q4[0], b = q4[1];
            qv[t][0]=a.x*qs; qv[t][1]=a.y*qs; qv[t][2]=a.z*qs; qv[t][3]=a.w*qs;
            qv[t][4]=b.x*qs; qv[t][5]=b.y*qs; qv[t][6]=b.z*qs; qv[t][7]=b.w*qs;
        } else {
            #pragma unroll
            for (int d = 0; d < PF_DPL; d++) qv[t][d] = 0.f;
        }
        #pragma unroll
        for (int d = 0; d < PF_PEPL; d++) qp[t][d] = live ? qr[d*PF_LANES+l]*qs : 0.f;
        #pragma unroll
        for (int d = 0; d < PF_DPL; d++) acc[t][d] = 0.f;
        m_run[t] = -1e30f; l_run[t] = 0.f;
    }

#define PFS_LOAD(CLAT, CPE, KV, KP)                                            \
    { const uint4 kr = *(const uint4 *)((CLAT) + l * PF_DPL);                  \
      KV[0]=__uint_as_float(kr.x<<16); KV[1]=__uint_as_float(kr.x&0xFFFF0000u);\
      KV[2]=__uint_as_float(kr.y<<16); KV[3]=__uint_as_float(kr.y&0xFFFF0000u);\
      KV[4]=__uint_as_float(kr.z<<16); KV[5]=__uint_as_float(kr.z&0xFFFF0000u);\
      KV[6]=__uint_as_float(kr.w<<16); KV[7]=__uint_as_float(kr.w&0xFFFF0000u);\
      _Pragma("unroll")                                                        \
      for (int d = 0; d < PF_PEPL; d++) KP[d] = pf_b2f((CPE)[d*PF_LANES + l]); }

#define PFS_STEP(KV, KP, LIVE_MASK)                                            \
    { _Pragma("unroll")                                                        \
      for (int t = 0; t < TQ; t++) {                                           \
        float s = 0.f;                                                         \
        _Pragma("unroll")                                                      \
        for (int d = 0; d < PF_DPL; d++)  s = fmaf(qv[t][d], KV[d], s);        \
        _Pragma("unroll")                                                      \
        for (int d = 0; d < PF_PEPL; d++) s = fmaf(qp[t][d], KP[d], s);        \
        s = pf_wave_sum(s);                                                    \
        s = (((LIVE_MASK) >> t) & 1) ? s : -1e30f;                             \
        const float mn = fmaxf(m_run[t], s);                                   \
        if (mn > m_run[t]) {                                                   \
            const float al = exp2f(m_run[t] - mn);                             \
            l_run[t] *= al;                                                    \
            _Pragma("unroll")                                                  \
            for (int d = 0; d < PF_DPL; d++) acc[t][d] *= al;                  \
            m_run[t] = mn;                                                     \
        }                                                                      \
        const float p = exp2f(s - mn);                                         \
        l_run[t] += p;                                                         \
        _Pragma("unroll")                                                      \
        for (int d = 0; d < PF_DPL; d++) acc[t][d] = fmaf(p, KV[d], acc[t][d]);\
      } }

    {   // sinks: visible to every query, always, selection or not
        const int FULL = (1 << TQ) - 1;
        float kv0[PF_DPL], kp0[PF_PEPL], kv1[PF_DPL], kp1[PF_PEPL];
        int i = 0;
        for (; i + 1 < PF_SINKS; i += 2) {
            PFS_LOAD(sink_lat + (size_t)i * PF_KVLORA,     sink_pe + (size_t)i * PF_PE,     kv0, kp0)
            PFS_LOAD(sink_lat + (size_t)(i+1) * PF_KVLORA, sink_pe + (size_t)(i+1) * PF_PE, kv1, kp1)
            PFS_STEP(kv0, kp0, FULL)
            PFS_STEP(kv1, kp1, FULL)
        }
        for (; i < PF_SINKS; i++) {
            PFS_LOAD(sink_lat + (size_t)i * PF_KVLORA, sink_pe + (size_t)i * PF_PE, kv0, kp0)
            PFS_STEP(kv0, kp0, FULL)
        }
    }
    {   // the tile's union, two keys in flight
        const int tile = blockIdx.x;
        const int n = ucnt[tile];
        const int32_t *ul = uni  + (size_t)tile * umax;
        const uint8_t *um = umask + (size_t)tile * umax;
        float kv0[PF_DPL], kp0[PF_PEPL], kv1[PF_DPL], kp1[PF_PEPL];
        int i = 0;
        // PIPELINED INDEX WALK (09-09). The ISA of the flat loop was a serial
        // chain: s_load the two slot indices, FULL s_waitcnt lgkmcnt(0), then
        // the key addresses, then the key loads - so every iteration exposed
        // the scalar-load latency before anything could issue. The union walk
        // has no aliasing hazard with its own indices, so hoist the NEXT pair's
        // index load to the top of the iteration: its wait lands a full
        // iteration later, behind this pair's FMA burst. The indices ride
        // SGPRs, which costs the TQ=2 kernel nothing in its VGPR-limited 3
        // waves/SIMD. Numerical semantics are bit-identical; only the memory
        // schedule changes.
        if (i + 3 < n) {
            int k0 = ul[0], k1 = ul[1];
            do {
                // volatile: the load must happen HERE, and LLVM cannot fold it
                // back into the induction walk (it re-derived the plain form
                // and re-serialised the wait - ISA-checked). The scalarized
                // s_load issues at the iteration top; its wait lands at the
                // next iteration's address math, behind this pair's FMA burst.
                const int n0 = ((const volatile int32_t*)ul)[i+2];
                const int n1 = ((const volatile int32_t*)ul)[i+3];
                PFS_LOAD(lat + (size_t)k0 * PF_KVLORA, kpe + (size_t)k0 * PF_PE, kv0, kp0)
                PFS_LOAD(lat + (size_t)k1 * PF_KVLORA, kpe + (size_t)k1 * PF_PE, kv1, kp1)
                k0 = n0; k1 = n1;
                PFS_STEP(kv0, kp0, (int)um[i])
                PFS_STEP(kv1, kp1, (int)um[i+1])
                i += 2;
            } while (i + 3 < n);
            // exit leaves one preloaded pair: i+1 < n holds (the last executed
            // iteration entered with i+3 < n and advanced by 2), and k0,k1 hold
            // exactly ul[i], ul[i+1]
            PFS_LOAD(lat + (size_t)k0 * PF_KVLORA, kpe + (size_t)k0 * PF_PE, kv0, kp0)
            PFS_LOAD(lat + (size_t)k1 * PF_KVLORA, kpe + (size_t)k1 * PF_PE, kv1, kp1)
            PFS_STEP(kv0, kp0, (int)um[i])
            PFS_STEP(kv1, kp1, (int)um[i+1])
            i += 2;
        }
        for (; i + 1 < n; i += 2) {
            const int k0 = ul[i], k1 = ul[i+1];
            PFS_LOAD(lat + (size_t)k0 * PF_KVLORA, kpe + (size_t)k0 * PF_PE, kv0, kp0)
            PFS_LOAD(lat + (size_t)k1 * PF_KVLORA, kpe + (size_t)k1 * PF_PE, kv1, kp1)
            PFS_STEP(kv0, kp0, (int)um[i])
            PFS_STEP(kv1, kp1, (int)um[i+1])
        }
        for (; i < n; i++) {
            const int k0 = ul[i];
            PFS_LOAD(lat + (size_t)k0 * PF_KVLORA, kpe + (size_t)k0 * PF_PE, kv0, kp0)
            PFS_STEP(kv0, kp0, (int)um[i])
        }
    }
#undef PFS_LOAD
#undef PFS_STEP

    #pragma unroll
    for (int t = 0; t < TQ; t++) {
        const int qi = q0 + t;
        if (qi >= T) continue;
        const float inv = l_run[t] > 0.f ? 1.f / l_run[t] : 0.f;
        float *o = out + ((size_t)qi * PF_NH + h) * PF_KVLORA;
        float4 *o4 = (float4 *)(o + l * PF_DPL);
        o4[0] = make_float4(acc[t][0]*inv, acc[t][1]*inv, acc[t][2]*inv, acc[t][3]*inv);
        o4[1] = make_float4(acc[t][4]*inv, acc[t][5]*inv, acc[t][6]*inv, acc[t][7]*inv);
    }
}

// ===========================================================================
// PACKED-UNION SPARSE ATTENTION (09-09)
//
// The union walk in k_pf_attn_sp is a dependent chain: scalar-load the slot
// index, FULL s_waitcnt lgkmcnt(0), address math, then the key loads - every
// iteration exposed the scalar-load latency before anything could issue
// (ISA-verified). Compiler-level prefetch of the next pair's indices re-derived
// and re-serialised: the scheduler phase-shifts the loop so each iteration
// consumes exactly what it just loaded. The chain exists because every key
// address needs an index.
//
// The fix is consumer-designed: k_pf_union already computes the ordered union,
// so it also EMITS a compacted key stream - lat 1024 B + pe 128 B + mask 16 B
// per slot, contiguous per tile. The attention walk then has NO index loads
// and NO per-key address math: three running pointers with power-of-two
// strides, the exact shape this compiler pipelines without help (the sinks
// loop is the proof: four loads back to back, staged waits). DRAM also sees
// one sequential stream per head sweep instead of scattered 1152 B gathers.
// Cost: the pack kernel touches the union's key bytes once (~2% of what the
// 48 head sweeps read), plus a one-block scan, plus ~612 MB of arena per card
// at chunk 256.
//
// Numerical semantics are IDENTICAL to k_pf_attn_sp: same keys, same order,
// same masks, same reduction. The gate for this variant is bit-exactness
// against the index path, not a tolerance.
// ===========================================================================

// exclusive scan of per-tile union counts -> entry offsets
__global__ void __launch_bounds__(256)
k_pf_pack_scan(const int32_t *__restrict__ ucnt, int nt,
               int32_t *__restrict__ packoff)
{
    __shared__ int sh[256];
    const int t = threadIdx.x;
    sh[t] = (t < nt) ? ucnt[t] : 0;
    __syncthreads();
    #pragma unroll
    for (int o = 1; o < 256; o <<= 1) {
        const int v = (t >= o) ? sh[t - o] : 0;
        __syncthreads();
        sh[t] += v;
        __syncthreads();
    }
    if (t < nt) packoff[t] = sh[t] - ucnt[t];
}

// copy each tile's union keys into the contiguous per-tile streams.
// lat: PF_KVLORA bf16 (1024 B) a slot, pe: PF_PE bf16 (128 B), mask: 16 B slot
// with the membership byte at offset 0.
__global__ void __launch_bounds__(256)
k_pf_union_pack(const int32_t *__restrict__ uni, const uint8_t *__restrict__ umask,
                const int32_t *__restrict__ ucnt, const int32_t *__restrict__ packoff,
                int umax,
                const pf_bf *__restrict__ lat, const pf_bf *__restrict__ kpe,
                pf_bf *__restrict__ lat_pack, pf_bf *__restrict__ pe_pack,
                uint8_t *__restrict__ mask_pack)
{
    const int tile = blockIdx.x;
    const int n = ucnt[tile];
    if (n <= 0) return;
    const int off = packoff[tile];
    const int32_t *ul = uni + (size_t)tile * umax;
    const uint8_t *um = umask + (size_t)tile * umax;
    // COALESCED (key, float4) SPLIT (09-09). The first shape gave each thread
    // one whole key: 256 threads at 1024-B stride, so every 16-B store touched
    // its own 128-B sector - 12.5% sector efficiency, measured 26 GB/s and
    // 20.4 s of dsauni at 64K. With thread t = (w = t>>6 keys, f = t&63
    // float4s) the wave's stores cover four CONSECUTIVE keys, 4 KB contiguous
    // per instruction, and the loads are the same 4 x 1024 B pattern the
    // attention walk already sustains.
    const int f = threadIdx.x & 63;
    const int w = threadIdx.x >> 6;
    for (int s0 = w; s0 < n; s0 += 4) {
        const int s = s0;
        const int k = ul[s];
        // +f inside a uint4* index: 16 B a step. The raw bf16 arithmetic
        // (f*4 = 8 B) overlapped the float4s - caught by the gate.
        ((uint4 *)(lat_pack + ((size_t)off + s) * PF_KVLORA))[f] =
            ((const uint4 *)(lat + (size_t)k * PF_KVLORA))[f];
        if (f < PF_PE / 8)
            ((uint4 *)(pe_pack + ((size_t)off + s) * PF_PE))[f] =
                ((const uint4 *)(kpe + (size_t)k * PF_PE))[f];
        if (f == 0)
            ((uint4 *)(mask_pack + ((size_t)off + s) * 16))[0] =
                make_uint4(um[s], 0u, 0u, 0u);
    }
}

// Sparse attention over the PACKED union. Same mathematics as k_pf_attn_sp -
// DPP butterfly, exp2 on a prescaled query, gated rescale, sinks first - but
// the union walk is three contiguous streams with power-of-two strides, so the
// inner loop is a pure pointer walk with zero dependent index loads.
template <int TQ>
__global__ void __launch_bounds__(PF_LANES)
k_pf_attn_spk(const float *__restrict__ qlat, const float *__restrict__ qpe,
              const pf_bf *__restrict__ lat_pack, const pf_bf *__restrict__ pe_pack,
              const uint8_t *__restrict__ mask_pack,
              const pf_bf *__restrict__ sink_lat, const pf_bf *__restrict__ sink_pe,
              const int32_t *__restrict__ ucnt, const int32_t *__restrict__ packoff,
              int T, float *__restrict__ out)
{
    const int h  = blockIdx.y;
    const int q0 = blockIdx.x * TQ;
    const int l  = threadIdx.x;
    const float qs = rsqrtf((float)PF_QK_DIM) * 1.44269504088896340736f;

    float qv[TQ][PF_DPL], qp[TQ][PF_PEPL], acc[TQ][PF_DPL];
    float m_run[TQ], l_run[TQ];
    #pragma unroll
    for (int t = 0; t < TQ; t++) {
        const int qi = q0 + t;
        const bool live = (qi < T);
        const float *ql = qlat + ((size_t)qi * PF_NH + h) * PF_KVLORA;
        const float *qr = qpe  + ((size_t)qi * PF_NH + h) * PF_PE;
        if (live) {
            const float4 *q4 = (const float4 *)(ql + l * PF_DPL);
            const float4 a = q4[0], b = q4[1];
            qv[t][0]=a.x*qs; qv[t][1]=a.y*qs; qv[t][2]=a.z*qs; qv[t][3]=a.w*qs;
            qv[t][4]=b.x*qs; qv[t][5]=b.y*qs; qv[t][6]=b.z*qs; qv[t][7]=b.w*qs;
        } else {
            #pragma unroll
            for (int d = 0; d < PF_DPL; d++) qv[t][d] = 0.f;
        }
        #pragma unroll
        for (int d = 0; d < PF_PEPL; d++) qp[t][d] = live ? qr[d*PF_LANES+l]*qs : 0.f;
        #pragma unroll
        for (int d = 0; d < PF_DPL; d++) acc[t][d] = 0.f;
        m_run[t] = -1e30f; l_run[t] = 0.f;
    }

#define PFSK_LOAD(CLAT, CPE, KV, KP)                                               { const uint4 kr = *(const uint4 *)((CLAT) + l * PF_DPL);                        KV[0]=__uint_as_float(kr.x<<16); KV[1]=__uint_as_float(kr.x&0xFFFF0000u);      KV[2]=__uint_as_float(kr.y<<16); KV[3]=__uint_as_float(kr.y&0xFFFF0000u);      KV[4]=__uint_as_float(kr.z<<16); KV[5]=__uint_as_float(kr.z&0xFFFF0000u);      KV[6]=__uint_as_float(kr.w<<16); KV[7]=__uint_as_float(kr.w&0xFFFF0000u);      _Pragma("unroll")                                                              for (int d = 0; d < PF_PEPL; d++) KP[d] = pf_b2f((CPE)[d*PF_LANES + l]); }

#define PFSK_STEP(KV, KP, LIVE_MASK)                                               { _Pragma("unroll")                                                              for (int t = 0; t < TQ; t++) {                                                   float s = 0.f;                                                                 _Pragma("unroll")                                                              for (int d = 0; d < PF_DPL; d++)  s = fmaf(qv[t][d], KV[d], s);                _Pragma("unroll")                                                              for (int d = 0; d < PF_PEPL; d++) s = fmaf(qp[t][d], KP[d], s);                s = pf_wave_sum(s);                                                            s = (((LIVE_MASK) >> t) & 1) ? s : -1e30f;                                     const float mn = fmaxf(m_run[t], s);                                           if (mn > m_run[t]) {                                                               const float al = exp2f(m_run[t] - mn);                                         l_run[t] *= al;                                                                _Pragma("unroll")                                                              for (int d = 0; d < PF_DPL; d++) acc[t][d] *= al;                              m_run[t] = mn;                                                             }                                                                              const float p = exp2f(s - mn);                                                 l_run[t] += p;                                                                 _Pragma("unroll")                                                              for (int d = 0; d < PF_DPL; d++) acc[t][d] = fmaf(p, KV[d], acc[t][d]);      } }

    {   // sinks: visible to every query, always, selection or not
        const int FULL = (1 << TQ) - 1;
        float kv0[PF_DPL], kp0[PF_PEPL], kv1[PF_DPL], kp1[PF_PEPL];
        int i = 0;
        for (; i + 1 < PF_SINKS; i += 2) {
            PFSK_LOAD(sink_lat + (size_t)i * PF_KVLORA,     sink_pe + (size_t)i * PF_PE,     kv0, kp0)
            PFSK_LOAD(sink_lat + (size_t)(i+1) * PF_KVLORA, sink_pe + (size_t)(i+1) * PF_PE, kv1, kp1)
            PFSK_STEP(kv0, kp0, FULL)
            PFSK_STEP(kv1, kp1, FULL)
        }
        for (; i < PF_SINKS; i++) {
            PFSK_LOAD(sink_lat + (size_t)i * PF_KVLORA, sink_pe + (size_t)i * PF_PE, kv0, kp0)
            PFSK_STEP(kv0, kp0, FULL)
        }
    }
    {   // the tile's packed union: three running pointers, no indices
        const int tile = blockIdx.x;
        const int n = ucnt[tile];
        const int off = packoff[tile];
        const pf_bf *lp = lat_pack + (size_t)off * PF_KVLORA;
        const pf_bf *pp = pe_pack  + (size_t)off * PF_PE;
        const uint8_t *mp = mask_pack + (size_t)off * 16;
        float kv0[PF_DPL], kp0[PF_PEPL], kv1[PF_DPL], kp1[PF_PEPL];
        int i = 0;
        for (; i + 1 < n; i += 2) {
            PFSK_LOAD(lp + (size_t)i * PF_KVLORA, pp + (size_t)i * PF_PE, kv0, kp0)
            PFSK_LOAD(lp + (size_t)(i+1) * PF_KVLORA, pp + (size_t)(i+1) * PF_PE, kv1, kp1)
            PFSK_STEP(kv0, kp0, (int)mp[(size_t)i * 16])
            PFSK_STEP(kv1, kp1, (int)mp[(size_t)(i+1) * 16])
        }
        for (; i < n; i++) {
            PFSK_LOAD(lp + (size_t)i * PF_KVLORA, pp + (size_t)i * PF_PE, kv0, kp0)
            PFSK_STEP(kv0, kp0, (int)mp[(size_t)i * 16])
        }
    }
#undef PFSK_LOAD
#undef PFSK_STEP

    #pragma unroll
    for (int t = 0; t < TQ; t++) {
        const int qi = q0 + t;
        if (qi >= T) continue;
        const float inv = l_run[t] > 0.f ? 1.f / l_run[t] : 0.f;
        float *o = out + ((size_t)qi * PF_NH + h) * PF_KVLORA;
        float4 *o4 = (float4 *)(o + l * PF_DPL);
        o4[0] = make_float4(acc[t][0]*inv, acc[t][1]*inv, acc[t][2]*inv, acc[t][3]*inv);
        o4[1] = make_float4(acc[t][4]*inv, acc[t][5]*inv, acc[t][6]*inv, acc[t][7]*inv);
    }
}

// ===========================================================================
// G=2 HEAD-SHARED SPARSE ATTENTION (09-09)
//
// The discriminator settled the trade: at 64K occupancy dominates traffic
// (TQ=2/3 waves 669 tok/s > TQ=4/2 waves 597 > TQ=8/1 wave 419), AND the
// attention kernel already runs at ~75-80% of its byte floor - so the next
// lever is traffic WITHOUT losing the third wave.
//
// The one structure that buys both: the DSA selection is per QUERY, not per
// head - all 48 heads of a query attend the SAME top-2048 keys. A block that
// serves TWO heads of ONE query (TQ=1, G=2) holds exactly the state of the
// TQ=2 kernel (2 x 8 latent dims of qv, 2 x 8 of acc, 2 m/l pairs - named
// state 36 registers, same as TQ=2) but each key load now feeds two heads, so
// the 48-head re-read factor halves: DSA key traffic drops ~34% at the same
// 3 waves/SIMD. No union and no masks exist here at all - the selection is the
// key list, every selected key attends, and the packed streams are addressed
// by a CONSTANT stride (q * want entries), so the block has zero per-block
// loads. Sliding-window layers stay on k_pf_attn2<2>, which this changes
// nothing about.
// ===========================================================================

// pack one query's selection into contiguous lat/pe streams
// grid (RC), block 256. Entry = 1024 B lat + 128 B pe. No masks: TQ=1 means
// every selected key attends, for BOTH heads sharing the tile.
__global__ void __launch_bounds__(256)
k_pf_sel_pack(const int32_t *__restrict__ sel, const int32_t *__restrict__ nsel,
              int T, int want,
              const pf_bf *__restrict__ lat, const pf_bf *__restrict__ kpe,
              pf_bf *__restrict__ lat_pack, pf_bf *__restrict__ pe_pack)
{
    const int q = blockIdx.x;
    if (q >= T) return;
    const int n = nsel[q];
    const int f = threadIdx.x & 63;
    const int w = threadIdx.x >> 6;
    const int32_t *sq = sel + (size_t)q * want;
    for (int s0 = w; s0 < n; s0 += 4) {
        const int k = sq[s0];
        ((uint4 *)(lat_pack + ((size_t)q * want + s0) * PF_KVLORA))[f] =
            ((const uint4 *)(lat + (size_t)k * PF_KVLORA))[f];
        if (f < PF_PE / 8)
            ((uint4 *)(pe_pack + ((size_t)q * want + s0) * PF_PE))[f] =
                ((const uint4 *)(kpe + (size_t)k * PF_PE))[f];
    }
}

// Sparse attention, G=2 heads x TQ=1 query per block, over the packed
// selection streams. Same mathematics as k_pf_attn_spk with the mask and the
// union machinery deleted and the head loop replacing the query loop.
__global__ void __launch_bounds__(PF_LANES)
k_pf_attn_spg2(const float *__restrict__ qlat, const float *__restrict__ qpe,
               const pf_bf *__restrict__ lat_pack, const pf_bf *__restrict__ pe_pack,
               const pf_bf *__restrict__ sink_lat, const pf_bf *__restrict__ sink_pe,
               const int32_t *__restrict__ nsel, int want, int T,
               float *__restrict__ out)
{
    const int hp = blockIdx.y;          // head pair, 0..23
    const int q  = blockIdx.x;          // query
    const int l  = threadIdx.x;
    const float qs = rsqrtf((float)PF_QK_DIM) * 1.44269504088896340736f;
    // the walk bound is the live selection count, clamped to the stream
    // capacity. One dependent load per BLOCK, not per key.
    const int n = min(nsel[q], want);

    float qv[2][PF_DPL], qp[2][PF_PEPL], acc[2][PF_DPL];
    float m_run[2], l_run[2];
    #pragma unroll
    for (int g = 0; g < 2; g++) {
        const int h = hp * 2 + g;
        const float *ql = qlat + ((size_t)q * PF_NH + h) * PF_KVLORA;
        const float *qr = qpe  + ((size_t)q * PF_NH + h) * PF_PE;
        const float4 *q4 = (const float4 *)(ql + l * PF_DPL);
        const float4 a = q4[0], b = q4[1];
        qv[g][0]=a.x*qs; qv[g][1]=a.y*qs; qv[g][2]=a.z*qs; qv[g][3]=a.w*qs;
        qv[g][4]=b.x*qs; qv[g][5]=b.y*qs; qv[g][6]=b.z*qs; qv[g][7]=b.w*qs;
        #pragma unroll
        for (int d = 0; d < PF_PEPL; d++) qp[g][d] = qr[d*PF_LANES+l]*qs;
        #pragma unroll
        for (int d = 0; d < PF_DPL; d++) acc[g][d] = 0.f;
        m_run[g] = -1e30f; l_run[g] = 0.f;
    }

#define PFSG_LOAD(CLAT, CPE, KV, KP)                                               { const uint4 kr = *(const uint4 *)((CLAT) + l * PF_DPL);                        KV[0]=__uint_as_float(kr.x<<16); KV[1]=__uint_as_float(kr.x&0xFFFF0000u);      KV[2]=__uint_as_float(kr.y<<16); KV[3]=__uint_as_float(kr.y&0xFFFF0000u);      KV[4]=__uint_as_float(kr.z<<16); KV[5]=__uint_as_float(kr.z&0xFFFF0000u);      KV[6]=__uint_as_float(kr.w<<16); KV[7]=__uint_as_float(kr.w&0xFFFF0000u);      _Pragma("unroll")                                                              for (int d = 0; d < PF_PEPL; d++) KP[d] = pf_b2f((CPE)[d*PF_LANES + l]); }

#define PFSG_STEP(KV, KP)                                                          { _Pragma("unroll")                                                              for (int g = 0; g < 2; g++) {                                                    float s = 0.f;                                                                 _Pragma("unroll")                                                              for (int d = 0; d < PF_DPL; d++)  s = fmaf(qv[g][d], KV[d], s);                _Pragma("unroll")                                                              for (int d = 0; d < PF_PEPL; d++) s = fmaf(qp[g][d], KP[d], s);                s = pf_wave_sum(s);                                                            const float mn = fmaxf(m_run[g], s);                                           if (mn > m_run[g]) {                                                               const float al = exp2f(m_run[g] - mn);                                         l_run[g] *= al;                                                                _Pragma("unroll")                                                              for (int d = 0; d < PF_DPL; d++) acc[g][d] *= al;                              m_run[g] = mn;                                                             }                                                                              const float p = exp2f(s - mn);                                                 l_run[g] += p;                                                                 _Pragma("unroll")                                                              for (int d = 0; d < PF_DPL; d++) acc[g][d] = fmaf(p, KV[d], acc[g][d]);      } }

    {   // sinks: visible to every head, always
        float kv0[PF_DPL], kp0[PF_PEPL], kv1[PF_DPL], kp1[PF_PEPL];
        int i = 0;
        for (; i + 1 < PF_SINKS; i += 2) {
            PFSG_LOAD(sink_lat + (size_t)i * PF_KVLORA,     sink_pe + (size_t)i * PF_PE,     kv0, kp0)
            PFSG_LOAD(sink_lat + (size_t)(i+1) * PF_KVLORA, sink_pe + (size_t)(i+1) * PF_PE, kv1, kp1)
            PFSG_STEP(kv0, kp0)
            PFSG_STEP(kv1, kp1)
        }
        for (; i < PF_SINKS; i++) {
            PFSG_LOAD(sink_lat + (size_t)i * PF_KVLORA, sink_pe + (size_t)i * PF_PE, kv0, kp0)
            PFSG_STEP(kv0, kp0)
        }
    }
    {   // the query's packed selection: two running pointers, constant stride
        const pf_bf *lp = lat_pack + (size_t)q * want * PF_KVLORA;
        const pf_bf *pp = pe_pack  + (size_t)q * want * PF_PE;
        float kv0[PF_DPL], kp0[PF_PEPL], kv1[PF_DPL], kp1[PF_PEPL];
        int i = 0;
        for (; i + 1 < n; i += 2) {
            PFSG_LOAD(lp + (size_t)i * PF_KVLORA, pp + (size_t)i * PF_PE, kv0, kp0)
            PFSG_LOAD(lp + (size_t)(i+1) * PF_KVLORA, pp + (size_t)(i+1) * PF_PE, kv1, kp1)
            PFSG_STEP(kv0, kp0)
            PFSG_STEP(kv1, kp1)
        }
        for (; i < n; i++) {
            PFSG_LOAD(lp + (size_t)i * PF_KVLORA, pp + (size_t)i * PF_PE, kv0, kp0)
            PFSG_STEP(kv0, kp0)
        }
    }
#undef PFSG_LOAD
#undef PFSG_STEP

    #pragma unroll
    for (int g = 0; g < 2; g++) {
        const int h = hp * 2 + g;
        const float inv = l_run[g] > 0.f ? 1.f / l_run[g] : 0.f;
        float *o = out + ((size_t)q * PF_NH + h) * PF_KVLORA;
        float4 *o4 = (float4 *)(o + l * PF_DPL);
        o4[0] = make_float4(acc[g][0]*inv, acc[g][1]*inv, acc[g][2]*inv, acc[g][3]*inv);
        o4[1] = make_float4(acc[g][4]*inv, acc[g][5]*inv, acc[g][6]*inv, acc[g][7]*inv);
    }
}

// TQ=8 exists to DISCRIMINATE, not to ship. Attention is 59.3% of card time at
// 64K and its arithmetic intensity is 4x1088 FMA against a 1152-byte key load =
// 3.8 FMA/byte, against a machine balance near 8.8 - so it should be MEMORY
// bound, and the grid (tiles, 48) means every key is read 48 times because MLA
// shares one KV head across 48 query heads.
//
// If traffic is the binding constraint, TQ=8 must WIN despite halving occupancy
// to 1 wave/SIMD, because it serves twice the queries per key load and the union
// grows only slightly. If occupancy is still binding, it loses. One A/B settles
// which, and it settles whether a cooperative multi-wave kernel - which was
// refuted on OCCUPANCY grounds and would now be justified on TRAFFIC grounds -
// is worth building at all.
template __global__ void k_pf_union<2>(const int32_t*, const int32_t*, int, int, int,
    int32_t*, uint8_t*, int32_t*, int);
template __global__ void k_pf_union<4>(const int32_t*, const int32_t*, int, int, int,
    int32_t*, uint8_t*, int32_t*, int);
template __global__ void k_pf_union<8>(const int32_t*, const int32_t*, int, int, int,
    int32_t*, uint8_t*, int32_t*, int);
template __global__ void k_pf_attn_sp<2>(const float*, const float*, const pf_bf*,
    const pf_bf*, const pf_bf*, const pf_bf*, const int32_t*, const uint8_t*,
    const int32_t*, int, int, float*);
template __global__ void k_pf_attn_sp<4>(const float*, const float*, const pf_bf*,
    const pf_bf*, const pf_bf*, const pf_bf*, const int32_t*, const uint8_t*,
    const int32_t*, int, int, float*);
template __global__ void k_pf_attn_sp<8>(const float*, const float*, const pf_bf*,
    const pf_bf*, const pf_bf*, const pf_bf*, const int32_t*, const uint8_t*,
    const int32_t*, int, int, float*);
template __global__ void k_pf_attn_spk<2>(const float*, const float*, const pf_bf*,
    const pf_bf*, const uint8_t*, const pf_bf*, const pf_bf*, const int32_t*,
    const int32_t*, int, float*);
template __global__ void k_pf_attn_spk<4>(const float*, const float*, const pf_bf*,
    const pf_bf*, const uint8_t*, const pf_bf*, const pf_bf*, const int32_t*,
    const int32_t*, int, float*);
template __global__ void k_pf_attn_spk<8>(const float*, const float*, const pf_bf*,
    const pf_bf*, const uint8_t*, const pf_bf*, const pf_bf*, const int32_t*,
    const int32_t*, int, float*);

#include "p92amd/p92_pf.h"
PF_ATTN_DISPATCH
