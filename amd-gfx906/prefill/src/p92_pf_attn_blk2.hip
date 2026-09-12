// Blocked MLA attention v2 for openPangu prefill on gfx906: ONE block per query holding all
// 48 heads, fp16 key tiles in LDS, packed-math scores.
//
// WHY v2. v1 (24 heads x 32 lanes, bf16 tiles) cut key traffic 12x and moved the attention bucket
// only 11 % at 1K: the production kernels were never traffic-bound at that context (the window
// sits in L2). They are ISSUE-bound: every bf16 element costs an unpack op before its FMA, and the
// cross-lane reductions ride the LDS path (ds_bpermute) inside a dependent chain that lockstep
// waves cannot hide. v1 had the same instruction mix per MAC.
//
// WHAT CHANGES. The tile is converted bf16 -> fp16 ONCE while it is staged (576 conversions a
// key, against 48 x 1088 unpacks a key in v1), so the score is 18 v_dot2_f32_f16 per lane (two
// MACs an instruction, no unpack) and the value product is fp16 x fp32 -> fp32 (v_fma_mix or
// cvt + fma; fp32 accumulation, never fp16). Sixteen lanes per head row: the row sum is four
// DPP row-rotate adds with no LDS-path op at all. All 48 heads in one 768-thread block, so each
// key is staged exactly once per query.
//
// GEOMETRY. Lane l (0..15) of head row h owns latent dims [32l, 32l+32) (four 16-byte LDS reads
// at a 16-byte lane stride) and rope dims [4l, 4l+4). Named state: q 16 packed + rope 2 packed +
// acc 32 fp32 + m/l 2 = 52 VGPRs; budget for 3 waves/SIMD (the whole 12-wave block resident) is 84.
//
// TILES, SINKS, MODES, MATHEMATICS: as v1 (TK keys per LDS buffer, two buffers, prefetch under
// the compute; sinks first; window [kstart, qabs] or the query's selection list; log2-unit scores,
// online softmax with a gated rescale, out = sum p * latent / sum p). Precision: q, K and V enter
// the products as fp16 (11-bit mantissa; the bf16 cache has 8), accumulations are fp32.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#define PFB2_LPR  16                      // lanes per head-row
#define PFB2_HB   24                      // heads per block (48 spilled 14 VGPRs at the 3-wave budget)
#define PFB2_NT   (PFB2_HB * PFB2_LPR)    // 384 threads, 6 waves
#define PFB2_TK   16                      // keys per LDS tile
#define PFB2_ROW  (PF_KVLORA + PF_PE)     // 576 halfs = 1152 B per key
#define PFB2_U4   (PFB2_ROW / 8)          // 72 x 16 B per key (1152 a tile: three passes over 384 threads)

typedef _Float16 pfb2_h2 __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ float pfb2_dot2(uint32_t a, uint32_t b, float c) {
    pfb2_h2 ha, hb;
    __builtin_memcpy(&ha, &a, 4); __builtin_memcpy(&hb, &b, 4);
    return __builtin_amdgcn_fdot2(ha, hb, c, false);
}
static __device__ __forceinline__ uint32_t pfb2_pack(float lo, float hi) {
    pfb2_h2 h; h.x = (_Float16)lo; h.y = (_Float16)hi;
    uint32_t u; __builtin_memcpy(&u, &h, 4); return u;
}
static __device__ __forceinline__ float pfb2_lo(uint32_t u) {
    pfb2_h2 h; __builtin_memcpy(&h, &u, 4); return (float)h.x;
}
static __device__ __forceinline__ float pfb2_hi(uint32_t u) {
    pfb2_h2 h; __builtin_memcpy(&h, &u, 4); return (float)h.y;
}
// four bf16 (two dwords) -> four fp16 (two dwords)
static __device__ __forceinline__ void pfb2_cvt2(uint32_t b0, uint32_t b1, uint32_t &o0, uint32_t &o1) {
    o0 = pfb2_pack(__uint_as_float(b0 << 16), __uint_as_float(b0 & 0xFFFF0000u));
    o1 = pfb2_pack(__uint_as_float(b1 << 16), __uint_as_float(b1 & 0xFFFF0000u));
}
// 16-lane row sum by DPP row rotations (row_ror:8,4,2,1); every lane ends with the full sum
#define PFB2_DPP(x, c) __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(x), __float_as_int(x), (c), 0xF, 0xF, false))
static __device__ __forceinline__ float pfb2_row_sum(float s) {
    s += PFB2_DPP(s, 0x128);   // row_ror:8
    s += PFB2_DPP(s, 0x124);   // row_ror:4
    s += PFB2_DPP(s, 0x122);   // row_ror:2
    s += PFB2_DPP(s, 0x121);   // row_ror:1
    return s;
}

// grid (T, 48/PFB2_HB), block PFB2_NT. Arguments as k_pf_attn_blk.
__global__ void __launch_bounds__(PFB2_NT)
k_pf_attn_blk2(const float *__restrict__ qlat, const float *__restrict__ qpe,
               const pf_bf *__restrict__ lat, const pf_bf *__restrict__ kpe,
               const pf_bf *__restrict__ sink_lat, const pf_bf *__restrict__ sink_pe,
               const int32_t *__restrict__ sel, const int32_t *__restrict__ nsel, int want,
               int chunk_base, int T, int win, float *__restrict__ out)
{
    __shared__ __align__(16) uint16_t Kt[2][PFB2_TK * PFB2_ROW];      // fp16, 2 x 18,432 B
    const int q   = blockIdx.x;
    const int tid = threadIdx.x;
    const int h   = blockIdx.y * PFB2_HB + (tid >> 4);   // head = row
    const int l   = tid & 15;                 // lane within the row
    const float qs = rsqrtf((float)PF_QK_DIM) * 1.44269504088896340736f;

    const int qabs = chunk_base + q;
    int nkeys, kfirst = 0;
    const int32_t *sq = nullptr;
    if (sel) { nkeys = min(nsel[q], want); sq = sel + (size_t)q * want; }
    else { kfirst = win ? max(qabs - win + 1, 0) : 0; nkeys = qabs - kfirst + 1; }
    const int nsink_tiles = PF_SINKS / PFB2_TK;
    const int ntiles = nsink_tiles + (nkeys + PFB2_TK - 1) / PFB2_TK;

    // ---- the query slice: 32 latent dims + 4 rope dims, prescaled, packed fp16 ----
    uint32_t qv[16], qp[2];
    {
        const float *ql = qlat + ((size_t)q * PF_NH + h) * PF_KVLORA + l * 32;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const float4 v = ((const float4 *)ql)[i];
            qv[2 * i]     = pfb2_pack(v.x * qs, v.y * qs);
            qv[2 * i + 1] = pfb2_pack(v.z * qs, v.w * qs);
        }
        const float4 r = *(const float4 *)(qpe + ((size_t)q * PF_NH + h) * PF_PE + l * 4);
        qp[0] = pfb2_pack(r.x * qs, r.y * qs); qp[1] = pfb2_pack(r.z * qs, r.w * qs);
    }
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.f;
    float m_run = -1e30f, l_run = 0.f;

    auto key_ptrs = [&](int t, int j, const pf_bf *&kl, const pf_bf *&kp) -> bool {
        if (t < nsink_tiles) { const int k = t * PFB2_TK + j;
            kl = sink_lat + (size_t)k * PF_KVLORA; kp = sink_pe + (size_t)k * PF_PE; return true; }
        const int i = (t - nsink_tiles) * PFB2_TK + j;
        if (i >= nkeys) return false;
        const int k = sel ? sq[i] : (kfirst + i);
        kl = lat + (size_t)k * PF_KVLORA; kp = kpe + (size_t)k * PF_PE; return true;
    };
    // staging: 1152 x 16 B a tile over 768 threads, converted to fp16 on the way into LDS
    uint4 st[3];
    auto fetch = [&](int t) {
        #pragma unroll
        for (int p = 0; p < 3; p++) {
            st[p] = make_uint4(0,0,0,0);
            const int idx = tid + p * PFB2_NT;
            if (idx < PFB2_TK * PFB2_U4) {
                const int j = idx / PFB2_U4, part = idx - j * PFB2_U4; const pf_bf *kl, *kp;
                if (key_ptrs(t, j, kl, kp))
                    st[p] = (part < PF_KVLORA / 8) ? *(const uint4 *)(kl + part * 8) : *(const uint4 *)(kp + (part - PF_KVLORA / 8) * 8);
            }
        }
    };
    auto store = [&](int buf) {
        #pragma unroll
        for (int p = 0; p < 3; p++) {
            const int idx = tid + p * PFB2_NT;
            if (idx < PFB2_TK * PFB2_U4) {
                const int j = idx / PFB2_U4, part = idx - j * PFB2_U4; uint4 o;
                pfb2_cvt2(st[p].x, st[p].y, o.x, o.y); pfb2_cvt2(st[p].z, st[p].w, o.z, o.w);
                *(uint4 *)(&Kt[buf][j * PFB2_ROW + part * 8]) = o;
            }
        }
    };

    fetch(0); store(0);
    __syncthreads();
    for (int t = 0; t < ntiles; t++) {
        const int buf = t & 1;
        if (t + 1 < ntiles) fetch(t + 1);
        const int nk = (t < nsink_tiles) ? PFB2_TK : min(PFB2_TK, nkeys - (t - nsink_tiles) * PFB2_TK);
        const uint16_t *Kb = Kt[buf];
        for (int j = 0; j < nk; j++) {
            const uint16_t *kr = Kb + j * PFB2_ROW + l * 32;
            const uint4 k0 = ((const uint4 *)kr)[0], k1 = ((const uint4 *)kr)[1];
            const uint4 k2 = ((const uint4 *)kr)[2], k3 = ((const uint4 *)kr)[3];
            const uint2 kp = *(const uint2 *)(Kb + j * PFB2_ROW + PF_KVLORA + l * 4);
            float s = 0.f;
            s = pfb2_dot2(qv[0],  k0.x, s); s = pfb2_dot2(qv[1],  k0.y, s); s = pfb2_dot2(qv[2],  k0.z, s); s = pfb2_dot2(qv[3],  k0.w, s);
            s = pfb2_dot2(qv[4],  k1.x, s); s = pfb2_dot2(qv[5],  k1.y, s); s = pfb2_dot2(qv[6],  k1.z, s); s = pfb2_dot2(qv[7],  k1.w, s);
            s = pfb2_dot2(qv[8],  k2.x, s); s = pfb2_dot2(qv[9],  k2.y, s); s = pfb2_dot2(qv[10], k2.z, s); s = pfb2_dot2(qv[11], k2.w, s);
            s = pfb2_dot2(qv[12], k3.x, s); s = pfb2_dot2(qv[13], k3.y, s); s = pfb2_dot2(qv[14], k3.z, s); s = pfb2_dot2(qv[15], k3.w, s);
            s = pfb2_dot2(qp[0], kp.x, s); s = pfb2_dot2(qp[1], kp.y, s);
            s = pfb2_row_sum(s);
            const float mn = fmaxf(m_run, s);
            if (mn > m_run) {
                const float al = exp2f(m_run - mn);
                l_run *= al;
                #pragma unroll
                for (int i = 0; i < 32; i++) acc[i] *= al;
                m_run = mn;
            }
            const float p = exp2f(s - mn);
            l_run += p;
            #define PFB2_PV(o, w) { acc[(o)] = fmaf(p, pfb2_lo(w), acc[(o)]); acc[(o) + 1] = fmaf(p, pfb2_hi(w), acc[(o) + 1]); }
            PFB2_PV(0, k0.x)  PFB2_PV(2, k0.y)  PFB2_PV(4, k0.z)  PFB2_PV(6, k0.w)
            PFB2_PV(8, k1.x)  PFB2_PV(10, k1.y) PFB2_PV(12, k1.z) PFB2_PV(14, k1.w)
            PFB2_PV(16, k2.x) PFB2_PV(18, k2.y) PFB2_PV(20, k2.z) PFB2_PV(22, k2.w)
            PFB2_PV(24, k3.x) PFB2_PV(26, k3.y) PFB2_PV(28, k3.z) PFB2_PV(30, k3.w)
            #undef PFB2_PV
        }
        if (t + 1 < ntiles) store(buf ^ 1);
        __syncthreads();
    }

    const float inv = l_run > 0.f ? 1.f / l_run : 0.f;
    float4 *o = (float4 *)(out + ((size_t)q * PF_NH + h) * PF_KVLORA + l * 32);
    #pragma unroll
    for (int i = 0; i < 8; i++)
        o[i] = make_float4(acc[4*i]*inv, acc[4*i+1]*inv, acc[4*i+2]*inv, acc[4*i+3]*inv);
}
