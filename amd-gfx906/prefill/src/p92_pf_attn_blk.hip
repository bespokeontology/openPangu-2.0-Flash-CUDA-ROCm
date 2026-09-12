// Blocked MLA attention for openPangu prefill on gfx906: the key is read ONCE per
// query (per head-half) and applied to 24 heads from LDS.
//
// WHY. MLA shares one 512-latent + 64-rope key across all 48 query heads, but the
// production kernels (k_pf_attn2: 2 queries x 1 head a block; k_pf_attn_spg2: 1
// query x 2 heads) re-read every key 24 times per query - 27.6 KB of L2/HBM
// traffic per (query, key) - and the receipts put them at 75-80% of that byte
// floor. The floor itself is the problem. A block that holds ONE query's 24 heads
// (12 waves, 32 lanes a head) stages each key tile into LDS once and sweeps it
// from there: 2 x 1152 B per (query, key) instead of 24 x 1152 B, and the LDS
// reads broadcast across the two heads of a wave.
//
// GEOMETRY. block = HB heads x 32 lanes = 768 threads, grid (T, 48/HB). Lane l of
// a head-row owns latent dims {8l..8l+7} u {256+8l..256+8l+7} (two 16-byte LDS
// reads at a 16-byte lane stride, conflict-free) and rope dims {2l, 2l+1}. Named
// state a lane: q 16 + qpe 2 + acc 16 + m/l 2 = 36 floats; the budget for 3
// waves/SIMD (12 resident waves = the whole block) is 84 VGPRs.
//
// TILES. TK keys x 1152 B of bf16 per LDS buffer, two buffers: the loads for tile
// t+1 are issued before tile t is consumed, stored after, one barrier a tile.
// Sinks come first (128, every query), then the keys: a contiguous causal range
// [kstart, qabs] (dense / sliding-window layers) or the query's selection list
// (DSA layers, gathered straight from the cache - no packed streams).
//
// MATHEMATICS. Identical to k_pf_attn2 / k_pf_attn_spg2: scores in log2 units
// (scale * log2e folded into the query), online softmax with a gated rescale,
// output = sum p * latent / sum p. Reduction order differs (32-lane butterfly),
// which this engine is explicitly allowed to do.
// The including TU must have included p92_pf_attn.hip first (PF_* constants, PF_DPPF).
#include <hip/hip_runtime.h>
#include <cstdint>

#define PFB_LPR   32                     // lanes per head-row
#define PFB_HB    24                     // heads per block
#define PFB_NT    (PFB_HB * PFB_LPR)     // 768 threads, 12 waves
#define PFB_TK    16                     // keys per LDS tile
#define PFB_ROW   (PF_KVLORA + PF_PE)    // 576 bf16 = 1152 B per key
#define PFB_U4    (PFB_ROW / 8)          // 72 uint4 per key

// 32-lane row sum: xor 16 and 4 on the LDS path, xor 8 / 2 / 1 on DPP.
static __device__ __forceinline__ float pfb_row_sum(float s) {
    s += __shfl_xor(s, 16, 32);
    { const float t = PF_DPPF(s, 0x140u); s += PF_DPPF(t, 0x141u); }  // xor 8
    s += __shfl_xor(s, 4, 32);
    s += PF_DPPF(s, 0x4Eu);                                           // xor 2
    s += PF_DPPF(s, 0xB1u);                                           // xor 1
    return s;
}

// grid (T, 48/PFB_HB), block PFB_NT.
//   qlat  [T][48][512] fp32   qpe [T][48][64] fp32
//   lat   [P][512] bf16 (absolute positions), kpe [P][64] bf16
//   sink_lat [128][512], sink_pe [128][64]
//   sel   nullptr: keys are the causal range [kstart, chunk_base + q];
//         else sel[q * want + i], i < min(nsel[q], want)  (DSA selection)
//   out   [T][48][512] fp32
__global__ void __launch_bounds__(PFB_NT)
k_pf_attn_blk(const float *__restrict__ qlat, const float *__restrict__ qpe,
              const pf_bf *__restrict__ lat, const pf_bf *__restrict__ kpe,
              const pf_bf *__restrict__ sink_lat, const pf_bf *__restrict__ sink_pe,
              const int32_t *__restrict__ sel, const int32_t *__restrict__ nsel, int want,
              int chunk_base, int T, int win, float *__restrict__ out)
{
    __shared__ __align__(16) pf_bf Kt[2][PFB_TK * PFB_ROW];          // 2 x 18,432 B
    const int q   = blockIdx.x;
    const int tid = threadIdx.x;
    const int r   = tid >> 5;                 // head within the block
    const int l   = tid & 31;                 // lane within the head-row
    const int h   = blockIdx.y * PFB_HB + r;
    const float qs = rsqrtf((float)PF_QK_DIM) * 1.44269504088896340736f;

    // ---- the key list ------------------------------------------------------
    const int qabs = chunk_base + q;
    int nkeys, kfirst = 0;
    const int32_t *sq = nullptr;
    if (sel) { nkeys = min(nsel[q], want); sq = sel + (size_t)q * want; }
    else { kfirst = win ? max(qabs - win + 1, 0) : 0; nkeys = qabs - kfirst + 1; }
    const int ntiles = (PF_SINKS + PFB_TK - 1) / PFB_TK + (nkeys + PFB_TK - 1) / PFB_TK;
    const int nsink_tiles = PF_SINKS / PFB_TK;

    // ---- the query slice, prescaled --------------------------------------
    float qv[16], qp[2], acc[16];
    {
        const float *ql = qlat + ((size_t)q * PF_NH + h) * PF_KVLORA;
        const float4 *a = (const float4 *)(ql + l * 8);
        const float4 *b = (const float4 *)(ql + 256 + l * 8);
        const float4 a0 = a[0], a1 = a[1], b0 = b[0], b1 = b[1];
        qv[0]=a0.x*qs; qv[1]=a0.y*qs; qv[2]=a0.z*qs; qv[3]=a0.w*qs;
        qv[4]=a1.x*qs; qv[5]=a1.y*qs; qv[6]=a1.z*qs; qv[7]=a1.w*qs;
        qv[8]=b0.x*qs; qv[9]=b0.y*qs; qv[10]=b0.z*qs; qv[11]=b0.w*qs;
        qv[12]=b1.x*qs; qv[13]=b1.y*qs; qv[14]=b1.z*qs; qv[15]=b1.w*qs;
        const float *qr = qpe + ((size_t)q * PF_NH + h) * PF_PE;
        qp[0] = qr[2 * l] * qs; qp[1] = qr[2 * l + 1] * qs;
        #pragma unroll
        for (int i = 0; i < 16; i++) acc[i] = 0.f;
    }
    float m_run = -1e30f, l_run = 0.f;

    // ---- tile staging: which key does tile t, slot j hold? ----------------
    auto key_ptrs = [&](int t, int j, const pf_bf *&kl, const pf_bf *&kp) -> bool {
        if (t < nsink_tiles) {
            const int k = t * PFB_TK + j;
            kl = sink_lat + (size_t)k * PF_KVLORA; kp = sink_pe + (size_t)k * PF_PE; return true;
        }
        const int i = (t - nsink_tiles) * PFB_TK + j;
        if (i >= nkeys) return false;
        const int k = sel ? sq[i] : (kfirst + i);
        kl = lat + (size_t)k * PF_KVLORA; kp = kpe + (size_t)k * PF_PE; return true;
    };
    // 1152 uint4 a tile over 768 threads: two passes, the second partial
    uint4 st0, st1;
    auto fetch = [&](int t) {
        st0 = make_uint4(0,0,0,0); st1 = make_uint4(0,0,0,0);
        { const int idx = tid, j = idx / PFB_U4, part = idx - j * PFB_U4;
          const pf_bf *kl, *kp;
          if (key_ptrs(t, j, kl, kp))
              st0 = (part < PF_KVLORA / 8) ? *(const uint4 *)(kl + part * 8)
                                           : *(const uint4 *)(kp + (part - PF_KVLORA / 8) * 8); }
        { const int idx = tid + PFB_NT;
          if (idx < PFB_TK * PFB_U4) {
              const int j = idx / PFB_U4, part = idx - j * PFB_U4;
              const pf_bf *kl, *kp;
              if (key_ptrs(t, j, kl, kp))
                  st1 = (part < PF_KVLORA / 8) ? *(const uint4 *)(kl + part * 8)
                                               : *(const uint4 *)(kp + (part - PF_KVLORA / 8) * 8); } }
    };
    auto store = [&](int buf) {
        { const int idx = tid, j = idx / PFB_U4, part = idx - j * PFB_U4;
          *(uint4 *)(&Kt[buf][j * PFB_ROW + part * 8]) = st0; }
        { const int idx = tid + PFB_NT;
          if (idx < PFB_TK * PFB_U4) { const int j = idx / PFB_U4, part = idx - j * PFB_U4;
              *(uint4 *)(&Kt[buf][j * PFB_ROW + part * 8]) = st1; } }
    };

    fetch(0); store(0);
    __syncthreads();
    for (int t = 0; t < ntiles; t++) {
        const int buf = t & 1;
        if (t + 1 < ntiles) fetch(t + 1);                       // in flight during this tile
        const int nk = (t < nsink_tiles) ? PFB_TK
                     : min(PFB_TK, nkeys - (t - nsink_tiles) * PFB_TK);
        const pf_bf *Kb = Kt[buf];
        for (int j = 0; j < nk; j++) {
            const pf_bf *kr = Kb + j * PFB_ROW;
            const uint4 ka = *(const uint4 *)(kr + l * 8);
            const uint4 kb = *(const uint4 *)(kr + 256 + l * 8);
            const uint32_t kpw = *(const uint32_t *)(kr + PF_KVLORA + 2 * l);
            float s = 0.f;
            s = fmaf(qv[0],  __uint_as_float(ka.x << 16), s); s = fmaf(qv[1],  __uint_as_float(ka.x & 0xFFFF0000u), s);
            s = fmaf(qv[2],  __uint_as_float(ka.y << 16), s); s = fmaf(qv[3],  __uint_as_float(ka.y & 0xFFFF0000u), s);
            s = fmaf(qv[4],  __uint_as_float(ka.z << 16), s); s = fmaf(qv[5],  __uint_as_float(ka.z & 0xFFFF0000u), s);
            s = fmaf(qv[6],  __uint_as_float(ka.w << 16), s); s = fmaf(qv[7],  __uint_as_float(ka.w & 0xFFFF0000u), s);
            s = fmaf(qv[8],  __uint_as_float(kb.x << 16), s); s = fmaf(qv[9],  __uint_as_float(kb.x & 0xFFFF0000u), s);
            s = fmaf(qv[10], __uint_as_float(kb.y << 16), s); s = fmaf(qv[11], __uint_as_float(kb.y & 0xFFFF0000u), s);
            s = fmaf(qv[12], __uint_as_float(kb.z << 16), s); s = fmaf(qv[13], __uint_as_float(kb.z & 0xFFFF0000u), s);
            s = fmaf(qv[14], __uint_as_float(kb.w << 16), s); s = fmaf(qv[15], __uint_as_float(kb.w & 0xFFFF0000u), s);
            s = fmaf(qp[0], __uint_as_float(kpw << 16), s);
            s = fmaf(qp[1], __uint_as_float(kpw & 0xFFFF0000u), s);
            s = pfb_row_sum(s);                                 // uniform across the 32 lanes of the row
            const float mn = fmaxf(m_run, s);
            if (mn > m_run) {                                   // rescale only when the max advances
                const float al = exp2f(m_run - mn);
                l_run *= al;
                #pragma unroll
                for (int i = 0; i < 16; i++) acc[i] *= al;
                m_run = mn;
            }
            const float p = exp2f(s - mn);
            l_run += p;
            acc[0]  = fmaf(p, __uint_as_float(ka.x << 16), acc[0]);  acc[1]  = fmaf(p, __uint_as_float(ka.x & 0xFFFF0000u), acc[1]);
            acc[2]  = fmaf(p, __uint_as_float(ka.y << 16), acc[2]);  acc[3]  = fmaf(p, __uint_as_float(ka.y & 0xFFFF0000u), acc[3]);
            acc[4]  = fmaf(p, __uint_as_float(ka.z << 16), acc[4]);  acc[5]  = fmaf(p, __uint_as_float(ka.z & 0xFFFF0000u), acc[5]);
            acc[6]  = fmaf(p, __uint_as_float(ka.w << 16), acc[6]);  acc[7]  = fmaf(p, __uint_as_float(ka.w & 0xFFFF0000u), acc[7]);
            acc[8]  = fmaf(p, __uint_as_float(kb.x << 16), acc[8]);  acc[9]  = fmaf(p, __uint_as_float(kb.x & 0xFFFF0000u), acc[9]);
            acc[10] = fmaf(p, __uint_as_float(kb.y << 16), acc[10]); acc[11] = fmaf(p, __uint_as_float(kb.y & 0xFFFF0000u), acc[11]);
            acc[12] = fmaf(p, __uint_as_float(kb.z << 16), acc[12]); acc[13] = fmaf(p, __uint_as_float(kb.z & 0xFFFF0000u), acc[13]);
            acc[14] = fmaf(p, __uint_as_float(kb.w << 16), acc[14]); acc[15] = fmaf(p, __uint_as_float(kb.w & 0xFFFF0000u), acc[15]);
        }
        if (t + 1 < ntiles) store(buf ^ 1);                    // waits for the prefetch, writes the other buffer
        __syncthreads();
    }

    const float inv = l_run > 0.f ? 1.f / l_run : 0.f;
    float *o = out + ((size_t)q * PF_NH + h) * PF_KVLORA;
    float4 *oa = (float4 *)(o + l * 8), *ob = (float4 *)(o + 256 + l * 8);
    oa[0] = make_float4(acc[0]*inv, acc[1]*inv, acc[2]*inv, acc[3]*inv);
    oa[1] = make_float4(acc[4]*inv, acc[5]*inv, acc[6]*inv, acc[7]*inv);
    ob[0] = make_float4(acc[8]*inv, acc[9]*inv, acc[10]*inv, acc[11]*inv);
    ob[1] = make_float4(acc[12]*inv, acc[13]*inv, acc[14]*inv, acc[15]*inv);
}
