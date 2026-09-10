#pragma once
// openPangu-2.0-Flash absorbed MLA decode attention, gfx906 wave64.
//
// Ported from the GLM gfx906 kernel dsa_decode_gfx906.hip:k_attend, which
// contributes the tiled single-pass online softmax: a tile of selected
// positions is loaded and scored together, every latent row is read exactly
// once and reused as its own V, and the running max and sum live in registers
// so there is no scores workspace in global memory.
//
// Four things differ from GLM and none of them are constants.
//
// 1. The rope term. GLM scores q.latent alone. openPangu carries a separate
//    64-dim rope key per position, so the score is q_lat.c_t + q_pe.k_pe_t.
//    It is folded in for free: the block already reduces a partial sum across
//    all 256 threads, so lanes 0..63 simply add their rope product into the
//    same partial. No extra barrier, no second reduction.
//
// 2. Sinks. Every layer prepends 128 sink positions, which are ordinary
//    scored entries drawn from a separate resident block rather than from the
//    KV cache. They run through the same online softmax before the body, so
//    the running max is already sane when the body starts.
//
// 3. Three addressing modes. A DSA layer (index % 3 == 0) takes a 2048-entry
//    selection list from the indexer; every other trunk layer takes a 512
//    sliding window; the window is circular in the cache. All three are the
//    same loop over a position list, so the mode is a host-side decision
//    about what the list contains, not a branch in the inner loop.
//
// 4. Geometry: 48 heads not 64, and the value side is a 128-dim head after
//    the up-projection rather than the 512-dim latent.
//
// One block per head, 256 threads, two latent dims a thread.
#include <hip/hip_runtime.h>
#include <cstdint>

#define P92_NH        48
#define P92_KVLORA   512
#define P92_PE        64
#define P92_SINKS    128
#define P92_QK_DIM   192          // qk_nope 128 + qk_rope 64
#define P92_NK       (P92_KVLORA / 256)     // 2 latent dims per thread
#define P92_ATT_TILE   4          // 8 overruns the gfx906 VGPR budget at 256 threads

typedef uint16_t p92_bf;
static __device__ __forceinline__ float p92_bf2f(p92_bf b) {
    return __uint_as_float((uint32_t)b << 16);
}

// Scored tile: accumulate q.latent into a per-thread partial, and on the first
// wave add the rope product for the same position.
#define P92_SCORE_TILE(GET_LAT, GET_PE, NT)                                    \
    for (int u = 0; u < (NT); u++) {                                           \
        float s = 0.f;                                                         \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < P92_NK; i++) {                                     \
            v[u][i] = GET_LAT(u, t + i * 256);                                 \
            s = fmaf(q[t + i * 256], v[u][i], s);                              \
        }                                                                      \
        if (t < P92_PE) s = fmaf(qpe[t], GET_PE(u, t), s);                     \
        a[u] = s;                                                              \
    }

// Reduce the 256 per-thread partials of each tile entry to one score.
#define P92_REDUCE_TILE(NT)                                                    \
    for (int u = 0; u < (NT); u++) {                                           \
        float r = a[u];                                                        \
        _Pragma("unroll")                                                      \
        for (int off = 32; off > 0; off >>= 1) r += __shfl_down(r, off, 64);   \
        if ((t & 63) == 0) sh[u * 4 + (t >> 6)] = r;                           \
    }                                                                          \
    __syncthreads();                                                           \
    if (t < (NT)) {                                                            \
        float s2 = 0.f;                                                        \
        _Pragma("unroll")                                                      \
        for (int w = 0; w < 4; w++) s2 += sh[t * 4 + w];                        \
        stile[t] = s2 * scale;                                                 \
    }                                                                          \
    __syncthreads();

// Fold a scored tile into the running online softmax.
#define P92_ACCUM_TILE(NT, SKIP)                                               \
    for (int u = 0; u < (NT); u++) {                                           \
        if (SKIP) continue;                                                    \
        const float s = stile[u];                                              \
        const float m_new = fmaxf(m_run, s);                                   \
        const float alpha = __expf(m_run - m_new);                             \
        const float p     = __expf(s - m_new);                                 \
        l_run = l_run * alpha + p;                                             \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < P92_NK; i++) acc[i] = acc[i] * alpha + p * v[u][i]; \
        m_run = m_new;                                                         \
    }                                                                          \
    __syncthreads();

// qlat  [48][512] fp32, absorbed query
// qpe_a [48][64]  fp32, rope query
// lat   [T][512]  bf16 kv cache latents      sink_lat [128][512] bf16
// kpe   [T][64]   bf16 rope key cache        sink_pe  [128][64]  bf16
// sel   [nsel]    int32 positions, -1 padded
// ctx   [48][512] fp32 out
__global__ void __launch_bounds__(256)
k_p92_mla_decode(const float *__restrict__ qlat, const float *__restrict__ qpe_a,
                 const p92_bf *__restrict__ lat, const p92_bf *__restrict__ kpe,
                 const p92_bf *__restrict__ sink_lat, const p92_bf *__restrict__ sink_pe,
                 const int *__restrict__ sel, int nsel, int nsink,
                 float *__restrict__ ctx) {
    const int h = blockIdx.x, t = threadIdx.x;
    const float *q   = qlat  + (size_t)h * P92_KVLORA;
    const float *qpe = qpe_a + (size_t)h * P92_PE;
    const float scale = rsqrtf((float)P92_QK_DIM);

    __shared__ float sh[P92_ATT_TILE * 4];
    __shared__ float stile[P92_ATT_TILE];

    float acc[P92_NK];
    #pragma unroll
    for (int i = 0; i < P92_NK; i++) acc[i] = 0.f;
    float m_run = -1e30f, l_run = 0.f;
    float v[P92_ATT_TILE][P92_NK], a[P92_ATT_TILE];

    // ---- sinks: always present, always dense ----------------------------
    for (int j0 = 0; j0 < nsink; j0 += P92_ATT_TILE) {
        int nt = nsink - j0; if (nt > P92_ATT_TILE) nt = P92_ATT_TILE;
#define SK_LAT(u, d) p92_bf2f(sink_lat[(size_t)(j0 + (u)) * P92_KVLORA + (d)])
#define SK_PE(u, d)  p92_bf2f(sink_pe [(size_t)(j0 + (u)) * P92_PE     + (d)])
        P92_SCORE_TILE(SK_LAT, SK_PE, nt)
#undef SK_LAT
#undef SK_PE
        P92_REDUCE_TILE(nt)
        P92_ACCUM_TILE(nt, false)
    }

    // ---- body: DSA selection list, or the sliding window, or full ------
    for (int j0 = 0; j0 < nsel; j0 += P92_ATT_TILE) {
        int nt = nsel - j0; if (nt > P92_ATT_TILE) nt = P92_ATT_TILE;
        int tok[P92_ATT_TILE];
        #pragma unroll
        for (int u = 0; u < P92_ATT_TILE; u++) tok[u] = (u < nt) ? sel[j0 + u] : -1;
#define BD_LAT(u, d) (tok[u] >= 0 ? p92_bf2f(lat[(size_t)tok[u] * P92_KVLORA + (d)]) : 0.f)
#define BD_PE(u, d)  (tok[u] >= 0 ? p92_bf2f(kpe[(size_t)tok[u] * P92_PE     + (d)]) : 0.f)
        P92_SCORE_TILE(BD_LAT, BD_PE, nt)
#undef BD_LAT
#undef BD_PE
        P92_REDUCE_TILE(nt)
        P92_ACCUM_TILE(nt, tok[u] < 0)
    }

    const float inv = (l_run > 0.f) ? 1.f / l_run : 0.f;
    #pragma unroll
    for (int i = 0; i < P92_NK; i++)
        ctx[(size_t)h * P92_KVLORA + t + i * 256] = acc[i] * inv;
}

// Build the body position list for one decode step.
//   DSA layer     : the indexer already wrote sel; this is not called.
//   sliding layer : the last `window` positions, circular in a cache of
//                   `cap` slots, oldest first so the order is deterministic.
__global__ void __launch_bounds__(256)
k_p92_mla_window(int kv_len, int window, int cap, int *__restrict__ sel,
                 int *__restrict__ nsel_out) {
    int n = kv_len < window ? kv_len : window;
    const int first = kv_len - n;
    for (int i = threadIdx.x; i < n; i += 256) sel[i] = (first + i) % cap;
    if (threadIdx.x == 0) *nsel_out = n;
}
