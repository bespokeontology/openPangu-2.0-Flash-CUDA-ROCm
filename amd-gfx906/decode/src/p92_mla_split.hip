#pragma once
// Split-K absorbed MLA decode. Replaces the block-per-head form, which ran at
// 3182 us/layer because 48 blocks on 60 CUs each walked 2176 positions
// serially behind two barriers per four positions.
//
// Here one WAVE owns one (head, split): the whole score reduction is a 64-lane
// shuffle, so the kernel takes no barrier at all, and the grid is heads x
// splits, which fills the machine. Each wave keeps its own online-softmax
// state and writes a partial (acc, m, l); a combine pass merges the splits
// with the standard rescale, which is exact.
#include <hip/hip_runtime.h>
#include <cstdint>
#ifndef P92_NH
#define P92_NH        48
#define P92_KVLORA   512
#define P92_PE        64
#define P92_SINKS    128
#define P92_QK_DIM   192
typedef uint16_t p92_bf;
static __device__ __forceinline__ float p92_bf2f(p92_bf b){ return __uint_as_float((uint32_t)b<<16); }
#endif

#include "p92amd/p92_mla_geometry.h"
#define P92_DPL      (P92_KVLORA / 64)     // 8 latent dims per lane
#define P92_STILE    4

// partials: [head][split][P92_KVLORA] acc, plus [head][split] m and l
__global__ void __launch_bounds__(256)
k_p92_mla_split(const float *__restrict__ qlat, const float *__restrict__ qpe_a,
                const p92_bf *__restrict__ lat, const p92_bf *__restrict__ kpe,
                const p92_bf *__restrict__ sink_lat, const p92_bf *__restrict__ sink_pe,
                const int *__restrict__ sel, int nsel, int nsink,
                float *__restrict__ pacc, float *__restrict__ pm, float *__restrict__ pl) {
    const int wave = threadIdx.x >> 6;                 // 4 waves per block
    const int lane = threadIdx.x & 63;
    const int h    = blockIdx.x;
    const int sp   = blockIdx.y * 4 + wave;            // gridDim.y * 4 == P92_SPLITS
    if (sp >= P92_SPLITS) return;

    const float *q   = qlat  + (size_t)h * P92_KVLORA;
    const float *qpe = qpe_a + (size_t)h * P92_PE;
    const float scale = rsqrtf((float)P92_QK_DIM);

    // The sinks go entirely to split 0; the body is divided evenly.
    const int total = nsel;
    const int per   = (total + P92_SPLITS - 1) / P92_SPLITS;
    const int lo    = sp * per;
    int hi = lo + per; if (hi > total) hi = total;

    float acc[P92_DPL];
    #pragma unroll
    for (int i = 0; i < P92_DPL; i++) acc[i] = 0.f;
    float m_run = -1e30f, l_run = 0.f;

    float qv[P92_DPL];
    #pragma unroll
    for (int i = 0; i < P92_DPL; i++) qv[i] = q[lane + i * 64];
    const float qp = qpe[lane];                        // 64 rope dims, one a lane

    float v[P92_STILE][P92_DPL], sc[P92_STILE];

    // ---- sinks, split 0 only --------------------------------------------
    if (sp == 0) {
        for (int j0 = 0; j0 < nsink; j0 += P92_STILE) {
            int nt = nsink - j0; if (nt > P92_STILE) nt = P92_STILE;
            for (int u = 0; u < nt; u++) {
                const p92_bf *c = sink_lat + (size_t)(j0 + u) * P92_KVLORA;
                float s = 0.f;
                #pragma unroll
                for (int i = 0; i < P92_DPL; i++) {
                    v[u][i] = p92_bf2f(c[lane + i * 64]);
                    s = fmaf(qv[i], v[u][i], s);
                }
                s = fmaf(qp, p92_bf2f(sink_pe[(size_t)(j0 + u) * P92_PE + lane]), s);
                #pragma unroll
                for (int off = 32; off > 0; off >>= 1) s += __shfl_down(s, off, 64);
                sc[u] = __shfl(s, 0, 64) * scale;
            }
            for (int u = 0; u < nt; u++) {
                const float mn = fmaxf(m_run, sc[u]);
                const float al = __expf(m_run - mn), p = __expf(sc[u] - mn);
                l_run = l_run * al + p;
                #pragma unroll
                for (int i = 0; i < P92_DPL; i++) acc[i] = acc[i] * al + p * v[u][i];
                m_run = mn;
            }
        }
    }

    // ---- body slice ------------------------------------------------------
    for (int j0 = lo; j0 < hi; j0 += P92_STILE) {
        int nt = hi - j0; if (nt > P92_STILE) nt = P92_STILE;
        int tk[P92_STILE];
        for (int u = 0; u < nt; u++) tk[u] = sel[j0 + u];
        for (int u = 0; u < nt; u++) {
            if (tk[u] < 0) { sc[u] = -1e30f;
                #pragma unroll
                for (int i = 0; i < P92_DPL; i++) v[u][i] = 0.f;
                continue; }
            const p92_bf *c = lat + (size_t)tk[u] * P92_KVLORA;
            float s = 0.f;
            #pragma unroll
            for (int i = 0; i < P92_DPL; i++) {
                v[u][i] = p92_bf2f(c[lane + i * 64]);
                s = fmaf(qv[i], v[u][i], s);
            }
            s = fmaf(qp, p92_bf2f(kpe[(size_t)tk[u] * P92_PE + lane]), s);
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1) s += __shfl_down(s, off, 64);
            sc[u] = __shfl(s, 0, 64) * scale;
        }
        for (int u = 0; u < nt; u++) {
            if (tk[u] < 0) continue;
            const float mn = fmaxf(m_run, sc[u]);
            const float al = __expf(m_run - mn), p = __expf(sc[u] - mn);
            l_run = l_run * al + p;
            #pragma unroll
            for (int i = 0; i < P92_DPL; i++) acc[i] = acc[i] * al + p * v[u][i];
            m_run = mn;
        }
    }

    const size_t base = ((size_t)h * P92_SPLITS + sp) * P92_KVLORA;
    #pragma unroll
    for (int i = 0; i < P92_DPL; i++) pacc[base + lane + i * 64] = acc[i];
    if (lane == 0) { pm[h * P92_SPLITS + sp] = m_run; pl[h * P92_SPLITS + sp] = l_run; }
}

// Merge the split partials: one block per head, 256 threads, two dims each.
__global__ void __launch_bounds__(256)
k_p92_mla_combine(const float *__restrict__ pacc, const float *__restrict__ pm,
                  const float *__restrict__ pl, float *__restrict__ ctx) {
    const int h = blockIdx.x, t = threadIdx.x;
    float m = -1e30f;
    #pragma unroll
    for (int s = 0; s < P92_SPLITS; s++) m = fmaxf(m, pm[h * P92_SPLITS + s]);
    float den = 0.f;
    #pragma unroll
    for (int s = 0; s < P92_SPLITS; s++) den += pl[h * P92_SPLITS + s] * __expf(pm[h * P92_SPLITS + s] - m);
    const float inv = den > 0.f ? 1.f / den : 0.f;
    #pragma unroll
    for (int i = 0; i < P92_KVLORA / 256; i++) {
        const int d = t + i * 256;
        float a = 0.f;
        #pragma unroll
        for (int s = 0; s < P92_SPLITS; s++)
            a += pacc[((size_t)h * P92_SPLITS + s) * P92_KVLORA + d] * __expf(pm[h * P92_SPLITS + s] - m);
        ctx[(size_t)h * P92_KVLORA + d] = a * inv;
    }
}
