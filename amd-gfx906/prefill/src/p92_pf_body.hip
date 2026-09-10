#include "p92amd/p92_pf_moe.h"

// State-shaped layer work, batched over T rows. No reuse inversion here: mHC,
// the router and RoPE have no expensive operand worth making resident, only
// per-row state. Widening is the correct transform for these, which is why they
// look like the decode kernels with a row index and the expert/attention
// kernels do not.
#define PF_MHC_H     4
#define PF_MHC_FLAT  10240
#define PF_MHC_MIX   24
#define PF_MHC_ITERS 20
#define PF_MHC_EPS   1e-6f
#define PF_NEXP_R    256

static __device__ __forceinline__ float pb_b2f(uint16_t b){
    return __uint_as_float((uint32_t)b << 16); }

// phi projection: 24 logits a row from the 10240-wide four-stream block.
// grid (PF_MHC_MIX, T), block 256. One block per (logit, row): the row is read
// once and reduced, so phi streams and the state is the reused operand - the
// opposite of the expert kernels because here the STATE is small and shared.
__global__ void __launch_bounds__(256)
k_pf_mhc_logits(const uint16_t *__restrict__ x, const uint16_t *__restrict__ phi,
                const uint16_t *__restrict__ gam, int T, float *__restrict__ out)
{
    const int m = blockIdx.x, row = blockIdx.y;
    if (row >= T) return;
    const uint16_t *xr = x + (size_t)row * PF_MHC_FLAT;
    const uint16_t *pr = phi + (size_t)m * PF_MHC_FLAT;
    float acc = 0.f;
    for (int i = threadIdx.x; i < PF_MHC_FLAT; i += 256)
        acc = fmaf(pb_b2f(xr[i]) * pb_b2f(gam[i]), pb_b2f(pr[i]), acc);
    #pragma unroll
    for (int o = 32; o; o >>= 1) acc += __shfl_down(acc, o, 64);
    __shared__ float red[4];
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = acc;
    __syncthreads();
    if (threadIdx.x == 0)
        out[(size_t)row * PF_MHC_MIX + m] = red[0] + red[1] + red[2] + red[3];
}

// Sinkhorn + collapse, one 64-lane wave a row. The 4x4 matrix lives in 16 of
// those lanes, exactly as decode, so the DPP substitutions that pay there pay
// here too - but this is 512 independent waves rather than one, so the DS ops
// are hidden and the decode result does not transfer. Left as shfl until a
// composed profile says otherwise.
__global__ void __launch_bounds__(64)
k_pf_mhc_collapse(const float *__restrict__ logits, const uint16_t *__restrict__ alpha,
                  const uint16_t *__restrict__ beta, int T, float *__restrict__ comb)
{
    const int row = blockIdx.x;
    if (row >= T) return;
    const int l = threadIdx.x;
    const bool on = l < PF_MHC_H * PF_MHC_H;
    const int i = l >> 2, j = l & 3;
    const float a2 = pb_b2f(alpha[2]);
    float v = on ? logits[(size_t)row * PF_MHC_MIX + 2 * PF_MHC_H + l] * a2
                       + pb_b2f(beta[2 * PF_MHC_H + l]) : -3.0e38f;
    float mx = v;
    mx = fmaxf(mx, __shfl_xor(mx, 1, 64));
    mx = fmaxf(mx, __shfl_xor(mx, 2, 64));
    const float e = on ? __expf(v - mx) : 0.f;
    float s = e;
    s += __shfl_xor(s, 1, 64); s += __shfl_xor(s, 2, 64);
    v = on ? e / s + PF_MHC_EPS : 0.f;
    #define PB_COL(acc) { acc = v; acc += __shfl_xor(acc,4,64); acc += __shfl_xor(acc,8,64); }
    #define PB_ROW(acc) { acc = v; acc += __shfl_xor(acc,1,64); acc += __shfl_xor(acc,2,64); }
    { float cs; PB_COL(cs); cs += PF_MHC_EPS; if (on) v /= cs; }
    for (int it = 0; it < PF_MHC_ITERS - 1; it++) {
        { float rs; PB_ROW(rs); rs += PF_MHC_EPS; if (on) v /= rs; }
        { float cs; PB_COL(cs); cs += PF_MHC_EPS; if (on) v /= cs; }
    }
    #undef PB_COL
    #undef PB_ROW
    if (on) comb[(size_t)row * 16 + i * PF_MHC_H + j] = v;
}

// Router: 256 sigmoid scores a row, correction bias for SELECTION ONLY, top-8,
// renormalise, scale by 2.5. One wave a row, four experts a lane - the decode
// structure, which is already right for this shape because the work per row is
// tiny and T supplies the parallelism.
__global__ void __launch_bounds__(64)
// e_score_correction_bias is stored fp32 by the loader (LayerW::router_bias is
// a float*), and decode consumes it as fp32. Declaring it bf16 here meant the
// bench cast a float array to uint16_t and every correction term was the low
// half of a float read as a bf16 - garbage on the term that decides which eight
// experts run. Selection is the one place a wrong value does not average out.
k_pf_router(const float *__restrict__ logits, const float *__restrict__ bias,
            int T, int32_t *__restrict__ ids, float *__restrict__ wts)
{
    const int row = blockIdx.x;
    if (row >= T) return;
    const int l = threadIdx.x;
    const float *lr = logits + (size_t)row * PF_NEXP_R;
    float sc[4], cor[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const int e = k * 64 + l;
        sc[k]  = 1.f / (1.f + __expf(-lr[e]));
        cor[k] = sc[k] + bias[e];
    }
    int32_t chosen[PF_TOPK]; float taken[PF_TOPK];
    #pragma unroll
    for (int t = 0; t < PF_TOPK; t++) {
        float bv = -1e30f; int bi = -1;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            bool used = false;
            for (int u = 0; u < t; u++) if (chosen[u] == k * 64 + l) used = true;
            if (!used && cor[k] > bv) { bv = cor[k]; bi = k * 64 + l; }
        }
        #pragma unroll
        for (int o = 32; o; o >>= 1) {
            const float ov = __shfl_xor(bv, o, 64);
            const int   oi = __shfl_xor(bi, o, 64);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        chosen[t] = bi;
        const int lane = bi & 63, slot = bi >> 6;
        taken[t] = __shfl(sc[slot], lane, 64);
    }
    if (l == 0) {
        float sum = 0.f;
        #pragma unroll
        for (int t = 0; t < PF_TOPK; t++) sum += taken[t];
        const float inv = sum > 0.f ? 2.5f / sum : 0.f;
        #pragma unroll
        for (int t = 0; t < PF_TOPK; t++) {
            ids[(size_t)row * PF_TOPK + t] = chosen[t];
            wts[(size_t)row * PF_TOPK + t] = taken[t] * inv;
        }
    }
}

// RoPE over T rows. Pairs (i, i+32) of the first 64 dims, per head.
// grid (heads, T), block 32.
__global__ void __launch_bounds__(32)
k_pf_rope(float *__restrict__ q, const float *__restrict__ cosv,
          const float *__restrict__ sinv, int T, int heads, int dim, int base)
{
    const int h = blockIdx.x, row = blockIdx.y;
    if (row >= T) return;
    const int i = threadIdx.x;
    float *v = q + ((size_t)row * heads + h) * dim;
    const float *c = cosv + (size_t)(base + row) * 32;
    const float *s = sinv + (size_t)(base + row) * 32;
    const float a = v[i], b = v[i + 32];
    v[i]      = a * c[i] - b * s[i];
    v[i + 32] = b * c[i] + a * s[i];
}

// Router logits, batched. [T][2560] against [PF_NEXP_R][2560] bf16.
// This was MISSING from the first wiring: k_pf_router consumed an allocated but
// never-written buffer, so it produced garbage expert ids and k_pf_group indexed
// its count array out of bounds - the first-light memory fault.
// grid (NEXP/4, T), block 256: four experts a block, one row.
__global__ void __launch_bounds__(256)
k_pf_route_gemv(const uint16_t *__restrict__ w, const float *__restrict__ x,
                int T, float *__restrict__ out)
{
    const int e0 = blockIdx.x * 4, row = blockIdx.y;
    if (row >= T) return;
    const float *xr = x + (size_t)row * PF_NEMBD;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i = threadIdx.x; i < PF_NEMBD; i += 256) {
        const float v = xr[i];
        #pragma unroll
        for (int k = 0; k < 4; k++)
            acc[k] = fmaf(pb_b2f(w[(size_t)(e0 + k) * PF_NEMBD + i]), v, acc[k]);
    }
    __shared__ float red[4][4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        float a = acc[k];
        #pragma unroll
        for (int o = 32; o; o >>= 1) a += __shfl_down(a, o, 64);
        if ((threadIdx.x & 63) == 0) red[k][threadIdx.x >> 6] = a;
    }
    __syncthreads();
    if (threadIdx.x < 4)
        out[(size_t)row * PF_NEXP_R + e0 + threadIdx.x] =
            red[threadIdx.x][0] + red[threadIdx.x][1] + red[threadIdx.x][2] + red[threadIdx.x][3];
}
