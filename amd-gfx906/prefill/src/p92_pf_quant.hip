#include "p92amd/p92_pf_moe.h"

// Batched activation quantiser for prefill.
//
// Decode quantises one 2560-vector per token. Prefill quantises T of them, and
// the layout obligation is the same: the packed-nibble ABI masks a dword to get
// the EVEN values of a 16-group and shifts to get the ODD ones, so the consumer
// wants each group written even-then-odd. That permutation lives HERE, in the
// producer, so k_pf_gateup stages nothing - the contract that let the decode
// expert kernel drop 9216 B of LDS and go from 5 to 8 waves/SIMD.
//
// One thread owns one 16-value group: 160 groups a row, so 160 of 256 threads
// are live and each does a contiguous 64-byte read. The alternative - 16 threads
// cooperating per group - needs a cross-lane max for every group and buys
// nothing at this width.
//
// Loads before arithmetic: four uint4 issued together, then the max, then the
// quantise. A per-value load-compute-load chain here would fence four times a
// group for no reason.
__global__ void __launch_bounds__(256)
k_pf_quant(const float *__restrict__ x, int T,
           int8_t *__restrict__ xq, float *__restrict__ xs)
{
    const int row = blockIdx.x;
    if (row >= T) return;
    const int g = threadIdx.x;                       // group index, 0..159
    if (g >= PF_NEMBD / 16) return;

    const float4 *src = (const float4 *)(x + (size_t)row * PF_NEMBD + (size_t)g * 16);
    float4 v0, v1, v2, v3;
    v0 = src[0]; v1 = src[1]; v2 = src[2]; v3 = src[3];   // all four, then use

    float m = 0.f;
    #define PF_MX(q) { m = fmaxf(m, fabsf((q).x)); m = fmaxf(m, fabsf((q).y)); \
                       m = fmaxf(m, fabsf((q).z)); m = fmaxf(m, fabsf((q).w)); }
    PF_MX(v0) PF_MX(v1) PF_MX(v2) PF_MX(v3)
    #undef PF_MX

    const float sc  = m > 0.f ? m * (1.f / 127.f) : 1.f;
    const float inv = m > 0.f ? 127.f / m : 0.f;
    // Written in the CONSUMER's order, not the natural one. k_pf_gateup's lane l
    // needs groups l, 32+l, 64+l, 96+l, 128+l, so storing them contiguously at
    // [l*5+i] turns five strided float loads in its token loop into one uint4
    // plus one float. Producer owns the layout - the same contract that keeps
    // the nibble permutation out of the consumer.
    xs[(size_t)row * (PF_NEMBD / 16) + (g % 32) * 5 + (g / 32)] = sc;

    // even-then-odd within the group: value j lands at (j&1) ? 8+(j>>1) : (j>>1)
    int8_t *dst = xq + (size_t)row * PF_NEMBD + (size_t)g * 16;
    const float f[16] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w,
                          v2.x, v2.y, v2.z, v2.w, v3.x, v3.y, v3.z, v3.w };
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        int q = (int)rintf(f[j] * inv);
        q = q < -128 ? -128 : (q > 127 ? 127 : q);
        dst[(j & 1) ? 8 + (j >> 1) : (j >> 1)] = (int8_t)q;
    }
}

// Batched RMS norm over T rows, fused with the quantise above where the
// consumer allows it. Kept separate for now: the prefill layer body needs the
// normed fp32 for the attention path as well as the quantised form for the
// expert path, so fusing would force a second write anyway.
__global__ void __launch_bounds__(256)
k_pf_rmsnorm(const float *__restrict__ x, const uint16_t *__restrict__ gamma,
             float *__restrict__ y, int T, float eps)
{
    const int row = blockIdx.x;
    if (row >= T) return;
    const float *xr = x + (size_t)row * PF_NEMBD;
    float ss = 0.f;
    #pragma unroll
    for (int i = 0; i < PF_NEMBD / 256; i++) {
        const float v = xr[i * 256 + threadIdx.x];
        ss = fmaf(v, v, ss);
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) ss += __shfl_down(ss, o, 64);
    __shared__ float red[4];
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = ss;
    __syncthreads();
    const float inv = rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float)PF_NEMBD + eps);
    float *yr = y + (size_t)row * PF_NEMBD;
    #pragma unroll
    for (int i = 0; i < PF_NEMBD / 256; i++) {
        const int d = i * 256 + threadIdx.x;
        yr[d] = xr[d] * inv * __uint_as_float((uint32_t)gamma[d] << 16);
    }
}
