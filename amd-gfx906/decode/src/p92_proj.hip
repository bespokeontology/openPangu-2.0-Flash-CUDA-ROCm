#pragma once
// openPangu attention projections, norms and rope, gfx906 wave64.
//
// Every extent here is a compile-time template argument with an explicit
// instantiation per real openPangu shape, so each is a distinct compiled kernel
// with folded bounds and a fully unrolled inner loop. There is no runtime-K
// entry point; a shape the model does not have cannot be called.
//
// The four NVFP4 projections and their group counts, from the artifact manifest:
//   q_a_proj  [ 1024, 2560]  160 groups   32 lanes x 5
//   q_b_proj  [ 9216, 1024]   64 groups   32 lanes x 2
//   kv_b_proj [12288,  512]   32 groups   32 lanes x 1
//   o_proj    [ 2560, 6144]  384 groups   64 lanes x 6
// kv_b is 12288 = 48 heads x (128 nope + 128 value), head-interleaved, and
// serves both the absorbed query and the value up-projection.
//
// kv_a_proj_with_mqa [576, 2560] has NO artifact record and stays bf16, so it
// gets a bf16 GEMV rather than the dot4 path. 576 = kv_lora 512 + rope 64.
//
// ACTIVATION SCALE GRANULARITY. Huawei's W8A8 contract uses ONE int8 scale per
// token, and that is correct FOR HUAWEI'S WEIGHTS, because jointfix smooths the
// per-channel outliers out of the activation and into the weights. Our arena is
// NVFP4 packed from the unsmoothed bf16, so the outliers are still in the
// activation, and a single scale is destructive. Measured on the captured
// layer-0 activations:
//
//   L0_INPUT_NORM, 49.3x outlier   one scale  rel_err 1.09e-01, 1951/2560 values
//                                             quantised to ZERO
//                                  per 16     rel_err 9.77e-03, 36 lost
//   L0_Q_NORM,      6.2x outlier   one scale  1.44e-02      per 16  6.06e-03
//   L0_KV_NORM,     3.7x outlier   one scale  8.33e-03      per 16  5.03e-03
//
// So the activation carries one scale per 16 values, which is exactly the
// granularity the NVFP4 weights already use, and costs one fmaf per group
// instead of a pure int32 row accumulation.
#include "p92_nvfp4_dot4.hip"

typedef uint16_t p92_pbf;
static __device__ __forceinline__ float p92_p2f(p92_pbf b){ return __uint_as_float((uint32_t)b<<16); }
static __device__ __forceinline__ p92_pbf p92_f2p(float f){
    uint32_t u=__float_as_uint(f); u += 0x7FFFu + ((u>>16)&1u); return (p92_pbf)(u>>16); }

// ---- NVFP4 projection, y[N] = W[N,K] . x --------------------------------
// LANES lanes own one row, STEPS = (K/16)/LANES groups each.
template <int K, int LANES>
__global__ void __launch_bounds__(256)
k_p92_proj(const uint8_t *__restrict__ W, const uint8_t *__restrict__ S,
           const int8_t *__restrict__ xq, const float *__restrict__ xs,
           float sc2, float *__restrict__ y) {   // xs has K/16 entries
    constexpr int GROUPS = K / 16;
    constexpr int STEPS  = GROUPS / LANES;
    constexpr int ROWS   = 256 / LANES;
    const int8_t *sx = xq;          // already permuted by k_p92_quant
    const int l   = threadIdx.x & (LANES - 1);
    const int row = blockIdx.x * ROWS + (threadIdx.x / LANES);
    const uint8_t *wr = W + (size_t)row * (K / 2);
    const uint8_t *sr = S + (size_t)row * GROUPS;

    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < STEPS; i++) {
        const int g = i * LANES + l;
        acc = fmaf(p92_ue4m3_d(sr[g]) * xs[g],
                   (float)p92_group_dot4(*(const uint2 *)(wr + (size_t)g * 8),
                                         *(const uint2 *)(sx + g * 16),
                                         *(const uint2 *)(sx + g * 16 + 8)), acc);
    }
    #pragma unroll
    for (int off = LANES / 2; off; off >>= 1) acc += __shfl_down(acc, off, 64);
    if (l == 0) y[row] = acc * 0.5f * sc2;
}

template __global__ void k_p92_proj<2560,32>(const uint8_t*,const uint8_t*,const int8_t*,const float*,float,float*);
template __global__ void k_p92_proj<1024,32>(const uint8_t*,const uint8_t*,const int8_t*,const float*,float,float*);
template __global__ void k_p92_proj< 512,32>(const uint8_t*,const uint8_t*,const int8_t*,const float*,float,float*);
template __global__ void k_p92_proj<6144,64>(const uint8_t*,const uint8_t*,const int8_t*,const float*,float,float*);

// ---- bf16 GEMV, for kv_a_proj_with_mqa [576, 2560] ----------------------
// bf16 stays bf16: it has no artifact record, so there is nothing to decode.
// One wave a row, 40 bf16 a lane as 10 uint2 loads.
template <int K>
__global__ void __launch_bounds__(256)
k_p92_bf16_gemv(const p92_pbf *__restrict__ W, const float *__restrict__ x,
                float *__restrict__ y) {
    __shared__ float sx[K];
    #pragma unroll
    for (int i = 0; i < K / 256; i++) sx[i * 256 + threadIdx.x] = x[i * 256 + threadIdx.x];
    __syncthreads();
    const int l = threadIdx.x & 63;
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    const uint2 *wr = (const uint2 *)(W + (size_t)row * K);
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < K / 256; i++) {
        const int c = i * 64 + l;                       // 4 bf16 a load
        const uint2 v = wr[c];
        const float *xp = sx + c * 4;
        acc = fmaf(p92_p2f((p92_pbf)(v.x & 0xFFFF)), xp[0], acc);
        acc = fmaf(p92_p2f((p92_pbf)(v.x >> 16)),    xp[1], acc);
        acc = fmaf(p92_p2f((p92_pbf)(v.y & 0xFFFF)), xp[2], acc);
        acc = fmaf(p92_p2f((p92_pbf)(v.y >> 16)),    xp[3], acc);
    }
    #pragma unroll
    for (int off = 32; off; off >>= 1) acc += __shfl_down(acc, off, 64);
    if (l == 0) y[row] = acc;
}
template __global__ void k_p92_bf16_gemv<2560>(const p92_pbf*,const float*,float*);

// ---- RMSNorm, y = x * rsqrt(mean(x^2) + eps) * gamma --------------------
// Widths openPangu actually uses: 512 kv_a_layernorm, 1024 q_a_layernorm,
// 2560 the four per-layer norms, 10240 block_post over all four streams.
template <int N>
__global__ void __launch_bounds__(256)
k_p92_rmsnorm(const float *__restrict__ x, const p92_pbf *__restrict__ gamma,
              float *__restrict__ y, float eps) {
    const int t = threadIdx.x, lane = t & 63, wv = t >> 6;
    __shared__ float red[4];
    float ss = 0.f;
    #pragma unroll
    for (int i = 0; i < N / 256; i++) { const float v = x[i * 256 + t]; ss = fmaf(v, v, ss); }
    #pragma unroll
    for (int o = 32; o; o >>= 1) ss += __shfl_down(ss, o, 64);
    if (lane == 0) red[wv] = ss;
    __syncthreads();
    const float inv = rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float)N + eps);
    #pragma unroll
    for (int i = 0; i < N / 256; i++) {
        const int d = i * 256 + t;
        y[d] = x[d] * inv * p92_p2f(gamma[d]);
    }
}
template __global__ void k_p92_rmsnorm<  512>(const float*,const p92_pbf*,float*,float);
template __global__ void k_p92_rmsnorm< 1024>(const float*,const p92_pbf*,float*,float);
template __global__ void k_p92_rmsnorm< 2560>(const float*,const p92_pbf*,float*,float);
template __global__ void k_p92_rmsnorm<10240>(const float*,const p92_pbf*,float*,float);

// ---- RMSNorm fused with the int8 quantiser -------------------------------
// The two ran back to back on the same 2560-wide vector: RMSNorm wrote it, the
// quantiser immediately read it again, and both are single-block kernels whose
// cost is dominated by launch and by one CU's bandwidth rather than by the 10 KB
// they touch. 185 norm launches and 92 quant launches a token, at 3.91 and 3.93
// us each.
//
// Fused, x is read once and both outputs are produced: the fp32 normalised
// vector, which the bf16 GEMVs and the router still need, and the int8 form in
// the even-then-odd order every NVFP4 consumer wants.
template <int N>
__global__ void __launch_bounds__(256)
k_p92_rmsnorm_quant(const float *__restrict__ x, const p92_pbf *__restrict__ gamma,
                    float *__restrict__ y, int8_t *__restrict__ q,
                    float *__restrict__ s, float eps) {
    const int t = threadIdx.x, lane = t & 63, wv = t >> 6;
    __shared__ float red[4];
    float v[N / 256];
    float ss = 0.f;
    #pragma unroll
    for (int i = 0; i < N / 256; i++) { v[i] = x[i * 256 + t]; ss = fmaf(v[i], v[i], ss); }
    #pragma unroll
    for (int o = 32; o; o >>= 1) ss += __shfl_down(ss, o, 64);
    if (lane == 0) red[wv] = ss;
    __syncthreads();
    const float inv = rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float)N + eps);
    #pragma unroll
    for (int i = 0; i < N / 256; i++) {
        const int d = i * 256 + t;
        y[d] = v[i] * inv * p92_p2f(gamma[d]);
    }
    __syncthreads();
    // one thread a group of 16: absmax in registers, no reduction
    constexpr int G = N / 16;
    for (int g = t; g < G; g += 256) {
        const float *xp = y + g * 16;   // just written, hot in L1; staging it in
                                        // LDS cost 10 KB a block and made the
                                        // fused kernel slower than the pair
        float m = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; i++) m = fmaxf(m, fabsf(xp[i]));
        const float sc = m > 0.f ? m * (1.f / 127.f) : 1.f;
        s[g] = sc;
        const float iv = 1.f / sc;
        int8_t *qp = q + g * 16;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int r = (int)rintf(xp[i] * iv);
            const int8_t val = (int8_t)(r < -128 ? -128 : (r > 127 ? 127 : r));
            qp[(i & 1) ? 8 + (i >> 1) : (i >> 1)] = val;
        }
    }
}
template __global__ void k_p92_rmsnorm_quant<1024>(const float*,const p92_pbf*,float*,int8_t*,float*,float);
template __global__ void k_p92_rmsnorm_quant<2560>(const float*,const p92_pbf*,float*,int8_t*,float*,float);

// ---- rope, non-interleaved -----------------------------------------------
// config rope_interleave is false, so a pair is (i, i + D/2) within the roped
// span. The indexer ropes only the first 64 of its 128 dims, as (i, i+32);
// the attention rope span is the full 64, as (i, i+32).
template <int D>
__global__ void __launch_bounds__(64)
k_p92_rope(float *__restrict__ v, const float *__restrict__ cosv,
           const float *__restrict__ sinv) {
    const int i = threadIdx.x;
    if (i >= D / 2) return;
    const float c = cosv[i], s = sinv[i];
    const float a = v[i], b = v[i + D / 2];
    v[i]         = a * c - b * s;
    v[i + D / 2] = b * c + a * s;
}
template __global__ void k_p92_rope<64>(float*,const float*,const float*);

// ---- quantise an fp32 vector to int8 with one scale per 16 values ---------
// One thread owns one group of 16, so the absmax is entirely in registers and
// there is no reduction at all.
template <int N>
__global__ void __launch_bounds__(256)
k_p92_quant(const float *__restrict__ x, int8_t *__restrict__ q, float *__restrict__ s) {
    constexpr int G = N / 16;
    for (int g = blockIdx.x * 256 + threadIdx.x; g < G; g += gridDim.x * 256) {
        const float *xp = x + g * 16;
        float m = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; i++) m = fmaxf(m, fabsf(xp[i]));
        const float sc = m > 0.f ? m * (1.f / 127.f) : 1.f;
        s[g] = sc;
        const float inv = 1.f / sc;
        int8_t *qp = q + g * 16;
        // Emitted even-then-odd within the group, the order every NVFP4
        // consumer wants, so no consumer stages it into LDS. o_proj alone had
        // 640 blocks each staging 6144 B - 3.93 MB of redundant traffic a call
        // against 8.85 MB of weight reads - and it capped occupancy at 4.
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int r = (int)rintf(xp[i] * inv);
            const int8_t v = (int8_t)(r < -128 ? -128 : (r > 127 ? 127 : r));
            qp[(i & 1) ? 8 + (i >> 1) : (i >> 1)] = v;
        }
    }
}
template __global__ void k_p92_quant< 512>(const float*,int8_t*,float*);
template __global__ void k_p92_quant<1024>(const float*,int8_t*,float*);
template __global__ void k_p92_quant<2560>(const float*,int8_t*,float*);
template __global__ void k_p92_quant<6144>(const float*,int8_t*,float*);
