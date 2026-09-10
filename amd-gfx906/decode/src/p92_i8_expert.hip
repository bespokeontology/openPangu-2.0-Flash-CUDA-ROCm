#pragma once
// openPangu-2.0-Flash W8A8 routed experts, gfx906 wave64.
//
// Contract (recovered from openpangu/openPangu-2.0-Flash-Int8 and the Windows
// CUDA pangu_under8_patch reference): int8 weights row-major [out, in], one
// symmetric bf16 scale per output row, no zero point; activations int8 with a
// SINGLE symmetric scale per token; int32 accumulation; requantisation after
// SwiGLU before down_proj.
//
// The single per-token activation scale is what separates this from the Qwen
// path, whose per-32-block scales force an fmaf inside the dot loop.  Here the
// whole row stays in int32 and the scale folds into one multiply at the end.
//
// Geometry is compiled in.  gate/up: K=2560 = 5 steps x 32 lanes x 16 B, one
// row per half wave, 8 rows per block, 128 blocks.  down: K=1024 = 8 steps x
// 8 lanes x 16 B, 32 rows per block, 80 blocks.  No runtime extents.
#include "p92amd/p92_i8_expert.h"

typedef int i8v4 __attribute__((ext_vector_type(4)));   // 16 int8 lanes

static __device__ __forceinline__ i8v4 ld16(const int8_t *p) {
    return *reinterpret_cast<const i8v4 *>(p);
}
static __device__ __forceinline__ i8v4 ld16nt(const int8_t *p) {   // weights: read once
    return __builtin_nontemporal_load(reinterpret_cast<const i8v4 *>(p));
}
// 16 int8 x 16 int8 -> int32 (v_dot4_i32_i8 x4), accumulating in place
static __device__ __forceinline__ int dot16(i8v4 w, i8v4 x, int acc) {
    acc = __builtin_amdgcn_sdot4(w.x, x.x, acc, false);
    acc = __builtin_amdgcn_sdot4(w.y, x.y, acc, false);
    acc = __builtin_amdgcn_sdot4(w.z, x.z, acc, false);
    acc = __builtin_amdgcn_sdot4(w.w, x.w, acc, false);
    return acc;
}
// integer cross-lane sums via ds_swizzle: no LDS, no barrier
static __device__ __forceinline__ int half_sum32(int v) {
    v += __builtin_amdgcn_ds_swizzle(v, 0x041F);
    v += __builtin_amdgcn_ds_swizzle(v, 0x081F);
    v += __builtin_amdgcn_ds_swizzle(v, 0x101F);
    v += __builtin_amdgcn_ds_swizzle(v, 0x201F);
    v += __builtin_amdgcn_ds_swizzle(v, 0x401F);
    return v;
}
static __device__ __forceinline__ int g8_sum32(int v) {
    v += __builtin_amdgcn_ds_swizzle(v, 0x041F);
    v += __builtin_amdgcn_ds_swizzle(v, 0x081F);
    v += __builtin_amdgcn_ds_swizzle(v, 0x101F);
    return v;
}

// ---- gate + up + SwiGLU, all eight selections ------------------------------
// One launch. Weights stream non-temporally; the activation row is read from
// L1 (256 threads share the same 2560 bytes). Accumulators stay int32 for the
// full K=2560 (bound 2560*127*127 = 41.3M, well inside int32).
__global__ void __launch_bounds__(256)
k_p92_i8_gateup(const int8_t *__restrict__ Wg, const float *__restrict__ Sg,
                const int8_t *__restrict__ Wu, const float *__restrict__ Su,
                const int8_t *__restrict__ xq, const float *__restrict__ xs,
                const int *__restrict__ slot, float *__restrict__ hidden) {
    const int k = blockIdx.y;
    const int s = slot[k];
    if (s < 0) return;                          // routed elsewhere this token

    const int l   = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const size_t off = (size_t)s * ((size_t)P92_I8_NFF * P92_I8_NEMBD)
                     + (size_t)row * P92_I8_NEMBD;
    const int8_t *wg = Wg + off + l * 16;
    const int8_t *wu = Wu + off + l * 16;

    i8v4 vg[5], vu[5];
    #pragma unroll
    for (int st = 0; st < 5; st++) {
        vg[st] = ld16nt(wg + st * 512);
        vu[st] = ld16nt(wu + st * 512);
    }
    int ag = 0, au = 0;
    #pragma unroll
    for (int st = 0; st < 5; st++) {
        const i8v4 xv = ld16(xq + st * 512 + l * 16);
        ag = dot16(vg[st], xv, ag);
        au = dot16(vu[st], xv, au);
    }
    ag = half_sum32(ag);
    au = half_sum32(au);
    if (l == 0) {
        const float xsc = xs[0];                                   // one per token
        const float g = (float)ag * xsc * Sg[(size_t)s * P92_I8_NFF + row];
        const float u = (float)au * xsc * Su[(size_t)s * P92_I8_NFF + row];
        hidden[(size_t)k * P92_I8_NFF + row] = g / (1.f + __expf(-g)) * u;
    }
}

// ---- requantisation boundary: one absmax/127 scale per selection -----------
// 1024 values, 256 threads, 4 each. Wave-level absmax then a 4-entry LDS
// reduction; the contract needs one scalar, not block scales.
__global__ void __launch_bounds__(256)
k_p92_i8_requant(const float *__restrict__ hidden, const int *__restrict__ slot,
                 int8_t *__restrict__ hq, float *__restrict__ hs) {
    const int k = blockIdx.x;
    if (slot[k] < 0) return;
    const size_t base = (size_t)k * P92_I8_NFF;

    float v[4];
    float m = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = hidden[base + i * 256 + threadIdx.x];
        m = fmaxf(m, fabsf(v[i]));
    }
    __shared__ float red[4];
    #pragma unroll
    for (int o = 32; o; o >>= 1) m = fmaxf(m, __shfl_down(m, o, 64));
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = m;
    __syncthreads();
    m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));

    const float sc = m > 0.f ? m * (1.f / 127.f) : 1.f;
    if (threadIdx.x == 0) hs[k] = sc;
    const float inv = 1.f / sc;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int q = (int)rintf(v[i] * inv);
        q = q < -128 ? -128 : (q > 127 ? 127 : q);
        hq[base + i * 256 + threadIdx.x] = (int8_t)q;
    }
}

// ---- down + router-weighted accumulate, all experts, single writer ---------
// 8 lanes own one row of 1024 bytes as chunks c = l + 8*i, i = 0..7, so the
// eight lanes read 128 contiguous bytes per step. One thread group walks every
// selection for its row: deterministic order, one writer per output, no atomics.
// y is seeded (shared expert) and both the int8 and nvfp4 down kernels add.
__global__ void __launch_bounds__(256)
k_p92_i8_downacc(const int8_t *__restrict__ Wd, const float *__restrict__ Sd,
                 const int8_t *__restrict__ hq, const float *__restrict__ hs,
                 const float *__restrict__ wts, const int *__restrict__ slot,
                 float *__restrict__ y) {
    __shared__ int   ssl[P92_I8_DISPATCH];
    __shared__ float swt[P92_I8_DISPATCH];
    if (threadIdx.x < P92_I8_DISPATCH) {
        ssl[threadIdx.x] = slot[threadIdx.x];
        swt[threadIdx.x] = wts[threadIdx.x];
    }
    __syncthreads();

    const int l   = threadIdx.x & 7;
    const int row = blockIdx.x * 32 + (threadIdx.x >> 3);
    float total = 0.f;

    #pragma unroll 2
    for (int k = 0; k < P92_I8_DISPATCH; k++) {
        const int s = ssl[k];
        if (s < 0) continue;
        const int8_t *wr = Wd + (size_t)s * ((size_t)P92_I8_NEMBD * P92_I8_NFF)
                              + (size_t)row * P92_I8_NFF;
        i8v4 v[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) v[i] = ld16nt(wr + (l + 8 * i) * 16);
        int acc = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            acc = dot16(v[i], ld16(hq + (size_t)k * P92_I8_NFF + (l + 8 * i) * 16), acc);
        acc = g8_sum32(acc);
        total = fmaf((float)acc * hs[k] * swt[k],
                     Sd[(size_t)s * P92_I8_NEMBD + row], total);
    }
    if (l == 0) y[row] += total;      // y seeded by the shared expert
}

// ---- per-token activation quantisation, residual width --------------------
__global__ void __launch_bounds__(256)
k_p92_i8_xquant(const float *__restrict__ x, int8_t *__restrict__ xq,
                float *__restrict__ xs) {
    float v[10];
    float m = 0.f;
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        v[i] = x[i * 256 + threadIdx.x];
        m = fmaxf(m, fabsf(v[i]));
    }
    __shared__ float red[4];
    #pragma unroll
    for (int o = 32; o; o >>= 1) m = fmaxf(m, __shfl_down(m, o, 64));
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = m;
    __syncthreads();
    m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));

    const float sc = m > 0.f ? m * (1.f / 127.f) : 1.f;
    if (threadIdx.x == 0) xs[0] = sc;
    const float inv = 1.f / sc;
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        int q = (int)rintf(v[i] * inv);
        q = q < -128 ? -128 : (q > 127 ? 127 : q);
        xq[i * 256 + threadIdx.x] = (int8_t)q;
    }
}

// ---- host entry points ----------------------------------------------------
void p92_i8_quantise(const float *x, int8_t *xq, float *xs, hipStream_t stream) {
    hipLaunchKernelGGL(k_p92_i8_xquant, dim3(1), dim3(256), 0, stream, x, xq, xs);
}

void p92_i8_expert_ffn(const P92I8ExpertArena &a,
                       const int8_t *xq, const float *xs,
                       const int *slot, const float *route_w,
                       float *hidden, int8_t *hq, float *hs,
                       float *y, hipStream_t stream) {
    hipLaunchKernelGGL(k_p92_i8_gateup, dim3(128, P92_I8_DISPATCH), dim3(256), 0, stream,
                       a.gate_w, a.gate_s, a.up_w, a.up_s, xq, xs, slot, hidden);
    hipLaunchKernelGGL(k_p92_i8_requant, dim3(P92_I8_DISPATCH), dim3(256), 0, stream,
                       hidden, slot, hq, hs);
    hipLaunchKernelGGL(k_p92_i8_downacc, dim3(80), dim3(256), 0, stream,
                       a.down_w, a.down_s, hq, hs, route_w, slot, y);
}
