#pragma once
// p92_nvfp4_wave64.hip - fused NVFP4 unpack/dequantise/accumulate GEMV kernels
// for openPangu-2.0-Flash routed and shared experts on gfx906 (MI50, wave64).
//
// Derived from the proven Qwen3.8 gfx906 kernels by shape specialisation.
// The E2M1/UE4M3 decode, the group dot, the wave64 sub-reduction geometry and
// the packing contract are copied unchanged; only the dimensions differ.
//
//   gate/up : W [1024, 2560]  K = 2560 -> 160 groups/row, 5 iters x 32 lanes
//   down    : W [2560, 1024]  K = 1024 ->  64 groups/row, 8 iters x  8 lanes
//
// gfx906 has no FP4/FP8/bf16 instructions; both formats are decoded by exact
// integer bit construction into f32. All cross-lane reductions use the full
// 64-lane mask with offsets <= half the sub-group width so 32-lane and 8-lane
// sub-reductions stay independent inside one wave64, with identical trip
// counts across the wave (no divergence around __shfl_down_sync).
//
// Host syntax check (no GPU, no HIP):
//   g++ -DP92_HOST_CHECK -I../include -fsyntax-only src/p92_nvfp4_wave64.hip
#include "p92amd/p92_nvfp4_wave64.h"
#include <math.h>

#if !defined(P92_HOST_CHECK) && defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 60000000
#define __shfl_down_sync(mask, value, delta) __shfl_down((value), (delta), warpSize)
#endif

#define WMASK 0xFFFFFFFFFFFFFFFFULL

// --- exact integer dequant, identical to the Qwen gfx906 kernels ------------
__device__ __forceinline__ float p92_e2m1(uint32_t nib) {
    uint32_t mag = nib & 7u, e = mag >> 1, m = mag & 1u;
    float v = e ? __uint_as_float(((e + 126u) << 23) | (m << 22))
                : (m ? 0.5f : 0.f);
    return (nib & 8u) ? -v : v;
}
__device__ __forceinline__ float p92_ue4m3(uint32_t b) {
    uint32_t e = b >> 3, m = b & 7u;
    if (e == 0) return (float)m * 0x1p-9f;
    if (e == 15 && m == 7) return INFINITY;
    return __uint_as_float(((e + 120u) << 23) | (m << 20));
}

// fp32 dot of one 16-value NVFP4 scale group against xs[0..15].
// Byte j holds value 2j in the LOW nibble and value 2j+1 in the HIGH nibble.
__device__ __forceinline__ float p92_group_dot(uint2 pk, const float *xs) {
    float a = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        uint32_t b0 = (pk.x >> (8 * j)) & 0xFFu;
        uint32_t b1 = (pk.y >> (8 * j)) & 0xFFu;
        a = fmaf(p92_e2m1(b0 & 15u), xs[2 * j],     a);
        a = fmaf(p92_e2m1(b0 >> 4),  xs[2 * j + 1], a);
        a = fmaf(p92_e2m1(b1 & 15u), xs[2 * j + 8], a);
        a = fmaf(p92_e2m1(b1 >> 4),  xs[2 * j + 9], a);
    }
    return a;
}

// ---------------------------------------------------------------------------
// gate / up : [1024, 2560]. Row per half-wave, P92_GU_ROWS_PER_BLOCK per block.
// Per row 160 groups; lane l (0..31) takes groups i*32 + l, i = 0..4.
// K is unchanged from the Qwen kernel, so the inner loop is identical; only
// the row count (and therefore the grid) grows from 640 to 1024.
// ---------------------------------------------------------------------------
__global__ void k_p92_nvfp4_gemv_1024x2560(const uint8_t *__restrict__ W,
                                           const uint8_t *__restrict__ S,
                                           const float *__restrict__ x,
                                           float *__restrict__ y,
                                           float scale2, int slot) {
    __shared__ float xs[P92_NEMBD];
    for (int i = threadIdx.x; i < P92_NEMBD; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    int l   = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint8_t *wr = W + (size_t)slot * P92_GU_W_BYTES + (size_t)row * (P92_NEMBD / 2);
    const uint8_t *sr = S + (size_t)slot * P92_GU_S_BYTES + (size_t)row * (P92_NEMBD / 16);
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 5; i++) {                 // 5 * 32 lanes = 160 groups
        int g = i * 32 + l;
        uint2 pk = *(const uint2 *)(wr + (size_t)g * 8);
        acc = fmaf(p92_ue4m3(sr[g]), p92_group_dot(pk, xs + g * 16), acc);
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(WMASK, acc, off);
    if (l == 0) y[row] = acc * scale2;
}

// ---------------------------------------------------------------------------
// down : [2560, 1024]. Row per eighth-wave, 32 rows per 256-thread block.
// Per row 64 groups; lane l (0..7) takes groups i*8 + l, i = 0..7.
// K grows 640 -> 1024, so the iteration count grows 5 -> 8.
// ---------------------------------------------------------------------------
__global__ void k_p92_nvfp4_gemv_2560x1024(const uint8_t *__restrict__ W,
                                           const uint8_t *__restrict__ S,
                                           const float *__restrict__ x,
                                           float *__restrict__ y,
                                           float scale2, int slot) {
    __shared__ float xs[P92_NFF];
    for (int i = threadIdx.x; i < P92_NFF; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    int l   = threadIdx.x & 7;
    int row = blockIdx.x * 32 + (threadIdx.x >> 3);
    const uint8_t *wr = W + (size_t)slot * P92_DN_W_BYTES + (size_t)row * (P92_NFF / 2);
    const uint8_t *sr = S + (size_t)slot * P92_DN_S_BYTES + (size_t)row * (P92_NFF / 16);
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {                 // 8 * 8 lanes = 64 groups
        int g = i * 8 + l;
        uint2 pk = *(const uint2 *)(wr + (size_t)g * 8);
        acc = fmaf(p92_ue4m3(sr[g]), p92_group_dot(pk, xs + g * 16), acc);
    }
    #pragma unroll
    for (int off = 4; off; off >>= 1)
        acc += __shfl_down_sync(WMASK, acc, off);
    if (l == 0) y[row] = acc * scale2;
}

// ---------------------------------------------------------------------------
// Fused gate + up + SiLU-mul for one expert slot:
//   hidden[r] = silu(dot(g_r, x) * s2g) * (dot(u_r, x) * s2u),  r < 1024
// One half-wave per row, both projections walked by the same lane so the
// activation vector is staged once.
// ---------------------------------------------------------------------------
__global__ void k_p92_nvfp4_gateup_1024x2560(const uint8_t *__restrict__ Wg,
                                             const uint8_t *__restrict__ Sg,
                                             const uint8_t *__restrict__ Wu,
                                             const uint8_t *__restrict__ Su,
                                             const float *__restrict__ x,
                                             float *__restrict__ h,
                                             float s2g, float s2u, int slot) {
    __shared__ float xs[P92_NEMBD];
    for (int i = threadIdx.x; i < P92_NEMBD; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    int l   = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const size_t wo = (size_t)slot * P92_GU_W_BYTES + (size_t)row * (P92_NEMBD / 2);
    const size_t so = (size_t)slot * P92_GU_S_BYTES + (size_t)row * (P92_NEMBD / 16);
    const uint8_t *gw = Wg + wo, *gs = Sg + so;
    const uint8_t *uw = Wu + wo, *us = Su + so;
    float ag = 0.f, au = 0.f;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        int g = i * 32 + l;
        const float *xg = xs + g * 16;
        ag = fmaf(p92_ue4m3(gs[g]), p92_group_dot(*(const uint2 *)(gw + (size_t)g * 8), xg), ag);
        au = fmaf(p92_ue4m3(us[g]), p92_group_dot(*(const uint2 *)(uw + (size_t)g * 8), xg), au);
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) {
        ag += __shfl_down_sync(WMASK, ag, off);
        au += __shfl_down_sync(WMASK, au, off);
    }
    if (l == 0) {
        float gv = ag * s2g, uv = au * s2u;
        h[row] = (gv / (1.f + __expf(-gv))) * uv;   // SiLU(gate) * up
    }
}
