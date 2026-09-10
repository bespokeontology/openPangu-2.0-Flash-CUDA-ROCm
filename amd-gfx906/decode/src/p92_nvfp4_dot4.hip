#pragma once
// NVFP4 cold-expert path rebuilt on v_dot4_i32_i8 for gfx906.
//
// The mechanism-1 kernel decodes each E2M1 nibble to fp32 and does 16 FMAs per
// 8 packed bytes: ~96 VALU per group of 16, which is why it measures 17.8 GB/s
// while the int8 path measures 731 GB/s. It is ALU bound, not stream bound.
//
// E2M1 magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6}. Doubled they are exactly
// {0,1,2,3,4,6,8,12}, integers, so a group can be decoded into int8 lanes and
// fed to v_dot4_i32_i8. Decode is two v_perm_b32 byte-table lookups plus a
// bitselect, four values at a time; the factor of two folds into the group
// scale. The weight arithmetic is therefore exact - identical values, integer
// products - and only the activation changes, from fp32 to the same int8
// per-token quantisation Huawei's contract already applies to the hot path.
//
// Nibble order in the artifact: byte j holds value 2j in the LOW nibble and
// 2j+1 in the HIGH nibble. Masking a packed dword therefore yields the even
// values and shifting yields the odd ones, so the activation is staged into
// LDS pre-permuted into even-then-odd halves per group and both sides line up
// without any per-group shuffling.
#include "p92amd/p92_nvfp4_wave64.h"
#include <hip/hip_runtime.h>

// magnitude tables, indexed by v_perm_b32: index 0-3 select bytes 0-3 of the
// second operand, index 4-7 select bytes 0-3 of the first.
#define P92_MAG_HI 0x0C080604u    // magnitudes 4..7 ->  4,  6,  8, 12
#define P92_MAG_LO 0x03020100u    // magnitudes 0..3 ->  0,  1,  2,  3
#define P92_NEG_HI 0xF4F8FAFCu    // magnitudes 4..7 -> -4, -6, -8,-12
#define P92_NEG_LO 0xFDFEFF00u    // magnitudes 0..3 -> -0, -1, -2, -3
#define P92_SGN_TBL 0x0000FF00u   // index 0 -> 0x00, index 1 -> 0xFF

// Four E2M1 nibbles held in the low half of each byte -> four int8 lanes
// carrying twice the represented value.
__device__ __forceinline__ uint32_t p92_dq4(uint32_t d) {
    const uint32_t mag = d & 0x07070707u;
    const uint32_t sel = (d >> 3) & 0x01010101u;
    const uint32_t pos = __builtin_amdgcn_perm(P92_MAG_HI, P92_MAG_LO, mag);
    const uint32_t neg = __builtin_amdgcn_perm(P92_NEG_HI, P92_NEG_LO, mag);
    const uint32_t msk = __builtin_amdgcn_perm(0u, P92_SGN_TBL, sel);
    return pos ^ ((pos ^ neg) & msk);          // v_bfi_b32
}

__device__ __forceinline__ float p92_ue4m3_d(uint32_t b) {
    const uint32_t e = b >> 3, m = b & 7u;
    if (e == 0) return (float)m * 0x1p-9f;
    if (e == 15 && m == 7) return INFINITY;
    return __uint_as_float(((e + 120u) << 23) | (m << 20));
}

static __device__ __forceinline__ int p92_d4(uint32_t w, uint32_t x, int a) {
    return __builtin_amdgcn_sdot4((int)w, (int)x, a, false);
}

// One 16-value scale group: 8 packed weight bytes against the pre-permuted
// int8 activation. Four decodes, four dot4, one int32 result.
__device__ __forceinline__ int p92_group_dot4(uint2 pk, const uint2 xe, const uint2 xo) {
    int a = 0;
    a = p92_d4(p92_dq4( pk.x       & 0x0F0F0F0Fu), xe.x, a);   // values 0,2,4,6
    a = p92_d4(p92_dq4((pk.x >> 4) & 0x0F0F0F0Fu), xo.x, a);   // values 1,3,5,7
    a = p92_d4(p92_dq4( pk.y       & 0x0F0F0F0Fu), xe.y, a);   // values 8,10,12,14
    a = p92_d4(p92_dq4((pk.y >> 4) & 0x0F0F0F0Fu), xo.y, a);   // values 9,11,13,15
    return a;
}

// Stage the quantised activation into LDS, even values then odd within each
// 16-value group, so the masked and shifted weight halves both match a
// contiguous dword.
__device__ __forceinline__ void p92_stage_perm(const int8_t *__restrict__ xq,
                                               int8_t *__restrict__ sx, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int g = i >> 4, j = i & 15;
        sx[(g << 4) + ((j & 1) ? 8 + (j >> 1) : (j >> 1))] = xq[i];
    }
}

// ---- gate + up + SwiGLU, cold experts -------------------------------------
// [1024, 2560]: 160 groups per row, one row per half wave, 8 rows per block.
__global__ void __launch_bounds__(256)
k_p92_f4_gateup(const uint8_t *__restrict__ Wg, const uint8_t *__restrict__ Sg,
                const uint8_t *__restrict__ Wu, const uint8_t *__restrict__ Su,
                const int8_t *__restrict__ xq, const float *__restrict__ xs,
                const int *__restrict__ slot, const float *__restrict__ s2,
                float *__restrict__ hidden) {   // s2 is [slot][3]: gate, up, down
    const int k = blockIdx.y;
    const int s = slot[k];
    if (s < 0) return;

    const int8_t *sx = xq;          // already permuted by k_p92_quant

    const int l   = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const uint8_t *wg = Wg + (size_t)s * P92_GU_W_BYTES + (size_t)row * (P92_NEMBD / 2);
    const uint8_t *wu = Wu + (size_t)s * P92_GU_W_BYTES + (size_t)row * (P92_NEMBD / 2);
    const uint8_t *sg = Sg + (size_t)s * P92_GU_S_BYTES + (size_t)row * (P92_NEMBD / 16);
    const uint8_t *su = Su + (size_t)s * P92_GU_S_BYTES + (size_t)row * (P92_NEMBD / 16);

    float ag = 0.f, au = 0.f;
    #pragma unroll 1
    for (int i = 0; i < 5; i++) {                       // 5 * 32 lanes = 160 groups
        const int g = i * 32 + l;
        const uint2 xe = *(const uint2 *)(sx + g * 16);
        const uint2 xo = *(const uint2 *)(sx + g * 16 + 8);
        const uint2 pg = *(const uint2 *)(wg + (size_t)g * 8);
        const uint2 pu = *(const uint2 *)(wu + (size_t)g * 8);
        // the per-16 activation scale belongs HERE, folded into the group scale;
        // dropping it cost a factor of xs on gate and again on up, and since the
        // intermediate is silu(gate)*up that is xs squared, about 1e3
        ag = fmaf(p92_ue4m3_d(sg[g]) * xs[g], (float)p92_group_dot4(pg, xe, xo), ag);
        au = fmaf(p92_ue4m3_d(su[g]) * xs[g], (float)p92_group_dot4(pu, xe, xo), au);
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) {
        ag += __shfl_down(ag, off, 64);
        au += __shfl_down(au, off, 64);
    }
    if (l == 0) {
        const float f = 0.5f;                          // 0.5 undoes the doubled table
        const float g = ag * f * s2[(size_t)s * 3 + 0];
        const float u = au * f * s2[(size_t)s * 3 + 1];
        hidden[(size_t)k * P92_NFF + row] = g / (1.f + __expf(-g)) * u;
    }
}

// ---- requantisation for the NVFP4 down path -------------------------------
// Separate from k_p92_i8_requant on purpose. The int8 down path consumes the
// hidden vector in natural order; the NVFP4 path consumes it even-then-odd
// within each 16-value group, because masking a packed dword yields the even
// values and shifting yields the odd. Writing that order HERE means the down
// kernel needs no LDS staging: every one of its 320 blocks used to stage all
// nine experts, 9216 B of LDS and 2.95 MB of redundant traffic a call, which
// capped its occupancy at 5 waves a SIMD. Without it the kernel is 30 VGPRs
// and 8 waves.
__global__ void __launch_bounds__(256)
k_p92_f4_requant(const float *__restrict__ hidden, const int *__restrict__ slot,
                 int8_t *__restrict__ hq, float *__restrict__ hs) {
    const int k = blockIdx.x;
    if (slot[k] < 0) return;
    const size_t base = (size_t)k * P92_NFF;
    float v[4]; float m = 0.f;
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
        const int idx = i * 256 + threadIdx.x;
        const int g = idx >> 4, j = idx & 15;
        hq[base + (g << 4) + ((j & 1) ? 8 + (j >> 1) : (j >> 1))] = (int8_t)q;
    }
}

// ---- down + router-weighted accumulate, cold experts ----------------------
// [2560, 1024]: 64 groups per row, one row per eighth wave, 32 rows per block.
__global__ void __launch_bounds__(256)
k_p92_f4_downacc(const uint8_t *__restrict__ Wd, const uint8_t *__restrict__ Sd,
                 const int8_t *__restrict__ hq, const float *__restrict__ hs,
                 const float *__restrict__ wts, const int *__restrict__ slot,
                 const float *__restrict__ s2, float *__restrict__ y) {
    __shared__ int   ssl[P92_DISPATCH];
    __shared__ float swt[P92_DISPATCH];

    if (threadIdx.x < P92_DISPATCH) {
        ssl[threadIdx.x] = slot[threadIdx.x];
        swt[threadIdx.x] = wts[threadIdx.x];
    }
    __syncthreads();

    // 32 lanes a row, not 8. Eight lanes made each step a 128-byte read and the
    // kernel ran at 166 GB/s while gate/up, which reads twice the bytes, ran at
    // 378. Widening the lane group makes each step a 512-byte transaction.
    // Swept end to end at 220 tokens: 8 lanes 46.08 tok/s, 16 lanes 51.22,
    // 32 lanes 51.32, 64 lanes 50.09 - past 32 the one-group reduction and the
    // redundant LDS staging across 640 blocks start to cost more than the
    // coalescing gains.
    const int l   = threadIdx.x % 32;
    const int row = blockIdx.x * 8 + (threadIdx.x / 32);
    float lane = 0.f;    // per-lane running total across experts

    #pragma unroll 2
    for (int k = 0; k < P92_DISPATCH; k++) {
        const int s = ssl[k];
        if (s < 0) continue;
        const uint8_t *wr = Wd + (size_t)s * P92_DN_W_BYTES + (size_t)row * (P92_NFF / 2);
        const uint8_t *sr = Sd + (size_t)s * P92_DN_S_BYTES + (size_t)row * (P92_NFF / 16);
        const int8_t *hx = hq + (size_t)k * P92_NFF;   // pre-permuted by the requant
        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < 2; i++) {                  // 2 * 32 lanes = 64 groups
            const int g = i * 32 + l;
            const uint2 xe = *(const uint2 *)(hx + g * 16);
            const uint2 xo = *(const uint2 *)(hx + g * 16 + 8);
            const uint2 pk = *(const uint2 *)(wr + (size_t)g * 8);
            acc = fmaf(p92_ue4m3_d(sr[g]), (float)p92_group_dot4(pk, xe, xo), acc);
        }
        // HOISTED REDUCTION. The per-expert scale is lane-uniform, so
        //   sum_k c_k * (sum_l acc_k[l])  ==  sum_l (sum_k acc_k[l] * c_k)
        // Accumulate per lane across all nine experts and reduce ONCE, turning
        // 45 ds_bpermute per thread into 5. Mathematically identical but it
        // REASSOCIATES: relative_l2 1.885e-07 against the per-expert form.
        lane = fmaf(acc * hs[k] * 0.5f * s2[(size_t)s * 3 + 2], swt[k], lane);
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) lane += __shfl_down(lane, off, 64);
    // Was '+=' onto a buffer the caller had just zeroed with a 10 KB
    // hipMemsetAsync. c.hidden holds the attention output at that point, so the
    // clear existed only to turn '+=' into '='. downacc is the sole writer of
    // the MoE result, so assign and drop the memset: 44 fillBufferAligned
    // launches a token, each ~3.37 us of almost pure launch overhead.
    if (l == 0) y[row] = lane;
}
