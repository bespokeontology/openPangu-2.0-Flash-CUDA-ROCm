#pragma once
// openPangu dense MLP, layers 0 and 1 only, gfx906 wave64.
//
// These two layers have a plain MLP instead of the MoE block, with
// moe_intermediate_size replaced by 9216:
//   gate_proj [9216, 2560]   up_proj [9216, 2560]   down_proj [2560, 9216]
// They are NVFP4 in the artifact and int8 in Huawei's checkpoint; the arena is
// all-NVFP4, so this is the NVFP4 path. They are NOT in the expert arena - the
// packer skips layers below 2 - and are loaded separately, 39.8 MB a layer.
//
// gate/up share the expert geometry exactly, because K is 2560 either way:
// 160 groups of 16, one row per half wave, 5 steps of 32 lanes. Only the row
// count changes, 1024 -> 9216, so the grid grows to 1152 blocks.
//
// down does NOT: K goes 1024 -> 9216, which is 576 groups rather than 64. Eight
// lanes a row would be 72 steps and leave a wave two thirds idle on the tail,
// so a whole wave takes one row at 9 groups a lane, four rows a block, 640
// blocks. That is the one real geometry change in this file.
#include "p92_nvfp4_dot4.hip"

#define P92_DENSE_FF   9216
#define P92_DENSE_GRP_GU  (P92_NEMBD / 16)      // 160 groups on gate/up
#define P92_DENSE_GRP_DN  (P92_DENSE_FF / 16)   // 576 groups on down

// ---- gate + up + SwiGLU --------------------------------------------------
__global__ void __launch_bounds__(256)
k_p92_dense_gateup(const uint8_t *__restrict__ Wg, const uint8_t *__restrict__ Sg,
                   const uint8_t *__restrict__ Wu, const uint8_t *__restrict__ Su,
                   const int8_t *__restrict__ xq, const float *__restrict__ xs,
                   float sc2g, float sc2u, float *__restrict__ hidden) {
    // xq arrives already even-then-odd from k_p92_quant, so staging it here
    // would permute it a SECOND time. That corrupted layers 0 and 1 - and
    // through them everything downstream - while every microgate still passed,
    // because each microtest builds its own activation instead of calling the
    // quantiser. p92_dense.hip:103 and the head still stage, correctly: their
    // inputs are k_p92_dense_requant output and a host-quantised vector, both
    // natural order.
    const int8_t *sx = xq;

    const int l   = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const uint8_t *wg = Wg + (size_t)row * (P92_NEMBD / 2);
    const uint8_t *wu = Wu + (size_t)row * (P92_NEMBD / 2);
    const uint8_t *sg = Sg + (size_t)row * (P92_NEMBD / 16);
    const uint8_t *su = Su + (size_t)row * (P92_NEMBD / 16);

    float ag = 0.f, au = 0.f;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        const int g = i * 32 + l;
        const uint2 xe = *(const uint2 *)(sx + g * 16);
        const uint2 xo = *(const uint2 *)(sx + g * 16 + 8);
        // per-16 activation scale, matching what k_p92_quant emits. This was
        // left on the old single-scale form when the projections and the
        // experts were converted, so layers 0 and 1 ran with the wrong
        // activation scaling: the contract gate measures cosine 0.769 against a
        // reference built from the producer's own output. Coherent generation
        // masked it because two of 46 layers are dense and the residual stream
        // renormalises.
        ag = fmaf(p92_ue4m3_d(sg[g]) * xs[g],
                  (float)p92_group_dot4(*(const uint2 *)(wg + (size_t)g * 8), xe, xo), ag);
        au = fmaf(p92_ue4m3_d(su[g]) * xs[g],
                  (float)p92_group_dot4(*(const uint2 *)(wu + (size_t)g * 8), xe, xo), au);
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) { ag += __shfl_down(ag, off, 64); au += __shfl_down(au, off, 64); }
    if (l == 0) {
        const float f = 0.5f;                      // 0.5 undoes the doubled E2M1 table
        const float g = ag * f * sc2g, u = au * f * sc2u;
        hidden[row] = g / (1.f + __expf(-g)) * u;
    }
}

// ---- requantise the 9216-wide intermediate -------------------------------
// One block, 1024 threads, 9 values each; absmax then a single scale, the same
// contract boundary the routed experts use.
__global__ void __launch_bounds__(1024)
k_p92_dense_requant(const float *__restrict__ hidden, int8_t *__restrict__ hq,
                    float *__restrict__ hs) {
    const int t = threadIdx.x;
    float v[9]; float m = 0.f;
    #pragma unroll
    for (int i = 0; i < 9; i++) { v[i] = hidden[i * 1024 + t]; m = fmaxf(m, fabsf(v[i])); }
    __shared__ float red[16];
    #pragma unroll
    for (int o = 32; o; o >>= 1) m = fmaxf(m, __shfl_down(m, o, 64));
    if ((t & 63) == 0) red[t >> 6] = m;
    __syncthreads();
    if (t == 0) {
        float g = red[0];
        #pragma unroll
        for (int i = 1; i < 16; i++) g = fmaxf(g, red[i]);
        hs[0] = g > 0.f ? g * (1.f / 127.f) : 1.f;
    }
    __syncthreads();
    const float inv = 1.f / hs[0];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        int q = (int)rintf(v[i] * inv);
        q = q < -128 ? -128 : (q > 127 ? 127 : q);
        hq[i * 1024 + t] = (int8_t)q;
    }
}

// ---- down ----------------------------------------------------------------
// 64 lanes a row, 9 groups a lane, 4 rows a block, 640 blocks. A full wave per
// row pays a six-shuffle reduction for every one of the 2560 rows and measured
// 324 GB/s against gate/up's 450; narrowing the lane group trades steps for
// shuffles. LDS holds the permuted 9216-byte intermediate, 9 KB.
__global__ void __launch_bounds__(256)
k_p92_dense_down(const uint8_t *__restrict__ Wd, const uint8_t *__restrict__ Sd,
                 const int8_t *__restrict__ hq, const float *__restrict__ hs,
                 float sc2, float *__restrict__ y) {
    __shared__ int8_t sh[P92_DENSE_FF];
    p92_stage_perm(hq, sh, P92_DENSE_FF);
    __syncthreads();

    const int l   = threadIdx.x & 63;
    const int row = blockIdx.x * 4 + (threadIdx.x / 64);
    const uint8_t *wr = Wd + (size_t)row * (P92_DENSE_FF / 2);
    const uint8_t *sr = Sd + (size_t)row * (P92_DENSE_FF / 16);

    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        const int g = i * 64 + l;
        const uint2 xe = *(const uint2 *)(sh + g * 16);
        const uint2 xo = *(const uint2 *)(sh + g * 16 + 8);
        acc = fmaf(p92_ue4m3_d(sr[g]),
                   (float)p92_group_dot4(*(const uint2 *)(wr + (size_t)g * 8), xe, xo), acc);
    }
    #pragma unroll
    for (int off = 32; off; off >>= 1) acc += __shfl_down(acc, off, 64);
    if (l == 0) y[row] = acc * hs[0] * 0.5f * sc2;
}
