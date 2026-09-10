#pragma once
// openPangu-2.0-Flash router, specialised for gfx906 wave64.
//
// Semantics from the shipped config and the reference implementation
// (pangu_v2_moe.py): scoring_func sigmoid, router_enable_expert_bias true,
// norm_topk_prob true, routed_scaling_factor 2.5, num_experts_per_tok 8,
// n_routed_experts 256. use_grouped_topk is set but num_expert_group and
// topk_group are both 1, so the grouping is degenerate and this is a plain
// top-8 over all 256.
//
// Three details separate it from the Qwen donor, which is otherwise the right
// template. Qwen softmaxes; openPangu takes a sigmoid per expert, so there is
// no max pass and no sum pass before selection. The correction bias is added
// for SELECTION ONLY - the returned weight is the uncorrected sigmoid score.
// And the renormalised weights are scaled by 2.5, which is what makes the
// oracle report weight_sum 2.5.
//
// Geometry is compiled in: 256 experts, K=2560, top-8. Router weights are
// bf16 and stay bf16; gfx906 has no bf16 arithmetic, so they are widened to
// fp32 by a shift, which costs one instruction per value and no memory.
#include <hip/hip_runtime.h>
#include <cstdint>

#define P92_RTR_NEXP  256
#define P92_RTR_NEMBD 2560
#define P92_RTR_TOPK  8
#define P92_RTR_SCALE 2.5f
#define P92_RTR_WAVE  64
#define WMASK 0xFFFFFFFFFFFFFFFFULL

// 2560 bf16 per row = 160 uint4 chunks; one wave per expert, 40 values a lane,
// 5 chunks of 8. Four experts per 256-thread block, 64 blocks.
#define RTR_CHUNKS 160
#define RTR_PER_LANE 5

__device__ __forceinline__ float p92_bf16(uint16_t b) {
    return __uint_as_float((uint32_t)b << 16);
}

__global__ void __launch_bounds__(256)
k_p92_route_gemv(const uint16_t *__restrict__ W, const float *__restrict__ x,
                 float *__restrict__ logits) {
    __shared__ float sx[P92_RTR_NEMBD];
    #pragma unroll
    for (int i = 0; i < P92_RTR_NEMBD / 256; i++)
        sx[i * 256 + threadIdx.x] = x[i * 256 + threadIdx.x];
    __syncthreads();

    const int lane   = threadIdx.x & (P92_RTR_WAVE - 1);
    const int expert = blockIdx.x * 4 + (threadIdx.x >> 6);
    const uint4 *wr  = reinterpret_cast<const uint4 *>(W + (size_t)expert * P92_RTR_NEMBD);

    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < RTR_PER_LANE; i++) {
        const int c = i * P92_RTR_WAVE + lane;          // 16-byte chunk = 8 bf16
        const uint4 v = wr[c];
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
        const float *xp = sx + c * 8;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            acc = fmaf(p92_bf16((uint16_t)(w[j] & 0xFFFFu)), xp[2 * j],     acc);
            acc = fmaf(p92_bf16((uint16_t)(w[j] >> 16)),     xp[2 * j + 1], acc);
        }
    }
    #pragma unroll
    for (int off = P92_RTR_WAVE / 2; off; off >>= 1) acc += __shfl_down(acc, off, 64);
    if (lane == 0) logits[expert] = acc;
}

// Single block, 256 threads, one expert each. Iterative argmax with the
// deterministic tie rule the AMD selectors use: equal score, lower index wins.
__global__ void __launch_bounds__(256)
k_p92_route_top8(const float *__restrict__ logits, const float *__restrict__ bias,
                 int *__restrict__ ids, float *__restrict__ wts) {
    const int t = threadIdx.x;
    __shared__ float sc[P92_RTR_NEXP];      // uncorrected sigmoid scores
    __shared__ float sel[P92_RTR_NEXP];     // score + bias, selection only
    __shared__ float rv[4]; __shared__ int ri[4];
    __shared__ float wsum; __shared__ int best;

    const float s = 1.f / (1.f + __expf(-logits[t]));
    sc[t] = s;
    sel[t] = s + bias[t];
    if (t == 0) wsum = 0.f;
    __syncthreads();

    for (int k = 0; k < P92_RTR_TOPK; k++) {
        float v = sel[t];
        int   i = t;
        #pragma unroll
        for (int off = P92_RTR_WAVE / 2; off; off >>= 1) {
            const float ov = __shfl_down(v, off, 64);
            const int   oi = __shfl_down(i, off, 64);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if ((t & 63) == 0) { rv[t >> 6] = v; ri[t >> 6] = i; }
        __syncthreads();
        if (t == 0) {
            float bv = rv[0]; int bi = ri[0];
            #pragma unroll
            for (int w = 1; w < 4; w++)
                if (rv[w] > bv || (rv[w] == bv && ri[w] < bi)) { bv = rv[w]; bi = ri[w]; }
            ids[k] = bi;
            wts[k] = sc[bi];                 // uncorrected score is the weight
            wsum  += sc[bi];
            best   = bi;
        }
        __syncthreads();
        if (t == best) sel[t] = -3.0e38f;    // exclude, do not disturb sc
        __syncthreads();
    }
    if (t == 0) {
        const float inv = wsum > 0.f ? P92_RTR_SCALE / wsum : 0.f;
        #pragma unroll
        for (int k = 0; k < P92_RTR_TOPK; k++) wts[k] *= inv;
    }
}
