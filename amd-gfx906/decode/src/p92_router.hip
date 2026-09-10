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
// Selecting 8 of 256 measured 25.00 us a call, 1.100 ms a token over the 44 MoE
// layers. Two structures were tried against that.
//
// A four-wave form with wave-local top-8 and a 32-candidate merge removed 23 of
// the 24 barriers and was SLOWER, 28.19 us: it trades barriers for 56 shuffles
// a wave, and on gfx906 a shuffle is ds_bpermute, the same LDS path, while
// three of the four waves idle through the merge. Recorded so it is not retried.
//
// This form uses ONE wave. Each lane holds four experts in registers, so a round
// is three register compares plus one 64-lane argmax, and there is no barrier,
// no LDS and no cross-wave traffic anywhere.
// Cross-lane reduction on the ALU path, not the LDS path.
//
// __shfl compiles to ds_bpermute on gfx906, which uses the LDS unit. In a
// one-wave kernel there is no other wave to hide its latency, and it dominated:
// measured with the round count varied, the kernel is a 2.23 us floor plus
// about 0.84 us a round, and replacing the cross-lane step with pure ALU of the
// same dependency depth drops the slope to 0.44. DPP row_shr/row_bcast does the
// same reduction on the ALU path and recovers almost all of it - 6.07 us
// against 5.90 for no cross-lane at all, and 8.96 for ds_bpermute.
//
// This is what AMD's own aiter mHC kernel uses. Note it matters HERE because
// the kernel is a single wave; in the big expert kernels many waves are in
// flight and the same latency is hidden.
//
// row_shr 1,2,4,8 reduces within each row of 16; row_bcast15 then row_bcast31
// combine the four rows, leaving the result in lane 63. `old` is the lane's own
// value so an out-of-range source is a max-identity rather than zero, which
// would be wrong for the -3.0e38 masked entries.
#define P92_DPP(x, ctrl) __builtin_amdgcn_update_dpp((x), (x), (ctrl), 0xF, 0xF, false)
#define P92_DPP_MAXIDX(C)                                                    \
    { const float ov = __int_as_float(P92_DPP(__float_as_int(bv), C));       \
      const int   oi = P92_DPP(bi, C);                                       \
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }

__global__ void __launch_bounds__(64)
k_p92_route_top8(const float *__restrict__ logits, const float *__restrict__ bias,
                 int *__restrict__ ids, float *__restrict__ wts) {
    const int lane = threadIdx.x;
    float sc[4], sel[4]; int idx[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int e = j * 64 + lane;
        const float s = 1.f / (1.f + __expf(-logits[e]));
        sc[j] = s;                       // uncorrected score is the weight
        sel[j] = s + bias[e];            // corrected, selection only
        idx[j] = e;
    }
    // Measured cost structure: a 2.23 us floor - launch, 512 loads, 256 sigmoids -
    // plus about 0.81 us for each of the eight rounds, so the dependent chain is
    // 6.5 of 8.9 us. Two dependent operations a round are removable.
    //
    // __shfl_xor instead of __shfl_down: a butterfly leaves the result in EVERY
    // lane, so the broadcast that __shfl_down needs before the owner can mask
    // its value out disappears.
    //
    // The weight lookup is deferred. Fetching sc[bi] cost a second broadcast a
    // round; instead the eight chosen ids are kept in registers and each lane
    // contributes the scores it owns in one pass at the end.
    int chosen[P92_RTR_TOPK];
    #pragma unroll
    for (int k = 0; k < P92_RTR_TOPK; k++) {
        float bv = sel[0]; int bi = idx[0];
        #pragma unroll
        for (int j = 1; j < 4; j++)
            if (sel[j] > bv || (sel[j] == bv && idx[j] < bi)) { bv = sel[j]; bi = idx[j]; }
        P92_DPP_MAXIDX(0x111) P92_DPP_MAXIDX(0x112)
        P92_DPP_MAXIDX(0x114) P92_DPP_MAXIDX(0x118)
        P92_DPP_MAXIDX(0x142) P92_DPP_MAXIDX(0x143)
        bi = __builtin_amdgcn_readlane(bi, 63);   // SALU broadcast from lane 63
        chosen[k] = bi;
        #pragma unroll
        for (int j = 0; j < 4; j++) if (idx[j] == bi) sel[j] = -3.0e38f;
    }
    // every lane hands over the scores of whichever chosen ids it owns; one
    // readlane a weight, on the scalar path
    float w[P92_RTR_TOPK];
    #pragma unroll
    for (int k = 0; k < P92_RTR_TOPK; k++) {
        float v = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; j++) if (idx[j] == chosen[k]) v = sc[j];
        w[k] = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), chosen[k] & 63));
    }
    if (lane == 0) {
        float wsum = 0.f;
        #pragma unroll
        for (int k = 0; k < P92_RTR_TOPK; k++) wsum += w[k];
        const float inv = wsum > 0.f ? P92_RTR_SCALE / wsum : 0.f;
        #pragma unroll
        for (int k = 0; k < P92_RTR_TOPK; k++) { ids[k] = chosen[k]; wts[k] = w[k] * inv; }
    }
}


