#pragma once
#include "p92_dsa_index.hip"
// openPangu lm_head, gfx906. [151552, 2560] untied, NVFP4 in the artifact with
// weight_scale_2 1.046e-04. At 218 MB it is the largest single per-token read
// in the model.
//
// The oracle does no special head kernel: model.cpp:246-247 is
// executor_.project(language_head_, normalized_, logits_) followed by
// cuda::argmax over kVocabulary. So the head is the ordinary NVFP4 projection
// at K=2560 - the same geometry as the expert and dense gate/up - with the row
// count grown to 151552 and no SwiGLU. Nothing new is needed but the grid.
//
// The one real question is the activation, and it turned out to matter. Every
// other NVFP4 consumer here takes the single int8 per-token quantisation, which
// is what makes the dot4 path possible. At the head that is not good enough:
// measured over 96 random activations, a single int8 component gave logits with
// 1.9e-02 relative error and flipped argmax against an fp32-activation
// reference on 3 of 96 trials. The head picks the emitted token, so a 3 percent
// token flip is a defect, not a tolerance.
//
// So the head activation carries TWO int8 components, x ~= s1*q1 + s2*q2, with
// q2 quantising the residual, giving about 14 bits of activation precision.
// Running that over all 151552 rows fixes the quality completely - argmax and
// top-5 both 96/96, logit error 1.9e-02 -> 5.1e-05 - but costs 483 -> 1620 us.
// The prediction that it would be nearly free because the nibble decode is
// shared was wrong: the second LDS staging and the halved occupancy dominate.
//
// So the two-component form is applied only where it changes an answer. The
// fast single-component pass ranks all 151552 rows, the top P92_HEAD_REFINE are
// recomputed at full precision, and argmax is taken over those. This is safe
// because the fast pass never put the true argmax outside its own top 5 in 96
// trials, and 256 is a wide margin on that.
#include "p92_nvfp4_dot4.hip"

#define P92_VOCAB 151552
#define P92_HEAD_REFINE 256
#define P92_HEAD_PER_BLOCK 32       // candidates a block, so staging happens 8 times not 256

// Fast pass: one activation component, all rows. Ranking only.
__global__ void __launch_bounds__(256)
k_p92_head_logits(const uint8_t *__restrict__ W, const uint8_t *__restrict__ S,
                  const int8_t *__restrict__ xq, const float *__restrict__ xs,
                  float sc2, float *__restrict__ logits) {
    __shared__ int8_t sx[P92_NEMBD];
    p92_stage_perm(xq, sx, P92_NEMBD);
    __syncthreads();

    const int l   = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= P92_VOCAB) return;
    const uint8_t *wr = W + (size_t)row * (P92_NEMBD / 2);
    const uint8_t *sr = S + (size_t)row * (P92_NEMBD / 16);

    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        const int g = i * 32 + l;
        const uint2 xe = *(const uint2 *)(sx + g * 16);
        const uint2 xo = *(const uint2 *)(sx + g * 16 + 8);
        acc = fmaf(p92_ue4m3_d(sr[g]),
                   (float)p92_group_dot4(*(const uint2 *)(wr + (size_t)g * 8), xe, xo), acc);
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_down(acc, off, 64);
    if (l == 0) logits[row] = acc * xs[0] * 0.5f * sc2;
}

// Refine pass: both activation components, candidate rows only. One block per
// candidate re-staged the same 5 KB of permuted activation 256 times and cost
// 817 us; a block now stages once and walks P92_HEAD_PER_BLOCK candidates.
__global__ void __launch_bounds__(256)
k_p92_head_refine(const uint8_t *__restrict__ W, const uint8_t *__restrict__ S,
                  const int8_t *__restrict__ xq, const int8_t *__restrict__ xq2,
                  const float *__restrict__ xs, const float *__restrict__ xs2,
                  float sc2, const int *__restrict__ cand, int ncand,
                  float *__restrict__ logits) {
    __shared__ int8_t sx[P92_NEMBD], sx2[P92_NEMBD];
    p92_stage_perm(xq,  sx,  P92_NEMBD);
    p92_stage_perm(xq2, sx2, P92_NEMBD);
    __syncthreads();

    // 32 lanes a candidate, 8 candidates in flight per block over the 160 groups
    const int l   = threadIdx.x & 31;
    const int sub = threadIdx.x >> 5;                       // 0..7
    for (int base = blockIdx.x * P92_HEAD_PER_BLOCK; base < (blockIdx.x + 1) * P92_HEAD_PER_BLOCK;
         base += 8) {
        const int c = base + sub;
        if (c >= ncand) break;
        const int row = cand[c];
        if (row < 0 || row >= P92_VOCAB) continue;
        const uint8_t *wr = W + (size_t)row * (P92_NEMBD / 2);
        const uint8_t *sr = S + (size_t)row * (P92_NEMBD / 16);
        float acc = 0.f, acc2 = 0.f;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            const int g = i * 32 + l;
            const uint2 pk = *(const uint2 *)(wr + (size_t)g * 8);
            const float gs = p92_ue4m3_d(sr[g]);
            acc  = fmaf(gs, (float)p92_group_dot4(pk, *(const uint2 *)(sx  + g * 16),
                                                      *(const uint2 *)(sx  + g * 16 + 8)), acc);
            acc2 = fmaf(gs, (float)p92_group_dot4(pk, *(const uint2 *)(sx2 + g * 16),
                                                      *(const uint2 *)(sx2 + g * 16 + 8)), acc2);
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) { acc += __shfl_down(acc, off, 64); acc2 += __shfl_down(acc2, off, 64); }
        if (l == 0) logits[row] = (acc * xs[0] + acc2 * xs2[0]) * 0.5f * sc2;
    }
}

// Deterministic greedy argmax: equal logits select the lower token id, which is
// the same tie rule the router and the DSA selector use.
#define P92_HEAD_RED 512
__global__ void __launch_bounds__(256)
k_p92_head_argmax_part(const float *__restrict__ logits,
                       float *__restrict__ pv, int *__restrict__ pi) {
    const int chunk = blockIdx.x;
    const int lo = chunk * P92_HEAD_RED;
    int hi = lo + P92_HEAD_RED; if (hi > P92_VOCAB) hi = P92_VOCAB;
    float bv = -3.0e38f; int bi = P92_VOCAB;
    for (int i = lo + threadIdx.x; i < hi; i += 256) {
        const float v = logits[i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) {
        const float ov = __shfl_down(bv, o, 64);
        const int   oi = __shfl_down(bi, o, 64);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float rv[4]; __shared__ int ri[4];
    if ((threadIdx.x & 63) == 0) { rv[threadIdx.x >> 6] = bv; ri[threadIdx.x >> 6] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = rv[0]; int idx = ri[0];
        #pragma unroll
        for (int w = 1; w < 4; w++)
            if (rv[w] > v || (rv[w] == v && ri[w] < idx)) { v = rv[w]; idx = ri[w]; }
        pv[chunk] = v; pi[chunk] = idx;
    }
}

__global__ void __launch_bounds__(256)
k_p92_head_argmax_final(const float *__restrict__ pv, const int *__restrict__ pi,
                        int chunks, int *__restrict__ out) {
    float bv = -3.0e38f; int bi = P92_VOCAB;
    for (int i = threadIdx.x; i < chunks; i += 256) {
        const float v = pv[i];
        if (v > bv || (v == bv && pi[i] < bi)) { bv = v; bi = pi[i]; }
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) {
        const float ov = __shfl_down(bv, o, 64);
        const int   oi = __shfl_down(bi, o, 64);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float rv[4]; __shared__ int ri[4];
    if ((threadIdx.x & 63) == 0) { rv[threadIdx.x >> 6] = bv; ri[threadIdx.x >> 6] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = rv[0]; int idx = ri[0];
        #pragma unroll
        for (int w = 1; w < 4; w++)
            if (rv[w] > v || (rv[w] == v && ri[w] < idx)) { v = rv[w]; idx = ri[w]; }
        *out = idx;
    }
}
