#pragma once
// openPangu mHC (multi-stream hyper-connections), gfx906 wave64.
//
// Donor: the GLM gfx906 kernel mhc_step_gfx906.hip, which is near-exact. The
// structure is confirmed by the checkpoint and the config rather than assumed:
//   phi.weight   [24, 10240]   GLM's fn [24, 16384]; 24 = (2 + H) * H with H = 4
//   branch_alpha [3]           GLM's per-group scale, for pre / post / comb
//   branch_beta  [24]          GLM's per-logit base
//   norm_gamma   [10240]       openPangu only; config mhc_use_gamma is true and
//                              the GLM flat step carries no gamma
//   config       mhc_num_stream 4, mhc_recur_norm 20 = the Sinkhorn iterations
// So the geometry delta is D 4096 -> 2560 and FLAT 16384 -> 10240, plus gamma.
//
// One deliberate departure from the donor. GLM's reexpand accumulates the
// four-term dot in fp64 to reproduce a torch bf16 rounding tie exactly. This
// port does not chase bit-exactness against any reference implementation, so
// the accumulation is fp32 and the gate is relative error against a scalar
// reference of the contract. That is the standing law for this port and it
// removes the fp64 path from the hot loop.
//
// mHC brackets a sublayer: collapse the four streams into one vector before
// attention or the MLP runs, then re-expand the sublayer's output back into
// four streams. So the three kernels are not free to be fused into one.
#include <hip/hip_runtime.h>
#include <cstdint>

#define P92_MHC_H      4                        // mhc_num_stream
#define P92_MHC_D      2560                     // hidden
#define P92_MHC_FLAT   (P92_MHC_H * P92_MHC_D)  // 10240
#define P92_MHC_MIX    24                       // (2 + H) * H
#define P92_MHC_ITERS  20                       // mhc_recur_norm
#define P92_MHC_HC_EPS  1e-6f
#define P92_MHC_RMS_EPS 1e-5f

typedef uint16_t p92_mbf;
static __device__ __forceinline__ float p92_mb2f(p92_mbf b){ return __uint_as_float((uint32_t)b<<16); }
static __device__ __forceinline__ p92_mbf p92_mf2b(float f){
    uint32_t u = __float_as_uint(f); u += 0x7FFFu + ((u >> 16) & 1u); return (p92_mbf)(u >> 16); }

#define P92_MHC_SPLIT 4

// ---- logits[24] = phi @ (rmsnorm(x) * gamma) -----------------------------
// Grid is (24 rows, 4 splits) so 96 blocks fill the 60 CUs; one block per row
// left the machine 40 percent occupied and measured 18.16 us against a 2.7 us
// byte floor. Partials are written per split and summed in the map kernel,
// which is scalar work anyway - that keeps the sum order fixed, where an
// atomicAdd into a single accumulator would not.
//
// The rmsnorm scale needs no kernel of its own. inv_rms is a scalar factor on
// the whole sum, so the GEMV can run unscaled and the map applies it at the
// end; and the four row-0 blocks are already reading exactly the four stream
// chunks the sum of squares needs. A standalone rms kernel measured 9.38 us,
// more than the GEMV it fed.
__global__ void __launch_bounds__(256)
k_p92_mhc_logits(const p92_mbf *__restrict__ x,      // [4][2560] streams
                 const p92_mbf *__restrict__ phi,    // [24][10240]
                 const p92_mbf *__restrict__ gamma,  // [10240]
                 float *__restrict__ partial,        // [24][4]
                 float *__restrict__ ssq_partial) {  // [4], written by row 0
    const int r = blockIdx.x, sp = blockIdx.y, t = threadIdx.x;
    const int lane = t & 63, wv = t >> 6;
    __shared__ float red[4], red2[4];
    const int per = P92_MHC_FLAT / P92_MHC_SPLIT;
    const int lo = sp * per, hi = lo + per;

    const p92_mbf *pr = phi + (size_t)r * P92_MHC_FLAT;
    float acc = 0.f, ss = 0.f;
    const bool do_ssq = (r == 0);
    // Four bf16 a load instead of one. Three streams are walked here - the phi
    // row, the stream block and gamma - and reading them a single 2-byte
    // element at a time issued twelve loads per four elements. As uint2 it is
    // three. The block's slice is 2560 elements over 256 threads, which divides
    // evenly by four, so no tail.
    const uint2 *pr4 = reinterpret_cast<const uint2 *>(pr + lo);
    const uint2 *x4  = reinterpret_cast<const uint2 *>(x  + lo);
    const uint2 *g4  = reinterpret_cast<const uint2 *>(gamma + lo);
    const int n4 = (hi - lo) / 4;
    for (int i = t; i < n4; i += 256) {
        const uint2 pv = pr4[i], xv2 = x4[i], gv = g4[i];
        const uint32_t pw[2] = {pv.x, pv.y}, xw[2] = {xv2.x, xv2.y}, gw[2] = {gv.x, gv.y};
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const float xa = p92_mb2f((p92_mbf)(xw[j] & 0xFFFFu));
            const float xb = p92_mb2f((p92_mbf)(xw[j] >> 16));
            acc = fmaf(p92_mb2f((p92_mbf)(pw[j] & 0xFFFFu)),
                       xa * p92_mb2f((p92_mbf)(gw[j] & 0xFFFFu)), acc);
            acc = fmaf(p92_mb2f((p92_mbf)(pw[j] >> 16)),
                       xb * p92_mb2f((p92_mbf)(gw[j] >> 16)), acc);
            if (do_ssq) { ss = fmaf(xa, xa, ss); ss = fmaf(xb, xb, ss); }
        }
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) { acc += __shfl_down(acc, o, 64); ss += __shfl_down(ss, o, 64); }
    if (lane == 0) { red[wv] = acc; red2[wv] = ss; }
    __syncthreads();
    if (t == 0) {
        partial[r * P92_MHC_SPLIT + sp] = red[0] + red[1] + red[2] + red[3];
        if (do_ssq) ssq_partial[sp] = red2[0] + red2[1] + red2[2] + red2[3];
    }
}

// ---- map + collapse ------------------------------------------------------
// The map is 24 logits and a 4x4 Sinkhorn: scalar work, one thread. The
// collapse that follows is 2560 wide and uses the whole block.
// Split follows the donor: [0,4) pre, [4,8) post, [8,24) comb, and branch_alpha
// indexes the three groups.
__global__ void __launch_bounds__(256)
k_p92_mhc_collapse(const float *__restrict__ partial,
                   const float *__restrict__ ssq_partial,
                   const p92_mbf *__restrict__ alpha,   // [3]  scale
                   const p92_mbf *__restrict__ beta,    // [24] base
                   const p92_mbf *__restrict__ x,       // [4][2560]
                   float *__restrict__ pre, float *__restrict__ post,
                   float *__restrict__ comb, float *__restrict__ collapsed) {
    __shared__ float spre[P92_MHC_H];
    // The map is a 4x4 Sinkhorn. Run serially in one lane it measured 33.30 us
    // of a 54.61 us mHC site - 20 iterations of row and column normalisation,
    // 640 dependent divisions in a single lane while the rest of the wave
    // stalls. Here lane i*4+j owns element (i,j): a row sum is two xor
    // shuffles within a group of four, a column sum is two more, and the
    // twenty iterations are forty shuffle pairs with no serial chain.
    __shared__ float logits[P92_MHC_MIX];
    __shared__ float s_inv;
    if (threadIdx.x == 0) {
        float ss = 0.f;
        #pragma unroll
        for (int sp = 0; sp < P92_MHC_SPLIT; sp++) ss += ssq_partial[sp];
        s_inv = rsqrtf(ss / (float)P92_MHC_FLAT + P92_MHC_RMS_EPS);
    }
    __syncthreads();
    if (threadIdx.x < P92_MHC_MIX) {
        float a = 0.f;
        #pragma unroll
        for (int sp = 0; sp < P92_MHC_SPLIT; sp++) a += partial[threadIdx.x * P92_MHC_SPLIT + sp];
        logits[threadIdx.x] = a * s_inv;          // the hoisted rmsnorm scale
    }
    __syncthreads();
    // AMD's own aiter mHC kernel runs the Sinkhorn on a wave CONCURRENT with the
    // collapse. The collapse consumes only `pre`, never `comb`, so it has no
    // reason to wait behind 20 sequential Sinkhorn iterations. pre and post are
    // computed first by four threads, then wave 0 runs the Sinkhorn while waves
    // 1-3 run the collapse.
    if (threadIdx.x < 64) {
        const int l = threadIdx.x;
        const float a0 = p92_mb2f(alpha[0]), a1 = p92_mb2f(alpha[1]);
        if (l < P92_MHC_H) {
            const float pv = 1.f / (1.f + __expf(-(logits[l] * a0 + p92_mb2f(beta[l]))))
                             + P92_MHC_HC_EPS;
            pre[l]  = pv; spre[l] = pv;
            post[l] = 2.f / (1.f + __expf(-(logits[P92_MHC_H + l] * a1
                                            + p92_mb2f(beta[P92_MHC_H + l]))));
        }
    }
    __syncthreads();                 // spre is now visible; the collapse can start
    if (threadIdx.x >= 64) {         // waves 1-3 collapse while wave 0 sinkhorns
        for (int d = threadIdx.x - 64; d < P92_MHC_D; d += 192) {
            float a = 0.f;
            #pragma unroll
            for (int h = 0; h < P92_MHC_H; h++)
                a = fmaf(spre[h], p92_mb2f(x[(size_t)h * P92_MHC_D + d]), a);
            collapsed[d] = a;        // fp32: the consumer is an RMSNorm
        }
    } else {
        const int l = threadIdx.x;
        const float a2 = p92_mb2f(alpha[2]);
        const bool on = l < P92_MHC_H * P92_MHC_H;
        const int i = l >> 2, j = l & 3;
        float v = on ? logits[2 * P92_MHC_H + l] * a2 + p92_mb2f(beta[2 * P92_MHC_H + l])
                     : -3.0e38f;
        // softmax over j, within each group of four lanes
        // gfx906: __shfl_xor lowers to ds_bpermute (DS path). This kernel is one
        // wave with nothing in flight to hide that latency, and it issues 82 of
        // them. DPP runs the same crossbar on the VALU operand path instead.
        // quad_perm [1,0,3,2] / [2,3,0,1] are exactly xor 1 / xor 2 over j.
        // row_ror:4 then row_ror:8 sums {l, l+4, l+8, l+12} mod 16, which is the
        // same SET of lanes as {l, l^4, l^8, l^12} since rotation by 4 preserves
        // bits 0-1. Every source lane is active here, so update_dpp never falls
        // back to the old operand and the sums cannot double-count. Validated lane-by-lane
        // against __shfl_xor before use; see receipts/10_GFX906_PLAYBOOK.md.
        #define P92_DPPF(x, c) __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(x), __float_as_int(x), (c), 0xF, 0xF, false))
        #define P92_QP_X1 0xB1
        #define P92_QP_X2 0x4E
        #define P92_ROR4  0x124
        #define P92_ROR8  0x128
        float mx = v;
        mx = fmaxf(mx, P92_DPPF(mx, P92_QP_X1));
        mx = fmaxf(mx, P92_DPPF(mx, P92_QP_X2));
        float e = on ? __expf(v - mx) : 0.f;
        float s = e;
        s += P92_DPPF(s, P92_QP_X1); s += P92_DPPF(s, P92_QP_X2);
        v = on ? e / s + P92_MHC_HC_EPS : 0.f;
        // one column pass, then ITERS-1 of (row, column)
        // COLSUM stays on ds_bpermute ON PURPOSE. row_ror:4/8 sums the same SET of
        // lanes but pairs them differently: for lane 4, xor gives (v4+v0)+(v12+v8)
        // while ror gives (v4+v8)+(v12+v0). Float add is not associative, so the
        // rotational form reassociates and perturbs generation. quad_perm above is
        // a true xor 1 / xor 2, so it is bit-exact and is the half we keep.
        #define P92_MIRROR 0x140
        #define P92_HALFMIR 0x141
        #define P92_XOR8(x) P92_DPPF(P92_DPPF((x), P92_MIRROR), P92_HALFMIR)
        #define P92_COLSUM(acc) { acc = v; acc += __shfl_xor(acc, 4, 64); acc += P92_XOR8(acc); }
        #define P92_ROWSUM(acc) { acc = v; acc += P92_DPPF(acc, P92_QP_X1); acc += P92_DPPF(acc, P92_QP_X2); }
        { float cs; P92_COLSUM(cs); cs += P92_MHC_HC_EPS; if (on) v /= cs; }
        for (int it = 0; it < P92_MHC_ITERS - 1; it++) {
            { float rs; P92_ROWSUM(rs); rs += P92_MHC_HC_EPS; if (on) v /= rs; }
            { float cs; P92_COLSUM(cs); cs += P92_MHC_HC_EPS; if (on) v /= cs; }
        }
        #undef P92_COLSUM
        #undef P92_ROWSUM
        if (on) comb[i * P92_MHC_H + j] = v;
    }
}

// ---- re-expand -----------------------------------------------------------
// y[h][d] = post[h] * m[d] + sum_k comb[k][h] * residual[k][d]
// fp32 throughout; the donor's fp64 term exists only to match a torch bf16
// rounding tie and is not wanted here.
__global__ void __launch_bounds__(256)
k_p92_mhc_reexpand(const float *__restrict__ post, const float *__restrict__ comb,
                   const float *__restrict__ m, const p92_mbf *__restrict__ residual,
                   p92_mbf *__restrict__ y) {
    const int d = blockIdx.x * 256 + threadIdx.x;
    if (d >= P92_MHC_D) return;
    const float mv = m[d];   // fp32 straight from the RMSNorm
    float res[P92_MHC_H];
    #pragma unroll
    for (int k = 0; k < P92_MHC_H; k++) res[k] = p92_mb2f(residual[(size_t)k * P92_MHC_D + d]);
    #pragma unroll
    for (int h = 0; h < P92_MHC_H; h++) {
        float a = post[h] * mv;
        #pragma unroll
        for (int k = 0; k < P92_MHC_H; k++) a = fmaf(comb[k * P92_MHC_H + h], res[k], a);
        y[(size_t)h * P92_MHC_D + d] = p92_mf2b(a);
    }
}

// ---- model-level merge ---------------------------------------------------
// model.merge_mhc_module has phi [4, 10240], branch_alpha_pre [1] and
// branch_beta_pre [4]: four logits, so it produces only the pre weights and
// collapses the four streams back to one. There is no post and no comb.
__global__ void __launch_bounds__(256)
k_p92_mhc_merge(const float *__restrict__ logits, const p92_mbf *__restrict__ alpha_pre,
                const p92_mbf *__restrict__ beta_pre, const p92_mbf *__restrict__ x,
                p92_mbf *__restrict__ out) {
    __shared__ float spre[P92_MHC_H];
    if (threadIdx.x < P92_MHC_H) {
        const float a = p92_mb2f(alpha_pre[0]);
        spre[threadIdx.x] = 1.f / (1.f + __expf(-(logits[threadIdx.x] * a
                                   + p92_mb2f(beta_pre[threadIdx.x])))) + P92_MHC_HC_EPS;
    }
    __syncthreads();
    for (int d = blockIdx.x * 256 + threadIdx.x; d < P92_MHC_D; d += gridDim.x * 256) {
        float acc = 0.f;
        #pragma unroll
        for (int h = 0; h < P92_MHC_H; h++)
            acc = fmaf(spre[h], p92_mb2f(x[(size_t)h * P92_MHC_D + d]), acc);
        out[d] = p92_mf2b(acc);
    }
}
