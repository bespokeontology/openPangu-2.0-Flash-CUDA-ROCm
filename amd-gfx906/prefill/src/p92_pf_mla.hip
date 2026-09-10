// Batched openPangu MLA / mHC / dense body for prefill on gfx906.
//
// These are the stages the first composed body was missing entirely: nothing
// produced q_abs or q_rope, nothing wrote the KV cache, and there was no
// o_proj, dense MLP or residual. Attention was reading allocated-but-unwritten
// scratch. Semantics are transcribed from the frozen decode engine
// (p92_mla_prep.hip, p92_mhc.hip, p92_dense.hip, p92_mome.hip); the shape is
// the prefill one - every kernel carries T rows and the weight-side operand is
// hoisted once and swept across rows where the shape allows it.
//
// Written for FIRST LIGHT, not for the roofline. The ranking pass comes after a
// composed number exists, because a stage that does not run cannot be ranked.
#include <hip/hip_runtime.h>
#include <cstdint>

#ifndef PM_GEOMETRY_H
#define PM_GEOMETRY_H
#define PM_HEADS    48
#define PM_LATENT  512
#define PM_ROPE     64
#define PM_NOPE    128
#define PM_VALUE   128
#define PM_QUP     192                  // per head: 128 nope + 64 rope
#define PM_UPPH    256                  // kv_b rows a head: 128 nope + 128 value
#define PM_THETA   6400000.0f
#define PM_MHC_H     4
#define PM_MHC_D  2560
#define PM_MHC_FLAT (PM_MHC_H * PM_MHC_D)
#define PM_MHC_MIX  24
#define PM_MHC_EPS  1e-6f
#define PM_RMS_EPS  1e-5f
#define PM_DENSE_FF 9216
#endif

static __device__ __forceinline__ float pm_b2f(uint16_t b) {
    return __uint_as_float((uint32_t)b << 16);
}
// round-to-nearest-even, matching k_p92_key_rope and the decode writers
static __device__ __forceinline__ uint16_t pm_f2b(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// ---- RoPE table for every position in the run, once -----------------------
// Decode rebuilds a 32-entry table per token. Prefill needs all of them and the
// table is tiny, so it is built once at startup rather than per chunk.
__global__ void __launch_bounds__(64)
k_pf_rope_table(float *__restrict__ cosv, float *__restrict__ sinv, int npos) {
    const int p = blockIdx.x, i = threadIdx.x;
    if (p >= npos || i >= PM_ROPE / 2) return;
    const float inv = __powf(PM_THETA, -2.0f * (float)i / (float)PM_ROPE);
    const float a = (float)p * inv;
    cosv[(size_t)p * 32 + i] = __cosf(a);
    sinv[(size_t)p * 32 + i] = __sinf(a);
}

// ---- RMS norm over K, batched --------------------------------------------
template <int K>
__global__ void __launch_bounds__(256)
k_pf_rmsnorm_k(const float *__restrict__ x, const uint16_t *__restrict__ gamma,
               float *__restrict__ y, int T, float eps) {
    const int row = blockIdx.x;
    if (row >= T) return;
    const float *xr = x + (size_t)row * K;
    float ss = 0.f;
    for (int i = threadIdx.x; i < K; i += 256) { const float v = xr[i]; ss = fmaf(v, v, ss); }
    #pragma unroll
    for (int o = 32; o; o >>= 1) ss += __shfl_down(ss, o, 64);
    __shared__ float red[4];
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = ss;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) inv = rsqrtf((red[0]+red[1]+red[2]+red[3]) / (float)K + eps);
    __syncthreads();
    float *yr = y + (size_t)row * K;
    for (int i = threadIdx.x; i < K; i += 256) yr[i] = xr[i] * inv * pm_b2f(gamma[i]);
}

// ---- int8 quantise over K, batched ---------------------------------------
// PRODUCER OWNS THE LAYOUT. Scales land lane-contiguous at (g%32)*PER + (g/32)
// and values even-then-odd inside each 16-group, exactly as k_pf_quant emits
// for 2560, because k_pf_proj<K> and k_pf_gateup consume that and nothing else.
template <int K>
__global__ void __launch_bounds__(256)
k_pf_quant_k(const float *__restrict__ x, int T,
             int8_t *__restrict__ xq, float *__restrict__ xs) {
    constexpr int GRP = K / 16, PER = GRP / 32;
    const int row = blockIdx.x;
    if (row >= T) return;
    for (int g = threadIdx.x; g < GRP; g += 256) {
        const float *src = x + (size_t)row * K + (size_t)g * 16;
        float f[16], m = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; j++) { f[j] = src[j]; m = fmaxf(m, fabsf(f[j])); }
        const float sc  = m > 0.f ? m * (1.f / 127.f) : 1.f;
        const float inv = m > 0.f ? 127.f / m : 0.f;
        xs[(size_t)row * GRP + (g % 32) * PER + (g / 32)] = sc;
        int8_t *dst = xq + (size_t)row * K + (size_t)g * 16;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int q = (int)rintf(f[j] * inv);
            q = q < -128 ? -128 : (q > 127 ? 127 : q);
            dst[(j & 1) ? 8 + (j >> 1) : (j >> 1)] = (int8_t)q;
        }
    }
}

// ---- bf16 GEMV, batched over rows ----------------------------------------
// grid ((N+3)/4, T), block 256: four output rows a block, one token. The
// activation row is staged once in LDS and shared by all four.
template <int K>
__global__ void __launch_bounds__(256)
k_pf_bf16_gemv_rows(const uint16_t *__restrict__ W, const float *__restrict__ x,
                    int T, int N, float *__restrict__ y) {
    const int t = blockIdx.y;
    if (t >= T) return;
    __shared__ float sx[K];
    for (int i = threadIdx.x; i < K; i += 256) sx[i] = x[(size_t)t * K + i];
    __syncthreads();                       // every thread arrives, then we bound
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= N) return;
    const int l = threadIdx.x & 63;
    const uint2 *wr = (const uint2 *)(W + (size_t)row * K);
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < K / 256; i++) {
        const int c = i * 64 + l;
        const uint2 v = wr[c];
        const float *xp = sx + c * 4;
        acc = fmaf(pm_b2f((uint16_t)(v.x & 0xFFFF)), xp[0], acc);
        acc = fmaf(pm_b2f((uint16_t)(v.x >> 16)),    xp[1], acc);
        acc = fmaf(pm_b2f((uint16_t)(v.y & 0xFFFF)), xp[2], acc);
        acc = fmaf(pm_b2f((uint16_t)(v.y >> 16)),    xp[3], acc);
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) acc += __shfl_down(acc, o, 64);
    if (l == 0) y[(size_t)t * N + row] = acc;
}

// ---- MoME depthwise causal conv, batched, with an explicit row stride -----
// The chunk carry is the whole point: tap 0 is the oldest sample and the two
// carried values cross the chunk boundary, so chunk k+1 continues chunk k's
// convolution rather than restarting it. kv_down is [T][576] but the conv runs
// over 512 channels, so the stride cannot be assumed equal to the channel count.
__global__ void __launch_bounds__(256)
k_pf_mome_st(float *__restrict__ x, const uint16_t *__restrict__ wt,
             uint16_t *__restrict__ carry, int T, int C, int stride) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    const float w0 = pm_b2f(wt[(size_t)c*3+0]), w1 = pm_b2f(wt[(size_t)c*3+1]),
                w2 = pm_b2f(wt[(size_t)c*3+2]);
    float m2 = pm_b2f(carry[(size_t)c*2+0]);      // t-2, oldest
    float m1 = pm_b2f(carry[(size_t)c*2+1]);      // t-1
    for (int t = 0; t < T; t++) {
        const float cur = x[(size_t)t * stride + c];
        x[(size_t)t * stride + c] = cur + fmaf(m2, w0, fmaf(m1, w1, cur * w2));
        m2 = m1; m1 = cur;
    }
    carry[(size_t)c*2+0] = pm_f2b(m2);
    carry[(size_t)c*2+1] = pm_f2b(m1);
}

// ---- MLA absorb: q_up nope half through kv_b -> latent query --------------
// grid (48, 2, T), block 256.
__global__ void __launch_bounds__(256)
k_pf_absorb(const float *__restrict__ qup, const uint16_t *__restrict__ kv_b,
            int T, float *__restrict__ qabs) {
    const int h = blockIdx.x, j = blockIdx.y * 256 + threadIdx.x, t = blockIdx.z;
    if (t >= T) return;
    __shared__ float qn[PM_NOPE];
    if (threadIdx.x < PM_NOPE)
        qn[threadIdx.x] = qup[((size_t)t * PM_HEADS + h) * PM_QUP + threadIdx.x];
    __syncthreads();
    const uint16_t *w = kv_b + (size_t)h * PM_UPPH * PM_LATENT + j;
    float acc = 0.f;
    #pragma unroll 8
    for (int i = 0; i < PM_NOPE; i++)
        acc = fmaf(qn[i], pm_b2f(w[(size_t)i * PM_LATENT]), acc);
    qabs[((size_t)t * PM_HEADS + h) * PM_LATENT + j] = acc;
}

// ---- query RoPE: extract the rope half of q_up and rotate ----------------
__global__ void __launch_bounds__(32)
k_pf_qrope(const float *__restrict__ qup, const float *__restrict__ cosv,
           const float *__restrict__ sinv, int T, int base, float *__restrict__ qrope) {
    const int h = blockIdx.x, t = blockIdx.y;
    if (t >= T) return;
    const int i = threadIdx.x;
    const float *src = qup + ((size_t)t * PM_HEADS + h) * PM_QUP + PM_NOPE;
    float *dst = qrope + ((size_t)t * PM_HEADS + h) * PM_ROPE;
    const float c = cosv[(size_t)(base + t) * 32 + i], s = sinv[(size_t)(base + t) * 32 + i];
    const float a = src[i], b = src[i + 32];
    dst[i] = a * c - b * s;
    dst[i + 32] = b * c + a * s;
}

// ---- normalise kv_down and publish this chunk's keys into the cache ------
// One block a token: RMS-norms the 512 latent dims with kv_norm and writes them
// as bf16 at the ABSOLUTE position, then rotates the 64 rope dims into the rope
// cache. This is what makes chunk k's keys visible to chunk k+1.
__global__ void __launch_bounds__(256)
k_pf_kv_cache(const float *__restrict__ kvdn, const uint16_t *__restrict__ kv_norm,
              const float *__restrict__ cosv, const float *__restrict__ sinv,
              int T, int base, int stride,
              uint16_t *__restrict__ cache_kv, uint16_t *__restrict__ cache_rope) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const float *src = kvdn + (size_t)t * stride;
    float ss = 0.f;
    for (int i = threadIdx.x; i < PM_LATENT; i += 256) { const float v = src[i]; ss = fmaf(v, v, ss); }
    #pragma unroll
    for (int o = 32; o; o >>= 1) ss += __shfl_down(ss, o, 64);
    __shared__ float red[4];
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = ss;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) inv = rsqrtf((red[0]+red[1]+red[2]+red[3]) / (float)PM_LATENT + PM_RMS_EPS);
    __syncthreads();
    uint16_t *dst = cache_kv + (size_t)(base + t) * PM_LATENT;
    for (int i = threadIdx.x; i < PM_LATENT; i += 256)
        dst[i] = pm_f2b(src[i] * inv * pm_b2f(kv_norm[i]));
    if (threadIdx.x < 32) {
        const int i = threadIdx.x;
        const float *rp = src + PM_LATENT;
        const float c = cosv[(size_t)(base + t) * 32 + i], s = sinv[(size_t)(base + t) * 32 + i];
        const float a = rp[i], b = rp[i + 32];
        uint16_t *kr = cache_rope + (size_t)(base + t) * PM_ROPE;
        kr[i]      = pm_f2b(a * c - b * s);
        kr[i + 32] = pm_f2b(b * c + a * s);
    }
}

// ---- latent context back up through kv_b's value half --------------------
__global__ void __launch_bounds__(256)
k_pf_value_up(const float *__restrict__ ctx, const uint16_t *__restrict__ kv_b,
              int T, float *__restrict__ out) {
    const int t = blockIdx.y;
    if (t >= T) return;
    const int lane = threadIdx.x & 63;
    const int idx  = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (idx >= PM_HEADS * PM_VALUE) return;
    const int h = idx / PM_VALUE, v = idx % PM_VALUE;
    const uint16_t *w = kv_b + ((size_t)h * PM_UPPH + PM_NOPE + v) * PM_LATENT;
    const float *c = ctx + ((size_t)t * PM_HEADS + h) * PM_LATENT;
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < PM_LATENT / 64; i++) { const int d = i * 64 + lane;
        acc = fmaf(c[d], pm_b2f(w[d]), acc); }
    #pragma unroll
    for (int o = 32; o; o >>= 1) acc += __shfl_down(acc, o, 64);
    if (lane == 0) out[(size_t)t * (PM_HEADS * PM_VALUE) + h * PM_VALUE + v] = acc;
}

// ---- mHC: hoisted RMS scale applied to the 24 mixing logits --------------
// Decode folds this into the collapse via ssq_partial. The prefill logits
// kernel does not emit it, so it is a separate pass over the row.
__global__ void __launch_bounds__(256)
k_pf_mhc_scale(const uint16_t *__restrict__ x, const uint16_t *__restrict__ gam,
               int T, float *__restrict__ logits) {
    const int row = blockIdx.x;
    if (row >= T) return;
    const uint16_t *xr = x + (size_t)row * PM_MHC_FLAT;
    float ss = 0.f;
    // ssq is over the RAW stream, NOT the gamma-weighted value. gamma scales
    // the mixing logits; it does not enter the norm those logits are divided
    // by (p92_mhc.hip: acc uses x*gamma, ss uses x alone). Squaring the
    // weighted value instead put a wrong scale on every mHC logit from layer 0
    // onward - prefill layer 0 came out 8.8 percent off decode on the same
    // token, growing to 55 percent by layer 1.
    for (int i = threadIdx.x; i < PM_MHC_FLAT; i += 256) {
        const float v = pm_b2f(xr[i]); ss = fmaf(v, v, ss); }
    #pragma unroll
    for (int o = 32; o; o >>= 1) ss += __shfl_down(ss, o, 64);
    __shared__ float red[4];
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        const float inv = rsqrtf((red[0]+red[1]+red[2]+red[3]) / (float)PM_MHC_FLAT + PM_RMS_EPS);
        float *lg = logits + (size_t)row * PM_MHC_MIX;
        for (int m = 0; m < PM_MHC_MIX; m++) lg[m] *= inv;
    }
}

// pre gates the collapse, post gates the re-expansion. Both are per stream.
__global__ void __launch_bounds__(64)
k_pf_mhc_prepost(const float *__restrict__ logits, const uint16_t *__restrict__ alpha,
                 const uint16_t *__restrict__ beta, int T,
                 float *__restrict__ pre, float *__restrict__ post) {
    const int row = blockIdx.x;
    if (row >= T) return;
    const int l = threadIdx.x;
    if (l >= PM_MHC_H) return;
    const float a0 = pm_b2f(alpha[0]), a1 = pm_b2f(alpha[1]);
    const float *lg = logits + (size_t)row * PM_MHC_MIX;
    pre [(size_t)row * PM_MHC_H + l] =
        1.f / (1.f + __expf(-(lg[l] * a0 + pm_b2f(beta[l])))) + PM_MHC_EPS;
    post[(size_t)row * PM_MHC_H + l] =
        2.f / (1.f + __expf(-(lg[PM_MHC_H + l] * a1 + pm_b2f(beta[PM_MHC_H + l]))));
}

// collapse the four streams into the layer input
__global__ void __launch_bounds__(256)
k_pf_mhc_mix(const uint16_t *__restrict__ x, const float *__restrict__ pre,
             int T, float *__restrict__ mixf) {
    const int d = blockIdx.x * 256 + threadIdx.x, row = blockIdx.y;
    if (row >= T || d >= PM_MHC_D) return;
    const uint16_t *xr = x + (size_t)row * PM_MHC_FLAT;
    const float *pr = pre + (size_t)row * PM_MHC_H;
    float a = 0.f;
    #pragma unroll
    for (int h = 0; h < PM_MHC_H; h++) a = fmaf(pr[h], pm_b2f(xr[(size_t)h * PM_MHC_D + d]), a);
    mixf[(size_t)row * PM_MHC_D + d] = a;
}

// re-expand one stream result back across the four, mixed by the Sinkhorn map
__global__ void __launch_bounds__(256)
k_pf_reexpand(const float *__restrict__ post, const float *__restrict__ comb,
              const float *__restrict__ m, const uint16_t *__restrict__ residual,
              int T, uint16_t *__restrict__ y) {
    const int d = blockIdx.x * 256 + threadIdx.x, row = blockIdx.y;
    if (row >= T || d >= PM_MHC_D) return;
    const float mv = m[(size_t)row * PM_MHC_D + d];
    const uint16_t *res = residual + (size_t)row * PM_MHC_FLAT;
    const float *cb = comb + (size_t)row * 16;
    const float *po = post + (size_t)row * PM_MHC_H;
    float r[PM_MHC_H];
    #pragma unroll
    for (int k = 0; k < PM_MHC_H; k++) r[k] = pm_b2f(res[(size_t)k * PM_MHC_D + d]);
    uint16_t *yr = y + (size_t)row * PM_MHC_FLAT;
    #pragma unroll
    for (int h = 0; h < PM_MHC_H; h++) {
        float a = po[h] * mv;
        #pragma unroll
        for (int k = 0; k < PM_MHC_H; k++) a = fmaf(cb[k * PM_MHC_H + h], r[k], a);
        yr[(size_t)h * PM_MHC_D + d] = pm_f2b(a);
    }
}

// ---- dense MLP gate/up, weights resident across the chunk's rows ---------
// Layers 0 and 1 only. Same reuse inversion as k_pf_gateup: the 2560 packed
// nibbles of an output row are hoisted once and swept against every token.
__global__ void __launch_bounds__(256)
k_pf_dense_gateup(const uint8_t *__restrict__ Wg, const uint8_t *__restrict__ Sg,
                  const uint8_t *__restrict__ Wu, const uint8_t *__restrict__ Su,
                  const int8_t *__restrict__ xq, const float *__restrict__ xs,
                  float s2g, float s2u, int T, float *__restrict__ hidden) {
    const int l   = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= PM_DENSE_FF) return;
    const uint8_t *wg = Wg + (size_t)row * (PM_MHC_D / 2);
    const uint8_t *wu = Wu + (size_t)row * (PM_MHC_D / 2);
    const uint8_t *sg = Sg + (size_t)row * (PM_MHC_D / 16);
    const uint8_t *su = Su + (size_t)row * (PM_MHC_D / 16);
    uint2 pg[5], pu[5]; uint8_t gb[5], ub[5]; float dg[5], du[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) { const int q = i * 32 + l;
        pg[i] = *(const uint2 *)(wg + (size_t)q * 8);
        pu[i] = *(const uint2 *)(wu + (size_t)q * 8); }
    #pragma unroll
    for (int i = 0; i < 5; i++) { const int q = i * 32 + l; gb[i] = sg[q]; ub[i] = su[q]; }
    #pragma unroll
    for (int i = 0; i < 5; i++) { dg[i] = p92_ue4m3_d(gb[i]); du[i] = p92_ue4m3_d(ub[i]); }
    for (int t = 0; t < T; t++) {
        const int8_t *sx = xq + (size_t)t * PM_MHC_D;
        const float  *ax = xs + (size_t)t * (PM_MHC_D / 16);
        float ag = 0.f, au = 0.f;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            const int q = i * 32 + l;
            const uint2 xe = *(const uint2 *)(sx + q * 16);
            const uint2 xo = *(const uint2 *)(sx + q * 16 + 8);
            const float a = ax[(q % 32) * 5 + (q / 32)];
            ag = fmaf(dg[i] * a, (float)p92_group_dot4(pg[i], xe, xo), ag);
            au = fmaf(du[i] * a, (float)p92_group_dot4(pu[i], xe, xo), au);
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) { ag += __shfl_down(ag, o, 64); au += __shfl_down(au, o, 64); }
        if (l == 0) {
            const float g = ag * 0.5f * s2g, u = au * 0.5f * s2u;   // 0.5 undoes doubled E2M1
            hidden[(size_t)t * PM_DENSE_FF + row] = g / (1.f + __expf(-g)) * u;
        }
    }
}

// ---- block_post_layernorm: normalises the whole four-stream block ---------
// 10240 wide, not 2560. Applied only on the layers listed in
// block_post_layernorm_idx, in place on the bf16 state.
__global__ void __launch_bounds__(256)
k_pf_block_post(uint16_t *__restrict__ x, const uint16_t *__restrict__ gamma, int T) {
    const int row = blockIdx.x;
    if (row >= T) return;
    uint16_t *xr = x + (size_t)row * PM_MHC_FLAT;
    float ss = 0.f;
    for (int i = threadIdx.x; i < PM_MHC_FLAT; i += 256) {
        const float v = pm_b2f(xr[i]); ss = fmaf(v, v, ss); }
    #pragma unroll
    for (int o = 32; o; o >>= 1) ss += __shfl_down(ss, o, 64);
    __shared__ float red[4];
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = ss;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0)
        inv = rsqrtf((red[0]+red[1]+red[2]+red[3]) / (float)PM_MHC_FLAT + PM_RMS_EPS);
    __syncthreads();
    for (int i = threadIdx.x; i < PM_MHC_FLAT; i += 256)
        xr[i] = pm_f2b(pm_b2f(xr[i]) * inv * pm_b2f(gamma[i]));
}
