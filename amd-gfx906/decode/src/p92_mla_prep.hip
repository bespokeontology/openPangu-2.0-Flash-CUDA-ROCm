#pragma once
// openPangu MLA query absorption, value up-projection and rope, gfx906.
//
// kv_b_proj is [12288, 512] = 48 heads x (128 nope up + 128 value up) x 512
// latent, head-interleaved, and it is BF16 from the auxiliary checkpoint rather
// than NVFP4 - the oracle's mla_prepare_query and mla_value_up both take
// const BFloat16* (model_ops.cu:701, :1073). It is used in BOTH directions:
//   absorbed query   q_abs[h][j] = sum_{i<128} q_nope[h][i] * kv_b[h*256+i][j]
//   value up         out[h][v]   = sum_{j<512} ctx[h][j]   * kv_b[h*256+128+v][j]
// The absorb sums over ROWS at a fixed column, which would be a stride-512
// gather per thread; laying threads across j instead makes each step a
// contiguous 512-wide read, so both directions stay coalesced.
//
// rope: config rope_interleave is false and the span is 64, so a pair is
// (i, i+32). theta 6,400,000.
#include <hip/hip_runtime.h>
#include <cstdint>

#define P92_MLA_HEADS  48
#define P92_MLA_NOPE  128
#define P92_MLA_ROPE   64
#define P92_MLA_LATENT 512
#define P92_MLA_VALUE 128
#define P92_MLA_UPPH  (P92_MLA_NOPE + P92_MLA_VALUE)      // 256 rows a head
#define P92_MLA_QUP   (P92_MLA_NOPE + P92_MLA_ROPE)       // 192 q_up dims a head
#define P92_MLA_THETA 6400000.0f

typedef uint16_t p92_mbf16;
static __device__ __forceinline__ float p92_m2f(p92_mbf16 b){ return __uint_as_float((uint32_t)b<<16); }

// ---- absorbed query ------------------------------------------------------
// grid (48 heads, 2), 256 threads: one thread owns one latent dim j and walks
// the 128 nope rows. Consecutive threads read consecutive j, so every step is a
// contiguous 512-element row read.
__global__ void __launch_bounds__(256)
k_p92_mla_absorb(const float *__restrict__ q_up, const p92_mbf16 *__restrict__ kv_b,
                 float *__restrict__ q_abs) {
    const int h = blockIdx.x;
    const int j = blockIdx.y * 256 + threadIdx.x;
    if (h >= P92_MLA_HEADS || j >= P92_MLA_LATENT) return;

    __shared__ float qn[P92_MLA_NOPE];
    if (threadIdx.x < P92_MLA_NOPE) qn[threadIdx.x] = q_up[h * P92_MLA_QUP + threadIdx.x];
    __syncthreads();

    const p92_mbf16 *w = kv_b + (size_t)h * P92_MLA_UPPH * P92_MLA_LATENT + j;
    float acc = 0.f;
    #pragma unroll 8
    for (int i = 0; i < P92_MLA_NOPE; i++)
        acc = fmaf(qn[i], p92_m2f(w[(size_t)i * P92_MLA_LATENT]), acc);
    q_abs[(size_t)h * P92_MLA_LATENT + j] = acc;
}

// ---- value up ------------------------------------------------------------
// One wave per output value, 8 latent dims a lane. 48 x 128 = 6144 outputs.
__global__ void __launch_bounds__(256)
k_p92_mla_value_up(const float *__restrict__ ctx, const p92_mbf16 *__restrict__ kv_b,
                   float *__restrict__ out) {
    const int lane = threadIdx.x & 63;
    const int idx  = blockIdx.x * 4 + (threadIdx.x >> 6);      // which (head, value)
    if (idx >= P92_MLA_HEADS * P92_MLA_VALUE) return;
    const int h = idx / P92_MLA_VALUE, v = idx % P92_MLA_VALUE;

    const p92_mbf16 *w = kv_b + ((size_t)h * P92_MLA_UPPH + P92_MLA_NOPE + v) * P92_MLA_LATENT;
    const float *c = ctx + (size_t)h * P92_MLA_LATENT;
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < P92_MLA_LATENT / 64; i++) {
        const int d = i * 64 + lane;
        acc = fmaf(c[d], p92_m2f(w[d]), acc);
    }
    #pragma unroll
    for (int off = 32; off; off >>= 1) acc += __shfl_down(acc, off, 64);
    if (lane == 0) out[(size_t)h * P92_MLA_VALUE + v] = acc;
}

// ---- rope tables and application -----------------------------------------
// Built per token, 32 pairs over the 64-dim span.
__global__ void __launch_bounds__(64)
k_p92_rope_table(int position, float *__restrict__ cosv, float *__restrict__ sinv) {
    const int i = threadIdx.x;
    if (i >= P92_MLA_ROPE / 2) return;
    const float inv = __powf(P92_MLA_THETA, -2.0f * (float)i / (float)P92_MLA_ROPE);
    const float a = (float)position * inv;
    cosv[i] = __cosf(a);
    sinv[i] = __sinf(a);
}

// q_up carries 48 heads of 192; the rope span is the last 64 of each head.
__global__ void __launch_bounds__(64)
k_p92_query_rope(const float *__restrict__ q_up, const float *__restrict__ cosv,
                 const float *__restrict__ sinv, float *__restrict__ q_rope) {
    const int h = blockIdx.x, i = threadIdx.x;
    if (i >= P92_MLA_ROPE / 2) return;
    const float *src = q_up + (size_t)h * P92_MLA_QUP + P92_MLA_NOPE;
    float *dst = q_rope + (size_t)h * P92_MLA_ROPE;
    const float c = cosv[i], s = sinv[i];
    const float a = src[i], b = src[i + P92_MLA_ROPE / 2];
    dst[i]                      = a * c - b * s;
    dst[i + P92_MLA_ROPE / 2]   = b * c + a * s;
}

// kv_down is [576] = 512 latent then 64 rope.
__global__ void __launch_bounds__(64)
k_p92_key_rope(const float *__restrict__ kv_down, const float *__restrict__ cosv,
               const float *__restrict__ sinv, p92_mbf16 *__restrict__ k_rope) {
    const int i = threadIdx.x;
    if (i >= P92_MLA_ROPE / 2) return;
    const float *src = kv_down + P92_MLA_LATENT;
    const float c = cosv[i], s = sinv[i];
    const float a = src[i], b = src[i + P92_MLA_ROPE / 2];
    const float o0 = a * c - b * s, o1 = b * c + a * s;
    uint32_t u0, u1; __builtin_memcpy(&u0,&o0,4); __builtin_memcpy(&u1,&o1,4);
    u0 += 0x7FFFu + ((u0 >> 16) & 1u); u1 += 0x7FFFu + ((u1 >> 16) & 1u);
    k_rope[i]                    = (p92_mbf16)(u0 >> 16);
    k_rope[i + P92_MLA_ROPE / 2] = (p92_mbf16)(u1 >> 16);
}

// fp32 -> bf16, for writing a vector into the bf16 KV cache
__global__ void __launch_bounds__(256)
k_p92_to_bf16(const float *__restrict__ x, p92_mbf16 *__restrict__ y, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    uint32_t u; __builtin_memcpy(&u,&x[i],4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    y[i] = (p92_mbf16)(u >> 16);
}
__global__ void __launch_bounds__(256)
k_p92_from_bf16(const p92_mbf16 *__restrict__ x, float *__restrict__ y, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    y[i] = p92_m2f(x[i]);
}
