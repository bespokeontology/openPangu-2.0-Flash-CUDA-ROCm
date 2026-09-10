#include "p92amd/p92_pf_moe.h"
#include "../../decode/src/p92_nvfp4_dot4.hip"

// Batched NVFP4 projection for prefill. Covers every dense projection in the
// layer with one template: q_a [1024,2560], q_b [9216,1024], kv_a [576,2560],
// o_proj [2560,6144].
//
// SAME INVERSION AS THE EXPERTS, and for the same reason: a projection weight is
// read once per token in decode and once per CHUNK here. Each 32-lane group owns
// one output row, holds that row's packed nibbles in REGISTERS, and sweeps all T
// tokens past them. Weight traffic becomes independent of T.
//
// This is NOT the decode kernel widened. k_p92_proj streams the weight past a
// single activation; inverting which operand is resident is the whole point and
// only pays because T > 1.
//
// K is the input width, in nibble groups of 16. A row is K/2 bytes, so at
// K=2560 a lane holds 40 B (five uint2) and at K=1024 it holds 16 B (two).
// Scales arrive lane-contiguous from the load-time preshuffle.
template <int K>
__global__ void __launch_bounds__(256)
k_pf_proj(const uint8_t *__restrict__ W, const uint8_t *__restrict__ S,
          const int8_t *__restrict__ xq, const float *__restrict__ xs,
          float s2, int T, int N, float *__restrict__ y)
{
    constexpr int GRP = K / 16;          // 16-value groups in a row
    constexpr int PER = GRP / 32;        // groups a lane
    const int l   = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= N) return;

    const uint8_t *w = W + (size_t)row * (K / 2);
    const uint8_t *s = S + (size_t)row * GRP;

    // hoist the whole weight row, once, then decode. Loads first: decoding a
    // branchy UE4M3 at the point of load fences every one of them.
    uint2 pk[PER]; uint8_t sb[PER]; float ds[PER];
    #pragma unroll
    for (int i = 0; i < PER; i++) pk[i] = *(const uint2 *)(w + (size_t)(i * 32 + l) * 8);
    #pragma unroll
    // SPEC ORDER, not lane-contiguous. The MoE arena is preshuffled by
    // k_pf_shuffle_scales_gu, so k_pf_gateup reads its scales at [l*PER+i];
    // projection weights come straight from nv_up and are never shuffled, so
    // reading them the same way paired the nibbles of group i*32+l with the
    // scale of group l*PER+i. Self-consistent, silently wrong, and worth about
    // 5 percent on q_lora against the decode engine on the same token.
    for (int i = 0; i < PER; i++) sb[i] = s[(size_t)(i * 32 + l)];
    #pragma unroll
    for (int i = 0; i < PER; i++) ds[i] = p92_ue4m3_d(sb[i]);
    const float f2 = 0.5f * s2;

    for (int t = 0; t < T; t++) {
        const int8_t *x = xq + (size_t)t * K;
        const float  *a = xs + (size_t)t * GRP;
        uint4 xv[PER]; float av[PER];
        #pragma unroll
        for (int i = 0; i < PER; i++) xv[i] = *(const uint4 *)(x + (size_t)(i * 32 + l) * 16);
        #pragma unroll
        for (int i = 0; i < PER; i++) av[i] = a[l * PER + i];    // lane-contiguous
        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < PER; i++) {
            const uint2 xe = make_uint2(xv[i].x, xv[i].y);
            const uint2 xo = make_uint2(xv[i].z, xv[i].w);
            acc = fmaf(ds[i] * av[i], (float)p92_group_dot4(pk[i], xe, xo), acc);
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) acc += __shfl_down(acc, off, 64);
        if (l == 0) y[(size_t)t * N + row] = acc * f2;
    }
}

template __global__ void k_pf_proj<2560>(const uint8_t*, const uint8_t*, const int8_t*,
    const float*, float, int, int, float*);
template __global__ void k_pf_proj<1024>(const uint8_t*, const uint8_t*, const int8_t*,
    const float*, float, int, int, float*);
template __global__ void k_pf_proj<6144>(const uint8_t*, const uint8_t*, const int8_t*,
    const float*, float, int, int, float*);

// Width-3 causal depthwise conv over the chunk, channel-major, tap 0 oldest,
// residual add, no activation, no bias. Genuinely batched rather than inverted:
// there is no expensive operand to make resident, only three taps a channel.
// The chunk's first two positions read carry-in state from the previous chunk,
// which is what makes this correct across chunk boundaries rather than only
// within one.
__global__ void __launch_bounds__(256)
k_pf_mome(float *__restrict__ x, const uint16_t *__restrict__ wt,
          float *__restrict__ carry, int T, int C)
{
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    const float w0 = __uint_as_float((uint32_t)wt[(size_t)c * 3 + 0] << 16);
    const float w1 = __uint_as_float((uint32_t)wt[(size_t)c * 3 + 1] << 16);
    const float w2 = __uint_as_float((uint32_t)wt[(size_t)c * 3 + 2] << 16);
    float m2 = carry[(size_t)c * 2 + 0];      // t-2
    float m1 = carry[(size_t)c * 2 + 1];      // t-1
    for (int t = 0; t < T; t++) {
        const float v = x[(size_t)t * C + c];
        x[(size_t)t * C + c] = v + (w0 * m2 + w1 * m1 + w2 * v);
        m2 = m1; m1 = v;                       // pre-conv values carry forward
    }
    carry[(size_t)c * 2 + 0] = m2;
    carry[(size_t)c * 2 + 1] = m1;
}
