#pragma once
// openPangu MoME depthwise causal convolutions, gfx906.
//
// Correction to receipts/00_PORT_MAP.md, which describes MoME as one conv and
// names the Qwen GDN short conv as the donor. There are THREE convs per
// attention block, at three different points and three different widths:
//
//   self_attn.qa_conv.weight          bf16 [1024, 1, 3]
//   self_attn.compresskv_conv.weight  bf16 [ 512, 1, 3]
//   self_attn.o_conv.weight           bf16 [6144, 1, 3]
//
// 147 tensors over 49 layers. They carry no record in the NVFP4 manifest, so
// they stay bf16, and there is no bias tensor for any of them.
//
// The donor is the CUDA oracle's own mome_conv3_kernel
// (openpangu-flash92-native/cuda/nvfp4_ops.cu:262-286), which is 25 lines of
// elementwise code with no cross-lane reduction and transliterates directly.
// Huawei's own MoME layer confirms the semantics independently: every call
// site passes residual_connection=1 and activation_mode='None', so the op is a
// residual add with no activation and no bias.
//
// Weights are CHANNEL-MAJOR, weight + channel*3, tap 0 oldest. Huawei's NPU
// loader transposes to tap-major [3, dim]; the oracle does not, and the
// checkpoint's native layout is what we read, so follow the oracle.
//
// One ordering detail that is easy to get wrong: the state shift stores the
// ORIGINAL input, before the output overwrites it in place.
#include <hip/hip_runtime.h>
#include <cstdint>

typedef uint16_t p92_cbf;
static __device__ __forceinline__ float p92_c2f(p92_cbf b){ return __uint_as_float((uint32_t)b<<16); }
static __device__ __forceinline__ p92_cbf p92_f2c(float f){
    uint32_t u = __float_as_uint(f); u += 0x7FFFu + ((u >> 16) & 1u); return (p92_cbf)(u >> 16); }

#define P92_MOME_QA   1024
#define P92_MOME_KV    512
#define P92_MOME_O    6144
#define P92_MOME_TAPS    3
#define P92_MOME_STATE   2      // taps - 1

// values is updated in place; state is [C][2], older then previous.
template <int C>
__global__ void __launch_bounds__(256)
k_p92_mome_conv3(p92_cbf *__restrict__ values, const p92_cbf *__restrict__ weight,
                 p92_cbf *__restrict__ state) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;                       // C is compile time, bound folds
    const float older    = p92_c2f(state[(size_t)c * 2]);
    const float previous = p92_c2f(state[(size_t)c * 2 + 1]);
    const p92_cbf cur_b  = values[c];
    const float current  = p92_c2f(cur_b);
    const p92_cbf *w = weight + (size_t)c * P92_MOME_TAPS;
    const float convolved = fmaf(older,    p92_c2f(w[0]),
                            fmaf(previous, p92_c2f(w[1]),
                                 current * p92_c2f(w[2])));
    state[(size_t)c * 2]     = state[(size_t)c * 2 + 1];
    state[(size_t)c * 2 + 1] = cur_b;         // the ORIGINAL input, not the output
    values[c] = p92_f2c(current + convolved); // residual add, no activation
}

// fp32-valued form. The oracle keeps values in bf16 because its whole engine
// does; ours carries fp32 between kernels, and converting to bf16 and back
// purely to cross this op cost two extra launches at each of three sites in
// every layer - 276 kernel launches a token doing nothing but changing format.
// The recurrent STATE stays bf16, which is what the contract actually fixes.
template <int C>
__global__ void __launch_bounds__(256)
k_p92_mome_conv3_f32(float *__restrict__ values, const p92_cbf *__restrict__ weight,
                     p92_cbf *__restrict__ state) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    const float older    = p92_c2f(state[(size_t)c * 2]);
    const float previous = p92_c2f(state[(size_t)c * 2 + 1]);
    const float current  = values[c];
    const p92_cbf *w = weight + (size_t)c * P92_MOME_TAPS;
    const float convolved = fmaf(older,    p92_c2f(w[0]),
                            fmaf(previous, p92_c2f(w[1]),
                                 current * p92_c2f(w[2])));
    state[(size_t)c * 2]     = state[(size_t)c * 2 + 1];
    state[(size_t)c * 2 + 1] = p92_f2c(current);
    values[c] = current + convolved;
}
template __global__ void k_p92_mome_conv3_f32<P92_MOME_QA>(float *, const p92_cbf *, p92_cbf *);
template __global__ void k_p92_mome_conv3_f32<P92_MOME_KV>(float *, const p92_cbf *, p92_cbf *);
template __global__ void k_p92_mome_conv3_f32<P92_MOME_O >(float *, const p92_cbf *, p92_cbf *);

template __global__ void k_p92_mome_conv3<P92_MOME_QA>(p92_cbf *, const p92_cbf *, p92_cbf *);
template __global__ void k_p92_mome_conv3<P92_MOME_KV>(p92_cbf *, const p92_cbf *, p92_cbf *);
template __global__ void k_p92_mome_conv3<P92_MOME_O >(p92_cbf *, const p92_cbf *, p92_cbf *);

// Rows form, for prefill: one thread per channel walks T positions carrying
// the three-deep ring in registers, so the state is loaded once at entry and
// stored once at exit rather than round-tripping per position. row_stride
// lets the kv site read a wider row than its channel count.
template <int C>
__global__ void __launch_bounds__(256)
k_p92_mome_conv3_rows(p92_cbf *__restrict__ values, int row_stride, int rows,
                      const p92_cbf *__restrict__ weight, p92_cbf *__restrict__ state) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    float older    = p92_c2f(state[(size_t)c * 2]);
    float previous = p92_c2f(state[(size_t)c * 2 + 1]);
    const p92_cbf *w = weight + (size_t)c * P92_MOME_TAPS;
    const float w0 = p92_c2f(w[0]), w1 = p92_c2f(w[1]), w2 = p92_c2f(w[2]);
    for (int r = 0; r < rows; r++) {
        p92_cbf *v = values + (size_t)r * row_stride + c;
        const float current = p92_c2f(*v);
        const float convolved = fmaf(older, w0, fmaf(previous, w1, current * w2));
        *v = p92_f2c(current + convolved);
        older = previous;
        previous = current;
    }
    state[(size_t)c * 2]     = p92_f2c(older);
    state[(size_t)c * 2 + 1] = p92_f2c(previous);
}

template __global__ void k_p92_mome_conv3_rows<P92_MOME_QA>(p92_cbf *, int, int, const p92_cbf *, p92_cbf *);
template __global__ void k_p92_mome_conv3_rows<P92_MOME_KV>(p92_cbf *, int, int, const p92_cbf *, p92_cbf *);
template __global__ void k_p92_mome_conv3_rows<P92_MOME_O >(p92_cbf *, int, int, const p92_cbf *, p92_cbf *);
