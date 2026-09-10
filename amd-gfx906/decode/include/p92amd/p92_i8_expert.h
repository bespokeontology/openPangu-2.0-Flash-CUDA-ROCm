// openPangu W8A8 routed-expert contract, recovered from the shipped
// openpangu/openPangu-2.0-Flash-Int8 checkpoint and the Windows CUDA
// pangu_under8_patch reference implementation.
//
// CONTRACT (hardware-neutral; see receipts/04_W8A8_CONTRACT.md):
//   weight        int8, row-major [outputs, inputs]
//   weight_scale  bf16, one per output row, SYMMETRIC (no zero point)
//   activation    int8, symmetric, ONE scale per token = max|x| / 127
//   accumulate    int32
//   dequantise    y[o] = (float)acc[o] * x_scale * w_scale[o]
//   requant       after SwiGLU, fresh per-token absmax/127 before down_proj
//
// Huawei quantises the routed experts only.  Shared expert, router, attention
// and the embedding stay bf16 in the shipped checkpoint.
#ifndef P92AMD_I8_EXPERT_H
#define P92AMD_I8_EXPERT_H

#include <hip/hip_runtime.h>
#include <cstdint>

#define P92_I8_NEMBD 2560
#define P92_I8_NFF   1024
#define P92_I8_TOPK  8
// 8 routed plus the shared expert, dispatched together in one pass
#define P92_I8_DISPATCH 9

// Per-expert byte strides in a packed resident arena.
#define P92_I8_GU_W_BYTES ((size_t)P92_I8_NFF * P92_I8_NEMBD)   // 2,621,440
#define P92_I8_DN_W_BYTES ((size_t)P92_I8_NEMBD * P92_I8_NFF)   // 2,621,440
#define P92_I8_GU_S_BYTES ((size_t)P92_I8_NFF * sizeof(float))  //     4,096
#define P92_I8_DN_S_BYTES ((size_t)P92_I8_NEMBD * sizeof(float))//    10,240

// One resident expert: gate, up, down weights plus their row scales.
#define P92_I8_EXPERT_BYTES (2 * P92_I8_GU_W_BYTES + P92_I8_DN_W_BYTES + \
                             2 * P92_I8_GU_S_BYTES + P92_I8_DN_S_BYTES)

struct P92I8ExpertArena {
    const int8_t *gate_w;   // [E][NFF  * NEMBD]
    const int8_t *up_w;     // [E][NFF  * NEMBD]
    const int8_t *down_w;   // [E][NEMBD * NFF ]
    const float  *gate_s;   // [E][NFF ]
    const float  *up_s;     // [E][NFF ]
    const float  *down_s;   // [E][NEMBD]
    int           experts;  // resident int8 expert count for this layer
};

// One resident-arena slot index per top-8 selection, or -1 when that expert is
// not resident in int8 for this layer (it goes down the nvfp4 path instead).
//
// Scratch: hidden [TOPK*NFF] f32, hq [TOPK*NFF] i8, hs [TOPK] f32.
void p92_i8_expert_ffn(const P92I8ExpertArena &arena,
                       const int8_t *xq, const float *xs,
                       const int *slot, const float *route_w,
                       float *hidden, int8_t *hq, float *hs,
                       float *y, hipStream_t stream);

// xs[0] = max|x| / 127 over the residual width; xq = round(x / xs[0])
void p92_i8_quantise(const float *x, int8_t *xq, float *xs, hipStream_t stream);

#endif
