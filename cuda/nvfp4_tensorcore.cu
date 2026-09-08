#include "p92/kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>

#include "flashinfer/gemm/cutlass_gemm_configs.h"
#include "flashinfer/gemm/fp4_gemm_template_sm120.h"

namespace flashinfer::gemm {

INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 32, 256, 1, 1, 1, _1SM, true)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 64, 256, 1, 1, 1, _1SM, true)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 1, 1, 1, _1SM, true)

}  // namespace flashinfer::gemm

namespace p92::cuda {
namespace {

using flashinfer::gemm::_1SM;

template <int TileN, bool StreamK>
std::size_t invoke(void* output,
                   const void* input,
                   const void* weight,
                   const void* input_scales,
                   const void* weight_scales,
                   const float* alpha,
                   int batch_rows,
                   int rows,
                   int columns,
                   void* workspace,
                   std::size_t workspace_bytes,
                   cudaStream_t stream) {
    using namespace flashinfer::gemm;
    CutlassGemmConfig config;
    if constexpr (StreamK) {
        return genericFp4GemmKernelLauncherStreamK<
            __nv_bfloat16, cute::Int<128>, cute::Int<TileN>, cute::Int<256>,
            cute::Int<1>, cute::Int<1>, cute::Int<1>, _1SM, true>(
                output, input, weight, input_scales, weight_scales, alpha,
                batch_rows, rows, columns, 1, config, static_cast<char*>(workspace),
                workspace_bytes, stream, nullptr);
    }
    return genericFp4GemmKernelLauncher<
        __nv_bfloat16, cute::Int<128>, cute::Int<TileN>, cute::Int<256>,
        cute::Int<1>, cute::Int<1>, cute::Int<1>, _1SM, true>(
            output, input, weight, input_scales, weight_scales, alpha,
            batch_rows, rows, columns, 1, config, static_cast<char*>(workspace),
            workspace_bytes, stream, nullptr);
}

std::size_t dispatch(void* output,
                     const void* input,
                     const void* weight,
                     const void* input_scales,
                     const void* weight_scales,
                     const float* alpha,
                     int batch_rows,
                     int rows,
                     int columns,
                     void* workspace,
                     std::size_t workspace_bytes,
                     int tactic,
                     cudaStream_t stream) {
    if (tactic == 6) {
        return invoke<32, true>(output, input, weight, input_scales, weight_scales,
                                alpha, batch_rows, rows, columns, workspace, workspace_bytes, stream);
    }
    if (tactic == 20) {
        return invoke<128, false>(output, input, weight, input_scales, weight_scales,
                                  alpha, batch_rows, rows, columns, workspace, workspace_bytes, stream);
    }
    if (tactic == 21) {
        return invoke<64, false>(output, input, weight, input_scales, weight_scales,
                                 alpha, batch_rows, rows, columns, workspace, workspace_bytes, stream);
    }
    if (tactic == 22) {
        return invoke<64, true>(output, input, weight, input_scales, weight_scales,
                                alpha, batch_rows, rows, columns, workspace, workspace_bytes, stream);
    }
    throw std::invalid_argument("unsupported P92 NVFP4 tensor-core tactic");
}

}  // namespace

std::size_t nvfp4_tensorcore_workspace_bytes(int rows, int columns, int tactic) {
    return nvfp4_tensorcore_workspace_bytes(1, rows, columns, tactic);
}

std::size_t nvfp4_tensorcore_workspace_bytes(int batch_rows, int rows, int columns, int tactic) {
    if (batch_rows <= 0 || rows <= 0 || columns <= 0) {
        throw std::invalid_argument("invalid P92 NVFP4 tensor-core shape");
    }
    return dispatch(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    batch_rows, rows, columns, nullptr, 0, tactic, nullptr);
}

cudaError_t nvfp4_tensorcore_gemv(const std::uint8_t* input,
                                  const std::uint8_t* weight,
                                  const std::uint8_t* input_scales_swizzled,
                                  const std::uint8_t* weight_scales_swizzled,
                                  BFloat16* output,
                                  int rows,
                                  int columns,
                                  const float* alpha,
                                  void* workspace,
                                  std::size_t workspace_bytes,
                                  int tactic,
                                  cudaStream_t stream) {
    return nvfp4_tensorcore_gemm(
        input, weight, input_scales_swizzled, weight_scales_swizzled, output,
        1, rows, columns, alpha, workspace, workspace_bytes, tactic, stream);
}

cudaError_t nvfp4_tensorcore_gemm(const std::uint8_t* input,
                                  const std::uint8_t* weight,
                                  const std::uint8_t* input_scales_swizzled,
                                  const std::uint8_t* weight_scales_swizzled,
                                  BFloat16* output,
                                  int batch_rows,
                                  int rows,
                                  int columns,
                                  const float* alpha,
                                  void* workspace,
                                  std::size_t workspace_bytes,
                                  int tactic,
                                  cudaStream_t stream) {
    try {
        if (batch_rows <= 0 || rows <= 0 || columns <= 0) return cudaErrorInvalidValue;
        dispatch(output, input, weight, input_scales_swizzled, weight_scales_swizzled,
                 alpha, batch_rows, rows, columns, workspace, workspace_bytes, tactic, stream);
        return cudaGetLastError();
    } catch (...) {
        return cudaErrorInvalidValue;
    }
}

}  // namespace p92::cuda
