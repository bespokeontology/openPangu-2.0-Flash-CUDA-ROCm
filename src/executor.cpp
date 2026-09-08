#include "p92/executor.h"

#include "p92/kernels.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace p92 {
namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

ProjectionExecutor::ProjectionExecutor() {
    check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
          "create projection stream");
}

ProjectionExecutor::~ProjectionExecutor() {
    for (cuda::Nvfp4GroupedPlan* plan : expert_down_plans_) {
        cuda::nvfp4_grouped_plan_destroy(plan);
    }
    for (cuda::Nvfp4GroupedPlan* plan : expert_gate_up_plans_) {
        cuda::nvfp4_grouped_plan_destroy(plan);
    }
    if (workspace_ != nullptr) cudaFree(workspace_);
    if (swizzled_scales_ != nullptr) cudaFree(swizzled_scales_);
    if (linear_scales_ != nullptr) cudaFree(linear_scales_);
    if (packed_input_ != nullptr) cudaFree(packed_input_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

void ProjectionExecutor::project_expert_gate_up(
    const cuda::Nvfp4ExpertProjection* expert_bank,
    const std::int32_t* expert_ids,
    const std::uint16_t* input,
    std::uint16_t* gate_output,
    std::uint16_t* up_output,
    int hidden,
    int intermediate,
    int batch_rows) {
    if (expert_bank == nullptr || expert_ids == nullptr || input == nullptr ||
        gate_output == nullptr || up_output == nullptr || hidden <= 0 || intermediate <= 0 ||
        batch_rows <= 0 || batch_rows >= static_cast<int>(expert_gate_up_plans_.size())) {
        throw std::invalid_argument("invalid grouped expert gate/up projection");
    }
    reserve(static_cast<std::uint32_t>(hidden), batch_rows, 0);
    check(cuda::nvfp4_quantize(
              input, packed_input_, linear_scales_, batch_rows * hidden, 1.0F, stream_),
          "quantize grouped expert input");
    check(cuda::nvfp4_swizzle_scales(
              linear_scales_, swizzled_scales_, batch_rows, hidden / 16, stream_),
          "swizzle grouped expert input scales");
    cuda::Nvfp4GroupedPlan*& plan =
        expert_gate_up_plans_[static_cast<std::size_t>(batch_rows)];
    if (plan == nullptr) {
        check(cuda::nvfp4_grouped_plan_create(
                  &plan, batch_rows * 16, intermediate, hidden),
              "create grouped expert gate/up plan");
    }
    check(cuda::nvfp4_grouped_rows(
              plan, packed_input_, swizzled_scales_, expert_bank,
              expert_ids, gate_output, up_output, 0, batch_rows, false, stream_),
          "execute grouped expert gate/up");
}

void ProjectionExecutor::project_expert_down(
    const cuda::Nvfp4ExpertProjection* expert_bank,
    const std::int32_t* expert_ids,
    const std::uint16_t* input_rows,
    std::uint16_t* output_rows,
    int hidden,
    int intermediate,
    int batch_rows) {
    if (expert_bank == nullptr || expert_ids == nullptr || input_rows == nullptr ||
        output_rows == nullptr || hidden <= 0 || intermediate <= 0 ||
        batch_rows <= 0 || batch_rows >= static_cast<int>(expert_down_plans_.size())) {
        throw std::invalid_argument("invalid grouped expert down projection");
    }
    constexpr int kExperts = 8;
    const int quantized_rows = batch_rows * kExperts;
    reserve(static_cast<std::uint32_t>(intermediate), quantized_rows, 0);
    check(cuda::nvfp4_quantize(
              input_rows, packed_input_, linear_scales_,
              quantized_rows * intermediate, 1.0F, stream_),
          "quantize grouped expert down input");
    check(cuda::nvfp4_swizzle_scales(
              linear_scales_, swizzled_scales_, quantized_rows, intermediate / 16, stream_),
          "swizzle grouped expert down input scales");
    cuda::Nvfp4GroupedPlan*& plan =
        expert_down_plans_[static_cast<std::size_t>(batch_rows)];
    if (plan == nullptr) {
        check(cuda::nvfp4_grouped_plan_create(
                  &plan, batch_rows * kExperts, hidden, intermediate),
              "create grouped expert down plan");
    }
    check(cuda::nvfp4_grouped_rows(
              plan, packed_input_, swizzled_scales_, expert_bank,
              expert_ids, output_rows, nullptr, 2, batch_rows, true, stream_),
          "execute grouped expert down");
}

void ProjectionExecutor::reserve(std::uint32_t columns, int batch_rows,
                                 std::size_t workspace_bytes) {
    if (batch_rows <= 0) throw std::invalid_argument("invalid projection batch rows");
    const std::size_t packed_bytes =
        static_cast<std::size_t>(batch_rows) * columns / 2;
    const std::size_t linear_scale_bytes =
        static_cast<std::size_t>(batch_rows) * columns / 16;
    const std::size_t swizzled_scale_bytes = cuda::nvfp4_swizzled_scale_bytes(
        batch_rows, static_cast<int>(columns / 16));
    const std::size_t scale_bytes = std::max(linear_scale_bytes, swizzled_scale_bytes);
    if (packed_bytes > packed_capacity_) {
        if (packed_input_ != nullptr) check(cudaFree(packed_input_), "free packed activation");
        check(cudaMalloc(reinterpret_cast<void**>(&packed_input_), packed_bytes),
              "allocate packed activation");
        packed_capacity_ = packed_bytes;
    }
    if (scale_bytes > scale_capacity_) {
        if (swizzled_scales_ != nullptr) check(cudaFree(swizzled_scales_), "free swizzled activation scales");
        if (linear_scales_ != nullptr) check(cudaFree(linear_scales_), "free linear activation scales");
        check(cudaMalloc(reinterpret_cast<void**>(&linear_scales_), scale_bytes),
              "allocate linear activation scales");
        check(cudaMalloc(reinterpret_cast<void**>(&swizzled_scales_), scale_bytes),
              "allocate swizzled activation scales");
        scale_capacity_ = scale_bytes;
    }
    if (workspace_bytes > workspace_capacity_) {
        if (workspace_ != nullptr) check(cudaFree(workspace_), "free projection workspace");
        check(cudaMalloc(&workspace_, std::max<std::size_t>(workspace_bytes, 1)),
              "allocate projection workspace");
        workspace_capacity_ = workspace_bytes;
    }
}

void ProjectionExecutor::project(const ResidentProjection& projection,
                                 const std::uint16_t* input,
                                 std::uint16_t* output) {
    project_rows(projection, input, output, 1);
}

void ProjectionExecutor::project_rows(const ResidentProjection& projection,
                                      const std::uint16_t* input,
                                      std::uint16_t* output,
                                      int batch_rows) {
    if (projection.weight == nullptr || projection.scales_swizzled == nullptr ||
        projection.alpha == nullptr || input == nullptr || output == nullptr ||
        projection.rows == 0 || projection.columns == 0 || projection.rows % 128 != 0 ||
        projection.columns % 16 != 0 || batch_rows <= 0 || batch_rows > 64) {
        throw std::invalid_argument("invalid resident NVFP4 projection");
    }
    const int tactic = batch_rows >= 16 ? 21
        : (batch_rows == 4 ? (projection.rows <= 1024 ? 21 : 6)
                           : (projection.columns > projection.rows ? 20 : 6));
    const std::size_t workspace_bytes = cuda::nvfp4_tensorcore_workspace_bytes(
        batch_rows, static_cast<int>(projection.rows),
        static_cast<int>(projection.columns), tactic);
    reserve(projection.columns, batch_rows, workspace_bytes);
    check(cuda::nvfp4_quantize(input, packed_input_, linear_scales_,
                               batch_rows * static_cast<int>(projection.columns),
                               1.0F, stream_),
          "quantize NVFP4 activation");
    check(cuda::nvfp4_swizzle_scales(linear_scales_, swizzled_scales_, batch_rows,
                                      static_cast<int>(projection.columns / 16), stream_),
          "swizzle NVFP4 activation scales");
    check(cuda::nvfp4_tensorcore_gemm(
              packed_input_, projection.weight, swizzled_scales_,
              projection.scales_swizzled, output,
              batch_rows,
              static_cast<int>(projection.rows), static_cast<int>(projection.columns),
              projection.alpha, workspace_, workspace_bytes, tactic, stream_),
          "execute resident NVFP4 projection");
}

void ProjectionExecutor::synchronize() {
    check(cudaStreamSynchronize(stream_), "synchronize projection stream");
}

}  // namespace p92
