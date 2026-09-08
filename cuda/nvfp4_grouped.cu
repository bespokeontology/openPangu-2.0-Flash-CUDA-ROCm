#include "p92/kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace p92::cuda {
namespace {

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementInput = cutlass::float_e2m1_t;
using ElementA = cutlass::nv_float4_t<ElementInput>;
using ElementB = cutlass::nv_float4_t<ElementInput>;
using ElementSF = cutlass::float_ue4m3_t;
using ElementD = cutlass::bfloat16_t;
using ElementC = void;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;
using ElementAccumulator = float;
using ElementCompute = float;
using Arch = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using TileShape = Shape<_256, _32, _256>;
using ClusterShape = Shape<_1, _1, _1>;
constexpr int kAlignmentA = 32;
constexpr int kAlignmentB = 32;
constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutD*, kAlignmentD,
    ElementD, LayoutD*, kAlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, OperatorClass,
    ElementA, LayoutA*, kAlignmentA,
    ElementB, LayoutB*, kAlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using PointerElementA = typename Gemm::ElementA;
using PointerElementB = typename Gemm::ElementB;
using PointerElementD = typename Gemm::EpilogueOutputOp::ElementOutput;
using StrideA = typename GemmKernel::InternalStrideA;
using StrideB = typename GemmKernel::InternalStrideB;
using StrideC = typename GemmKernel::InternalStrideC;
using StrideD = typename GemmKernel::InternalStrideD;
using LayoutSFA = typename CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename CollectiveMainloop::InternalLayoutSFB;
using BlockScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
using UnderlyingShape = typename ProblemShape::UnderlyingProblemShape;

template <class T>
cudaError_t allocate_device(T** pointer, std::size_t count) {
    return cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(T));
}

__global__ void prepare_grouped_pointers_kernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_scales,
    const Nvfp4ExpertProjection* expert_bank,
    const std::int32_t* expert_ids,
    PointerElementA const** ptr_a,
    PointerElementB const** ptr_b,
    ElementSF const** ptr_sfa,
    ElementSF const** ptr_sfb,
    float const** ptr_alpha,
    PointerElementD** ptr_d,
    BFloat16* first_output,
    BFloat16* second_output,
    int groups,
    int rows,
    int columns,
    int projection_kind,
    int batch_rows,
    bool expert_distinct_input_rows) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    constexpr int kSelectedExperts = 8;
    const int projections_per_row = second_output != nullptr
        ? 2 * kSelectedExperts : kSelectedExperts;
    const int batch_row = group / projections_per_row;
    if (batch_row >= batch_rows) return;
    const int within_row = group % projections_per_row;
    const int slot = within_row % kSelectedExperts;
    const bool second = second_output != nullptr && within_row >= kSelectedExperts;
    const int kind = projection_kind + (second ? 1 : 0);
    const int expert = expert_ids[batch_row * kSelectedExperts + slot];
    int ordered_group = group;
    if (batch_rows > 1) {
        const int key = kind * 256 + expert;
        ordered_group = 0;
        for (int other = 0; other < groups; ++other) {
            const int other_row = other / projections_per_row;
            const int other_within = other % projections_per_row;
            const int other_slot = other_within % kSelectedExperts;
            const bool other_second =
                second_output != nullptr && other_within >= kSelectedExperts;
            const int other_kind = projection_kind + (other_second ? 1 : 0);
            const int other_expert =
                expert_ids[other_row * kSelectedExperts + other_slot];
            const int other_key = other_kind * 256 + other_expert;
            ordered_group += other_key < key || (other_key == key && other < group);
        }
    }
    const Nvfp4ExpertProjection projection = expert_bank[expert * 3 + kind];
    const int input_row = expert_distinct_input_rows
        ? batch_row * kSelectedExperts + slot : batch_row;
    const std::size_t input_offset = static_cast<std::size_t>(input_row) * columns / 2;
    const std::size_t input_scale_offset = static_cast<std::size_t>(input_row) * 16;
    ptr_a[ordered_group] = reinterpret_cast<PointerElementA const*>(projection.weight);
    ptr_b[ordered_group] = reinterpret_cast<PointerElementB const*>(packed_input + input_offset);
    ptr_sfa[ordered_group] = reinterpret_cast<ElementSF const*>(projection.scales_swizzled);
    ptr_sfb[ordered_group] = reinterpret_cast<ElementSF const*>(input_scales + input_scale_offset);
    ptr_alpha[ordered_group] = projection.alpha;
    BFloat16* destination = second ? second_output : first_output;
    ptr_d[ordered_group] = reinterpret_cast<PointerElementD*>(
        destination +
            static_cast<std::size_t>(batch_row * kSelectedExperts + slot) * rows);
}

}  // namespace

struct Nvfp4GroupedPlan {
    Gemm gemm;
    int groups = 0;
    int rows = 0;
    int columns = 0;
    std::vector<UnderlyingShape> host_shapes;
    UnderlyingShape* problem_shapes = nullptr;
    PointerElementA const** ptr_a = nullptr;
    PointerElementB const** ptr_b = nullptr;
    ElementSF const** ptr_sfa = nullptr;
    ElementSF const** ptr_sfb = nullptr;
    float const** ptr_alpha = nullptr;
    PointerElementD** ptr_d = nullptr;
    StrideA* stride_a = nullptr;
    StrideB* stride_b = nullptr;
    StrideC* stride_c = nullptr;
    StrideD* stride_d = nullptr;
    LayoutSFA* layout_sfa = nullptr;
    LayoutSFB* layout_sfb = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;
};

namespace {

void release_plan(Nvfp4GroupedPlan* plan) noexcept {
    if (plan == nullptr) return;
    if (plan->workspace != nullptr) cudaFree(plan->workspace);
    if (plan->layout_sfb != nullptr) cudaFree(plan->layout_sfb);
    if (plan->layout_sfa != nullptr) cudaFree(plan->layout_sfa);
    if (plan->stride_d != nullptr) cudaFree(plan->stride_d);
    if (plan->stride_c != nullptr) cudaFree(plan->stride_c);
    if (plan->stride_b != nullptr) cudaFree(plan->stride_b);
    if (plan->stride_a != nullptr) cudaFree(plan->stride_a);
    if (plan->ptr_d != nullptr) cudaFree(plan->ptr_d);
    if (plan->ptr_alpha != nullptr) cudaFree(plan->ptr_alpha);
    if (plan->ptr_sfb != nullptr) cudaFree(plan->ptr_sfb);
    if (plan->ptr_sfa != nullptr) cudaFree(plan->ptr_sfa);
    if (plan->ptr_b != nullptr) cudaFree(plan->ptr_b);
    if (plan->ptr_a != nullptr) cudaFree(plan->ptr_a);
    if (plan->problem_shapes != nullptr) cudaFree(plan->problem_shapes);
    delete plan;
}

typename Gemm::Arguments make_arguments(Nvfp4GroupedPlan* plan) {
    typename CollectiveEpilogue::FusionCallbacks::Arguments fusion_args;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha = 0.0F;
    fusion_args.beta = 0.0F;
    fusion_args.alpha_ptr_array = plan->ptr_alpha;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {_0{}, _0{}, 1};
    fusion_args.dBeta = {_0{}, _0{}, 0};
    cutlass::KernelHardwareInfo hardware;
    hardware.device_id = 0;
    hardware.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename GemmKernel::TileSchedulerArguments scheduler;
    scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;
    return typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {plan->groups, plan->problem_shapes, plan->host_shapes.data()},
        {plan->ptr_a, plan->stride_a,
         plan->ptr_b, plan->stride_b,
         plan->ptr_sfa, plan->layout_sfa,
         plan->ptr_sfb, plan->layout_sfb},
        {fusion_args, nullptr, plan->stride_c, plan->ptr_d, plan->stride_d},
        hardware, scheduler};
}

}  // namespace

cudaError_t nvfp4_grouped_plan_create(Nvfp4GroupedPlan** output,
                                      int groups,
                                      int rows,
                                      int columns) {
    if (output == nullptr || groups <= 0 || rows <= 0 || columns <= 0 ||
        rows % 128 != 0 || columns % 16 != 0) {
        return cudaErrorInvalidValue;
    }
    *output = nullptr;
    Nvfp4GroupedPlan* plan = new (std::nothrow) Nvfp4GroupedPlan;
    if (plan == nullptr) return cudaErrorMemoryAllocation;
    plan->groups = groups;
    plan->rows = rows;
    plan->columns = columns;
    plan->host_shapes.assign(static_cast<std::size_t>(groups), UnderlyingShape{rows, 1, columns});
    std::vector<StrideA> host_stride_a(groups);
    std::vector<StrideB> host_stride_b(groups);
    std::vector<StrideC> host_stride_c(groups);
    std::vector<StrideD> host_stride_d(groups);
    std::vector<LayoutSFA> host_layout_sfa(groups);
    std::vector<LayoutSFB> host_layout_sfb(groups);
    const auto shape4 = make_shape(rows, 1, columns, 1);
    for (int group = 0; group < groups; ++group) {
        host_stride_a[group] = cutlass::make_cute_packed_stride(StrideA{}, {rows, columns, 1});
        host_stride_b[group] = cutlass::make_cute_packed_stride(StrideB{}, {1, columns, 1});
        host_stride_c[group] = cutlass::make_cute_packed_stride(StrideC{}, {rows, 1, 1});
        host_stride_d[group] = cutlass::make_cute_packed_stride(StrideD{}, {rows, 1, 1});
        host_layout_sfa[group] = BlockScaledConfig::tile_atom_to_shape_SFA(shape4);
        host_layout_sfb[group] = BlockScaledConfig::tile_atom_to_shape_SFB(shape4);
    }
    auto fail = [&](cudaError_t status) {
        release_plan(plan);
        return status;
    };
    cudaError_t status = allocate_device(&plan->problem_shapes, groups);
    if (status != cudaSuccess) return fail(status);
    status = allocate_device(&plan->ptr_a, groups);
    if (status != cudaSuccess) return fail(status);
    status = allocate_device(&plan->ptr_b, groups);
    if (status != cudaSuccess) return fail(status);
    status = allocate_device(&plan->ptr_sfa, groups);
    if (status != cudaSuccess) return fail(status);
    status = allocate_device(&plan->ptr_sfb, groups);
    if (status != cudaSuccess) return fail(status);
    status = allocate_device(&plan->ptr_d, groups);
    if (status != cudaSuccess) return fail(status);
    status = allocate_device(&plan->ptr_alpha, groups);
    if (status != cudaSuccess) return fail(status);
    status = allocate_device(&plan->stride_a, groups);
    if (status != cudaSuccess) return fail(status);
    status = allocate_device(&plan->stride_b, groups);
    if (status != cudaSuccess) return fail(status);
    status = allocate_device(&plan->stride_c, groups);
    if (status != cudaSuccess) return fail(status);
    status = allocate_device(&plan->stride_d, groups);
    if (status != cudaSuccess) return fail(status);
    status = allocate_device(&plan->layout_sfa, groups);
    if (status != cudaSuccess) return fail(status);
    status = allocate_device(&plan->layout_sfb, groups);
    if (status != cudaSuccess) return fail(status);
    status = cudaMemcpy(plan->problem_shapes, plan->host_shapes.data(),
                        static_cast<std::size_t>(groups) * sizeof(UnderlyingShape),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return fail(status);
    status = cudaMemcpy(plan->stride_a, host_stride_a.data(),
                        static_cast<std::size_t>(groups) * sizeof(StrideA), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return fail(status);
    status = cudaMemcpy(plan->stride_b, host_stride_b.data(),
                        static_cast<std::size_t>(groups) * sizeof(StrideB), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return fail(status);
    status = cudaMemcpy(plan->stride_c, host_stride_c.data(),
                        static_cast<std::size_t>(groups) * sizeof(StrideC), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return fail(status);
    status = cudaMemcpy(plan->stride_d, host_stride_d.data(),
                        static_cast<std::size_t>(groups) * sizeof(StrideD), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return fail(status);
    status = cudaMemcpy(plan->layout_sfa, host_layout_sfa.data(),
                        static_cast<std::size_t>(groups) * sizeof(LayoutSFA), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return fail(status);
    status = cudaMemcpy(plan->layout_sfb, host_layout_sfb.data(),
                        static_cast<std::size_t>(groups) * sizeof(LayoutSFB), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return fail(status);
    try {
        auto arguments = make_arguments(plan);
        plan->workspace_bytes = Gemm::get_workspace_size(arguments);
        if (plan->workspace_bytes != 0) {
            status = cudaMalloc(&plan->workspace, plan->workspace_bytes);
            if (status != cudaSuccess) return fail(status);
        }
        const cutlass::Status implementable = plan->gemm.can_implement(arguments);
        if (implementable != cutlass::Status::kSuccess) return fail(cudaErrorNotSupported);
        const cutlass::Status initialized =
            plan->gemm.initialize(arguments, plan->workspace, nullptr);
        if (initialized != cutlass::Status::kSuccess) return fail(cudaErrorInvalidValue);
    } catch (...) {
        return fail(cudaErrorInvalidValue);
    }
    *output = plan;
    return cudaSuccess;
}

void nvfp4_grouped_plan_destroy(Nvfp4GroupedPlan* plan) noexcept {
    release_plan(plan);
}

cudaError_t nvfp4_grouped_m1(Nvfp4GroupedPlan* plan,
                             const std::uint8_t* packed_input,
                             const std::uint8_t* input_scales_swizzled,
                             const Nvfp4ExpertProjection* expert_bank,
                             const std::int32_t* expert_ids,
                             BFloat16* first_output,
                             BFloat16* second_output,
                             int projection_kind,
                             bool distinct_input_rows,
                             cudaStream_t stream) {
    return nvfp4_grouped_rows(
        plan, packed_input, input_scales_swizzled, expert_bank, expert_ids,
        first_output, second_output, projection_kind, 1, distinct_input_rows, stream);
}

cudaError_t nvfp4_grouped_rows(Nvfp4GroupedPlan* plan,
                               const std::uint8_t* packed_input,
                               const std::uint8_t* input_scales_swizzled,
                               const Nvfp4ExpertProjection* expert_bank,
                               const std::int32_t* expert_ids,
                               BFloat16* first_output,
                               BFloat16* second_output,
                               int projection_kind,
                               int batch_rows,
                               bool expert_distinct_input_rows,
                               cudaStream_t stream) {
    if (plan == nullptr || packed_input == nullptr || input_scales_swizzled == nullptr ||
        expert_bank == nullptr || expert_ids == nullptr || first_output == nullptr ||
        projection_kind < 0 || projection_kind > 2 || batch_rows <= 0 ||
        plan->groups != batch_rows * (second_output != nullptr ? 16 : 8)) {
        return cudaErrorInvalidValue;
    }
    prepare_grouped_pointers_kernel<<<(plan->groups + 31) / 32, 32, 0, stream>>>(
        packed_input, input_scales_swizzled, expert_bank, expert_ids,
        plan->ptr_a, plan->ptr_b, plan->ptr_sfa, plan->ptr_sfb, plan->ptr_alpha,
        plan->ptr_d,
        first_output, second_output, plan->groups, plan->rows, plan->columns,
        projection_kind, batch_rows, expert_distinct_input_rows);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    try {
        const cutlass::Status launched = plan->gemm.run(stream);
        return launched == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorLaunchFailure;
    } catch (...) {
        return cudaErrorInvalidValue;
    }
}

}  // namespace p92::cuda
