#pragma once

#include "p92/resident.h"
#include "p92/kernels.h"

#include <cstddef>
#include <cstdint>
#include <array>

#include <cuda_runtime_api.h>

namespace p92 {

class ProjectionExecutor final {
public:
    ProjectionExecutor();
    ~ProjectionExecutor();

    ProjectionExecutor(const ProjectionExecutor&) = delete;
    ProjectionExecutor& operator=(const ProjectionExecutor&) = delete;
    ProjectionExecutor(ProjectionExecutor&&) = delete;
    ProjectionExecutor& operator=(ProjectionExecutor&&) = delete;

    void project(const ResidentProjection& projection,
                 const std::uint16_t* input,
                 std::uint16_t* output);
    void project_rows(const ResidentProjection& projection,
                      const std::uint16_t* input,
                      std::uint16_t* output,
                      int batch_rows);
    void project_expert_gate_up(const cuda::Nvfp4ExpertProjection* expert_bank,
                                const std::int32_t* expert_ids,
                                const std::uint16_t* input,
                                std::uint16_t* gate_output,
                                std::uint16_t* up_output,
                                int hidden,
                                int intermediate,
                                int batch_rows = 1);
    void project_expert_down(const cuda::Nvfp4ExpertProjection* expert_bank,
                             const std::int32_t* expert_ids,
                             const std::uint16_t* input_rows,
                             std::uint16_t* output_rows,
                             int hidden,
                             int intermediate,
                             int batch_rows = 1);
    void synchronize();
    cudaStream_t stream() const noexcept { return stream_; }

private:
    void reserve(std::uint32_t columns, int batch_rows,
                 std::size_t workspace_bytes);

    cudaStream_t stream_ = nullptr;
    std::uint8_t* packed_input_ = nullptr;
    std::uint8_t* linear_scales_ = nullptr;
    std::uint8_t* swizzled_scales_ = nullptr;
    void* workspace_ = nullptr;
    std::size_t packed_capacity_ = 0;
    std::size_t scale_capacity_ = 0;
    std::size_t workspace_capacity_ = 0;
    std::array<cuda::Nvfp4GroupedPlan*, 5> expert_gate_up_plans_ {};
    std::array<cuda::Nvfp4GroupedPlan*, 5> expert_down_plans_ {};
};

}  // namespace p92
