#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace p92::cuda {

using BFloat16 = std::uint16_t;

struct Nvfp4ExpertProjection {
    const std::uint8_t* weight = nullptr;
    const std::uint8_t* scales_swizzled = nullptr;
    const float* alpha = nullptr;
};

struct Nvfp4GroupedPlan;
struct Bf16MlaPlan;

cudaError_t bf16_mla_plan_create(Bf16MlaPlan** plan);

void bf16_mla_plan_destroy(Bf16MlaPlan* plan) noexcept;

cudaError_t nvfp4_grouped_plan_create(Nvfp4GroupedPlan** plan,
                                      int groups,
                                      int rows,
                                      int columns);

void nvfp4_grouped_plan_destroy(Nvfp4GroupedPlan* plan) noexcept;

cudaError_t nvfp4_grouped_m1(Nvfp4GroupedPlan* plan,
                             const std::uint8_t* packed_input,
                             const std::uint8_t* input_scales_swizzled,
                             const Nvfp4ExpertProjection* expert_bank,
                             const std::int32_t* expert_ids,
                             BFloat16* first_output,
                             BFloat16* second_output,
                             int projection_kind,
                             bool distinct_input_rows,
                             cudaStream_t stream = nullptr);

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
                               cudaStream_t stream = nullptr);

cudaError_t expert_combine(const BFloat16* expert_rows,
                           const float* route_weights,
                           BFloat16* output,
                           int experts,
                           int elements,
                           cudaStream_t stream = nullptr);

cudaError_t expert_combine_rows(const BFloat16* expert_rows,
                                const float* route_weights,
                                BFloat16* output,
                                int batch_rows,
                                int experts,
                                int elements,
                                cudaStream_t stream = nullptr);

cudaError_t repeat_streams(const BFloat16* input,
                           BFloat16* output,
                           int rows,
                           int elements,
                           int streams,
                           cudaStream_t stream = nullptr);

cudaError_t rmsnorm(const BFloat16* input,
                    const BFloat16* weight,
                    BFloat16* output,
                    int rows,
                    int elements,
                    float epsilon,
                    cudaStream_t stream = nullptr);

cudaError_t rmsnorm_strided(const BFloat16* input,
                            int input_stride,
                            const BFloat16* weight,
                            BFloat16* output,
                            int output_stride,
                            int rows,
                            int elements,
                            float epsilon,
                            cudaStream_t stream = nullptr);

cudaError_t add_inplace(BFloat16* destination,
                        const BFloat16* source,
                        int elements,
                        cudaStream_t stream = nullptr);

cudaError_t swiglu(const BFloat16* gate,
                   const BFloat16* up,
                   BFloat16* output,
                   int elements,
                   cudaStream_t stream = nullptr);

cudaError_t bf16_gemv(const BFloat16* input,
                      const BFloat16* weight,
                      BFloat16* output,
                      int rows,
                      int columns,
                      cudaStream_t stream = nullptr);

cudaError_t bf16_gemm(const BFloat16* input,
                      const BFloat16* weight,
                      BFloat16* output,
                      int batch_rows,
                      int rows,
                      int columns,
                      cudaStream_t stream = nullptr);

cudaError_t bf16_gemv_f32(const BFloat16* input,
                          const BFloat16* weight,
                          float* output,
                          int rows,
                          int columns,
                          cudaStream_t stream = nullptr);

cudaError_t bf16_gemm_f32(const BFloat16* input,
                          const BFloat16* weight,
                          float* output,
                          int batch_rows,
                          int rows,
                          int columns,
                          cudaStream_t stream = nullptr);

cudaError_t add_scaled_inplace(BFloat16* destination,
                               const BFloat16* source,
                               const float* scale,
                               int elements,
                               cudaStream_t stream = nullptr);

cudaError_t argmax(const BFloat16* values,
                   int elements,
                   std::int32_t* index,
                   cudaStream_t stream = nullptr);

cudaError_t argmax_rows(const BFloat16* values,
                        int rows,
                        int elements,
                        std::int32_t* indices,
                        cudaStream_t stream = nullptr);

cudaError_t mome_conv3_step(BFloat16* values,
                            const BFloat16* weight,
                            BFloat16* state,
                            int channels,
                            cudaStream_t stream = nullptr);

cudaError_t mome_conv3_rows(BFloat16* values,
                            const BFloat16* weight,
                            BFloat16* state,
                            int rows,
                            int channels,
                            cudaStream_t stream = nullptr);

cudaError_t mome_conv3_rows_strided(BFloat16* values,
                                    int row_stride,
                                    const BFloat16* weight,
                                    BFloat16* state,
                                    int rows,
                                    int channels,
                                    cudaStream_t stream = nullptr);

cudaError_t mome_conv3_rows_strided_capture(BFloat16* values,
                                            int row_stride,
                                            const BFloat16* weight,
                                            BFloat16* state,
                                            BFloat16* state_after_rows,
                                            int rows,
                                            int channels,
                                            cudaStream_t stream = nullptr);

cudaError_t mome_commit_captured_row(BFloat16* state,
                                     const BFloat16* captured_rows,
                                     int layers,
                                     int rows,
                                     int row,
                                     int query_channels,
                                     int kv_channels,
                                     int output_channels,
                                     cudaStream_t stream = nullptr);

cudaError_t router_top8(const float* logits,
                        const float* correction_bias,
                        std::int32_t* expert_ids,
                        float* expert_weights,
                        float routed_scale,
                        cudaStream_t stream = nullptr);

cudaError_t router_top8_rows(const float* logits,
                             const float* correction_bias,
                             std::int32_t* expert_ids,
                             float* expert_weights,
                             int rows,
                             float routed_scale,
                             cudaStream_t stream = nullptr);

cudaError_t gather_rows(const BFloat16* input,
                        const std::int32_t* row_indices,
                        BFloat16* output,
                        int selected_rows,
                        int elements,
                        cudaStream_t stream = nullptr);

cudaError_t scatter_add_scaled_rows(BFloat16* destination,
                                    const BFloat16* source,
                                    const std::int32_t* row_indices,
                                    const float* scales,
                                    int selected_rows,
                                    int elements,
                                    cudaStream_t stream = nullptr);

cudaError_t mla_prepare_query(const BFloat16* q_up,
                              const BFloat16* kv_up_weight,
                              BFloat16* q_absorbed,
                              BFloat16* q_rope,
                              int position,
                              float rope_theta,
                              cudaStream_t stream = nullptr);

cudaError_t mla_prepare_query_rows(const BFloat16* q_up,
                                   const BFloat16* kv_up_weight,
                                   BFloat16* q_absorbed,
                                   BFloat16* q_rope,
                                   int batch_rows,
                                   int start_position,
                                   float rope_theta,
                                   cudaStream_t stream = nullptr);

cudaError_t mla_prepare_query_rope_rows(const BFloat16* q_up,
                                        BFloat16* q_rope,
                                        int batch_rows,
                                        int start_position,
                                        float rope_theta,
                                        cudaStream_t stream = nullptr);

cudaError_t mla_absorb_query_rows_tensorcore(Bf16MlaPlan* plan,
                                             const BFloat16* q_up,
                                             const BFloat16* kv_up_weight,
                                             BFloat16* gathered_input,
                                             BFloat16* head_major_output,
                                             BFloat16* q_absorbed,
                                             int batch_rows,
                                             cudaStream_t stream = nullptr);

cudaError_t mla_prepare_key_rope(const BFloat16* kv_down,
                                 BFloat16* key_rope,
                                 int position,
                                 float rope_theta,
                                 cudaStream_t stream = nullptr);

cudaError_t mla_prepare_key_rope_rows(const BFloat16* kv_down,
                                      BFloat16* key_rope,
                                      int batch_rows,
                                      int start_position,
                                      float rope_theta,
                                      cudaStream_t stream = nullptr);

cudaError_t mla_store_cache_rows(BFloat16* cache_kv,
                                 BFloat16* cache_rope,
                                 const BFloat16* row_kv,
                                 const BFloat16* row_rope,
                                 int start_position,
                                 int rows,
                                 int cache_capacity,
                                 cudaStream_t stream = nullptr);

cudaError_t mla_transfer_cache_rows(bool restore,
                                    BFloat16* live_kv,
                                    BFloat16* live_rope,
                                    BFloat16* snapshot_kv,
                                    BFloat16* snapshot_rope,
                                    int start_position,
                                    int snapshot_rows,
                                    int first_row,
                                    int row_count,
                                    int cache_capacity,
                                    cudaStream_t stream = nullptr);

cudaError_t mla_attention_decode(const BFloat16* q_absorbed,
                                 const BFloat16* q_rope,
                                 const BFloat16* sink_kv,
                                 const BFloat16* sink_rope,
                                 const BFloat16* cache_kv,
                                 const BFloat16* cache_rope,
                                 int cache_start,
                                 int cache_count,
                                 BFloat16* latent_output,
                                 float softmax_scale,
                                 cudaStream_t stream = nullptr);

cudaError_t mla_attention_decode_tiled(const BFloat16* q_absorbed,
                                       const BFloat16* q_rope,
                                       const BFloat16* sink_kv,
                                       const BFloat16* sink_rope,
                                       const BFloat16* cache_kv,
                                       const BFloat16* cache_rope,
                                       int cache_start,
                                       int cache_count,
                                       BFloat16* latent_output,
                                       float softmax_scale,
                                       cudaStream_t stream = nullptr);

cudaError_t mla_attention_decode_circular(const BFloat16* q_absorbed,
                                          const BFloat16* q_rope,
                                          const BFloat16* sink_kv,
                                          const BFloat16* sink_rope,
                                          const BFloat16* cache_kv,
                                          const BFloat16* cache_rope,
                                          int cache_start,
                                          int cache_count,
                                          int cache_capacity,
                                          BFloat16* latent_output,
                                          float softmax_scale,
                                          cudaStream_t stream = nullptr);

cudaError_t mla_attention_decode_circular_tiled(const BFloat16* q_absorbed,
                                                const BFloat16* q_rope,
                                                const BFloat16* sink_kv,
                                                const BFloat16* sink_rope,
                                                const BFloat16* cache_kv,
                                                const BFloat16* cache_rope,
                                                int cache_start,
                                                int cache_count,
                                                int cache_capacity,
                                                BFloat16* latent_output,
                                                float softmax_scale,
                                                cudaStream_t stream = nullptr);

cudaError_t mla_attention_decode_indexed(const BFloat16* q_absorbed,
                                         const BFloat16* q_rope,
                                         const BFloat16* sink_kv,
                                         const BFloat16* sink_rope,
                                         const BFloat16* cache_kv,
                                         const BFloat16* cache_rope,
                                         const std::int32_t* indices,
                                         int selected_count,
                                         BFloat16* latent_output,
                                         float softmax_scale,
                                         cudaStream_t stream = nullptr);

cudaError_t mla_attention_decode_indexed_tiled(const BFloat16* q_absorbed,
                                               const BFloat16* q_rope,
                                               const BFloat16* sink_kv,
                                               const BFloat16* sink_rope,
                                               const BFloat16* cache_kv,
                                               const BFloat16* cache_rope,
                                               const std::int32_t* indices,
                                               int selected_count,
                                               BFloat16* latent_output,
                                               float softmax_scale,
                                               cudaStream_t stream = nullptr);

cudaError_t mla_attention_prefill_rows(const BFloat16* q_absorbed,
                                       const BFloat16* q_rope,
                                       const BFloat16* sink_kv,
                                       const BFloat16* sink_rope,
                                       const BFloat16* cache_kv,
                                       const BFloat16* cache_rope,
                                       int start_position,
                                       int batch_rows,
                                       BFloat16* latent_output,
                                       float softmax_scale,
                                       cudaStream_t stream = nullptr);

cudaError_t mla_attention_prefill_rows_tiled(const BFloat16* q_absorbed,
                                             const BFloat16* q_rope,
                                             const BFloat16* sink_kv,
                                             const BFloat16* sink_rope,
                                             const BFloat16* cache_kv,
                                             const BFloat16* cache_rope,
                                             int start_position,
                                             int batch_rows,
                                             BFloat16* latent_output,
                                             float softmax_scale,
                                             cudaStream_t stream = nullptr);

cudaError_t mla_value_up(const BFloat16* latent,
                         const BFloat16* kv_up_weight,
                         BFloat16* output,
                         cudaStream_t stream = nullptr);

cudaError_t mla_value_up_rows(const BFloat16* latent,
                              const BFloat16* kv_up_weight,
                              BFloat16* output,
                              int batch_rows,
                              cudaStream_t stream = nullptr);

cudaError_t mla_value_up_rows_tensorcore(Bf16MlaPlan* plan,
                                         const BFloat16* latent,
                                         const BFloat16* kv_up_weight,
                                         BFloat16* gathered_input,
                                         BFloat16* head_major_output,
                                         BFloat16* output,
                                         int batch_rows,
                                         cudaStream_t stream = nullptr);

cudaError_t dsa_rope(BFloat16* query,
                     BFloat16* key,
                     int position,
                     float rope_theta,
                     cudaStream_t stream = nullptr);

cudaError_t dsa_rope_rows(BFloat16* query,
                          BFloat16* key,
                          int batch_rows,
                          int start_position,
                          float rope_theta,
                          cudaStream_t stream = nullptr);

std::size_t dsa_sort_workspace_bytes(int positions);

cudaError_t dsa_select(const BFloat16* query,
                       const BFloat16* head_weights,
                       const BFloat16* key_cache,
                       int positions,
                       float* scores_in,
                       float* scores_out,
                       std::int32_t* indices_in,
                       std::int32_t* indices_out,
                       void* sort_workspace,
                       std::size_t sort_workspace_bytes,
                       cudaStream_t stream = nullptr);

cudaError_t mhc_pre(const BFloat16* streams,
                    const BFloat16* phi,
                    const BFloat16* norm_gamma,
                    const BFloat16* branch_alpha,
                    const BFloat16* branch_beta,
                    BFloat16* mixed,
                    float* h_post,
                    float* h_res,
                    int rows,
                    float norm_epsilon,
                    cudaStream_t stream = nullptr);

cudaError_t mhc_pre_parallel(const BFloat16* streams,
                             const BFloat16* phi,
                             const BFloat16* norm_gamma,
                             const BFloat16* branch_alpha,
                             const BFloat16* branch_beta,
                             BFloat16* mixed,
                             float* h_post,
                             float* h_res,
                             float* inverse_scratch,
                             float* mixes_scratch,
                             int rows,
                             float norm_epsilon,
                             cudaStream_t stream = nullptr);

cudaError_t mhc_post(const BFloat16* hidden,
                     const BFloat16* residual_streams,
                     const float* h_post,
                     const float* h_res,
                     BFloat16* output_streams,
                     int rows,
                     cudaStream_t stream = nullptr);

cudaError_t mhc_merge(const BFloat16* streams,
                      const BFloat16* phi,
                      const BFloat16* norm_gamma,
                      const BFloat16* branch_alpha,
                      const BFloat16* branch_beta,
                      BFloat16* output,
                      int rows,
                      float norm_epsilon,
                      cudaStream_t stream = nullptr);

cudaError_t nvfp4_quantize(const BFloat16* input,
                           std::uint8_t* output,
                           std::uint8_t* scales,
                           int elements,
                           float global_scale,
                           cudaStream_t stream = nullptr);

cudaError_t nvfp4_gemv(const std::uint8_t* input,
                       const std::uint8_t* weight,
                       const std::uint8_t* input_scales,
                       const std::uint8_t* weight_scales,
                       BFloat16* output,
                       int rows,
                       int columns,
                       float alpha,
                       cudaStream_t stream = nullptr);

std::size_t nvfp4_swizzled_scale_bytes(int rows, int scale_columns);

cudaError_t nvfp4_swizzle_scales(const std::uint8_t* linear,
                                 std::uint8_t* swizzled,
                                 int rows,
                                 int scale_columns,
                                 cudaStream_t stream = nullptr);

std::size_t nvfp4_tensorcore_workspace_bytes(int rows, int columns, int tactic);
std::size_t nvfp4_tensorcore_workspace_bytes(int batch_rows, int rows, int columns, int tactic);

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
                                  cudaStream_t stream = nullptr);

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
                                  cudaStream_t stream = nullptr);

}  // namespace p92::cuda
