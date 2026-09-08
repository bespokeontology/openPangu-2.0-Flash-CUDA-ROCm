#include "p92/kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

namespace p92::cuda {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float bf16_to_float(BFloat16 bits) {
    __nv_bfloat16_raw raw;
    raw.x = bits;
    return __bfloat162float(static_cast<__nv_bfloat16>(raw));
}

__device__ __forceinline__ BFloat16 float_to_bf16(float value) {
    return static_cast<__nv_bfloat16_raw>(__float2bfloat16_rn(value)).x;
}

__device__ __forceinline__ float decode_fp4(std::uint8_t bits) {
    __nv_fp4_e2m1 value;
    value.__x = bits & 0x0fU;
    return static_cast<float>(value);
}

__device__ __forceinline__ float decode_e4m3(std::uint8_t bits) {
    __nv_fp8_e4m3 value;
    value.__x = bits;
    return static_cast<float>(value);
}

__device__ float block_sum(float value, float* warp_sums) {
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffU, value, offset);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = warp == 0 && lane < blockDim.x / 32 ? warp_sums[lane] : 0.0F;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffU, value, offset);
        if (lane == 0) warp_sums[0] = value;
    }
    __syncthreads();
    return warp_sums[0];
}

__global__ void repeat_streams_kernel(const BFloat16* input,
                                      BFloat16* output,
                                      int rows,
                                      int elements,
                                      int streams) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * elements * streams;
    if (index >= total) return;
    const int row_stride = elements * streams;
    const int row = index / row_stride;
    const int column = index - row * row_stride;
    output[index] = input[static_cast<std::size_t>(row) * elements + column % elements];
}

__global__ void rmsnorm_kernel(const BFloat16* input,
                               const BFloat16* weight,
                               BFloat16* output,
                               int rows,
                               int elements,
                               int input_stride,
                               int output_stride,
                               float epsilon) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    input += static_cast<std::size_t>(row) * input_stride;
    output += static_cast<std::size_t>(row) * output_stride;
    __shared__ float warp_sums[8];
    float square_sum = 0.0F;
    for (int i = threadIdx.x; i < elements; i += blockDim.x) {
        const float value = bf16_to_float(input[i]);
        square_sum = fmaf(value, value, square_sum);
    }
    square_sum = block_sum(square_sum, warp_sums);
    const float inverse = rsqrtf(square_sum / static_cast<float>(elements) + epsilon);
    for (int i = threadIdx.x; i < elements; i += blockDim.x) {
        output[i] = float_to_bf16(bf16_to_float(input[i]) * inverse * bf16_to_float(weight[i]));
    }
}

__global__ void add_kernel(BFloat16* destination, const BFloat16* source, int elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) destination[i] = float_to_bf16(bf16_to_float(destination[i]) + bf16_to_float(source[i]));
}

__global__ void swiglu_kernel(const BFloat16* gate,
                              const BFloat16* up,
                              BFloat16* output,
                              int elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const float value = bf16_to_float(gate[i]);
    output[i] = float_to_bf16((value / (1.0F + expf(-value))) * bf16_to_float(up[i]));
}

__global__ void expert_combine_kernel(
    const BFloat16* expert_rows,
    const float* route_weights,
    BFloat16* output,
    int experts,
    int elements) {
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= elements) return;
    float value = 0.0F;
    for (int slot = 0; slot < experts; ++slot) {
        value = fmaf(bf16_to_float(
                         expert_rows[static_cast<std::size_t>(slot) * elements + element]),
                     route_weights[slot], value);
    }
    output[element] = float_to_bf16(value);
}

__global__ void expert_combine_rows_kernel(
    const BFloat16* expert_rows,
    const float* route_weights,
    BFloat16* output,
    int experts,
    int elements) {
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (element >= elements) return;
    float value = 0.0F;
    const std::size_t route_base = static_cast<std::size_t>(row) * experts;
    const std::size_t output_base = static_cast<std::size_t>(row) * elements;
    for (int slot = 0; slot < experts; ++slot) {
        value = fmaf(
            bf16_to_float(expert_rows[(route_base + slot) * elements + element]),
            route_weights[route_base + slot], value);
    }
    output[output_base + element] = float_to_bf16(value);
}

__global__ void bf16_gemv_kernel(const BFloat16* input,
                                 const BFloat16* weight,
                                 BFloat16* output,
                                 int rows,
                                 int columns) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float warp_sums[8];
    const BFloat16* row_weight = weight + static_cast<std::size_t>(row) * columns;
    float value = 0.0F;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        value = fmaf(bf16_to_float(input[column]), bf16_to_float(row_weight[column]), value);
    }
    value = block_sum(value, warp_sums);
    if (threadIdx.x == 0) output[row] = float_to_bf16(value);
}

__global__ void bf16_gemv_f32_kernel(const BFloat16* input,
                                     const BFloat16* weight,
                                     float* output,
                                     int rows,
                                     int columns) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float warp_sums[8];
    const BFloat16* row_weight = weight + static_cast<std::size_t>(row) * columns;
    float value = 0.0F;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        value = fmaf(bf16_to_float(input[column]), bf16_to_float(row_weight[column]), value);
    }
    value = block_sum(value, warp_sums);
    if (threadIdx.x == 0) output[row] = value;
}

template <bool OutputF32>
__global__ void bf16_gemm_kernel(const BFloat16* input,
                                 const BFloat16* weight,
                                 void* output,
                                 int batch_rows,
                                 int rows,
                                 int columns) {
    const int output_row = blockIdx.x;
    if (output_row >= rows) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x / 32;
    const BFloat16* row_weight = weight + static_cast<std::size_t>(output_row) * columns;
    for (int batch_row = warp; batch_row < batch_rows; batch_row += warps) {
        const BFloat16* row_input = input + static_cast<std::size_t>(batch_row) * columns;
        float value = 0.0F;
        for (int column = lane; column < columns; column += 32) {
            value = fmaf(bf16_to_float(row_input[column]),
                         bf16_to_float(row_weight[column]), value);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) {
            const std::size_t index = static_cast<std::size_t>(batch_row) * rows + output_row;
            if constexpr (OutputF32) static_cast<float*>(output)[index] = value;
            else static_cast<BFloat16*>(output)[index] = float_to_bf16(value);
        }
    }
}

__global__ void add_scaled_kernel(BFloat16* destination,
                                  const BFloat16* source,
                                  const float* scale,
                                  int elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) {
        destination[i] = float_to_bf16(
            bf16_to_float(destination[i]) + bf16_to_float(source[i]) * *scale);
    }
}

struct MaximumPair {
    float value;
    int index;
};

__device__ __forceinline__ MaximumPair better(MaximumPair left, MaximumPair right) {
    return right.value > left.value || (right.value == left.value && right.index < left.index)
        ? right : left;
}

__global__ void argmax_kernel(const BFloat16* values, int elements, std::int32_t* index) {
    const int row = blockIdx.x;
    values += static_cast<std::size_t>(row) * elements;
    MaximumPair local {-FLT_MAX, 0};
    for (int i = threadIdx.x; i < elements; i += blockDim.x) {
        local = better(local, {bf16_to_float(values[i]), i});
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        MaximumPair other {
            __shfl_down_sync(0xffffffffU, local.value, offset),
            __shfl_down_sync(0xffffffffU, local.index, offset),
        };
        local = better(local, other);
    }
    __shared__ MaximumPair warps[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warps[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = lane < blockDim.x / 32 ? warps[lane] : MaximumPair {-FLT_MAX, 0};
        for (int offset = 16; offset > 0; offset >>= 1) {
            MaximumPair other {
                __shfl_down_sync(0xffffffffU, local.value, offset),
                __shfl_down_sync(0xffffffffU, local.index, offset),
            };
            local = better(local, other);
        }
        if (lane == 0) index[row] = local.index;
    }
}

__global__ void mome_conv3_kernel(BFloat16* values,
                                  const BFloat16* weight,
                                  BFloat16* state,
                                  int channels,
                                  BFloat16* state_after) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const float older = bf16_to_float(state[static_cast<std::size_t>(channel) * 2]);
    const float previous = bf16_to_float(state[static_cast<std::size_t>(channel) * 2 + 1]);
    const float current = bf16_to_float(values[channel]);
    const BFloat16* channel_weight = weight + static_cast<std::size_t>(channel) * 3;
    const float convolved =
        fmaf(older, bf16_to_float(channel_weight[0]),
             fmaf(previous, bf16_to_float(channel_weight[1]),
                  current * bf16_to_float(channel_weight[2])));
    state[static_cast<std::size_t>(channel) * 2] = state[static_cast<std::size_t>(channel) * 2 + 1];
    state[static_cast<std::size_t>(channel) * 2 + 1] = values[channel];
    if (state_after != nullptr) {
        state_after[static_cast<std::size_t>(channel) * 2] =
            state[static_cast<std::size_t>(channel) * 2];
        state_after[static_cast<std::size_t>(channel) * 2 + 1] =
            state[static_cast<std::size_t>(channel) * 2 + 1];
    }
    values[channel] = float_to_bf16(current + convolved);
}

__global__ void mome_conv3_rows_kernel(BFloat16* values,
                                       int row_stride,
                                       const BFloat16* weight,
                                       BFloat16* state,
                                       BFloat16* state_after_rows,
                                       int rows,
                                       int channels) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    BFloat16 older_bits = state[static_cast<std::size_t>(channel) * 2];
    BFloat16 previous_bits = state[static_cast<std::size_t>(channel) * 2 + 1];
    const BFloat16* channel_weight = weight + static_cast<std::size_t>(channel) * 3;
    const float first_weight = bf16_to_float(channel_weight[0]);
    const float second_weight = bf16_to_float(channel_weight[1]);
    const float current_weight = bf16_to_float(channel_weight[2]);
    for (int row = 0; row < rows; ++row) {
        BFloat16* row_values = values + static_cast<std::size_t>(row) * row_stride;
        const BFloat16 current_bits = row_values[channel];
        const float current = bf16_to_float(current_bits);
        const float convolved =
            fmaf(bf16_to_float(older_bits), first_weight,
                 fmaf(bf16_to_float(previous_bits), second_weight,
                      current * current_weight));
        older_bits = previous_bits;
        previous_bits = current_bits;
        if (state_after_rows != nullptr) {
            BFloat16* row_state = state_after_rows +
                static_cast<std::size_t>(row) * 2 * channels;
            row_state[static_cast<std::size_t>(channel) * 2] = older_bits;
            row_state[static_cast<std::size_t>(channel) * 2 + 1] = previous_bits;
        }
        row_values[channel] = float_to_bf16(current + convolved);
    }
    state[static_cast<std::size_t>(channel) * 2] = older_bits;
    state[static_cast<std::size_t>(channel) * 2 + 1] = previous_bits;
}

__global__ void mome_commit_captured_row_kernel(BFloat16* state,
                                                const BFloat16* captured_rows,
                                                int layers,
                                                int rows,
                                                int row,
                                                int query_channels,
                                                int kv_channels,
                                                int output_channels) {
    const int state_elements = 2 * (query_channels + kv_channels + output_channels);
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= layers * state_elements) return;
    const int layer = index / state_elements;
    const int feature = index - layer * state_elements;
    const int query_elements = 2 * query_channels;
    const int kv_elements = 2 * kv_channels;
    const int output_elements = 2 * output_channels;
    const BFloat16* layer_rows = captured_rows +
        static_cast<std::size_t>(layer) * rows * state_elements;
    std::size_t source = 0;
    if (feature < query_elements) {
        source = static_cast<std::size_t>(row) * query_elements + feature;
    } else if (feature < query_elements + kv_elements) {
        source = static_cast<std::size_t>(rows) * query_elements +
            static_cast<std::size_t>(row) * kv_elements + feature - query_elements;
    } else {
        source = static_cast<std::size_t>(rows) * (query_elements + kv_elements) +
            static_cast<std::size_t>(row) * output_elements +
            feature - query_elements - kv_elements;
    }
    state[index] = layer_rows[source];
}

__global__ void router_top8_kernel(const float* logits,
                                   const float* correction_bias,
                                   std::int32_t* expert_ids,
                                   float* expert_weights,
                                   int rows,
                                   float routed_scale) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    logits += static_cast<std::size_t>(row) * 256;
    expert_ids += static_cast<std::size_t>(row) * 8;
    expert_weights += static_cast<std::size_t>(row) * 8;
    __shared__ float raw[256];
    __shared__ float corrected[256];
    __shared__ float warp_best_score[8];
    __shared__ int warp_best_id[8];
    __shared__ int selected_ids[8];
    const int expert = threadIdx.x;
    if (expert < 256) {
        raw[expert] = 1.0F / (1.0F + expf(-logits[expert]));
        corrected[expert] = raw[expert] + correction_bias[expert];
    }
    __syncthreads();
    for (int slot = 0; slot < 8; ++slot) {
        bool used = false;
        #pragma unroll
        for (int prior = 0; prior < slot; ++prior) {
            used = used || selected_ids[prior] == expert;
        }
        float best_score = used ? -FLT_MAX : corrected[expert];
        int best_id = expert;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_score = __shfl_down_sync(0xffffffffU, best_score, offset);
            const int other_id = __shfl_down_sync(0xffffffffU, best_id, offset);
            if (other_score > best_score ||
                (other_score == best_score && other_id < best_id)) {
                best_score = other_score;
                best_id = other_id;
            }
        }
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (lane == 0) {
            warp_best_score[warp] = best_score;
            warp_best_id[warp] = best_id;
        }
        __syncthreads();
        if (warp == 0) {
            best_score = lane < 8 ? warp_best_score[lane] : -FLT_MAX;
            best_id = lane < 8 ? warp_best_id[lane] : 256 + lane;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_score = __shfl_down_sync(0xffffffffU, best_score, offset);
                const int other_id = __shfl_down_sync(0xffffffffU, best_id, offset);
                if (other_score > best_score ||
                    (other_score == best_score && other_id < best_id)) {
                    best_score = other_score;
                    best_id = other_id;
                }
            }
            if (lane == 0) {
                selected_ids[slot] = best_id;
                expert_ids[slot] = best_id;
                expert_weights[slot] = raw[best_id];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float total = 0.0F;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) total += expert_weights[slot];
        const float multiplier = routed_scale / total;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) expert_weights[slot] *= multiplier;
    }
}

__global__ void mhc_inverse_kernel(const BFloat16* streams,
                                   float* inverse,
                                   int rows,
                                   float norm_epsilon) {
    constexpr int kFlat = 4 * 2560;
    const int row = blockIdx.x;
    if (row >= rows) return;
    streams += static_cast<std::size_t>(row) * kFlat;
    __shared__ float warp_sums[8];
    float square_sum = 0.0F;
    for (int index = threadIdx.x; index < kFlat; index += blockDim.x) {
        const float value = bf16_to_float(streams[index]);
        square_sum = fmaf(value, value, square_sum);
    }
    square_sum = block_sum(square_sum, warp_sums);
    if (threadIdx.x == 0) {
        inverse[row] = rsqrtf(square_sum / static_cast<float>(kFlat) + norm_epsilon);
    }
}

__global__ void mhc_phi_parallel_kernel(const BFloat16* streams,
                                        const BFloat16* phi,
                                        const BFloat16* norm_gamma,
                                        const float* inverse,
                                        float* mixes,
                                        int rows) {
    constexpr int kFlat = 4 * 2560;
    constexpr int kMixes = 24;
    const int row = blockIdx.x / kMixes;
    const int mix = blockIdx.x - row * kMixes;
    if (row >= rows) return;
    streams += static_cast<std::size_t>(row) * kFlat;
    phi += static_cast<std::size_t>(mix) * kFlat;
    float sum = 0.0F;
    const float row_inverse = inverse[row];
    for (int index = threadIdx.x; index < kFlat; index += 32) {
        const float normalized =
            bf16_to_float(streams[index]) * row_inverse * bf16_to_float(norm_gamma[index]);
        sum = fmaf(normalized, bf16_to_float(phi[index]), sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    if (threadIdx.x == 0) mixes[static_cast<std::size_t>(row) * kMixes + mix] = sum;
}

__global__ void mhc_phi_paired_kernel(const BFloat16* streams,
                                      const BFloat16* phi,
                                      const BFloat16* norm_gamma,
                                      const float* inverse,
                                      float* mixes,
                                      int rows) {
    constexpr int kFlat = 4 * 2560;
    constexpr int kMixes = 24;
    constexpr int kMixesPerBlock = 2;
    __shared__ float normalized[kFlat];
    const int blocks_per_row = kMixes / kMixesPerBlock;
    const int row = blockIdx.x / blocks_per_row;
    const int mix_base = (blockIdx.x - row * blocks_per_row) * kMixesPerBlock;
    if (row >= rows) return;
    streams += static_cast<std::size_t>(row) * kFlat;
    const float row_inverse = inverse[row];
    for (int index = threadIdx.x; index < kFlat; index += blockDim.x) {
        normalized[index] =
            bf16_to_float(streams[index]) * row_inverse * bf16_to_float(norm_gamma[index]);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int mix = mix_base + warp;
    const BFloat16* mix_phi = phi + static_cast<std::size_t>(mix) * kFlat;
    float sum = 0.0F;
    for (int index = lane; index < kFlat; index += 32) {
        sum = fmaf(normalized[index], bf16_to_float(mix_phi[index]), sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    if (lane == 0) mixes[static_cast<std::size_t>(row) * kMixes + mix] = sum;
}

__global__ void mhc_coefficients_kernel(const BFloat16* branch_alpha,
                                        const BFloat16* branch_beta,
                                        float* mixes,
                                        float* h_post,
                                        float* h_res,
                                        int rows) {
    constexpr int kStreams = 4;
    constexpr int kMixes = 24;
    const int row = blockIdx.x;
    if (threadIdx.x != 0 || row >= rows) return;
    float* row_mixes = mixes + static_cast<std::size_t>(row) * kMixes;
    h_post += static_cast<std::size_t>(row) * kStreams;
    h_res += static_cast<std::size_t>(row) * kStreams * kStreams;
    const float alpha_pre = bf16_to_float(branch_alpha[0]);
    const float alpha_post = bf16_to_float(branch_alpha[1]);
    const float alpha_res = bf16_to_float(branch_alpha[2]);
    for (int stream = 0; stream < kStreams; ++stream) {
        row_mixes[stream] = 1.0F / (1.0F + expf(-(row_mixes[stream] * alpha_pre +
                                                    bf16_to_float(branch_beta[stream])))) + 1.0e-6F;
        h_post[stream] = 2.0F / (1.0F + expf(-(row_mixes[4 + stream] * alpha_post +
                                                  bf16_to_float(branch_beta[4 + stream]))));
    }
    for (int source = 0; source < kStreams; ++source) {
        float row_max = -FLT_MAX;
        for (int destination = 0; destination < kStreams; ++destination) {
            const int index = source * kStreams + destination;
            h_res[index] = row_mixes[8 + index] * alpha_res +
                           bf16_to_float(branch_beta[8 + index]);
            row_max = fmaxf(row_max, h_res[index]);
        }
        float total = 0.0F;
        for (int destination = 0; destination < kStreams; ++destination) {
            const int index = source * kStreams + destination;
            h_res[index] = expf(h_res[index] - row_max) + 1.0e-6F;
            total += h_res[index];
        }
        for (int destination = 0; destination < kStreams; ++destination) {
            h_res[source * kStreams + destination] /= total;
        }
    }
    for (int iteration = 0; iteration < 20; ++iteration) {
        for (int destination = 0; destination < kStreams; ++destination) {
            float total = 1.0e-6F;
            for (int source = 0; source < kStreams; ++source) {
                total += h_res[source * kStreams + destination];
            }
            for (int source = 0; source < kStreams; ++source) {
                h_res[source * kStreams + destination] /= total;
            }
        }
        if (iteration + 1 == 20) break;
        for (int source = 0; source < kStreams; ++source) {
            float total = 1.0e-6F;
            for (int destination = 0; destination < kStreams; ++destination) {
                total += h_res[source * kStreams + destination];
            }
            for (int destination = 0; destination < kStreams; ++destination) {
                h_res[source * kStreams + destination] /= total;
            }
        }
    }
}

__global__ void mhc_mix_parallel_kernel(const BFloat16* streams,
                                        const float* mixes,
                                        BFloat16* mixed,
                                        int rows) {
    constexpr int kHidden = 2560;
    constexpr int kStreams = 4;
    constexpr int kFlat = kHidden * kStreams;
    constexpr int kMixes = 24;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * kHidden;
    if (index >= total) return;
    const int row = index / kHidden;
    const int feature = index - row * kHidden;
    const BFloat16* row_streams = streams + static_cast<std::size_t>(row) * kFlat;
    const float* h_pre = mixes + static_cast<std::size_t>(row) * kMixes;
    float value = 0.0F;
    #pragma unroll
    for (int stream = 0; stream < kStreams; ++stream) {
        value = fmaf(h_pre[stream], bf16_to_float(row_streams[stream * kHidden + feature]), value);
    }
    mixed[index] = float_to_bf16(value);
}

__global__ void gather_rows_kernel(const BFloat16* input,
                                   const std::int32_t* row_indices,
                                   BFloat16* output,
                                   int selected_rows,
                                   int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = selected_rows * elements;
    if (index >= total) return;
    const int selected_row = index / elements;
    const int column = index - selected_row * elements;
    const int source_row = row_indices[selected_row];
    output[index] = input[static_cast<std::size_t>(source_row) * elements + column];
}

__global__ void scatter_add_scaled_rows_kernel(BFloat16* destination,
                                               const BFloat16* source,
                                               const std::int32_t* row_indices,
                                               const float* scales,
                                               int selected_rows,
                                               int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = selected_rows * elements;
    if (index >= total) return;
    const int selected_row = index / elements;
    const int column = index - selected_row * elements;
    const int destination_row = row_indices[selected_row];
    BFloat16* value = destination +
        static_cast<std::size_t>(destination_row) * elements + column;
    *value = float_to_bf16(
        bf16_to_float(*value) + bf16_to_float(source[index]) * scales[selected_row]);
}

__device__ float warp_sum_value(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffU, value, offset);
    return value;
}

__global__ void mhc_pre_kernel(const BFloat16* streams,
                               const BFloat16* phi,
                               const BFloat16* norm_gamma,
                               const BFloat16* branch_alpha,
                               const BFloat16* branch_beta,
                               BFloat16* mixed,
                               float* h_post,
                               float* h_res,
                               int rows,
                               float norm_epsilon) {
    constexpr int kHidden = 2560;
    constexpr int kStreams = 4;
    constexpr int kFlat = kHidden * kStreams;
    constexpr int kMixes = 24;
    const int row = blockIdx.x;
    if (row >= rows) return;
    streams += static_cast<std::size_t>(row) * kFlat;
    mixed += static_cast<std::size_t>(row) * kHidden;
    h_post += static_cast<std::size_t>(row) * kStreams;
    h_res += static_cast<std::size_t>(row) * kStreams * kStreams;
    __shared__ float warp_sums[8];
    __shared__ float inverse;
    __shared__ float mixes[kMixes];
    __shared__ float h_pre[kStreams];
    float square_sum = 0.0F;
    for (int i = threadIdx.x; i < kFlat; i += blockDim.x) {
        const float value = bf16_to_float(streams[i]);
        square_sum = fmaf(value, value, square_sum);
    }
    square_sum = block_sum(square_sum, warp_sums);
    if (threadIdx.x == 0) inverse = rsqrtf(square_sum / static_cast<float>(kFlat) + norm_epsilon);
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int mix_base = 0; mix_base < kMixes; mix_base += 8) {
        const int mix = mix_base + warp;
        float sum = 0.0F;
        if (mix < kMixes) {
            const BFloat16* row_phi = phi + static_cast<std::size_t>(mix) * kFlat;
            for (int i = lane; i < kFlat; i += 32) {
                const float normalized = bf16_to_float(streams[i]) * inverse * bf16_to_float(norm_gamma[i]);
                sum = fmaf(normalized, bf16_to_float(row_phi[i]), sum);
            }
            sum = warp_sum_value(sum);
            if (lane == 0) mixes[mix] = sum;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float alpha_pre = bf16_to_float(branch_alpha[0]);
        const float alpha_post = bf16_to_float(branch_alpha[1]);
        const float alpha_res = bf16_to_float(branch_alpha[2]);
        for (int stream = 0; stream < kStreams; ++stream) {
            h_pre[stream] = 1.0F / (1.0F + expf(-(mixes[stream] * alpha_pre +
                                                       bf16_to_float(branch_beta[stream])))) + 1.0e-6F;
            h_post[stream] = 2.0F / (1.0F + expf(-(mixes[4 + stream] * alpha_post +
                                                        bf16_to_float(branch_beta[4 + stream]))));
        }
        for (int source = 0; source < kStreams; ++source) {
            float row_max = -FLT_MAX;
            for (int destination = 0; destination < kStreams; ++destination) {
                const int index = source * kStreams + destination;
                h_res[index] = mixes[8 + index] * alpha_res + bf16_to_float(branch_beta[8 + index]);
                row_max = fmaxf(row_max, h_res[index]);
            }
            float total = 0.0F;
            for (int destination = 0; destination < kStreams; ++destination) {
                const int index = source * kStreams + destination;
                h_res[index] = expf(h_res[index] - row_max) + 1.0e-6F;
                total += h_res[index];
            }
            for (int destination = 0; destination < kStreams; ++destination) {
                h_res[source * kStreams + destination] /= total;
            }
        }
        for (int iteration = 0; iteration < 20; ++iteration) {
            for (int destination = 0; destination < kStreams; ++destination) {
                float total = 1.0e-6F;
                for (int source = 0; source < kStreams; ++source) total += h_res[source * kStreams + destination];
                for (int source = 0; source < kStreams; ++source) h_res[source * kStreams + destination] /= total;
            }
            if (iteration + 1 == 20) break;
            for (int source = 0; source < kStreams; ++source) {
                float total = 1.0e-6F;
                for (int destination = 0; destination < kStreams; ++destination) total += h_res[source * kStreams + destination];
                for (int destination = 0; destination < kStreams; ++destination) h_res[source * kStreams + destination] /= total;
            }
        }
    }
    __syncthreads();
    for (int feature = threadIdx.x; feature < kHidden; feature += blockDim.x) {
        float value = 0.0F;
        #pragma unroll
        for (int stream = 0; stream < kStreams; ++stream) {
            value = fmaf(h_pre[stream], bf16_to_float(streams[stream * kHidden + feature]), value);
        }
        mixed[feature] = float_to_bf16(value);
    }
}

__global__ void mhc_post_kernel(const BFloat16* hidden,
                                const BFloat16* residual,
                                const float* h_post,
                                const float* h_res,
                                BFloat16* output,
                                int rows) {
    constexpr int kHidden = 2560;
    constexpr int kStreams = 4;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * kStreams * kHidden;
    if (index >= total) return;
    const int row_stride = kStreams * kHidden;
    const int row = index / row_stride;
    const int within = index - row * row_stride;
    const int destination = within / kHidden;
    const int feature = within - destination * kHidden;
    const BFloat16* row_residual = residual + static_cast<std::size_t>(row) * row_stride;
    float value = h_post[static_cast<std::size_t>(row) * kStreams + destination] *
                  bf16_to_float(hidden[static_cast<std::size_t>(row) * kHidden + feature]);
    #pragma unroll
    for (int source = 0; source < kStreams; ++source) {
        value = fmaf(h_res[(static_cast<std::size_t>(row) * kStreams + source) * kStreams + destination],
                     bf16_to_float(row_residual[source * kHidden + feature]), value);
    }
    output[index] = float_to_bf16(value);
}

__global__ void mhc_merge_kernel(const BFloat16* streams,
                                 const BFloat16* phi,
                                 const BFloat16* norm_gamma,
                                 const BFloat16* branch_alpha,
                                 const BFloat16* branch_beta,
                                 BFloat16* output,
                                 int rows,
                                 float norm_epsilon) {
    constexpr int kHidden = 2560;
    constexpr int kStreams = 4;
    constexpr int kFlat = kHidden * kStreams;
    const int row = blockIdx.x;
    if (row >= rows) return;
    streams += static_cast<std::size_t>(row) * kFlat;
    output += static_cast<std::size_t>(row) * kHidden;
    __shared__ float warp_sums[8];
    __shared__ float inverse;
    __shared__ float weights[kStreams];
    float square_sum = 0.0F;
    for (int i = threadIdx.x; i < kFlat; i += blockDim.x) {
        const float value = bf16_to_float(streams[i]);
        square_sum = fmaf(value, value, square_sum);
    }
    square_sum = block_sum(square_sum, warp_sums);
    if (threadIdx.x == 0) inverse = rsqrtf(square_sum / static_cast<float>(kFlat) + norm_epsilon);
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp < kStreams) {
        float sum = 0.0F;
        const BFloat16* row_phi = phi + static_cast<std::size_t>(warp) * kFlat;
        for (int i = lane; i < kFlat; i += 32) {
            const float normalized = bf16_to_float(streams[i]) * inverse * bf16_to_float(norm_gamma[i]);
            sum = fmaf(normalized, bf16_to_float(row_phi[i]), sum);
        }
        sum = warp_sum_value(sum);
        if (lane == 0) {
            weights[warp] = 1.0F / (1.0F + expf(-(sum * bf16_to_float(branch_alpha[0]) +
                                                      bf16_to_float(branch_beta[warp]))));
        }
    }
    __syncthreads();
    for (int feature = threadIdx.x; feature < kHidden; feature += blockDim.x) {
        float value = 0.0F;
        #pragma unroll
        for (int stream = 0; stream < kStreams; ++stream) {
            value = fmaf(weights[stream], bf16_to_float(streams[stream * kHidden + feature]), value);
        }
        output[feature] = float_to_bf16(value);
    }
}

__global__ void quantize(const BFloat16* input,
                         std::uint8_t* output,
                         std::uint8_t* scales,
                         int groups,
                         float global_scale) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    const BFloat16* block = input + group * 16;
    float maximum = 0.0F;
    #pragma unroll
    for (int i = 0; i < 16; ++i) maximum = fmaxf(maximum, fabsf(bf16_to_float(block[i])));
    const __nv_fp8_e4m3 scale(maximum * (global_scale / 6.0F));
    scales[group] = scale.__x;
    const float decoded_scale = static_cast<float>(scale);
    const float multiplier = decoded_scale == 0.0F ? 0.0F : global_scale / decoded_scale;
    std::uint8_t* packed = output + group * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float2 pair = make_float2(
            bf16_to_float(block[2 * i]) * multiplier,
            bf16_to_float(block[2 * i + 1]) * multiplier);
        packed[i] = __nv_fp4x2_e2m1(pair).__x;
    }
}

__global__ void gemv(const std::uint8_t* input,
                     const std::uint8_t* weight,
                     const std::uint8_t* input_scales,
                     const std::uint8_t* weight_scales,
                     BFloat16* output,
                     int rows,
                     int columns,
                     float alpha) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float warp_sums[8];
    float sum = 0.0F;
    const std::uint8_t* row_weight = weight + static_cast<std::size_t>(row) * (columns / 2);
    const std::uint8_t* row_scale = weight_scales + static_cast<std::size_t>(row) * (columns / 16);
    for (int element = threadIdx.x; element < columns; element += blockDim.x) {
        const int shift = (element & 1) * 4;
        const float input_value = decode_fp4(input[element / 2] >> shift);
        const float weight_value = decode_fp4(row_weight[element / 2] >> shift);
        const int group = element / 16;
        sum = fmaf(input_value * decode_e4m3(input_scales[group]),
                   weight_value * decode_e4m3(row_scale[group]), sum);
    }
    sum = block_sum(sum, warp_sums);
    if (threadIdx.x == 0) output[row] = float_to_bf16(sum * alpha);
}

__global__ void swizzle(const std::uint8_t* linear,
                        std::uint8_t* swizzled,
                        int rows,
                        int scale_columns,
                        int padded_scale_columns) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = rows * scale_columns;
    if (index >= elements) return;
    const int row = index / scale_columns;
    const int column = index - row * scale_columns;
    const int row_block = row / 128;
    const int row_in_block = row - row_block * 128;
    const int row_quarter = row_in_block / 32;
    const int row_in_quarter = row_in_block - row_quarter * 32;
    const int column_block = column / 4;
    const int column_in_block = column - column_block * 4;
    const int column_blocks = padded_scale_columns / 4;
    const std::size_t destination =
        (static_cast<std::size_t>(row_block) * column_blocks + column_block) * 512 +
        row_in_quarter * 16 + row_quarter * 4 + column_in_block;
    swizzled[destination] = linear[index];
}

}  // namespace

cudaError_t repeat_streams(const BFloat16* input,
                           BFloat16* output,
                           int rows,
                           int elements,
                           int streams,
                           cudaStream_t stream) {
    if (rows <= 0 || elements <= 0 || streams <= 0) return cudaErrorInvalidValue;
    const int total = rows * elements * streams;
    repeat_streams_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        input, output, rows, elements, streams);
    return cudaGetLastError();
}

cudaError_t rmsnorm(const BFloat16* input,
                    const BFloat16* weight,
                    BFloat16* output,
                    int rows,
                    int elements,
                    float epsilon,
                    cudaStream_t stream) {
    if (rows <= 0 || elements <= 0 || !(epsilon > 0.0F)) return cudaErrorInvalidValue;
    rmsnorm_kernel<<<rows, kThreads, 0, stream>>>(
        input, weight, output, rows, elements, elements, elements, epsilon);
    return cudaGetLastError();
}

cudaError_t rmsnorm_strided(const BFloat16* input,
                            int input_stride,
                            const BFloat16* weight,
                            BFloat16* output,
                            int output_stride,
                            int rows,
                            int elements,
                            float epsilon,
                            cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr ||
        input_stride < elements || output_stride < elements || rows <= 0 ||
        elements <= 0 || !(epsilon > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    rmsnorm_kernel<<<rows, kThreads, 0, stream>>>(
        input, weight, output, rows, elements, input_stride, output_stride, epsilon);
    return cudaGetLastError();
}

cudaError_t add_inplace(BFloat16* destination,
                        const BFloat16* source,
                        int elements,
                        cudaStream_t stream) {
    if (elements <= 0) return cudaErrorInvalidValue;
    add_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(destination, source, elements);
    return cudaGetLastError();
}

cudaError_t swiglu(const BFloat16* gate,
                   const BFloat16* up,
                   BFloat16* output,
                   int elements,
                   cudaStream_t stream) {
    if (elements <= 0) return cudaErrorInvalidValue;
    swiglu_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(gate, up, output, elements);
    return cudaGetLastError();
}

cudaError_t expert_combine(const BFloat16* expert_rows,
                           const float* route_weights,
                           BFloat16* output,
                           int experts,
                           int elements,
                           cudaStream_t stream) {
    if (expert_rows == nullptr || route_weights == nullptr || output == nullptr ||
        experts <= 0 || elements <= 0) {
        return cudaErrorInvalidValue;
    }
    expert_combine_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        expert_rows, route_weights, output, experts, elements);
    return cudaGetLastError();
}

cudaError_t expert_combine_rows(const BFloat16* expert_rows,
                                const float* route_weights,
                                BFloat16* output,
                                int batch_rows,
                                int experts,
                                int elements,
                                cudaStream_t stream) {
    if (expert_rows == nullptr || route_weights == nullptr || output == nullptr ||
        batch_rows <= 0 || experts <= 0 || elements <= 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid((elements + kThreads - 1) / kThreads, batch_rows);
    expert_combine_rows_kernel<<<grid, kThreads, 0, stream>>>(
        expert_rows, route_weights, output, experts, elements);
    return cudaGetLastError();
}

cudaError_t bf16_gemv(const BFloat16* input,
                      const BFloat16* weight,
                      BFloat16* output,
                      int rows,
                      int columns,
                      cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr || rows <= 0 || columns <= 0) {
        return cudaErrorInvalidValue;
    }
    bf16_gemv_kernel<<<rows, kThreads, 0, stream>>>(input, weight, output, rows, columns);
    return cudaGetLastError();
}

cudaError_t bf16_gemm(const BFloat16* input,
                      const BFloat16* weight,
                      BFloat16* output,
                      int batch_rows,
                      int rows,
                      int columns,
                      cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr ||
        batch_rows <= 0 || rows <= 0 || columns <= 0) {
        return cudaErrorInvalidValue;
    }
    bf16_gemm_kernel<false><<<rows, kThreads, 0, stream>>>(
        input, weight, output, batch_rows, rows, columns);
    return cudaGetLastError();
}

cudaError_t bf16_gemv_f32(const BFloat16* input,
                          const BFloat16* weight,
                          float* output,
                          int rows,
                          int columns,
                          cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr || rows <= 0 || columns <= 0) {
        return cudaErrorInvalidValue;
    }
    bf16_gemv_f32_kernel<<<rows, kThreads, 0, stream>>>(input, weight, output, rows, columns);
    return cudaGetLastError();
}

cudaError_t bf16_gemm_f32(const BFloat16* input,
                          const BFloat16* weight,
                          float* output,
                          int batch_rows,
                          int rows,
                          int columns,
                          cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr ||
        batch_rows <= 0 || rows <= 0 || columns <= 0) {
        return cudaErrorInvalidValue;
    }
    bf16_gemm_kernel<true><<<rows, kThreads, 0, stream>>>(
        input, weight, output, batch_rows, rows, columns);
    return cudaGetLastError();
}

cudaError_t add_scaled_inplace(BFloat16* destination,
                               const BFloat16* source,
                               const float* scale,
                               int elements,
                               cudaStream_t stream) {
    if (destination == nullptr || source == nullptr || scale == nullptr || elements <= 0) {
        return cudaErrorInvalidValue;
    }
    add_scaled_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        destination, source, scale, elements);
    return cudaGetLastError();
}

cudaError_t argmax(const BFloat16* values,
                   int elements,
                   std::int32_t* index,
                   cudaStream_t stream) {
    if (values == nullptr || elements <= 0 || index == nullptr) return cudaErrorInvalidValue;
    argmax_kernel<<<1, kThreads, 0, stream>>>(values, elements, index);
    return cudaGetLastError();
}

cudaError_t argmax_rows(const BFloat16* values,
                        int rows,
                        int elements,
                        std::int32_t* indices,
                        cudaStream_t stream) {
    if (values == nullptr || rows <= 0 || elements <= 0 || indices == nullptr) {
        return cudaErrorInvalidValue;
    }
    argmax_kernel<<<rows, kThreads, 0, stream>>>(values, elements, indices);
    return cudaGetLastError();
}

cudaError_t mome_conv3_step(BFloat16* values,
                            const BFloat16* weight,
                            BFloat16* state,
                            int channels,
                            cudaStream_t stream) {
    if (channels <= 0) return cudaErrorInvalidValue;
    mome_conv3_kernel<<<(channels + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        values, weight, state, channels, nullptr);
    return cudaGetLastError();
}

cudaError_t mome_conv3_rows(BFloat16* values,
                            const BFloat16* weight,
                            BFloat16* state,
                            int rows,
                            int channels,
                            cudaStream_t stream) {
    return mome_conv3_rows_strided(
        values, channels, weight, state, rows, channels, stream);
}

cudaError_t mome_conv3_rows_strided(BFloat16* values,
                                    int row_stride,
                                    const BFloat16* weight,
                                    BFloat16* state,
                                    int rows,
                                    int channels,
                                    cudaStream_t stream) {
    if (values == nullptr || weight == nullptr || state == nullptr ||
        rows <= 0 || channels <= 0 || row_stride < channels) {
        return cudaErrorInvalidValue;
    }
    mome_conv3_rows_kernel<<<(channels + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        values, row_stride, weight, state, nullptr, rows, channels);
    return cudaGetLastError();
}

cudaError_t mome_conv3_rows_strided_capture(BFloat16* values,
                                            int row_stride,
                                            const BFloat16* weight,
                                            BFloat16* state,
                                            BFloat16* state_after_rows,
                                            int rows,
                                            int channels,
                                            cudaStream_t stream) {
    if (values == nullptr || weight == nullptr || state == nullptr ||
        state_after_rows == nullptr || rows <= 0 || channels <= 0 ||
        row_stride < channels) {
        return cudaErrorInvalidValue;
    }
    mome_conv3_rows_kernel<<<(channels + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        values, row_stride, weight, state, state_after_rows, rows, channels);
    return cudaGetLastError();
}

cudaError_t mome_commit_captured_row(BFloat16* state,
                                     const BFloat16* captured_rows,
                                     int layers,
                                     int rows,
                                     int row,
                                     int query_channels,
                                     int kv_channels,
                                     int output_channels,
                                     cudaStream_t stream) {
    if (state == nullptr || captured_rows == nullptr || layers <= 0 || rows <= 0 ||
        row < 0 || row >= rows || query_channels <= 0 || kv_channels <= 0 ||
        output_channels <= 0) {
        return cudaErrorInvalidValue;
    }
    const int state_elements = 2 * (query_channels + kv_channels + output_channels);
    const int total = layers * state_elements;
    mome_commit_captured_row_kernel<<<(total + kThreads - 1) / kThreads,
                                      kThreads, 0, stream>>>(
        state, captured_rows, layers, rows, row,
        query_channels, kv_channels, output_channels);
    return cudaGetLastError();
}

cudaError_t router_top8(const float* logits,
                        const float* correction_bias,
                        std::int32_t* expert_ids,
                        float* expert_weights,
                        float routed_scale,
                        cudaStream_t stream) {
    if (!(routed_scale > 0.0F)) return cudaErrorInvalidValue;
    router_top8_kernel<<<1, 256, 0, stream>>>(
        logits, correction_bias, expert_ids, expert_weights, 1, routed_scale);
    return cudaGetLastError();
}

cudaError_t router_top8_rows(const float* logits,
                             const float* correction_bias,
                             std::int32_t* expert_ids,
                             float* expert_weights,
                             int rows,
                             float routed_scale,
                             cudaStream_t stream) {
    if (logits == nullptr || correction_bias == nullptr || expert_ids == nullptr ||
        expert_weights == nullptr || rows <= 0 || !(routed_scale > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    router_top8_kernel<<<rows, 256, 0, stream>>>(
        logits, correction_bias, expert_ids, expert_weights, rows, routed_scale);
    return cudaGetLastError();
}

cudaError_t gather_rows(const BFloat16* input,
                        const std::int32_t* row_indices,
                        BFloat16* output,
                        int selected_rows,
                        int elements,
                        cudaStream_t stream) {
    if (input == nullptr || row_indices == nullptr || output == nullptr ||
        selected_rows <= 0 || elements <= 0) {
        return cudaErrorInvalidValue;
    }
    const int total = selected_rows * elements;
    gather_rows_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        input, row_indices, output, selected_rows, elements);
    return cudaGetLastError();
}

cudaError_t scatter_add_scaled_rows(BFloat16* destination,
                                    const BFloat16* source,
                                    const std::int32_t* row_indices,
                                    const float* scales,
                                    int selected_rows,
                                    int elements,
                                    cudaStream_t stream) {
    if (destination == nullptr || source == nullptr || row_indices == nullptr ||
        scales == nullptr || selected_rows <= 0 || elements <= 0) {
        return cudaErrorInvalidValue;
    }
    const int total = selected_rows * elements;
    scatter_add_scaled_rows_kernel<<<
        (total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            destination, source, row_indices, scales, selected_rows, elements);
    return cudaGetLastError();
}

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
                    cudaStream_t stream) {
    if (rows <= 0 || !(norm_epsilon > 0.0F)) return cudaErrorInvalidValue;
    mhc_pre_kernel<<<rows, kThreads, 0, stream>>>(
        streams, phi, norm_gamma, branch_alpha, branch_beta, mixed, h_post, h_res, rows, norm_epsilon);
    return cudaGetLastError();
}

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
                             cudaStream_t stream) {
    if (streams == nullptr || phi == nullptr || norm_gamma == nullptr ||
        branch_alpha == nullptr || branch_beta == nullptr || mixed == nullptr ||
        h_post == nullptr || h_res == nullptr || inverse_scratch == nullptr ||
        mixes_scratch == nullptr || rows <= 0 || !(norm_epsilon > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    mhc_inverse_kernel<<<rows, kThreads, 0, stream>>>(
        streams, inverse_scratch, rows, norm_epsilon);
    if (rows >= 3) {
        mhc_phi_paired_kernel<<<rows * 12, 64, 0, stream>>>(
            streams, phi, norm_gamma, inverse_scratch, mixes_scratch, rows);
    } else {
        mhc_phi_parallel_kernel<<<rows * 24, 32, 0, stream>>>(
            streams, phi, norm_gamma, inverse_scratch, mixes_scratch, rows);
    }
    mhc_coefficients_kernel<<<rows, 1, 0, stream>>>(
        branch_alpha, branch_beta, mixes_scratch, h_post, h_res, rows);
    const int elements = rows * 2560;
    mhc_mix_parallel_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        streams, mixes_scratch, mixed, rows);
    return cudaGetLastError();
}

cudaError_t mhc_post(const BFloat16* hidden,
                     const BFloat16* residual_streams,
                     const float* h_post,
                     const float* h_res,
                     BFloat16* output_streams,
                     int rows,
                     cudaStream_t stream) {
    if (rows <= 0) return cudaErrorInvalidValue;
    const int elements = rows * 4 * 2560;
    mhc_post_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        hidden, residual_streams, h_post, h_res, output_streams, rows);
    return cudaGetLastError();
}

cudaError_t mhc_merge(const BFloat16* streams,
                      const BFloat16* phi,
                      const BFloat16* norm_gamma,
                      const BFloat16* branch_alpha,
                      const BFloat16* branch_beta,
                      BFloat16* output,
                      int rows,
                      float norm_epsilon,
                      cudaStream_t stream) {
    if (rows <= 0 || !(norm_epsilon > 0.0F)) return cudaErrorInvalidValue;
    mhc_merge_kernel<<<rows, kThreads, 0, stream>>>(
        streams, phi, norm_gamma, branch_alpha, branch_beta, output, rows, norm_epsilon);
    return cudaGetLastError();
}

cudaError_t nvfp4_quantize(const BFloat16* input,
                           std::uint8_t* output,
                           std::uint8_t* scales,
                           int elements,
                           float global_scale,
                           cudaStream_t stream) {
    if (elements <= 0 || elements % 16 != 0 || !(global_scale > 0.0F)) return cudaErrorInvalidValue;
    const int groups = elements / 16;
    quantize<<<(groups + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        input, output, scales, groups, global_scale);
    return cudaGetLastError();
}

cudaError_t nvfp4_gemv(const std::uint8_t* input,
                       const std::uint8_t* weight,
                       const std::uint8_t* input_scales,
                       const std::uint8_t* weight_scales,
                       BFloat16* output,
                       int rows,
                       int columns,
                       float alpha,
                       cudaStream_t stream) {
    if (rows <= 0 || columns <= 0 || columns % 16 != 0) return cudaErrorInvalidValue;
    gemv<<<rows, kThreads, 0, stream>>>(
        input, weight, input_scales, weight_scales, output, rows, columns, alpha);
    return cudaGetLastError();
}

std::size_t nvfp4_swizzled_scale_bytes(int rows, int scale_columns) {
    if (rows <= 0 || scale_columns <= 0) return 0;
    const std::size_t row_blocks = static_cast<std::size_t>(rows + 127) / 128;
    const std::size_t padded_columns = static_cast<std::size_t>(scale_columns + 3) & ~std::size_t{3};
    return row_blocks * padded_columns * 128;
}

cudaError_t nvfp4_swizzle_scales(const std::uint8_t* linear,
                                 std::uint8_t* swizzled,
                                 int rows,
                                 int scale_columns,
                                 cudaStream_t stream) {
    const std::size_t bytes = nvfp4_swizzled_scale_bytes(rows, scale_columns);
    if (bytes == 0) return cudaErrorInvalidValue;
    cudaError_t status = cudaMemsetAsync(swizzled, 0, bytes, stream);
    if (status != cudaSuccess) return status;
    const int elements = rows * scale_columns;
    const int padded_columns = (scale_columns + 3) & ~3;
    swizzle<<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        linear, swizzled, rows, scale_columns, padded_columns);
    return cudaGetLastError();
}

}  // namespace p92::cuda
