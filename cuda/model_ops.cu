#include "p92/kernels.h"

#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <cfloat>
#include <cstddef>
#include <new>

namespace p92::cuda {

struct Bf16MlaPlan {
    cublasHandle_t handle = nullptr;
};

namespace {

constexpr int kHeads = 48;
constexpr int kNope = 128;
constexpr int kRope = 64;
constexpr int kLatent = 512;
constexpr int kValue = 128;
constexpr int kUpPerHead = kNope + kValue;
constexpr int kQueryPerHead = kNope + kRope;
constexpr int kSinks = 128;

cudaError_t cublas_result(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS) return cudaSuccess;
    if (status == CUBLAS_STATUS_ALLOC_FAILED) return cudaErrorMemoryAllocation;
    if (status == CUBLAS_STATUS_INVALID_VALUE) return cudaErrorInvalidValue;
    return cudaErrorUnknown;
}

__device__ __forceinline__ float to_float(BFloat16 bits) {
    __nv_bfloat16_raw raw;
    raw.x = bits;
    return __bfloat162float(static_cast<__nv_bfloat16>(raw));
}

__device__ __forceinline__ BFloat16 to_bf16(float value) {
    return static_cast<__nv_bfloat16_raw>(__float2bfloat16_rn(value)).x;
}

__device__ float block_sum(float value, float* warp_sums) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = warp == 0 && lane < blockDim.x / 32 ? warp_sums[lane] : 0.0F;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) warp_sums[0] = value;
    }
    __syncthreads();
    return warp_sums[0];
}

__global__ void absorb_query_kernel(const BFloat16* q_up,
                                    const BFloat16* kv_up_weight,
                                    BFloat16* q_absorbed) {
    const int head = blockIdx.x;
    const int latent = threadIdx.x + blockIdx.y * blockDim.x;
    if (head >= kHeads || latent >= kLatent) return;
    const BFloat16* query = q_up + head * kQueryPerHead;
    float value = 0.0F;
    #pragma unroll 4
    for (int nope = 0; nope < kNope; ++nope) {
        const BFloat16* weight = kv_up_weight +
            static_cast<std::size_t>(head * kUpPerHead + nope) * kLatent;
        value = fmaf(to_float(query[nope]), to_float(weight[latent]), value);
    }
    q_absorbed[head * kLatent + latent] = to_bf16(value);
}

__global__ void absorb_query_rows_kernel(const BFloat16* q_up,
                                         const BFloat16* kv_up_weight,
                                         BFloat16* q_absorbed,
                                         int batch_rows) {
    __shared__ BFloat16 query_shared[kNope];
    const int head = blockIdx.x;
    const int latent = threadIdx.x + blockIdx.y * blockDim.x;
    const int row = blockIdx.z;
    if (head >= kHeads || latent >= kLatent || row >= batch_rows) return;
    const BFloat16* query = q_up +
        static_cast<std::size_t>(row) * kHeads * kQueryPerHead + head * kQueryPerHead;
    if (threadIdx.x < kNope) query_shared[threadIdx.x] = query[threadIdx.x];
    __syncthreads();
    float value = 0.0F;
    #pragma unroll 4
    for (int nope = 0; nope < kNope; ++nope) {
        const BFloat16* weight = kv_up_weight +
            static_cast<std::size_t>(head * kUpPerHead + nope) * kLatent;
        value = fmaf(to_float(query_shared[nope]), to_float(weight[latent]), value);
    }
    q_absorbed[(static_cast<std::size_t>(row) * kHeads + head) * kLatent + latent] =
        to_bf16(value);
}

__global__ void gather_query_nope_rows_kernel(const BFloat16* q_up,
                                              BFloat16* gathered,
                                              int batch_rows) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int element = threadIdx.x;
    if (head >= kHeads || row >= batch_rows || element >= kNope) return;
    gathered[(static_cast<std::size_t>(head) * batch_rows + row) * kNope + element] =
        q_up[(static_cast<std::size_t>(row) * kHeads + head) * kQueryPerHead + element];
}

__global__ void scatter_absorbed_rows_kernel(const BFloat16* head_major,
                                             BFloat16* row_major,
                                             int batch_rows) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int element = threadIdx.x + blockIdx.z * blockDim.x;
    if (head >= kHeads || row >= batch_rows || element >= kLatent) return;
    row_major[(static_cast<std::size_t>(row) * kHeads + head) * kLatent + element] =
        head_major[(static_cast<std::size_t>(head) * batch_rows + row) * kLatent + element];
}

__global__ void query_rope_kernel(const BFloat16* q_up,
                                  BFloat16* q_rope,
                                  int position,
                                  float theta) {
    const int head = blockIdx.x;
    const int pair = threadIdx.x;
    if (head >= kHeads || pair >= kRope / 2) return;
    const BFloat16* source = q_up + head * kQueryPerHead + kNope;
    const float frequency = powf(theta, -static_cast<float>(pair) / (kRope / 2));
    const float angle = static_cast<float>(position) * frequency;
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    const float first = to_float(source[pair]);
    const float second = to_float(source[pair + kRope / 2]);
    q_rope[head * kRope + pair] = to_bf16(first * cosine - second * sine);
    q_rope[head * kRope + pair + kRope / 2] = to_bf16(second * cosine + first * sine);
}

__global__ void query_rope_rows_kernel(const BFloat16* q_up,
                                       BFloat16* q_rope,
                                       int batch_rows,
                                       int start_position,
                                       float theta) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int pair = threadIdx.x;
    if (head >= kHeads || row >= batch_rows || pair >= kRope / 2) return;
    const BFloat16* source = q_up +
        static_cast<std::size_t>(row) * kHeads * kQueryPerHead +
        head * kQueryPerHead + kNope;
    const float frequency = powf(theta, -static_cast<float>(pair) / (kRope / 2));
    const float angle = static_cast<float>(start_position + row) * frequency;
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    const float first = to_float(source[pair]);
    const float second = to_float(source[pair + kRope / 2]);
    BFloat16* destination = q_rope +
        (static_cast<std::size_t>(row) * kHeads + head) * kRope;
    destination[pair] = to_bf16(first * cosine - second * sine);
    destination[pair + kRope / 2] = to_bf16(second * cosine + first * sine);
}

__global__ void key_rope_kernel(const BFloat16* kv_down,
                                BFloat16* key_rope,
                                int position,
                                float theta) {
    const int pair = threadIdx.x;
    if (pair >= kRope / 2) return;
    const BFloat16* source = kv_down + kLatent;
    const float frequency = powf(theta, -static_cast<float>(pair) / (kRope / 2));
    const float angle = static_cast<float>(position) * frequency;
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    const float first = to_float(source[pair]);
    const float second = to_float(source[pair + kRope / 2]);
    key_rope[pair] = to_bf16(first * cosine - second * sine);
    key_rope[pair + kRope / 2] = to_bf16(second * cosine + first * sine);
}

__global__ void key_rope_rows_kernel(const BFloat16* kv_down,
                                     BFloat16* key_rope,
                                     int batch_rows,
                                     int start_position,
                                     float theta) {
    const int row = blockIdx.x;
    const int pair = threadIdx.x;
    if (row >= batch_rows || pair >= kRope / 2) return;
    const BFloat16* source = kv_down +
        static_cast<std::size_t>(row) * (kLatent + kRope) + kLatent;
    const float frequency = powf(theta, -static_cast<float>(pair) / (kRope / 2));
    const float angle = static_cast<float>(start_position + row) * frequency;
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    const float first = to_float(source[pair]);
    const float second = to_float(source[pair + kRope / 2]);
    BFloat16* destination = key_rope + static_cast<std::size_t>(row) * kRope;
    destination[pair] = to_bf16(first * cosine - second * sine);
    destination[pair + kRope / 2] = to_bf16(second * cosine + first * sine);
}

__global__ void store_cache_rows_kernel(BFloat16* cache_kv,
                                        BFloat16* cache_rope,
                                        const BFloat16* row_kv,
                                        const BFloat16* row_rope,
                                        int start_position,
                                        int rows,
                                        int cache_capacity) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int kRowElements = kLatent + kRope;
    if (index >= rows * kRowElements) return;
    const int row = index / kRowElements;
    const int feature = index - row * kRowElements;
    const int slot = (start_position + row) % cache_capacity;
    if (feature < kLatent) {
        cache_kv[static_cast<std::size_t>(slot) * kLatent + feature] =
            row_kv[static_cast<std::size_t>(row) * kLatent + feature];
    } else {
        const int rope_feature = feature - kLatent;
        cache_rope[static_cast<std::size_t>(slot) * kRope + rope_feature] =
            row_rope[static_cast<std::size_t>(row) * kRope + rope_feature];
    }
}

__global__ void transfer_cache_rows_kernel(bool restore,
                                           BFloat16* live_kv,
                                           BFloat16* live_rope,
                                           BFloat16* snapshot_kv,
                                           BFloat16* snapshot_rope,
                                           int start_position,
                                           int snapshot_rows,
                                           int first_row,
                                           int row_count,
                                           int cache_capacity) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int kRowElements = kLatent + kRope;
    if (index >= row_count * kRowElements) return;
    const int local_row = index / kRowElements;
    const int row = first_row + local_row;
    const int feature = index - local_row * kRowElements;
    const int slot = (start_position + row) % cache_capacity;
    if (feature < kLatent) {
        BFloat16* live = live_kv + static_cast<std::size_t>(slot) * kLatent + feature;
        BFloat16* saved = snapshot_kv +
            static_cast<std::size_t>(row) * kLatent + feature;
        if (restore) *live = *saved;
        else *saved = *live;
    } else {
        const int rope_feature = feature - kLatent;
        BFloat16* live = live_rope +
            static_cast<std::size_t>(slot) * kRope + rope_feature;
        BFloat16* saved = snapshot_rope +
            static_cast<std::size_t>(row) * kRope + rope_feature;
        if (restore) *live = *saved;
        else *saved = *live;
    }
}

__global__ void latent_attention_kernel(const BFloat16* q_absorbed,
                                        const BFloat16* q_rope,
                                        const BFloat16* sink_kv,
                                        const BFloat16* sink_rope,
                                        const BFloat16* cache_kv,
                                        const BFloat16* cache_rope,
                                        int cache_start,
                                        int cache_count,
                                        int cache_capacity,
                                        const std::int32_t* selected_indices,
                                        BFloat16* latent_output,
                                        float softmax_scale) {
    const int head = blockIdx.x;
    if (head >= kHeads) return;
    __shared__ float warp_sums[8];
    __shared__ float old_factor;
    __shared__ float current_factor;
    __shared__ float denominator;
    __shared__ float running_max;
    if (threadIdx.x == 0) {
        denominator = 0.0F;
        running_max = -FLT_MAX;
    }
    __syncthreads();

    float accumulator[2] = {0.0F, 0.0F};
    const BFloat16* query_latent = q_absorbed + head * kLatent;
    const BFloat16* query_rope = q_rope + head * kRope;
    const int total = kSinks + cache_count;
    for (int key_index = 0; key_index < total; ++key_index) {
        const bool sink = key_index < kSinks;
        const int logical_index = key_index - kSinks;
        const int cache_index = selected_indices != nullptr
            ? selected_indices[logical_index]
            : (cache_capacity > 0 ? (cache_start + logical_index) % cache_capacity
                                  : cache_start + logical_index);
        const BFloat16* key_latent = sink
            ? sink_kv + static_cast<std::size_t>(key_index) * kLatent
            : cache_kv + static_cast<std::size_t>(cache_index) * kLatent;
        const BFloat16* key_rope = sink
            ? sink_rope + static_cast<std::size_t>(key_index) * kRope
            : cache_rope + static_cast<std::size_t>(cache_index) * kRope;

        float dot = 0.0F;
        for (int i = threadIdx.x; i < kLatent; i += blockDim.x) {
            dot = fmaf(to_float(query_latent[i]), to_float(key_latent[i]), dot);
        }
        for (int i = threadIdx.x; i < kRope; i += blockDim.x) {
            dot = fmaf(to_float(query_rope[i]), to_float(key_rope[i]), dot);
        }
        const float score = block_sum(dot, warp_sums) * softmax_scale;
        if (threadIdx.x == 0) {
            const float next_max = fmaxf(running_max, score);
            old_factor = denominator == 0.0F ? 0.0F : expf(running_max - next_max);
            current_factor = expf(score - next_max);
            denominator = denominator * old_factor + current_factor;
            running_max = next_max;
        }
        __syncthreads();
        #pragma unroll
        for (int lane = 0; lane < 2; ++lane) {
            const int feature = threadIdx.x + lane * blockDim.x;
            if (feature < kLatent) {
                accumulator[lane] = accumulator[lane] * old_factor +
                    to_float(key_latent[feature]) * current_factor;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int lane = 0; lane < 2; ++lane) {
        const int feature = threadIdx.x + lane * blockDim.x;
        if (feature < kLatent) {
            latent_output[head * kLatent + feature] =
                to_bf16(accumulator[lane] / denominator);
        }
    }
}

__global__ void latent_attention_tiled_kernel(const BFloat16* q_absorbed,
                                              const BFloat16* q_rope,
                                              const BFloat16* sink_kv,
                                              const BFloat16* sink_rope,
                                              const BFloat16* cache_kv,
                                              const BFloat16* cache_rope,
                                              int cache_start,
                                              int cache_count,
                                              int query_rows,
                                              int cache_count_step,
                                              int cache_capacity,
                                              const std::int32_t* selected_indices,
                                              BFloat16* latent_output,
                                              float softmax_scale) {
    constexpr int kTile = 64;
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (head >= kHeads || row >= query_rows) return;
    cache_count += row * cache_count_step;
    __shared__ float scores[kTile];
    __shared__ float weights[kTile];
    __shared__ float warp_sums[8];
    __shared__ float old_factor;
    __shared__ float denominator;
    __shared__ float running_max;
    if (threadIdx.x == 0) {
        denominator = 0.0F;
        running_max = -FLT_MAX;
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const BFloat16* query_latent = q_absorbed +
        (static_cast<std::size_t>(row) * kHeads + head) * kLatent;
    const BFloat16* query_position = q_rope +
        (static_cast<std::size_t>(row) * kHeads + head) * kRope;
    float accumulator[2] = {0.0F, 0.0F};
    const int total = kSinks + cache_count;
    for (int tile_start = 0; tile_start < total; tile_start += kTile) {
        const int tile_count = min(kTile, total - tile_start);
        for (int local_key = warp; local_key < tile_count; local_key += 8) {
            const int key_index = tile_start + local_key;
            const bool sink = key_index < kSinks;
            const int logical_index = key_index - kSinks;
            const int cache_index = selected_indices != nullptr
                ? selected_indices[logical_index]
                : (cache_capacity > 0 ? (cache_start + logical_index) % cache_capacity
                                      : cache_start + logical_index);
            const BFloat16* key_latent = sink
                ? sink_kv + static_cast<std::size_t>(key_index) * kLatent
                : cache_kv + static_cast<std::size_t>(cache_index) * kLatent;
            const BFloat16* key_position = sink
                ? sink_rope + static_cast<std::size_t>(key_index) * kRope
                : cache_rope + static_cast<std::size_t>(cache_index) * kRope;
            float dot = 0.0F;
            for (int dimension = lane; dimension < kLatent; dimension += 32) {
                dot = fmaf(to_float(query_latent[dimension]),
                           to_float(key_latent[dimension]), dot);
            }
            for (int dimension = lane; dimension < kRope; dimension += 32) {
                dot = fmaf(to_float(query_position[dimension]),
                           to_float(key_position[dimension]), dot);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(0xffffffffU, dot, offset);
            }
            if (lane == 0) scores[local_key] = dot * softmax_scale;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float tile_maximum = -FLT_MAX;
            for (int local_key = 0; local_key < tile_count; ++local_key) {
                tile_maximum = fmaxf(tile_maximum, scores[local_key]);
            }
            const float next_maximum = fmaxf(running_max, tile_maximum);
            old_factor = denominator == 0.0F ? 0.0F : expf(running_max - next_maximum);
            running_max = next_maximum;
        }
        __syncthreads();
        float weight = 0.0F;
        if (threadIdx.x < tile_count) {
            weight = expf(scores[threadIdx.x] - running_max);
            weights[threadIdx.x] = weight;
        }
        const float tile_sum = block_sum(weight, warp_sums);
        if (threadIdx.x == 0) denominator = denominator * old_factor + tile_sum;
        __syncthreads();

        #pragma unroll
        for (int output_lane = 0; output_lane < 2; ++output_lane) {
            const int feature = threadIdx.x + output_lane * blockDim.x;
            if (feature >= kLatent) continue;
            float value = accumulator[output_lane] * old_factor;
            for (int local_key = 0; local_key < tile_count; ++local_key) {
                const int key_index = tile_start + local_key;
                const bool sink = key_index < kSinks;
                const int logical_index = key_index - kSinks;
                const int cache_index = selected_indices != nullptr
                    ? selected_indices[logical_index]
                    : (cache_capacity > 0 ? (cache_start + logical_index) % cache_capacity
                                          : cache_start + logical_index);
                const BFloat16* key_latent = sink
                    ? sink_kv + static_cast<std::size_t>(key_index) * kLatent
                    : cache_kv + static_cast<std::size_t>(cache_index) * kLatent;
                value = fmaf(to_float(key_latent[feature]), weights[local_key], value);
            }
            accumulator[output_lane] = value;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int output_lane = 0; output_lane < 2; ++output_lane) {
        const int feature = threadIdx.x + output_lane * blockDim.x;
        if (feature < kLatent) {
            latent_output[(static_cast<std::size_t>(row) * kHeads + head) *
                              kLatent + feature] =
                to_bf16(accumulator[output_lane] / denominator);
        }
    }
}

[[maybe_unused]] __global__ void latent_attention_rows_tiled_kernel(const BFloat16* q_absorbed,
                                                    const BFloat16* q_rope,
                                                    const BFloat16* sink_kv,
                                                    const BFloat16* sink_rope,
                                                    const BFloat16* cache_kv,
                                                    const BFloat16* cache_rope,
                                                    int start_position,
                                                    int batch_rows,
                                                    BFloat16* latent_output,
                                                    float softmax_scale) {
    constexpr int kTile = 64;
    constexpr int kMaximumRows = 4;
    const int head = blockIdx.x;
    if (head >= kHeads || batch_rows > kMaximumRows) return;
    __shared__ BFloat16 key_latent[kTile][kLatent];
    __shared__ BFloat16 key_rope[kTile][kRope];
    __shared__ BFloat16 query[kMaximumRows][kLatent + kRope];
    __shared__ float scores[kTile];
    __shared__ float weights[kTile];
    __shared__ float warp_sums[8];
    __shared__ float old_factor[kMaximumRows];
    __shared__ float denominator[kMaximumRows];
    __shared__ float running_max[kMaximumRows];

    for (int index = threadIdx.x;
         index < batch_rows * (kLatent + kRope);
         index += blockDim.x) {
        const int row = index / (kLatent + kRope);
        const int feature = index - row * (kLatent + kRope);
        if (feature < kLatent) {
            query[row][feature] = q_absorbed[
                (static_cast<std::size_t>(row) * kHeads + head) * kLatent + feature];
        } else {
            query[row][feature] = q_rope[
                (static_cast<std::size_t>(row) * kHeads + head) * kRope +
                feature - kLatent];
        }
    }
    if (threadIdx.x < batch_rows) {
        denominator[threadIdx.x] = 0.0F;
        running_max[threadIdx.x] = -FLT_MAX;
    }
    __syncthreads();

    float accumulator[kMaximumRows][2] {};
    const int maximum_total = kSinks + start_position + batch_rows;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int tile_start = 0; tile_start < maximum_total; tile_start += kTile) {
        const int tile_count = min(kTile, maximum_total - tile_start);
        for (int index = threadIdx.x;
             index < tile_count * (kLatent + kRope);
             index += blockDim.x) {
            const int local_key = index / (kLatent + kRope);
            const int feature = index - local_key * (kLatent + kRope);
            const int key_index = tile_start + local_key;
            const bool sink = key_index < kSinks;
            const int cache_index = key_index - kSinks;
            if (feature < kLatent) {
                key_latent[local_key][feature] = sink
                    ? sink_kv[static_cast<std::size_t>(key_index) * kLatent + feature]
                    : cache_kv[static_cast<std::size_t>(cache_index) * kLatent + feature];
            } else {
                const int rope_feature = feature - kLatent;
                key_rope[local_key][rope_feature] = sink
                    ? sink_rope[static_cast<std::size_t>(key_index) * kRope + rope_feature]
                    : cache_rope[static_cast<std::size_t>(cache_index) * kRope + rope_feature];
            }
        }
        __syncthreads();

        for (int row = 0; row < batch_rows; ++row) {
            const int row_total = kSinks + start_position + row + 1;
            const int row_tile_count = max(0, min(tile_count, row_total - tile_start));
            for (int local_key = warp; local_key < row_tile_count; local_key += 8) {
                float dot = 0.0F;
                for (int feature = lane; feature < kLatent; feature += 32) {
                    dot = fmaf(to_float(query[row][feature]),
                               to_float(key_latent[local_key][feature]), dot);
                }
                for (int feature = lane; feature < kRope; feature += 32) {
                    dot = fmaf(to_float(query[row][kLatent + feature]),
                               to_float(key_rope[local_key][feature]), dot);
                }
                for (int offset = 16; offset > 0; offset >>= 1) {
                    dot += __shfl_down_sync(0xffffffffU, dot, offset);
                }
                if (lane == 0) scores[local_key] = dot * softmax_scale;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                float tile_maximum = -FLT_MAX;
                for (int local_key = 0; local_key < row_tile_count; ++local_key) {
                    tile_maximum = fmaxf(tile_maximum, scores[local_key]);
                }
                const float next_maximum = fmaxf(running_max[row], tile_maximum);
                old_factor[row] = denominator[row] == 0.0F
                    ? 0.0F : expf(running_max[row] - next_maximum);
                running_max[row] = next_maximum;
            }
            __syncthreads();
            float weight = 0.0F;
            if (threadIdx.x < row_tile_count) {
                weight = expf(scores[threadIdx.x] - running_max[row]);
                weights[threadIdx.x] = weight;
            }
            const float tile_sum = block_sum(weight, warp_sums);
            if (threadIdx.x == 0) {
                denominator[row] = denominator[row] * old_factor[row] + tile_sum;
            }
            __syncthreads();
            #pragma unroll
            for (int output_lane = 0; output_lane < 2; ++output_lane) {
                const int feature = threadIdx.x + output_lane * blockDim.x;
                if (feature >= kLatent) continue;
                float value = accumulator[row][output_lane] * old_factor[row];
                for (int local_key = 0; local_key < row_tile_count; ++local_key) {
                    value = fmaf(to_float(key_latent[local_key][feature]),
                                 weights[local_key], value);
                }
                accumulator[row][output_lane] = value;
            }
            __syncthreads();
        }
    }
    for (int row = 0; row < batch_rows; ++row) {
        #pragma unroll
        for (int output_lane = 0; output_lane < 2; ++output_lane) {
            const int feature = threadIdx.x + output_lane * blockDim.x;
            if (feature < kLatent) {
                latent_output[(static_cast<std::size_t>(row) * kHeads + head) *
                                  kLatent + feature] =
                    to_bf16(accumulator[row][output_lane] / denominator[row]);
            }
        }
    }
}

__global__ void value_up_kernel(const BFloat16* latent,
                                const BFloat16* kv_up_weight,
                                BFloat16* output) {
    const int head = blockIdx.x;
    const int value_index = threadIdx.x;
    if (head >= kHeads || value_index >= kValue) return;
    const BFloat16* input = latent + head * kLatent;
    const BFloat16* weight = kv_up_weight +
        static_cast<std::size_t>(head * kUpPerHead + kNope + value_index) * kLatent;
    float sum = 0.0F;
    #pragma unroll 4
    for (int i = 0; i < kLatent; ++i) {
        sum = fmaf(to_float(input[i]), to_float(weight[i]), sum);
    }
    output[head * kValue + value_index] = to_bf16(sum);
}

__global__ void value_up_rows_kernel(const BFloat16* latent,
                                     const BFloat16* kv_up_weight,
                                     BFloat16* output,
                                     int batch_rows) {
    __shared__ BFloat16 latent_shared[kLatent];
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int value_index = threadIdx.x;
    if (head >= kHeads || row >= batch_rows || value_index >= kValue) return;
    const BFloat16* input = latent +
        (static_cast<std::size_t>(row) * kHeads + head) * kLatent;
    #pragma unroll
    for (int index = threadIdx.x; index < kLatent; index += blockDim.x) {
        latent_shared[index] = input[index];
    }
    __syncthreads();
    const BFloat16* weight = kv_up_weight +
        static_cast<std::size_t>(head * kUpPerHead + kNope + value_index) * kLatent;
    float sum = 0.0F;
    #pragma unroll 4
    for (int i = 0; i < kLatent; ++i) {
        sum = fmaf(to_float(latent_shared[i]), to_float(weight[i]), sum);
    }
    output[(static_cast<std::size_t>(row) * kHeads + head) * kValue + value_index] =
        to_bf16(sum);
}

__global__ void gather_latent_rows_kernel(const BFloat16* latent,
                                          BFloat16* gathered,
                                          int batch_rows) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (head >= kHeads || row >= batch_rows) return;
    for (int element = threadIdx.x; element < kLatent; element += blockDim.x) {
        gathered[(static_cast<std::size_t>(head) * batch_rows + row) * kLatent + element] =
            latent[(static_cast<std::size_t>(row) * kHeads + head) * kLatent + element];
    }
}

__global__ void scatter_value_rows_kernel(const BFloat16* head_major,
                                          BFloat16* row_major,
                                          int batch_rows) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int element = threadIdx.x;
    if (head >= kHeads || row >= batch_rows || element >= kValue) return;
    row_major[(static_cast<std::size_t>(row) * kHeads + head) * kValue + element] =
        head_major[(static_cast<std::size_t>(head) * batch_rows + row) * kValue + element];
}

}  // namespace

cudaError_t bf16_mla_plan_create(Bf16MlaPlan** plan) {
    if (plan == nullptr || *plan != nullptr) return cudaErrorInvalidValue;
    Bf16MlaPlan* created = new (std::nothrow) Bf16MlaPlan;
    if (created == nullptr) return cudaErrorMemoryAllocation;
    const cublasStatus_t status = cublasCreate(&created->handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        delete created;
        return cublas_result(status);
    }
    const cublasStatus_t math_status = cublasSetMathMode(
        created->handle, CUBLAS_TENSOR_OP_MATH);
    if (math_status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(created->handle);
        delete created;
        return cublas_result(math_status);
    }
    *plan = created;
    return cudaSuccess;
}

void bf16_mla_plan_destroy(Bf16MlaPlan* plan) noexcept {
    if (plan == nullptr) return;
    if (plan->handle != nullptr) cublasDestroy(plan->handle);
    delete plan;
}

cudaError_t mla_prepare_query(const BFloat16* q_up,
                              const BFloat16* kv_up_weight,
                              BFloat16* q_absorbed,
                              BFloat16* q_rope,
                              int position,
                              float rope_theta,
                              cudaStream_t stream) {
    if (q_up == nullptr || kv_up_weight == nullptr || q_absorbed == nullptr ||
        q_rope == nullptr || position < 0 || !(rope_theta > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    absorb_query_kernel<<<dim3(kHeads, 2), 256, 0, stream>>>(q_up, kv_up_weight, q_absorbed);
    query_rope_kernel<<<kHeads, 32, 0, stream>>>(q_up, q_rope, position, rope_theta);
    return cudaGetLastError();
}

cudaError_t mla_prepare_query_rows(const BFloat16* q_up,
                                   const BFloat16* kv_up_weight,
                                   BFloat16* q_absorbed,
                                   BFloat16* q_rope,
                                   int batch_rows,
                                   int start_position,
                                   float rope_theta,
                                   cudaStream_t stream) {
    if (q_up == nullptr || kv_up_weight == nullptr || q_absorbed == nullptr ||
        q_rope == nullptr || batch_rows <= 0 || start_position < 0 ||
        !(rope_theta > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    absorb_query_rows_kernel<<<dim3(kHeads, 2, batch_rows), 256, 0, stream>>>(
        q_up, kv_up_weight, q_absorbed, batch_rows);
    query_rope_rows_kernel<<<dim3(kHeads, batch_rows), 32, 0, stream>>>(
        q_up, q_rope, batch_rows, start_position, rope_theta);
    return cudaGetLastError();
}

cudaError_t mla_prepare_query_rope_rows(const BFloat16* q_up,
                                        BFloat16* q_rope,
                                        int batch_rows,
                                        int start_position,
                                        float rope_theta,
                                        cudaStream_t stream) {
    if (q_up == nullptr || q_rope == nullptr || batch_rows <= 0 ||
        start_position < 0 || !(rope_theta > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    query_rope_rows_kernel<<<dim3(kHeads, batch_rows), 32, 0, stream>>>(
        q_up, q_rope, batch_rows, start_position, rope_theta);
    return cudaGetLastError();
}

cudaError_t mla_absorb_query_rows_tensorcore(Bf16MlaPlan* plan,
                                             const BFloat16* q_up,
                                             const BFloat16* kv_up_weight,
                                             BFloat16* gathered_input,
                                             BFloat16* head_major_output,
                                             BFloat16* q_absorbed,
                                             int batch_rows,
                                             cudaStream_t stream) {
    if (plan == nullptr || plan->handle == nullptr || q_up == nullptr ||
        kv_up_weight == nullptr || gathered_input == nullptr ||
        head_major_output == nullptr || q_absorbed == nullptr || batch_rows <= 0) {
        return cudaErrorInvalidValue;
    }
    gather_query_nope_rows_kernel<<<dim3(kHeads, batch_rows), kNope, 0, stream>>>(
        q_up, gathered_input, batch_rows);
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) return launch_status;
    cublasStatus_t status = cublasSetStream(plan->handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) return cublas_result(status);
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    status = cublasGemmStridedBatchedEx(
        plan->handle, CUBLAS_OP_N, CUBLAS_OP_N,
        kLatent, batch_rows, kNope,
        &alpha,
        kv_up_weight, CUDA_R_16BF, kLatent,
        static_cast<long long>(kUpPerHead) * kLatent,
        gathered_input, CUDA_R_16BF, kNope,
        static_cast<long long>(batch_rows) * kNope,
        &beta,
        head_major_output, CUDA_R_16BF, kLatent,
        static_cast<long long>(batch_rows) * kLatent,
        kHeads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) return cublas_result(status);
    scatter_absorbed_rows_kernel<<<dim3(kHeads, batch_rows, 2), 256, 0, stream>>>(
        head_major_output, q_absorbed, batch_rows);
    return cudaGetLastError();
}

cudaError_t mla_prepare_key_rope(const BFloat16* kv_down,
                                 BFloat16* key_rope,
                                 int position,
                                 float rope_theta,
                                 cudaStream_t stream) {
    if (kv_down == nullptr || key_rope == nullptr || position < 0 || !(rope_theta > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    key_rope_kernel<<<1, 32, 0, stream>>>(kv_down, key_rope, position, rope_theta);
    return cudaGetLastError();
}

cudaError_t mla_prepare_key_rope_rows(const BFloat16* kv_down,
                                      BFloat16* key_rope,
                                      int batch_rows,
                                      int start_position,
                                      float rope_theta,
                                      cudaStream_t stream) {
    if (kv_down == nullptr || key_rope == nullptr || batch_rows <= 0 ||
        start_position < 0 || !(rope_theta > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    key_rope_rows_kernel<<<batch_rows, 32, 0, stream>>>(
        kv_down, key_rope, batch_rows, start_position, rope_theta);
    return cudaGetLastError();
}

cudaError_t mla_store_cache_rows(BFloat16* cache_kv,
                                 BFloat16* cache_rope,
                                 const BFloat16* row_kv,
                                 const BFloat16* row_rope,
                                 int start_position,
                                 int rows,
                                 int cache_capacity,
                                 cudaStream_t stream) {
    if (cache_kv == nullptr || cache_rope == nullptr || row_kv == nullptr ||
        row_rope == nullptr || start_position < 0 || rows <= 0 ||
        cache_capacity <= 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int elements = rows * (kLatent + kRope);
    store_cache_rows_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        cache_kv, cache_rope, row_kv, row_rope,
        start_position, rows, cache_capacity);
    return cudaGetLastError();
}

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
                                    cudaStream_t stream) {
    if (live_kv == nullptr || live_rope == nullptr || snapshot_kv == nullptr ||
        snapshot_rope == nullptr || start_position < 0 || snapshot_rows <= 0 ||
        first_row < 0 || row_count <= 0 || first_row + row_count > snapshot_rows ||
        cache_capacity <= 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 256;
    const int elements = row_count * (kLatent + kRope);
    transfer_cache_rows_kernel<<<
        (elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            restore, live_kv, live_rope, snapshot_kv, snapshot_rope,
            start_position, snapshot_rows, first_row, row_count, cache_capacity);
    return cudaGetLastError();
}

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
                                 cudaStream_t stream) {
    if (q_absorbed == nullptr || q_rope == nullptr || sink_kv == nullptr ||
        sink_rope == nullptr || cache_kv == nullptr || cache_rope == nullptr ||
        latent_output == nullptr || cache_start < 0 || cache_count <= 0 ||
        !(softmax_scale > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    latent_attention_kernel<<<kHeads, 256, 0, stream>>>(
        q_absorbed, q_rope, sink_kv, sink_rope, cache_kv, cache_rope,
        cache_start, cache_count, 0, nullptr, latent_output, softmax_scale);
    return cudaGetLastError();
}

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
                                          cudaStream_t stream) {
    if (q_absorbed == nullptr || q_rope == nullptr || sink_kv == nullptr ||
        sink_rope == nullptr || cache_kv == nullptr || cache_rope == nullptr ||
        latent_output == nullptr || cache_start < 0 || cache_count <= 0 ||
        cache_capacity <= 0 || cache_start >= cache_capacity || cache_count > cache_capacity ||
        !(softmax_scale > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    latent_attention_kernel<<<kHeads, 256, 0, stream>>>(
        q_absorbed, q_rope, sink_kv, sink_rope, cache_kv, cache_rope,
        cache_start, cache_count, cache_capacity, nullptr, latent_output, softmax_scale);
    return cudaGetLastError();
}

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
                                         cudaStream_t stream) {
    if (q_absorbed == nullptr || q_rope == nullptr || sink_kv == nullptr ||
        sink_rope == nullptr || cache_kv == nullptr || cache_rope == nullptr ||
        indices == nullptr || selected_count <= 0 || latent_output == nullptr ||
        !(softmax_scale > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    latent_attention_kernel<<<kHeads, 256, 0, stream>>>(
        q_absorbed, q_rope, sink_kv, sink_rope, cache_kv, cache_rope,
        0, selected_count, 0, indices, latent_output, softmax_scale);
    return cudaGetLastError();
}

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
                                       cudaStream_t stream) {
    if (q_absorbed == nullptr || q_rope == nullptr || sink_kv == nullptr ||
        sink_rope == nullptr || cache_kv == nullptr || cache_rope == nullptr ||
        start_position < 0 || batch_rows <= 0 || latent_output == nullptr ||
        !(softmax_scale > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    for (int row = 0; row < batch_rows; ++row) {
        latent_attention_kernel<<<kHeads, 256, 0, stream>>>(
            q_absorbed + static_cast<std::size_t>(row) * kHeads * kLatent,
            q_rope + static_cast<std::size_t>(row) * kHeads * kRope,
            sink_kv, sink_rope, cache_kv, cache_rope,
            0, start_position + row + 1, 0, nullptr,
            latent_output + static_cast<std::size_t>(row) * kHeads * kLatent,
            softmax_scale);
    }
    return cudaGetLastError();
}

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
                                       cudaStream_t stream) {
    if (q_absorbed == nullptr || q_rope == nullptr || sink_kv == nullptr ||
        sink_rope == nullptr || cache_kv == nullptr || cache_rope == nullptr ||
        latent_output == nullptr || cache_start < 0 || cache_count <= 0 ||
        !(softmax_scale > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    latent_attention_tiled_kernel<<<kHeads, 256, 0, stream>>>(
        q_absorbed, q_rope, sink_kv, sink_rope, cache_kv, cache_rope,
        cache_start, cache_count, 1, 0, 0, nullptr, latent_output, softmax_scale);
    return cudaGetLastError();
}

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
                                                cudaStream_t stream) {
    if (q_absorbed == nullptr || q_rope == nullptr || sink_kv == nullptr ||
        sink_rope == nullptr || cache_kv == nullptr || cache_rope == nullptr ||
        latent_output == nullptr || cache_start < 0 || cache_count <= 0 ||
        cache_capacity <= 0 || cache_start >= cache_capacity || cache_count > cache_capacity ||
        !(softmax_scale > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    latent_attention_tiled_kernel<<<kHeads, 256, 0, stream>>>(
        q_absorbed, q_rope, sink_kv, sink_rope, cache_kv, cache_rope,
        cache_start, cache_count, 1, 0, cache_capacity, nullptr,
        latent_output, softmax_scale);
    return cudaGetLastError();
}

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
                                               cudaStream_t stream) {
    if (q_absorbed == nullptr || q_rope == nullptr || sink_kv == nullptr ||
        sink_rope == nullptr || cache_kv == nullptr || cache_rope == nullptr ||
        indices == nullptr || selected_count <= 0 || latent_output == nullptr ||
        !(softmax_scale > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    latent_attention_tiled_kernel<<<kHeads, 256, 0, stream>>>(
        q_absorbed, q_rope, sink_kv, sink_rope, cache_kv, cache_rope,
        0, selected_count, 1, 0, 0, indices, latent_output, softmax_scale);
    return cudaGetLastError();
}

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
                                             cudaStream_t stream) {
    if (q_absorbed == nullptr || q_rope == nullptr || sink_kv == nullptr ||
        sink_rope == nullptr || cache_kv == nullptr || cache_rope == nullptr ||
        start_position < 0 || batch_rows <= 0 || latent_output == nullptr ||
        !(softmax_scale > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    if (batch_rows <= 4) {
        latent_attention_tiled_kernel<<<dim3(kHeads, batch_rows), 256, 0, stream>>>(
            q_absorbed, q_rope, sink_kv, sink_rope, cache_kv, cache_rope,
            0, start_position + 1, batch_rows, 1, 0, nullptr,
            latent_output, softmax_scale);
        return cudaGetLastError();
    }
    for (int row = 0; row < batch_rows; ++row) {
        latent_attention_tiled_kernel<<<kHeads, 256, 0, stream>>>(
            q_absorbed + static_cast<std::size_t>(row) * kHeads * kLatent,
            q_rope + static_cast<std::size_t>(row) * kHeads * kRope,
            sink_kv, sink_rope, cache_kv, cache_rope,
            0, start_position + row + 1, 1, 0, 0, nullptr,
            latent_output + static_cast<std::size_t>(row) * kHeads * kLatent,
            softmax_scale);
    }
    return cudaGetLastError();
}

cudaError_t mla_value_up(const BFloat16* latent,
                         const BFloat16* kv_up_weight,
                         BFloat16* output,
                         cudaStream_t stream) {
    if (latent == nullptr || kv_up_weight == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }
    value_up_kernel<<<kHeads, 128, 0, stream>>>(latent, kv_up_weight, output);
    return cudaGetLastError();
}

cudaError_t mla_value_up_rows(const BFloat16* latent,
                              const BFloat16* kv_up_weight,
                              BFloat16* output,
                              int batch_rows,
                              cudaStream_t stream) {
    if (latent == nullptr || kv_up_weight == nullptr || output == nullptr ||
        batch_rows <= 0) {
        return cudaErrorInvalidValue;
    }
    value_up_rows_kernel<<<dim3(kHeads, batch_rows), 128, 0, stream>>>(
        latent, kv_up_weight, output, batch_rows);
    return cudaGetLastError();
}

cudaError_t mla_value_up_rows_tensorcore(Bf16MlaPlan* plan,
                                         const BFloat16* latent,
                                         const BFloat16* kv_up_weight,
                                         BFloat16* gathered_input,
                                         BFloat16* head_major_output,
                                         BFloat16* output,
                                         int batch_rows,
                                         cudaStream_t stream) {
    if (plan == nullptr || plan->handle == nullptr || latent == nullptr ||
        kv_up_weight == nullptr || gathered_input == nullptr ||
        head_major_output == nullptr || output == nullptr || batch_rows <= 0) {
        return cudaErrorInvalidValue;
    }
    gather_latent_rows_kernel<<<dim3(kHeads, batch_rows), 256, 0, stream>>>(
        latent, gathered_input, batch_rows);
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) return launch_status;
    cublasStatus_t status = cublasSetStream(plan->handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) return cublas_result(status);
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const BFloat16* value_weight = kv_up_weight +
        static_cast<std::size_t>(kNope) * kLatent;
    status = cublasGemmStridedBatchedEx(
        plan->handle, CUBLAS_OP_T, CUBLAS_OP_N,
        kValue, batch_rows, kLatent,
        &alpha,
        value_weight, CUDA_R_16BF, kLatent,
        static_cast<long long>(kUpPerHead) * kLatent,
        gathered_input, CUDA_R_16BF, kLatent,
        static_cast<long long>(batch_rows) * kLatent,
        &beta,
        head_major_output, CUDA_R_16BF, kValue,
        static_cast<long long>(batch_rows) * kValue,
        kHeads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) return cublas_result(status);
    scatter_value_rows_kernel<<<dim3(kHeads, batch_rows), kValue, 0, stream>>>(
        head_major_output, output, batch_rows);
    return cudaGetLastError();
}

}  // namespace p92::cuda
