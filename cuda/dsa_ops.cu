#include "p92/kernels.h"

#include <cuda_bf16.h>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace p92::cuda {
namespace {

constexpr int kHeads = 24;
constexpr int kHeadSize = 128;
constexpr int kRope = 64;

__device__ __forceinline__ float to_float(BFloat16 bits) {
    __nv_bfloat16_raw raw;
    raw.x = bits;
    return __bfloat162float(static_cast<__nv_bfloat16>(raw));
}

__device__ __forceinline__ BFloat16 to_bf16(float value) {
    return static_cast<__nv_bfloat16_raw>(__float2bfloat16_rn(value)).x;
}

__global__ void dsa_rope_kernel(BFloat16* query,
                                BFloat16* key,
                                int position,
                                float theta) {
    const int pair = threadIdx.x;
    if (pair >= kRope / 2) return;
    const float frequency = powf(theta, -static_cast<float>(pair) / (kRope / 2));
    const float angle = static_cast<float>(position) * frequency;
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    for (int head = 0; head < kHeads; ++head) {
        BFloat16* row = query + head * kHeadSize;
        const float first = to_float(row[pair]);
        const float second = to_float(row[pair + kRope / 2]);
        row[pair] = to_bf16(first * cosine - second * sine);
        row[pair + kRope / 2] = to_bf16(second * cosine + first * sine);
    }
    const float first = to_float(key[pair]);
    const float second = to_float(key[pair + kRope / 2]);
    key[pair] = to_bf16(first * cosine - second * sine);
    key[pair + kRope / 2] = to_bf16(second * cosine + first * sine);
}

__global__ void dsa_rope_rows_kernel(BFloat16* query,
                                     BFloat16* key,
                                     int batch_rows,
                                     int start_position,
                                     float theta) {
    const int row_index = blockIdx.x;
    const int pair = threadIdx.x;
    if (row_index >= batch_rows || pair >= kRope / 2) return;
    const float frequency = powf(theta, -static_cast<float>(pair) / (kRope / 2));
    const float angle = static_cast<float>(start_position + row_index) * frequency;
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    BFloat16* row_query = query +
        static_cast<std::size_t>(row_index) * kHeads * kHeadSize;
    for (int head = 0; head < kHeads; ++head) {
        BFloat16* row = row_query + head * kHeadSize;
        const float first = to_float(row[pair]);
        const float second = to_float(row[pair + kRope / 2]);
        row[pair] = to_bf16(first * cosine - second * sine);
        row[pair + kRope / 2] = to_bf16(second * cosine + first * sine);
    }
    BFloat16* row_key = key + static_cast<std::size_t>(row_index) * kHeadSize;
    const float first = to_float(row_key[pair]);
    const float second = to_float(row_key[pair + kRope / 2]);
    row_key[pair] = to_bf16(first * cosine - second * sine);
    row_key[pair + kRope / 2] = to_bf16(second * cosine + first * sine);
}

__global__ void dsa_score_kernel(const BFloat16* query,
                                 const BFloat16* head_weights,
                                 const BFloat16* key_cache,
                                 int positions,
                                 float* scores,
                                 std::int32_t* indices) {
    const int position = blockIdx.x;
    if (position >= positions) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float head_scores[kHeads];
    const BFloat16* key = key_cache + static_cast<std::size_t>(position) * kHeadSize;
    for (int base = 0; base < kHeads; base += 8) {
        const int head = base + warp;
        float value = 0.0F;
        if (head < kHeads) {
            const BFloat16* row = query + head * kHeadSize;
            #pragma unroll 4
            for (int i = lane; i < kHeadSize; i += 32) {
                value = fmaf(to_float(row[i]), to_float(key[i]), value);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffffU, value, offset);
            }
            if (lane == 0) head_scores[head] = fmaxf(value, 0.0F);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float score = 0.0F;
        #pragma unroll
        for (int head = 0; head < kHeads; ++head) {
            score = fmaf(to_float(head_weights[head]), head_scores[head], score);
        }
        scores[position] = score;
        indices[position] = position;
    }
}

}  // namespace

cudaError_t dsa_rope(BFloat16* query,
                     BFloat16* key,
                     int position,
                     float rope_theta,
                     cudaStream_t stream) {
    if (query == nullptr || key == nullptr || position < 0 || !(rope_theta > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    dsa_rope_kernel<<<1, 32, 0, stream>>>(query, key, position, rope_theta);
    return cudaGetLastError();
}

cudaError_t dsa_rope_rows(BFloat16* query,
                          BFloat16* key,
                          int batch_rows,
                          int start_position,
                          float rope_theta,
                          cudaStream_t stream) {
    if (query == nullptr || key == nullptr || batch_rows <= 0 ||
        start_position < 0 || !(rope_theta > 0.0F)) {
        return cudaErrorInvalidValue;
    }
    dsa_rope_rows_kernel<<<batch_rows, 32, 0, stream>>>(
        query, key, batch_rows, start_position, rope_theta);
    return cudaGetLastError();
}

std::size_t dsa_sort_workspace_bytes(int positions) {
    if (positions <= 0) return 0;
    std::size_t bytes = 0;
    const cudaError_t status = cub::DeviceRadixSort::SortPairsDescending(
        nullptr, bytes, static_cast<float*>(nullptr), static_cast<float*>(nullptr),
        static_cast<std::int32_t*>(nullptr), static_cast<std::int32_t*>(nullptr), positions);
    return status == cudaSuccess ? bytes : 0;
}

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
                       cudaStream_t stream) {
    if (query == nullptr || head_weights == nullptr || key_cache == nullptr || positions <= 0 ||
        scores_in == nullptr || scores_out == nullptr || indices_in == nullptr ||
        indices_out == nullptr || sort_workspace == nullptr) {
        return cudaErrorInvalidValue;
    }
    std::size_t required = 0;
    cudaError_t status = cub::DeviceRadixSort::SortPairsDescending(
        nullptr, required, scores_in, scores_out, indices_in, indices_out, positions, 0, 8 * sizeof(float), stream);
    if (status != cudaSuccess) return status;
    if (sort_workspace_bytes < required) return cudaErrorInvalidValue;
    dsa_score_kernel<<<positions, 256, 0, stream>>>(
        query, head_weights, key_cache, positions, scores_in, indices_in);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    return cub::DeviceRadixSort::SortPairsDescending(
        sort_workspace, sort_workspace_bytes, scores_in, scores_out,
        indices_in, indices_out, positions, 0, 8 * sizeof(float), stream);
}

}  // namespace p92::cuda
