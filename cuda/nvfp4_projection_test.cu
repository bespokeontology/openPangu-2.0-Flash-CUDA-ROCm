#include "p92/catalog.h"
#include "p92/kernels.h"
#include "p92/nvfp4_artifact.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <class T>
class DeviceBuffer final {
public:
    explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
        if (elements != 0) check(cudaMalloc(reinterpret_cast<void**>(&data_), elements * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (data_ != nullptr) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() noexcept { return data_; }
    const T* get() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return elements_ * sizeof(T); }
private:
    T* data_ = nullptr;
    std::size_t elements_ = 0;
};

__global__ void pack_weight(const __nv_bfloat16* input,
                            std::uint8_t* output,
                            std::uint8_t* scales,
                            int groups,
                            float global_scale) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    const __nv_bfloat16* block = input + static_cast<std::size_t>(group) * 16;
    float maximum = 0.0F;
    #pragma unroll
    for (int i = 0; i < 16; ++i) maximum = fmaxf(maximum, fabsf(__bfloat162float(block[i])));
    const __nv_fp8_e4m3 block_scale(maximum * (global_scale / 6.0F));
    scales[group] = block_scale.__x;
    const float decoded_scale = static_cast<float>(block_scale);
    const float multiplier = decoded_scale == 0.0F ? 0.0F : global_scale / decoded_scale;
    std::uint8_t* packed = output + static_cast<std::size_t>(group) * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float2 pair = make_float2(
            __bfloat162float(block[2 * i]) * multiplier,
            __bfloat162float(block[2 * i + 1]) * multiplier);
        packed[i] = __nv_fp4x2_e2m1(pair).__x;
    }
}

__device__ float warp_sum(float value) {
    for (int offset = 16; offset != 0; offset /= 2) value += __shfl_down_sync(0xffffffffU, value, offset);
    return value;
}

__global__ void bf16_gemv(const __nv_bfloat16* input,
                          const __nv_bfloat16* weight,
                          float* output,
                          int rows,
                          int columns) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0F;
    const __nv_bfloat16* row_weight = weight + static_cast<std::size_t>(row) * columns;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        sum = fmaf(__bfloat162float(input[column]), __bfloat162float(row_weight[column]), sum);
    }
    sum = warp_sum(sum);
    __shared__ float warps[8];
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float total = threadIdx.x < 8 ? warps[threadIdx.x] : 0.0F;
        total = warp_sum(total);
        if (threadIdx.x == 0) output[row] = total;
    }
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

struct Metrics {
    double cosine = 0.0;
    double maximum_absolute = 0.0;
};

Metrics compare(const std::vector<std::uint16_t>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("comparison size mismatch");
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    double maximum = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double a = bf16_to_float(actual[i]);
        const double b = expected[i];
        if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("non-finite projection result");
        dot += a * b;
        actual_norm += a * a;
        expected_norm += b * b;
        maximum = std::max(maximum, std::abs(a - b));
    }
    return {dot / std::sqrt(actual_norm * expected_norm), maximum};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: p92_nvfp4_projection_test CHECKPOINT [NVFP4_ARTIFACT]\n";
        return 2;
    }
    try {
        p92::Catalog catalog;
        catalog.open(argv[1]);
        const std::string tensor_name = "model.layers.0.self_attn.q_a_proj.weight";
        const p92::Tensor& tensor = catalog.find(tensor_name);
        if (tensor.dtype != p92::DType::bf16 || tensor.shape.size() != 2) {
            throw std::runtime_error("unexpected real projection contract");
        }
        const int rows = static_cast<int>(tensor.shape[0]);
        const int columns = static_cast<int>(tensor.shape[1]);
        if (rows % 128 != 0 || columns % 16 != 0) throw std::runtime_error("projection is not NVFP4 aligned");

        std::vector<std::uint16_t> input(static_cast<std::size_t>(columns));
        for (int i = 0; i < columns; ++i) {
            const float value = std::sin(static_cast<float>(i) * 0.013F) +
                                0.25F * std::cos(static_cast<float>(i) * 0.0031F);
            input[static_cast<std::size_t>(i)] = float_to_bf16(value);
        }
        std::unique_ptr<p92::Nvfp4Artifact> artifact;
        const p92::Nvfp4Record* artifact_record = nullptr;
        float weight_amax = 0.0F;
        float weight_scale_2 = 0.0F;
        float weight_global_scale = 0.0F;
        if (argc == 3) {
            artifact = std::make_unique<p92::Nvfp4Artifact>();
            artifact->open(argv[2]);
            artifact_record = &artifact->find(tensor_name);
            if (artifact_record->rows != static_cast<std::uint32_t>(rows) ||
                artifact_record->columns != static_cast<std::uint32_t>(columns)) {
                throw std::runtime_error("artifact/source projection shape mismatch");
            }
            weight_scale_2 = artifact_record->weight_scale_2;
            weight_global_scale = 1.0F / weight_scale_2;
        } else {
            const auto* source_weight = reinterpret_cast<const std::uint16_t*>(catalog.data(tensor));
            for (std::size_t i = 0; i < tensor.nbytes / sizeof(std::uint16_t); ++i) {
                weight_amax = std::max(weight_amax, std::abs(bf16_to_float(source_weight[i])));
            }
            if (!(weight_amax > 0.0F) || !std::isfinite(weight_amax)) {
                throw std::runtime_error("invalid real weight amax");
            }
            weight_scale_2 = weight_amax / (6.0F * 448.0F);
            weight_global_scale = 1.0F / weight_scale_2;
        }

        DeviceBuffer<std::uint16_t> d_input(input.size());
        DeviceBuffer<std::uint16_t> d_weight(tensor.nbytes / sizeof(std::uint16_t));
        DeviceBuffer<std::uint8_t> d_input_packed(static_cast<std::size_t>(columns) / 2);
        DeviceBuffer<std::uint8_t> d_input_scales(static_cast<std::size_t>(columns) / 16);
        DeviceBuffer<std::uint8_t> d_weight_packed(static_cast<std::size_t>(rows) * columns / 2);
        DeviceBuffer<std::uint8_t> d_weight_scales(static_cast<std::size_t>(rows) * columns / 16);
        DeviceBuffer<std::uint16_t> d_software(static_cast<std::size_t>(rows));
        DeviceBuffer<std::uint16_t> d_tensorcore(static_cast<std::size_t>(rows));
        DeviceBuffer<float> d_bf16_reference(static_cast<std::size_t>(rows));
        DeviceBuffer<float> d_alpha(1);

        check(cudaMemcpy(d_input.get(), input.data(), d_input.bytes(), cudaMemcpyHostToDevice), "copy input");
        check(cudaMemcpy(d_weight.get(), catalog.data(tensor), d_weight.bytes(), cudaMemcpyHostToDevice), "copy weight");
        const float alpha = weight_scale_2;
        check(cudaMemcpy(d_alpha.get(), &alpha, sizeof(alpha), cudaMemcpyHostToDevice), "copy alpha");

        if (artifact_record != nullptr) {
            check(cudaMemcpy(d_weight_packed.get(), artifact->weight(*artifact_record),
                             d_weight_packed.bytes(), cudaMemcpyHostToDevice),
                  "copy artifact packed weight");
            check(cudaMemcpy(d_weight_scales.get(), artifact->scales(*artifact_record),
                             d_weight_scales.bytes(), cudaMemcpyHostToDevice),
                  "copy artifact weight scales");
        } else {
            const int groups = rows * columns / 16;
            pack_weight<<<(groups + 255) / 256, 256>>>(
                reinterpret_cast<const __nv_bfloat16*>(d_weight.get()), d_weight_packed.get(),
                d_weight_scales.get(), groups, weight_global_scale);
            check(cudaGetLastError(), "pack real Huawei weight");
        }
        check(p92::cuda::nvfp4_quantize(d_input.get(), d_input_packed.get(), d_input_scales.get(),
                                            columns, 1.0F), "pack activation");
        check(p92::cuda::nvfp4_gemv(
                  d_input_packed.get(), d_weight_packed.get(), d_input_scales.get(),
                  d_weight_scales.get(), d_software.get(), rows, columns, weight_scale_2),
              "software NVFP4 projection");

        const std::size_t input_scale_swizzled_bytes =
            p92::cuda::nvfp4_swizzled_scale_bytes(1, columns / 16);
        const std::size_t weight_scale_swizzled_bytes =
            p92::cuda::nvfp4_swizzled_scale_bytes(rows, columns / 16);
        DeviceBuffer<std::uint8_t> d_input_scales_swizzled(input_scale_swizzled_bytes);
        DeviceBuffer<std::uint8_t> d_weight_scales_swizzled(weight_scale_swizzled_bytes);
        check(p92::cuda::nvfp4_swizzle_scales(
                  d_input_scales.get(), d_input_scales_swizzled.get(), 1, columns / 16),
              "swizzle activation scales");
        check(p92::cuda::nvfp4_swizzle_scales(
                  d_weight_scales.get(), d_weight_scales_swizzled.get(), rows, columns / 16),
              "swizzle weight scales");
        const int tactic = 6;
        const std::size_t workspace_bytes =
            p92::cuda::nvfp4_tensorcore_workspace_bytes(rows, columns, tactic);
        DeviceBuffer<std::uint8_t> workspace(workspace_bytes);
        check(p92::cuda::nvfp4_tensorcore_gemv(
                  d_input_packed.get(), d_weight_packed.get(), d_input_scales_swizzled.get(),
                  d_weight_scales_swizzled.get(), d_tensorcore.get(), rows, columns,
                  d_alpha.get(), workspace.get(), workspace_bytes, tactic),
              "SM121 tensor-core projection");

        bf16_gemv<<<rows, 256>>>(reinterpret_cast<const __nv_bfloat16*>(d_input.get()),
                                 reinterpret_cast<const __nv_bfloat16*>(d_weight.get()),
                                 d_bf16_reference.get(), rows, columns);
        check(cudaGetLastError(), "BF16 reference projection");
        check(cudaDeviceSynchronize(), "projection synchronize");

        std::vector<std::uint16_t> software(static_cast<std::size_t>(rows));
        std::vector<std::uint16_t> tensorcore(static_cast<std::size_t>(rows));
        std::vector<float> reference(static_cast<std::size_t>(rows));
        check(cudaMemcpy(software.data(), d_software.get(), d_software.bytes(), cudaMemcpyDeviceToHost),
              "read software output");
        check(cudaMemcpy(tensorcore.data(), d_tensorcore.get(), d_tensorcore.bytes(), cudaMemcpyDeviceToHost),
              "read tensor-core output");
        check(cudaMemcpy(reference.data(), d_bf16_reference.get(), d_bf16_reference.bytes(), cudaMemcpyDeviceToHost),
              "read BF16 output");
        std::vector<float> software_float(software.size());
        for (std::size_t i = 0; i < software.size(); ++i) software_float[i] = bf16_to_float(software[i]);
        const Metrics tensorcore_parity = compare(tensorcore, software_float);
        const Metrics quant_quality = compare(tensorcore, reference);
        if (tensorcore_parity.cosine < 0.9999 || quant_quality.cosine < 0.97) {
            throw std::runtime_error("NVFP4 projection quality gate failed");
        }
        std::cout << "P92_NVFP4_PROJECTION_OK tensor=" << tensor_name
                  << " n=" << rows << " k=" << columns
                  << " packed_bytes=" << d_weight_packed.bytes()
                  << " scale_bytes=" << d_weight_scales.bytes()
                  << " source=" << (artifact_record == nullptr ? "live_pack" : "artifact")
                  << " weight_amax=" << weight_amax
                  << " weight_scale_2=" << weight_scale_2
                  << " tc_vs_software_cos=" << tensorcore_parity.cosine
                  << " tc_vs_bf16_cos=" << quant_quality.cosine
                  << " tc_vs_bf16_max_abs=" << quant_quality.maximum_absolute
                  << " workspace_bytes=" << workspace_bytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_NVFP4_PROJECTION_ERROR " << error.what() << '\n';
        return 1;
    }
}
