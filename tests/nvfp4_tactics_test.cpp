#include "p92/kernels.h"
#include "p92/resident.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
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
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                         std::max<std::size_t>(count, 1) * sizeof(T)),
              "allocate tactic buffer");
    }
    ~DeviceBuffer() { if (pointer_ != nullptr) cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const noexcept { return pointer_; }
private:
    T* pointer_ = nullptr;
};

std::uint16_t to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

float to_float(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: p92_nvfp4_tactics_test NVFP4_ARTIFACT TENSOR BATCH_ROWS\n";
        return 2;
    }
    try {
        const int batch_rows = std::stoi(argv[3]);
        if (batch_rows <= 0 || batch_rows > 64) {
            throw std::invalid_argument("BATCH_ROWS must be in [1,64]");
        }
        p92::SelectiveWeights weights;
        weights.open(argv[1]);
        const p92::ResidentProjection projection = weights.load(argv[2]);
        const std::size_t input_values =
            static_cast<std::size_t>(batch_rows) * projection.columns;
        const std::size_t output_values =
            static_cast<std::size_t>(batch_rows) * projection.rows;
        std::vector<std::uint16_t> input(input_values);
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = to_bf16(
                0.17F * std::sin(static_cast<float>(index) * 0.0013F) +
                0.09F * std::cos(static_cast<float>(index) * 0.0007F));
        }
        DeviceBuffer<std::uint16_t> device_input(input_values);
        DeviceBuffer<std::uint8_t> packed(input_values / 2);
        DeviceBuffer<std::uint8_t> linear_scales(input_values / 16);
        DeviceBuffer<std::uint8_t> swizzled_scales(
            p92::cuda::nvfp4_swizzled_scale_bytes(
                batch_rows, static_cast<int>(projection.columns / 16)));
        DeviceBuffer<std::uint16_t> output(output_values);
        check(cudaMemcpy(device_input.get(), input.data(), input.size() * sizeof(std::uint16_t),
                         cudaMemcpyHostToDevice),
              "upload tactic input");
        check(p92::cuda::nvfp4_quantize(
                  device_input.get(), packed.get(), linear_scales.get(),
                  static_cast<int>(input_values), 1.0F),
              "quantize tactic input");
        check(p92::cuda::nvfp4_swizzle_scales(
                  linear_scales.get(), swizzled_scales.get(), batch_rows,
                  static_cast<int>(projection.columns / 16)),
              "swizzle tactic input scales");
        check(cudaDeviceSynchronize(), "synchronize tactic preparation");

        std::vector<std::uint16_t> reference;
        for (const int tactic : {6, 20, 21, 22}) {
            const std::size_t workspace_bytes = p92::cuda::nvfp4_tensorcore_workspace_bytes(
                batch_rows, static_cast<int>(projection.rows),
                static_cast<int>(projection.columns), tactic);
            DeviceBuffer<std::uint8_t> workspace(workspace_bytes);
            const auto launch = [&]() {
                check(p92::cuda::nvfp4_tensorcore_gemm(
                          packed.get(), projection.weight, swizzled_scales.get(),
                          projection.scales_swizzled, output.get(), batch_rows,
                          static_cast<int>(projection.rows),
                          static_cast<int>(projection.columns), projection.alpha,
                          workspace.get(), workspace_bytes, tactic),
                      "run tactic GEMM");
            };
            launch();
            check(cudaDeviceSynchronize(), "warm tactic GEMM");
            cudaEvent_t begin = nullptr;
            cudaEvent_t end = nullptr;
            check(cudaEventCreate(&begin), "create tactic begin event");
            check(cudaEventCreate(&end), "create tactic end event");
            check(cudaEventRecord(begin), "record tactic begin event");
            constexpr int kIterations = 20;
            for (int iteration = 0; iteration < kIterations; ++iteration) launch();
            check(cudaEventRecord(end), "record tactic end event");
            check(cudaEventSynchronize(end), "synchronize tactic end event");
            float total_ms = 0.0F;
            check(cudaEventElapsedTime(&total_ms, begin, end), "measure tactic");
            check(cudaEventDestroy(end), "destroy tactic end event");
            check(cudaEventDestroy(begin), "destroy tactic begin event");
            std::vector<std::uint16_t> result(output_values);
            check(cudaMemcpy(result.data(), output.get(), result.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost),
                  "read tactic result");
            double cosine = 1.0;
            double maximum = 0.0;
            if (reference.empty()) {
                reference = result;
            } else {
                double dot = 0.0;
                double left_square = 0.0;
                double right_square = 0.0;
                for (std::size_t index = 0; index < result.size(); ++index) {
                    const double left = to_float(reference[index]);
                    const double right = to_float(result[index]);
                    dot += left * right;
                    left_square += left * left;
                    right_square += right * right;
                    maximum = std::max(maximum, std::abs(left - right));
                }
                cosine = dot / std::sqrt(left_square * right_square);
                if (cosine < 0.999) throw std::runtime_error("tactic numerical gate failed");
            }
            std::cout << "P92_NVFP4_TACTIC tensor=" << argv[2]
                      << " shape=" << projection.rows << 'x' << projection.columns
                      << " m=" << batch_rows
                      << " tactic=" << tactic
                      << " ms=" << total_ms / kIterations
                      << " cosine=" << cosine
                      << " max_abs=" << maximum << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_NVFP4_TACTIC_ERROR " << error.what() << '\n';
        return 1;
    }
}
