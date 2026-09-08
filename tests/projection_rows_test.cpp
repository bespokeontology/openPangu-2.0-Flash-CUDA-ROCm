#include "p92/executor.h"
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

constexpr int kBatchRows = 64;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

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

template <class T>
class DeviceBuffer final {
public:
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                         std::max<std::size_t>(count, 1) * sizeof(T)),
              "allocate projection-row test buffer");
    }
    ~DeviceBuffer() { if (pointer_ != nullptr) cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const noexcept { return pointer_; }
private:
    T* pointer_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: p92_projection_rows_test NVFP4_ARTIFACT\n";
        return 2;
    }
    try {
        p92::SelectiveWeights weights;
        weights.open(argv[1]);
        const p92::ResidentProjection projection =
            weights.load("model.layers.0.self_attn.q_a_proj.weight");
        const std::size_t input_values =
            static_cast<std::size_t>(kBatchRows) * projection.columns;
        const std::size_t output_values =
            static_cast<std::size_t>(kBatchRows) * projection.rows;
        std::vector<std::uint16_t> input(input_values);
        for (int row = 0; row < kBatchRows; ++row) {
            for (std::uint32_t column = 0; column < projection.columns; ++column) {
                const float value = 0.17F * std::sin((row + 1) * (column + 3) * 0.0013F) +
                                    0.09F * std::cos((row + 7) * (column + 1) * 0.0007F);
                input[static_cast<std::size_t>(row) * projection.columns + column] =
                    to_bf16(value);
            }
        }
        DeviceBuffer<std::uint16_t> device_input(input_values);
        DeviceBuffer<std::uint16_t> block_output(output_values);
        DeviceBuffer<std::uint16_t> serial_output(output_values);
        check(cudaMemcpy(device_input.get(), input.data(), input.size() * sizeof(std::uint16_t),
                         cudaMemcpyHostToDevice),
              "upload projection rows");

        p92::ProjectionExecutor executor;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        check(cudaEventCreate(&start), "create start event");
        check(cudaEventCreate(&stop), "create stop event");

        executor.project_rows(projection, device_input.get(), block_output.get(), kBatchRows);
        executor.synchronize();
        check(cudaEventRecord(start, executor.stream()), "record block start");
        executor.project_rows(projection, device_input.get(), block_output.get(), kBatchRows);
        check(cudaEventRecord(stop, executor.stream()), "record block stop");
        executor.synchronize();
        float block_ms = 0.0F;
        check(cudaEventElapsedTime(&block_ms, start, stop), "measure block projection");

        check(cudaEventRecord(start, executor.stream()), "record serial start");
        for (int row = 0; row < kBatchRows; ++row) {
            executor.project(
                projection,
                device_input.get() + static_cast<std::size_t>(row) * projection.columns,
                serial_output.get() + static_cast<std::size_t>(row) * projection.rows);
        }
        check(cudaEventRecord(stop, executor.stream()), "record serial stop");
        executor.synchronize();
        float serial_ms = 0.0F;
        check(cudaEventElapsedTime(&serial_ms, start, stop), "measure serial projections");
        check(cudaEventDestroy(stop), "destroy stop event");
        check(cudaEventDestroy(start), "destroy start event");

        std::vector<std::uint16_t> block(output_values);
        std::vector<std::uint16_t> serial(output_values);
        check(cudaMemcpy(block.data(), block_output.get(), block.size() * sizeof(std::uint16_t),
                         cudaMemcpyDeviceToHost),
              "read block output");
        check(cudaMemcpy(serial.data(), serial_output.get(), serial.size() * sizeof(std::uint16_t),
                         cudaMemcpyDeviceToHost),
              "read serial output");
        double dot = 0.0;
        double block_norm = 0.0;
        double serial_norm = 0.0;
        double maximum = 0.0;
        for (std::size_t i = 0; i < output_values; ++i) {
            const double left = to_float(block[i]);
            const double right = to_float(serial[i]);
            if (!std::isfinite(left) || !std::isfinite(right)) {
                throw std::runtime_error("non-finite M=64 projection output");
            }
            dot += left * right;
            block_norm += left * left;
            serial_norm += right * right;
            maximum = std::max(maximum, std::abs(left - right));
        }
        const double cosine = dot / std::sqrt(block_norm * serial_norm);
        if (cosine < 0.999) throw std::runtime_error("M=64 projection cosine gate failed");
        std::cout << "P92_PROJECTION_ROWS_OK rows=" << kBatchRows
                  << " shape=" << projection.rows << 'x' << projection.columns
                  << " cosine=" << cosine
                  << " max_abs=" << maximum
                  << " block_ms=" << block_ms
                  << " serial_ms=" << serial_ms
                  << " speedup=" << serial_ms / block_ms << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_PROJECTION_ROWS_ERROR " << error.what() << '\n';
        return 1;
    }
}
