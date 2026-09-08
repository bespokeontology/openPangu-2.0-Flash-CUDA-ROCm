#include "p92/model.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: p92_decode_profile CHECKPOINT NVFP4_ARTIFACT\n";
        return 2;
    }
    try {
        constexpr std::int32_t bos = 148899;
        p92::Model model(argv[1], argv[2], 64, p92::Model::WeightMode::selective);
        const std::int32_t first = model.forward(bos);
        const std::int32_t expected = model.forward(first);
        model.reset();
        const std::int32_t repeated_first = model.forward(bos);
        if (repeated_first != first) throw std::runtime_error("profile warmup changed first token");
        if (cudaProfilerStart() != cudaSuccess) throw std::runtime_error("cudaProfilerStart failed");
        const auto started = std::chrono::steady_clock::now();
        const std::int32_t measured = model.forward(first);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        if (cudaProfilerStop() != cudaSuccess) throw std::runtime_error("cudaProfilerStop failed");
        if (measured != expected) throw std::runtime_error("profiled token changed prediction");
        std::cout << "P92_DECODE_PROFILE_OK input=" << first
                  << " next=" << measured
                  << " wall_ms=" << seconds * 1000.0
                  << " tok_s=" << 1.0 / seconds
                  << " projection_bytes=" << model.projection_bytes()
                  << " auxiliary_bytes=" << model.auxiliary_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_DECODE_PROFILE_ERROR " << error.what() << '\n';
        return 1;
    }
}
