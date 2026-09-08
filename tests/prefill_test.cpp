#include "p92/model.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: p92_prefill_test CHECKPOINT NVFP4_ARTIFACT\n";
        return 2;
    }
    try {
        constexpr std::int32_t bos = 148899;
        std::int32_t first = -1;
        std::int32_t serial_prediction = -1;
        double serial_cold_seconds = 0.0;
        double serial_warm_seconds = 0.0;
        {
            p92::Model serial(argv[1], argv[2], 64,
                              p92::Model::WeightMode::selective);
            const auto started = std::chrono::steady_clock::now();
            first = serial.forward(bos);
            serial_prediction = serial.forward(first);
            serial_cold_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            serial.reset();
            const auto warm_started = std::chrono::steady_clock::now();
            const std::int32_t repeated_first = serial.forward(bos);
            const std::int32_t repeated_prediction = serial.forward(repeated_first);
            serial_warm_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - warm_started).count();
            if (repeated_first != first || repeated_prediction != serial_prediction) {
                throw std::runtime_error("serial target path is not deterministic after reset");
            }
        }

        const std::vector<std::int32_t> tokens {bos, first};
        p92::Model block(argv[1], argv[2], 64,
                         p92::Model::WeightMode::selective);
        const auto cold_started = std::chrono::steady_clock::now();
        const std::int32_t block_prediction = block.prefill(tokens);
        const double cold_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - cold_started).count();
        if (block.position() != tokens.size()) {
            throw std::runtime_error("prefill position did not advance by its row count");
        }
        block.reset();
        const auto warm_started = std::chrono::steady_clock::now();
        const std::int32_t repeated_prediction = block.prefill(tokens);
        const double warm_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - warm_started).count();
        if (block_prediction != repeated_prediction) {
            throw std::runtime_error("prefill block is not deterministic after reset");
        }
        if (block_prediction != serial_prediction) {
            throw std::runtime_error(
                "prefill prediction differs from serial target path: serial=" +
                std::to_string(serial_prediction) + " block=" +
                std::to_string(block_prediction));
        }
        std::cout << "P92_PREFILL_OK rows=" << tokens.size()
                  << " first=" << first
                  << " prediction=" << block_prediction
                  << " serial_cold_s=" << serial_cold_seconds
                  << " serial_warm_s=" << serial_warm_seconds
                  << " block_cold_s=" << cold_seconds
                  << " block_warm_s=" << warm_seconds
                  << " warm_speedup=" << serial_warm_seconds / warm_seconds
                  << " projection_bytes=" << block.projection_bytes()
                  << " auxiliary_bytes=" << block.auxiliary_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_PREFILL_ERROR " << error.what() << '\n';
        return 1;
    }
}
