#include "p92/model.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: p92_full_forward_test CHECKPOINT NVFP4_ARTIFACT [resident]\n";
        return 2;
    }
    try {
        const auto load_start = std::chrono::steady_clock::now();
        const bool resident = argc == 4 && std::string_view(argv[3]) == "resident";
        p92::Model model(argv[1], argv[2], 64,
                         resident ? p92::Model::WeightMode::fully_resident
                                  : p92::Model::WeightMode::selective);
        const double load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - load_start).count();
        const auto forward_start = std::chrono::steady_clock::now();
        const std::int32_t token = model.forward(148899);
        const double forward_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - forward_start).count();
        const auto second_start = std::chrono::steady_clock::now();
        const std::int32_t second = model.forward(token);
        const double second_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - second_start).count();
        std::cout << "P92_FULL_FORWARD_OK input=148899 next=" << token
                  << " second=" << second
                  << " position=" << model.position()
                  << " projection_bytes=" << model.projection_bytes()
                  << " auxiliary_bytes=" << model.auxiliary_bytes()
                  << " load_s=" << load_seconds
                  << " first_forward_s=" << forward_seconds
                  << " second_forward_s=" << second_seconds
                  << " second_tok_s=" << 1.0 / second_seconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_FULL_FORWARD_ERROR " << error.what() << '\n';
        return 1;
    }
}
