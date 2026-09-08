#include "p92/resident.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: p92_resident_test NVFP4_ARTIFACT\n";
        return 2;
    }
    try {
        const auto started = std::chrono::steady_clock::now();
        p92::ResidentWeights weights;
        weights.open(argv[1]);
        const p92::ResidentProjection q_a =
            weights.find("model.layers.0.self_attn.q_a_proj.weight");
        const p92::ResidentProjection expert =
            weights.find("model.layers.48.mlp.experts.255.down_proj.weight");
        const p92::ResidentProjection head = weights.find("lm_head.weight");
        if (q_a.rows != 1024 || q_a.columns != 2560 ||
            expert.rows != 2560 || expert.columns != 1024 ||
            head.rows != 151552 || head.columns != 2560 ||
            q_a.weight == nullptr || q_a.scales_swizzled == nullptr || q_a.alpha == nullptr) {
            throw std::runtime_error("resident projection binding mismatch");
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "P92_RESIDENT_OK tensors=" << weights.tensor_count()
                  << " resident_bytes=" << weights.resident_bytes()
                  << " load_s=" << seconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_RESIDENT_ERROR " << error.what() << '\n';
        return 1;
    }
}
