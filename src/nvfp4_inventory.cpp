#include "p92/catalog.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

bool eligible(const p92::Tensor& tensor) {
    return tensor.dtype == p92::DType::bf16 && tensor.shape.size() == 2 &&
           tensor.shape[0] % 128 == 0 && tensor.shape[1] % 16 == 0 &&
           ends_with(tensor.name, ".weight") && tensor.name != "model.embed_tokens.weight";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: p92_nvfp4_inventory CHECKPOINT\n";
        return 2;
    }
    try {
        p92::Catalog catalog;
        catalog.open(argv[1]);
        std::uint64_t bf16_bytes = 0;
        std::uint64_t packed_bytes = 0;
        std::uint64_t scale_bytes = 0;
        std::size_t tensors = 0;
        for (std::size_t i = 0; i < catalog.tensor_count(); ++i) {
            const p92::Tensor& tensor = catalog.at(i);
            if (!eligible(tensor)) continue;
            ++tensors;
            bf16_bytes += tensor.nbytes;
            packed_bytes += tensor.nbytes / 4;
            scale_bytes += tensor.nbytes / 32;
        }
        if (tensors == 0 || bf16_bytes == 0) throw std::runtime_error("empty NVFP4 inventory");
        std::cout << "P92_NVFP4_INVENTORY_OK tensors=" << tensors
                  << " bf16_bytes=" << bf16_bytes
                  << " packed_bytes=" << packed_bytes
                  << " scale_bytes=" << scale_bytes
                  << " artifact_bytes=" << packed_bytes + scale_bytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_NVFP4_INVENTORY_ERROR " << error.what() << '\n';
        return 1;
    }
}
