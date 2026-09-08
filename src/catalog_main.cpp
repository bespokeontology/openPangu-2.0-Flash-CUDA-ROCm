#include "p92/catalog.h"

#include <exception>
#include <iostream>
#include <string_view>

namespace {

void print_tensor(const p92::Catalog& catalog, std::string_view name) {
    const p92::Tensor& tensor = catalog.find(name);
    std::cout << name << " dtype=" << p92::dtype_name(tensor.dtype) << " shape=[";
    for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << tensor.shape[i];
    }
    std::cout << "] shard=" << tensor.shard + 1 << " bytes=" << tensor.nbytes << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: p92_catalog_inspect CHECKPOINT [TENSOR ...]\n";
        return 2;
    }
    try {
        p92::Catalog catalog;
        catalog.open(argv[1]);
        constexpr std::size_t kExpectedTensors = 37587;
        constexpr std::size_t kExpectedShards = 50;
        constexpr std::uint64_t kExpectedPayload = 200272648050ULL;
        if (catalog.tensor_count() != kExpectedTensors || catalog.shard_count() != kExpectedShards ||
            catalog.payload_bytes() != kExpectedPayload) {
            throw std::runtime_error("official checkpoint identity mismatch");
        }
        std::cout << "P92_CATALOG_OK tensors=" << catalog.tensor_count()
                  << " shards=" << catalog.shard_count()
                  << " payload_bytes=" << catalog.payload_bytes() << '\n';
        if (argc > 2) {
            for (int i = 2; i < argc; ++i) print_tensor(catalog, argv[i]);
        } else {
            print_tensor(catalog, "model.embed_tokens.weight");
            print_tensor(catalog, "model.layers.0.self_attn.q_a_proj.weight");
            print_tensor(catalog, "model.layers.2.mlp.experts.0.gate_proj.weight");
            print_tensor(catalog, "model.layers.45.self_attn.indexer.wq_b.weight");
            print_tensor(catalog, "model.layers.48.mlp.experts.255.down_proj.weight");
            print_tensor(catalog, "lm_head.weight");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_CATALOG_ERROR " << error.what() << '\n';
        return 1;
    }
}
