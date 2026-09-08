#include "p92/nvfp4_artifact.h"

#include <exception>
#include <iostream>
#include <string_view>

namespace {

void print(const p92::Nvfp4Artifact& artifact, std::string_view name) {
    const p92::Nvfp4Record& record = artifact.find(name);
    std::cout << name << " shape=[" << record.rows << ',' << record.columns
              << "] weight_offset=" << record.weight_offset
              << " scale_offset=" << record.scale_offset
              << " weight_scale_2=" << record.weight_scale_2 << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: p92_nvfp4_artifact_inspect ARTIFACT_DIRECTORY\n";
        return 2;
    }
    try {
        p92::Nvfp4Artifact artifact;
        artifact.open(argv[1]);
        if (artifact.tensor_count() != 36528 || artifact.source_payload_bytes() != 200272648050ULL) {
            throw std::runtime_error("official NVFP4 artifact identity mismatch");
        }
        std::cout << "P92_NVFP4_ARTIFACT_OK tensors=" << artifact.tensor_count()
                  << " weight_bytes=" << artifact.weight_bytes()
                  << " scale_bytes=" << artifact.scale_bytes()
                  << " source_payload_bytes=" << artifact.source_payload_bytes() << '\n';
        print(artifact, "model.layers.0.self_attn.q_a_proj.weight");
        print(artifact, "model.layers.48.mlp.experts.255.down_proj.weight");
        print(artifact, "lm_head.weight");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_NVFP4_ARTIFACT_ERROR " << error.what() << '\n';
        return 1;
    }
}
