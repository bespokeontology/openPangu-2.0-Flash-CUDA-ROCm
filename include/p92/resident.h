#pragma once

#include "p92/nvfp4_artifact.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>

#include <cuda_runtime_api.h>

namespace p92 {

struct ResidentProjection {
    const std::uint8_t* weight = nullptr;
    const std::uint8_t* scales_swizzled = nullptr;
    const float* alpha = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
};

class ResidentWeights final {
public:
    ResidentWeights() = default;
    ~ResidentWeights();

    ResidentWeights(const ResidentWeights&) = delete;
    ResidentWeights& operator=(const ResidentWeights&) = delete;
    ResidentWeights(ResidentWeights&&) = delete;
    ResidentWeights& operator=(ResidentWeights&&) = delete;

    void open(const std::filesystem::path& artifact_directory);
    void close() noexcept;
    ResidentProjection find(std::string_view name) const;

    std::size_t tensor_count() const noexcept { return artifact_.tensor_count(); }
    std::uint64_t resident_bytes() const noexcept { return resident_bytes_; }

private:
    Nvfp4Artifact artifact_;
    std::uint8_t* weights_ = nullptr;
    std::uint8_t* scales_ = nullptr;
    float* alphas_ = nullptr;
    std::uint64_t resident_bytes_ = 0;
};

class SelectiveWeights final {
public:
    SelectiveWeights() = default;
    ~SelectiveWeights();

    SelectiveWeights(const SelectiveWeights&) = delete;
    SelectiveWeights& operator=(const SelectiveWeights&) = delete;
    SelectiveWeights(SelectiveWeights&&) = delete;
    SelectiveWeights& operator=(SelectiveWeights&&) = delete;

    void open(const std::filesystem::path& artifact_directory);
    void close() noexcept;
    ResidentProjection load(std::string_view name);
    std::uint64_t resident_bytes() const noexcept { return resident_bytes_; }

private:
    struct Allocation {
        std::uint8_t* weight = nullptr;
        std::uint8_t* scales = nullptr;
        float* alpha = nullptr;
        std::uint32_t rows = 0;
        std::uint32_t columns = 0;
        std::uint64_t bytes = 0;
    };

    Nvfp4Artifact artifact_;
    std::unordered_map<std::string, Allocation> allocations_;
    std::uint64_t resident_bytes_ = 0;
};

}  // namespace p92
