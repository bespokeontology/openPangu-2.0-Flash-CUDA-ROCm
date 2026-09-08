#pragma once

#include "p92/catalog.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace p92 {

struct DeviceTensor {
    const std::uint16_t* data = nullptr;
    std::vector<std::uint64_t> shape;
    std::uint64_t bytes = 0;
};

struct DeviceFloatTensor {
    const float* data = nullptr;
    std::vector<std::uint64_t> shape;
    std::uint64_t bytes = 0;
};

class AuxiliaryWeights final {
public:
    AuxiliaryWeights() = default;
    ~AuxiliaryWeights();

    AuxiliaryWeights(const AuxiliaryWeights&) = delete;
    AuxiliaryWeights& operator=(const AuxiliaryWeights&) = delete;
    AuxiliaryWeights(AuxiliaryWeights&&) = delete;
    AuxiliaryWeights& operator=(AuxiliaryWeights&&) = delete;

    void open(const std::filesystem::path& checkpoint);
    void close() noexcept;
    DeviceTensor load_bf16(std::string_view name);
    DeviceFloatTensor load_f32(std::string_view name);
    const Catalog& catalog() const noexcept { return catalog_; }
    std::uint64_t resident_bytes() const noexcept { return resident_bytes_; }

private:
    struct Allocation {
        void* device = nullptr;
        std::vector<std::uint64_t> shape;
        std::uint64_t bytes = 0;
    };

    Catalog catalog_;
    std::unordered_map<std::string, Allocation> allocations_;
    std::uint64_t resident_bytes_ = 0;
};

}  // namespace p92
