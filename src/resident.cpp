#include "p92/resident.h"

#include "p92/kernels.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace p92 {
namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class ReadOnlyMapping final {
public:
    explicit ReadOnlyMapping(const std::filesystem::path& path) {
        const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) throw std::runtime_error("cannot open " + path.string());
        struct stat info {};
        if (::fstat(descriptor, &info) != 0 || info.st_size <= 0) {
            ::close(descriptor);
            throw std::runtime_error("cannot stat " + path.string());
        }
        bytes_ = static_cast<std::size_t>(info.st_size);
        pointer_ = ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, descriptor, 0);
        ::close(descriptor);
        if (pointer_ == MAP_FAILED) {
            pointer_ = nullptr;
            throw std::runtime_error("cannot map " + path.string());
        }
    }
    ~ReadOnlyMapping() { if (pointer_ != nullptr) ::munmap(pointer_, bytes_); }
    ReadOnlyMapping(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;
    const std::byte* data() const noexcept { return static_cast<const std::byte*>(pointer_); }
    std::size_t bytes() const noexcept { return bytes_; }
private:
    void* pointer_ = nullptr;
    std::size_t bytes_ = 0;
};

}  // namespace

ResidentWeights::~ResidentWeights() {
    close();
}

void ResidentWeights::close() noexcept {
    if (alphas_ != nullptr) cudaFree(alphas_);
    if (scales_ != nullptr) cudaFree(scales_);
    if (weights_ != nullptr) cudaFree(weights_);
    alphas_ = nullptr;
    scales_ = nullptr;
    weights_ = nullptr;
    resident_bytes_ = 0;
    artifact_.close();
}

void ResidentWeights::open(const std::filesystem::path& artifact_directory) {
    close();
    cudaStream_t stream = nullptr;
    std::uint8_t* staging = nullptr;
    try {
        artifact_.open(artifact_directory);
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create resident upload stream");
        check(cudaMalloc(reinterpret_cast<void**>(&weights_), artifact_.weight_bytes()),
              "allocate resident NVFP4 weights");
        check(cudaMalloc(reinterpret_cast<void**>(&scales_), artifact_.scale_bytes()),
              "allocate resident NVFP4 scales");
        check(cudaMalloc(reinterpret_cast<void**>(&alphas_), artifact_.tensor_count() * sizeof(float)),
              "allocate resident projection scales");

        constexpr std::size_t kCopyChunk = 1ULL << 30;
        const std::byte* source_weights = artifact_.weight(artifact_.at(0)) - artifact_.at(0).weight_offset;
        for (std::uint64_t offset = 0; offset < artifact_.weight_bytes(); offset += kCopyChunk) {
            const std::size_t bytes = static_cast<std::size_t>(
                std::min<std::uint64_t>(kCopyChunk, artifact_.weight_bytes() - offset));
            check(cudaMemcpyAsync(weights_ + offset, source_weights + offset, bytes,
                                  cudaMemcpyHostToDevice, stream),
                  "upload resident NVFP4 weights");
        }

        std::vector<float> host_alphas(artifact_.tensor_count());
        const std::filesystem::path swizzled_path = artifact_directory / "scales.swizzled.e4m3";
        const bool have_swizzled = std::filesystem::is_regular_file(swizzled_path);
        std::size_t maximum_scale_bytes = 0;
        for (std::size_t i = 0; i < artifact_.tensor_count(); ++i) {
            const Nvfp4Record& record = artifact_.at(i);
            const std::size_t linear_bytes =
                static_cast<std::size_t>(record.rows) * record.columns / 16;
            const std::size_t swizzled_bytes =
                cuda::nvfp4_swizzled_scale_bytes(record.rows, record.columns / 16);
            if (linear_bytes != swizzled_bytes) {
                throw std::runtime_error("NVFP4 scale layout changes artifact extent: " +
                                         std::string(artifact_.name(record)));
            }
            maximum_scale_bytes = std::max(maximum_scale_bytes, linear_bytes);
            host_alphas[i] = record.weight_scale_2;
        }
        if (have_swizzled) {
            ReadOnlyMapping swizzled(swizzled_path);
            if (swizzled.bytes() != artifact_.scale_bytes()) {
                throw std::runtime_error("pre-swizzled scale file size mismatch");
            }
            for (std::uint64_t offset = 0; offset < artifact_.scale_bytes(); offset += kCopyChunk) {
                const std::size_t bytes = static_cast<std::size_t>(
                    std::min<std::uint64_t>(kCopyChunk, artifact_.scale_bytes() - offset));
                check(cudaMemcpyAsync(scales_ + offset, swizzled.data() + offset, bytes,
                                      cudaMemcpyHostToDevice, stream),
                      "upload pre-swizzled NVFP4 scales");
            }
        } else {
            check(cudaMalloc(reinterpret_cast<void**>(&staging), maximum_scale_bytes),
                  "allocate NVFP4 scale staging");
            for (std::size_t i = 0; i < artifact_.tensor_count(); ++i) {
                const Nvfp4Record& record = artifact_.at(i);
                const std::size_t scale_bytes =
                    static_cast<std::size_t>(record.rows) * record.columns / 16;
                check(cudaMemcpyAsync(staging, artifact_.scales(record), scale_bytes,
                                      cudaMemcpyHostToDevice, stream),
                      "upload linear NVFP4 scales");
                check(cuda::nvfp4_swizzle_scales(
                          staging, scales_ + record.scale_offset,
                          static_cast<int>(record.rows), static_cast<int>(record.columns / 16), stream),
                      "swizzle resident NVFP4 scales");
            }
        }
        check(cudaMemcpyAsync(alphas_, host_alphas.data(), host_alphas.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream),
              "upload projection alpha values");
        check(cudaStreamSynchronize(stream), "finish resident NVFP4 upload");
        if (staging != nullptr) {
            check(cudaFree(staging), "free NVFP4 scale staging");
            staging = nullptr;
        }
        check(cudaStreamDestroy(stream), "destroy resident upload stream");
        stream = nullptr;
        resident_bytes_ = artifact_.weight_bytes() + artifact_.scale_bytes() +
                          artifact_.tensor_count() * sizeof(float);
    } catch (...) {
        if (staging != nullptr) cudaFree(staging);
        if (stream != nullptr) cudaStreamDestroy(stream);
        close();
        throw;
    }
}

ResidentProjection ResidentWeights::find(std::string_view name) const {
    const Nvfp4Record& record = artifact_.find(name);
    const std::size_t index = static_cast<std::size_t>(&record - &artifact_.at(0));
    return {
        weights_ + record.weight_offset,
        scales_ + record.scale_offset,
        alphas_ + index,
        record.rows,
        record.columns,
    };
}

SelectiveWeights::~SelectiveWeights() {
    close();
}

void SelectiveWeights::close() noexcept {
    for (auto& [name, allocation] : allocations_) {
        static_cast<void>(name);
        if (allocation.alpha != nullptr) cudaFree(allocation.alpha);
        if (allocation.scales != nullptr) cudaFree(allocation.scales);
        if (allocation.weight != nullptr) cudaFree(allocation.weight);
    }
    allocations_.clear();
    resident_bytes_ = 0;
    artifact_.close();
}

void SelectiveWeights::open(const std::filesystem::path& artifact_directory) {
    close();
    artifact_.open(artifact_directory);
}

ResidentProjection SelectiveWeights::load(std::string_view name) {
    const auto existing = allocations_.find(std::string(name));
    if (existing != allocations_.end()) {
        const Allocation& allocation = existing->second;
        return {allocation.weight, allocation.scales, allocation.alpha,
                allocation.rows, allocation.columns};
    }

    const Nvfp4Record& record = artifact_.find(name);
    Allocation allocation;
    allocation.rows = record.rows;
    allocation.columns = record.columns;
    const std::size_t weight_bytes =
        static_cast<std::size_t>(record.rows) * record.columns / 2;
    const std::size_t scale_bytes =
        static_cast<std::size_t>(record.rows) * record.columns / 16;
    std::uint8_t* linear_scales = nullptr;
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&allocation.weight), weight_bytes),
              "allocate selected NVFP4 weight");
        check(cudaMalloc(reinterpret_cast<void**>(&allocation.scales), scale_bytes),
              "allocate selected NVFP4 scales");
        check(cudaMalloc(reinterpret_cast<void**>(&allocation.alpha), sizeof(float)),
              "allocate selected NVFP4 alpha");
        check(cudaMalloc(reinterpret_cast<void**>(&linear_scales), scale_bytes),
              "allocate selected linear scales");
        check(cudaMemcpy(allocation.weight, artifact_.weight(record), weight_bytes,
                         cudaMemcpyHostToDevice),
              "upload selected NVFP4 weight");
        check(cudaMemcpy(linear_scales, artifact_.scales(record), scale_bytes,
                         cudaMemcpyHostToDevice),
              "upload selected NVFP4 scales");
        check(cudaMemcpy(allocation.alpha, &record.weight_scale_2, sizeof(float),
                         cudaMemcpyHostToDevice),
              "upload selected NVFP4 alpha");
        check(cuda::nvfp4_swizzle_scales(
                  linear_scales, allocation.scales,
                  static_cast<int>(record.rows), static_cast<int>(record.columns / 16)),
              "swizzle selected NVFP4 scales");
        check(cudaDeviceSynchronize(), "finish selected NVFP4 projection");
        check(cudaFree(linear_scales), "free selected linear scales");
        linear_scales = nullptr;
        allocation.bytes = weight_bytes + scale_bytes + sizeof(float);
    } catch (...) {
        if (linear_scales != nullptr) cudaFree(linear_scales);
        if (allocation.alpha != nullptr) cudaFree(allocation.alpha);
        if (allocation.scales != nullptr) cudaFree(allocation.scales);
        if (allocation.weight != nullptr) cudaFree(allocation.weight);
        throw;
    }
    resident_bytes_ += allocation.bytes;
    const auto [inserted, ok] = allocations_.emplace(std::string(name), std::move(allocation));
    if (!ok) throw std::runtime_error("selected projection insertion failed");
    const Allocation& stored = inserted->second;
    return {stored.weight, stored.scales, stored.alpha, stored.rows, stored.columns};
}

}  // namespace p92
