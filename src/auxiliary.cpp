#include "p92/auxiliary.h"

#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

namespace p92 {
namespace {

void check(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

AuxiliaryWeights::~AuxiliaryWeights() {
    close();
}

void AuxiliaryWeights::close() noexcept {
    for (auto& [name, allocation] : allocations_) {
        static_cast<void>(name);
        if (allocation.device != nullptr) cudaFree(allocation.device);
    }
    allocations_.clear();
    resident_bytes_ = 0;
    catalog_.close();
}

void AuxiliaryWeights::open(const std::filesystem::path& checkpoint) {
    close();
    catalog_.open(checkpoint);
}

DeviceTensor AuxiliaryWeights::load_bf16(std::string_view name) {
    const auto existing = allocations_.find(std::string(name));
    if (existing != allocations_.end()) {
        return {static_cast<const std::uint16_t*>(existing->second.device),
                existing->second.shape, existing->second.bytes};
    }
    const Tensor& tensor = catalog_.find(name);
    if (tensor.dtype != DType::bf16 || tensor.nbytes == 0) {
        throw std::runtime_error("expected nonempty BF16 tensor: " + std::string(name));
    }
    Allocation allocation;
    allocation.shape = tensor.shape;
    allocation.bytes = tensor.nbytes;
    check(cudaMalloc(&allocation.device, tensor.nbytes),
          "allocate auxiliary tensor " + std::string(name));
    try {
        check(cudaMemcpy(allocation.device, catalog_.data(tensor), tensor.nbytes,
                         cudaMemcpyHostToDevice),
              "upload auxiliary tensor " + std::string(name));
    } catch (...) {
        cudaFree(allocation.device);
        throw;
    }
    resident_bytes_ += tensor.nbytes;
    const auto [inserted, ok] = allocations_.emplace(std::string(name), std::move(allocation));
    if (!ok) throw std::runtime_error("auxiliary tensor insertion failed");
    return {static_cast<const std::uint16_t*>(inserted->second.device),
            inserted->second.shape, inserted->second.bytes};
}

DeviceFloatTensor AuxiliaryWeights::load_f32(std::string_view name) {
    const auto existing = allocations_.find(std::string(name));
    if (existing != allocations_.end()) {
        return {static_cast<const float*>(existing->second.device),
                existing->second.shape, existing->second.bytes};
    }
    const Tensor& tensor = catalog_.find(name);
    if (tensor.dtype != DType::f32 || tensor.nbytes == 0) {
        throw std::runtime_error("expected nonempty F32 tensor: " + std::string(name));
    }
    Allocation allocation;
    allocation.shape = tensor.shape;
    allocation.bytes = tensor.nbytes;
    check(cudaMalloc(&allocation.device, tensor.nbytes),
          "allocate auxiliary tensor " + std::string(name));
    try {
        check(cudaMemcpy(allocation.device, catalog_.data(tensor), tensor.nbytes,
                         cudaMemcpyHostToDevice),
              "upload auxiliary tensor " + std::string(name));
    } catch (...) {
        cudaFree(allocation.device);
        throw;
    }
    resident_bytes_ += tensor.nbytes;
    const auto [inserted, ok] = allocations_.emplace(std::string(name), std::move(allocation));
    if (!ok) throw std::runtime_error("auxiliary tensor insertion failed");
    return {static_cast<const float*>(inserted->second.device),
            inserted->second.shape, inserted->second.bytes};
}

}  // namespace p92
