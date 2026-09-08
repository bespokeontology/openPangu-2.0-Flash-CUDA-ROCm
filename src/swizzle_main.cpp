#include "p92/kernels.h"
#include "p92/nvfp4_artifact.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void write_all(int descriptor, const void* data, std::size_t bytes, std::uint64_t offset) {
    const auto* source = static_cast<const std::byte*>(data);
    std::size_t written = 0;
    while (written < bytes) {
        const ssize_t count = ::pwrite(descriptor, source + written, bytes - written,
                                       static_cast<off_t>(offset + written));
        if (count < 0) throw std::system_error(errno, std::generic_category(), "write swizzled scales");
        if (count == 0) throw std::runtime_error("zero-byte write to swizzled scale file");
        written += static_cast<std::size_t>(count);
    }
}

template <class T>
class DeviceBuffer final {
public:
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&pointer_), std::max<std::size_t>(count, 1) * sizeof(T)),
              "allocate scale conversion buffer");
    }
    ~DeviceBuffer() { if (pointer_ != nullptr) cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const noexcept { return pointer_; }
private:
    T* pointer_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: p92_nvfp4_swizzle NVFP4_ARTIFACT\n";
        return 2;
    }
    int descriptor = -1;
    std::filesystem::path partial;
    try {
        const std::filesystem::path directory = argv[1];
        const std::filesystem::path output = directory / "scales.swizzled.e4m3";
        partial = directory / "scales.swizzled.e4m3.partial";
        if (std::filesystem::exists(output) || std::filesystem::exists(partial)) {
            throw std::runtime_error("swizzled scale output already exists");
        }
        p92::Nvfp4Artifact artifact;
        artifact.open(directory);
        std::size_t maximum = 0;
        for (std::size_t i = 0; i < artifact.tensor_count(); ++i) {
            const p92::Nvfp4Record& record = artifact.at(i);
            maximum = std::max(maximum, static_cast<std::size_t>(record.rows) * record.columns / 16);
        }
        DeviceBuffer<std::uint8_t> linear(maximum);
        DeviceBuffer<std::uint8_t> swizzled(maximum);
        std::vector<std::uint8_t> host(maximum);
        descriptor = ::open(partial.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (descriptor < 0) throw std::system_error(errno, std::generic_category(), "create swizzled scales");
        if (::ftruncate(descriptor, static_cast<off_t>(artifact.scale_bytes())) != 0) {
            throw std::system_error(errno, std::generic_category(), "size swizzled scales");
        }
        const auto started = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < artifact.tensor_count(); ++i) {
            const p92::Nvfp4Record& record = artifact.at(i);
            const std::size_t bytes = static_cast<std::size_t>(record.rows) * record.columns / 16;
            check(cudaMemcpy(linear.get(), artifact.scales(record), bytes, cudaMemcpyHostToDevice),
                  "upload linear scales");
            check(p92::cuda::nvfp4_swizzle_scales(
                      linear.get(), swizzled.get(), static_cast<int>(record.rows),
                      static_cast<int>(record.columns / 16)),
                  "swizzle scale tensor");
            check(cudaMemcpy(host.data(), swizzled.get(), bytes, cudaMemcpyDeviceToHost),
                  "read swizzled scales");
            write_all(descriptor, host.data(), bytes, record.scale_offset);
            if ((i + 1) % 512 == 0 || i + 1 == artifact.tensor_count()) {
                const double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                std::cerr << "P92_SWIZZLE progress=" << i + 1 << '/' << artifact.tensor_count()
                          << " elapsed_s=" << seconds << '\n';
            }
        }
        if (::fsync(descriptor) != 0) throw std::system_error(errno, std::generic_category(), "sync swizzled scales");
        if (::close(descriptor) != 0) throw std::system_error(errno, std::generic_category(), "close swizzled scales");
        descriptor = -1;
        std::filesystem::rename(partial, output);
        std::cout << "P92_SWIZZLE_OK tensors=" << artifact.tensor_count()
                  << " bytes=" << artifact.scale_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        if (descriptor >= 0) ::close(descriptor);
        if (!partial.empty() && std::filesystem::exists(partial)) std::filesystem::remove(partial);
        std::cerr << "P92_SWIZZLE_ERROR " << error.what() << '\n';
        return 1;
    }
}
