#include "p92/catalog.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr char kMagic[8] = {'P', '9', '2', 'F', 'P', '4', '1', '\0'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint64_t kAlignment = 4096;

#pragma pack(push, 1)
struct ManifestHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t record_bytes;
    std::uint32_t tensor_count;
    std::uint32_t reserved;
    std::uint64_t weight_bytes;
    std::uint64_t scale_bytes;
    std::uint64_t source_payload_bytes;
};

struct ManifestRecord {
    char name[96];
    std::uint32_t rows;
    std::uint32_t columns;
    std::uint64_t weight_offset;
    std::uint64_t scale_offset;
    float weight_scale_2;
    std::uint32_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(ManifestHeader) == 48);
static_assert(sizeof(ManifestRecord) == 128);

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::uint64_t align_up(std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - (kAlignment - 1)) {
        throw std::runtime_error("artifact offset overflow");
    }
    return (value + kAlignment - 1) & ~(kAlignment - 1);
}

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

bool eligible(const p92::Tensor& tensor) {
    return tensor.dtype == p92::DType::bf16 && tensor.shape.size() == 2 &&
           tensor.shape[0] % 128 == 0 && tensor.shape[1] % 16 == 0 &&
           ends_with(tensor.name, ".weight") && tensor.name != "model.embed_tokens.weight";
}

void write_all(int fd, const void* data, std::size_t bytes, std::uint64_t offset) {
    const auto* current = static_cast<const std::byte*>(data);
    while (bytes != 0) {
        const std::size_t chunk = std::min<std::size_t>(bytes, 1U << 30);
        const ssize_t written = ::pwrite(fd, current, chunk, static_cast<off_t>(offset));
        if (written < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "pwrite NVFP4 artifact");
        }
        if (written == 0) throw std::runtime_error("short zero-byte artifact write");
        current += written;
        bytes -= static_cast<std::size_t>(written);
        offset += static_cast<std::uint64_t>(written);
    }
}

class ExclusiveFile final {
public:
    explicit ExclusiveFile(const std::filesystem::path& path) : path_(path) {
        fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "create " + path.string());
    }
    ~ExclusiveFile() { if (fd_ >= 0) ::close(fd_); }
    ExclusiveFile(const ExclusiveFile&) = delete;
    ExclusiveFile& operator=(const ExclusiveFile&) = delete;
    int get() const noexcept { return fd_; }
    void finish() {
        if (::fsync(fd_) != 0) throw std::system_error(errno, std::generic_category(), "fsync " + path_.string());
        if (::close(fd_) != 0) {
            fd_ = -1;
            throw std::system_error(errno, std::generic_category(), "close " + path_.string());
        }
        fd_ = -1;
    }
private:
    int fd_ = -1;
    std::filesystem::path path_;
};

template <class T>
class DeviceBuffer final {
public:
    explicit DeviceBuffer(std::size_t elements) : bytes_(elements * sizeof(T)) {
        if (bytes_ != 0) check(cudaMalloc(reinterpret_cast<void**>(&data_), bytes_), "cudaMalloc pack buffer");
    }
    ~DeviceBuffer() { if (data_ != nullptr) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() noexcept { return data_; }
    std::size_t bytes() const noexcept { return bytes_; }
private:
    T* data_ = nullptr;
    std::size_t bytes_ = 0;
};

__global__ void amax_bf16(const __nv_bfloat16* input, std::size_t elements, unsigned* maximum) {
    unsigned local = 0;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        local = max(local, __float_as_uint(fabsf(__bfloat162float(input[i]))));
    }
    if (local != 0) atomicMax(maximum, local);
}

__global__ void pack_weight(const __nv_bfloat16* input,
                            std::uint8_t* output,
                            std::uint8_t* scales,
                            std::size_t groups,
                            float global_scale) {
    for (std::size_t group = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         group < groups;
         group += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const __nv_bfloat16* block = input + group * 16;
        float maximum = 0.0F;
        #pragma unroll
        for (int i = 0; i < 16; ++i) maximum = fmaxf(maximum, fabsf(__bfloat162float(block[i])));
        const __nv_fp8_e4m3 scale(maximum * (global_scale / 6.0F));
        scales[group] = scale.__x;
        const float decoded_scale = static_cast<float>(scale);
        const float multiplier = decoded_scale == 0.0F ? 0.0F : global_scale / decoded_scale;
        std::uint8_t* packed = output + group * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float2 pair = make_float2(
                __bfloat162float(block[2 * i]) * multiplier,
                __bfloat162float(block[2 * i + 1]) * multiplier);
            packed[i] = __nv_fp4x2_e2m1(pair).__x;
        }
    }
}

void rename_complete(const std::filesystem::path& partial, const std::filesystem::path& final) {
    std::error_code error;
    std::filesystem::rename(partial, final, error);
    if (error) throw std::system_error(error, "rename completed NVFP4 artifact");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: p92_nvfp4_pack CHECKPOINT OUTPUT_DIRECTORY\n";
        return 2;
    }
    try {
        const std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);
        const std::filesystem::path weights_partial = output / "weights.nvfp4.partial";
        const std::filesystem::path scales_partial = output / "scales.e4m3.partial";
        const std::filesystem::path manifest_partial = output / "manifest.bin.partial";
        const std::filesystem::path weights_final = output / "weights.nvfp4";
        const std::filesystem::path scales_final = output / "scales.e4m3";
        const std::filesystem::path manifest_final = output / "manifest.bin";
        for (const auto& path : {weights_final, scales_final, manifest_final}) {
            if (std::filesystem::exists(path)) throw std::runtime_error("completed artifact already exists: " + path.string());
        }

        p92::Catalog catalog;
        catalog.open(argv[1]);
        std::vector<const p92::Tensor*> tensors;
        std::size_t maximum_source = 0;
        std::size_t maximum_packed = 0;
        std::size_t maximum_scales = 0;
        for (std::size_t i = 0; i < catalog.tensor_count(); ++i) {
            const p92::Tensor& tensor = catalog.at(i);
            if (!eligible(tensor)) continue;
            tensors.push_back(&tensor);
            maximum_source = std::max(maximum_source, static_cast<std::size_t>(tensor.nbytes));
            maximum_packed = std::max(maximum_packed, static_cast<std::size_t>(tensor.nbytes / 4));
            maximum_scales = std::max(maximum_scales, static_cast<std::size_t>(tensor.nbytes / 32));
        }
        if (tensors.empty()) throw std::runtime_error("no NVFP4-eligible tensors");

        DeviceBuffer<std::uint8_t> source(maximum_source);
        DeviceBuffer<std::uint8_t> packed(maximum_packed);
        DeviceBuffer<std::uint8_t> scales(maximum_scales);
        DeviceBuffer<unsigned> maximum(1);
        std::vector<std::uint8_t> host_packed(maximum_packed);
        std::vector<std::uint8_t> host_scales(maximum_scales);
        std::vector<ManifestRecord> records;
        records.reserve(tensors.size());
        ExclusiveFile weight_file(weights_partial);
        ExclusiveFile scale_file(scales_partial);
        std::uint64_t weight_offset = 0;
        std::uint64_t scale_offset = 0;
        std::uint64_t source_done = 0;
        const auto started = std::chrono::steady_clock::now();

        for (std::size_t index = 0; index < tensors.size(); ++index) {
            const p92::Tensor& tensor = *tensors[index];
            const std::size_t source_bytes = static_cast<std::size_t>(tensor.nbytes);
            const std::size_t packed_bytes = source_bytes / 4;
            const std::size_t scale_bytes = source_bytes / 32;
            const std::size_t elements = source_bytes / sizeof(std::uint16_t);
            const std::size_t groups = elements / 16;
            check(cudaMemcpy(source.get(), catalog.data(tensor), source_bytes, cudaMemcpyHostToDevice), "copy BF16 tensor");
            check(cudaMemset(maximum.get(), 0, sizeof(unsigned)), "clear tensor amax");
            const int reduction_blocks = static_cast<int>(std::min<std::size_t>((elements + 255) / 256, 65535));
            amax_bf16<<<reduction_blocks, 256>>>(
                reinterpret_cast<const __nv_bfloat16*>(source.get()), elements, maximum.get());
            check(cudaGetLastError(), "launch tensor amax");
            unsigned maximum_bits = 0;
            check(cudaMemcpy(&maximum_bits, maximum.get(), sizeof(maximum_bits), cudaMemcpyDeviceToHost), "read tensor amax");
            float tensor_amax = 0.0F;
            std::memcpy(&tensor_amax, &maximum_bits, sizeof(tensor_amax));
            if (!(tensor_amax > 0.0F) || !std::isfinite(tensor_amax)) {
                throw std::runtime_error("invalid amax for " + tensor.name);
            }
            const float weight_scale_2 = tensor_amax / (6.0F * 448.0F);
            const float global_scale = 1.0F / weight_scale_2;
            const int pack_blocks = static_cast<int>(std::min<std::size_t>((groups + 255) / 256, 65535));
            pack_weight<<<pack_blocks, 256>>>(
                reinterpret_cast<const __nv_bfloat16*>(source.get()), packed.get(), scales.get(), groups,
                global_scale);
            check(cudaGetLastError(), "launch NVFP4 pack");
            check(cudaMemcpy(host_packed.data(), packed.get(), packed_bytes, cudaMemcpyDeviceToHost), "read packed tensor");
            check(cudaMemcpy(host_scales.data(), scales.get(), scale_bytes, cudaMemcpyDeviceToHost), "read packed scales");

            weight_offset = align_up(weight_offset);
            scale_offset = align_up(scale_offset);
            write_all(weight_file.get(), host_packed.data(), packed_bytes, weight_offset);
            write_all(scale_file.get(), host_scales.data(), scale_bytes, scale_offset);
            if (tensor.name.size() >= sizeof(ManifestRecord::name)) {
                throw std::runtime_error("tensor name is too long: " + tensor.name);
            }
            ManifestRecord record {};
            std::memcpy(record.name, tensor.name.data(), tensor.name.size());
            record.rows = static_cast<std::uint32_t>(tensor.shape[0]);
            record.columns = static_cast<std::uint32_t>(tensor.shape[1]);
            record.weight_offset = weight_offset;
            record.scale_offset = scale_offset;
            record.weight_scale_2 = weight_scale_2;
            records.push_back(record);
            weight_offset += packed_bytes;
            scale_offset += scale_bytes;
            source_done += source_bytes;

            if ((index + 1) % 256 == 0 || index + 1 == tensors.size()) {
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                std::cerr << "P92_NVFP4_PACK progress=" << index + 1 << '/' << tensors.size()
                          << " source_gib=" << static_cast<double>(source_done) / (1ULL << 30)
                          << " elapsed_s=" << elapsed << '\n';
            }
        }
        weight_file.finish();
        scale_file.finish();

        ExclusiveFile manifest_file(manifest_partial);
        ManifestHeader header {};
        std::memcpy(header.magic, kMagic, sizeof(kMagic));
        header.version = kVersion;
        header.record_bytes = sizeof(ManifestRecord);
        header.tensor_count = static_cast<std::uint32_t>(records.size());
        header.weight_bytes = weight_offset;
        header.scale_bytes = scale_offset;
        header.source_payload_bytes = catalog.payload_bytes();
        write_all(manifest_file.get(), &header, sizeof(header), 0);
        write_all(manifest_file.get(), records.data(), records.size() * sizeof(ManifestRecord), sizeof(header));
        manifest_file.finish();

        rename_complete(weights_partial, weights_final);
        rename_complete(scales_partial, scales_final);
        rename_complete(manifest_partial, manifest_final);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "P92_NVFP4_PACK_OK tensors=" << records.size()
                  << " weight_bytes=" << weight_offset
                  << " scale_bytes=" << scale_offset
                  << " source_payload_bytes=" << catalog.payload_bytes()
                  << " elapsed_s=" << elapsed << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_NVFP4_PACK_ERROR " << error.what() << '\n';
        return 1;
    }
}
