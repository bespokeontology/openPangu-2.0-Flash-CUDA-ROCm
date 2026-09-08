#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace p92 {

enum class DType {
    bf16,
    f16,
    f32,
    f64,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    boolean,
};

struct Tensor {
    std::string name;
    std::uint32_t shard = 0;
    DType dtype = DType::bf16;
    std::vector<std::uint64_t> shape;
    std::uint64_t offset = 0;
    std::uint64_t nbytes = 0;
};

class Catalog final {
public:
    Catalog() = default;
    ~Catalog();

    Catalog(const Catalog&) = delete;
    Catalog& operator=(const Catalog&) = delete;
    Catalog(Catalog&&) = delete;
    Catalog& operator=(Catalog&&) = delete;

    void open(const std::filesystem::path& checkpoint);
    void close() noexcept;

    const Tensor& find(std::string_view name) const;
    const Tensor& at(std::size_t index) const;
    const std::byte* data(const Tensor& tensor) const;

    std::size_t tensor_count() const noexcept { return tensors_.size(); }
    std::size_t shard_count() const noexcept { return shards_.size(); }
    std::uint64_t payload_bytes() const noexcept { return payload_bytes_; }

private:
    struct Mapping {
        void* base = nullptr;
        std::size_t bytes = 0;
        std::filesystem::path path;
    };

    std::vector<Mapping> shards_;
    std::vector<Tensor> tensors_;
    std::uint64_t payload_bytes_ = 0;
};

std::string_view dtype_name(DType dtype) noexcept;

}  // namespace p92

