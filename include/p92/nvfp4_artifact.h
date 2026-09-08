#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>

namespace p92 {

#pragma pack(push, 1)
struct Nvfp4ManifestHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t record_bytes;
    std::uint32_t tensor_count;
    std::uint32_t reserved;
    std::uint64_t weight_bytes;
    std::uint64_t scale_bytes;
    std::uint64_t source_payload_bytes;
};

struct Nvfp4Record {
    char name[96];
    std::uint32_t rows;
    std::uint32_t columns;
    std::uint64_t weight_offset;
    std::uint64_t scale_offset;
    float weight_scale_2;
    std::uint32_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(Nvfp4ManifestHeader) == 48);
static_assert(sizeof(Nvfp4Record) == 128);

class Nvfp4Artifact final {
public:
    Nvfp4Artifact() = default;
    ~Nvfp4Artifact();

    Nvfp4Artifact(const Nvfp4Artifact&) = delete;
    Nvfp4Artifact& operator=(const Nvfp4Artifact&) = delete;
    Nvfp4Artifact(Nvfp4Artifact&&) = delete;
    Nvfp4Artifact& operator=(Nvfp4Artifact&&) = delete;

    void open(const std::filesystem::path& directory);
    void close() noexcept;

    const Nvfp4Record& find(std::string_view name) const;
    const Nvfp4Record& at(std::size_t index) const;
    std::string_view name(const Nvfp4Record& record) const;
    const std::byte* weight(const Nvfp4Record& record) const;
    const std::byte* scales(const Nvfp4Record& record) const;

    std::size_t tensor_count() const noexcept;
    std::uint64_t weight_bytes() const noexcept;
    std::uint64_t scale_bytes() const noexcept;
    std::uint64_t source_payload_bytes() const noexcept;

private:
    struct Mapping {
        void* base = nullptr;
        std::size_t bytes = 0;
    };

    Mapping manifest_;
    Mapping weights_;
    Mapping scales_;
    const Nvfp4ManifestHeader* header_ = nullptr;
    const Nvfp4Record* records_ = nullptr;
};

}  // namespace p92

