#include "p92/nvfp4_artifact.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace p92 {
namespace {

constexpr char kMagic[8] = {'P', '9', '2', 'F', 'P', '4', '1', '\0'};
constexpr std::uint32_t kVersion = 1;

std::pair<void*, std::size_t> map_read_only(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "open " + path.string());
    struct stat info {};
    if (::fstat(fd, &info) != 0 || info.st_size <= 0) {
        const int error = errno;
        ::close(fd);
        throw std::system_error(error, std::generic_category(), "fstat " + path.string());
    }
    if (static_cast<std::uintmax_t>(info.st_size) > std::numeric_limits<std::size_t>::max()) {
        ::close(fd);
        throw std::runtime_error("artifact file is too large: " + path.string());
    }
    const std::size_t bytes = static_cast<std::size_t>(info.st_size);
    void* base = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    const int error = errno;
    ::close(fd);
    if (base == MAP_FAILED) throw std::system_error(error, std::generic_category(), "mmap " + path.string());
    return {base, bytes};
}

std::string_view record_name(const Nvfp4Record& record) {
    const void* end = std::memchr(record.name, '\0', sizeof(record.name));
    if (end == nullptr) throw std::runtime_error("unterminated NVFP4 tensor name");
    return {record.name, static_cast<std::size_t>(static_cast<const char*>(end) - record.name)};
}

}  // namespace

Nvfp4Artifact::~Nvfp4Artifact() {
    close();
}

void Nvfp4Artifact::close() noexcept {
    for (Mapping* mapping : {&manifest_, &weights_, &scales_}) {
        if (mapping->base != nullptr) ::munmap(mapping->base, mapping->bytes);
        *mapping = {};
    }
    header_ = nullptr;
    records_ = nullptr;
}

void Nvfp4Artifact::open(const std::filesystem::path& directory) {
    close();
    try {
        auto mapped = map_read_only(directory / "manifest.bin");
        manifest_ = {mapped.first, mapped.second};
        mapped = map_read_only(directory / "weights.nvfp4");
        weights_ = {mapped.first, mapped.second};
        mapped = map_read_only(directory / "scales.e4m3");
        scales_ = {mapped.first, mapped.second};
        if (manifest_.bytes < sizeof(Nvfp4ManifestHeader)) throw std::runtime_error("short NVFP4 manifest");
        header_ = static_cast<const Nvfp4ManifestHeader*>(manifest_.base);
        const std::uint64_t expected_manifest = sizeof(Nvfp4ManifestHeader) +
            static_cast<std::uint64_t>(header_->tensor_count) * sizeof(Nvfp4Record);
        if (std::memcmp(header_->magic, kMagic, sizeof(kMagic)) != 0 ||
            header_->version != kVersion || header_->record_bytes != sizeof(Nvfp4Record) ||
            header_->reserved != 0 || expected_manifest != manifest_.bytes) {
            throw std::runtime_error("NVFP4 manifest contract mismatch");
        }
        if (header_->weight_bytes != weights_.bytes || header_->scale_bytes != scales_.bytes) {
            throw std::runtime_error("NVFP4 artifact file-size mismatch");
        }
        records_ = reinterpret_cast<const Nvfp4Record*>(
            static_cast<const std::byte*>(manifest_.base) + sizeof(Nvfp4ManifestHeader));
        std::string_view previous;
        for (std::uint32_t i = 0; i < header_->tensor_count; ++i) {
            const Nvfp4Record& record = records_[i];
            const std::string_view current = record_name(record);
            if (current.empty() || (i != 0 && previous >= current) || record.reserved != 0 ||
                record.rows == 0 || record.columns == 0 || record.rows % 128 != 0 ||
                record.columns % 16 != 0 || !(record.weight_scale_2 > 0.0F)) {
                throw std::runtime_error("invalid NVFP4 record: " + std::string(current));
            }
            const std::uint64_t packed_bytes =
                static_cast<std::uint64_t>(record.rows) * record.columns / 2;
            const std::uint64_t scale_bytes =
                static_cast<std::uint64_t>(record.rows) * record.columns / 16;
            if (record.weight_offset > weights_.bytes || packed_bytes > weights_.bytes - record.weight_offset ||
                record.scale_offset > scales_.bytes || scale_bytes > scales_.bytes - record.scale_offset) {
                throw std::runtime_error("NVFP4 record lies outside artifact: " + std::string(current));
            }
            previous = current;
        }
    } catch (...) {
        close();
        throw;
    }
}

const Nvfp4Record& Nvfp4Artifact::find(std::string_view key) const {
    if (records_ == nullptr) throw std::logic_error("NVFP4 artifact is not open");
    const Nvfp4Record* begin = records_;
    const Nvfp4Record* end = records_ + header_->tensor_count;
    const auto it = std::lower_bound(begin, end, key,
        [](const Nvfp4Record& record, std::string_view wanted) { return record_name(record) < wanted; });
    if (it == end || record_name(*it) != key) throw std::runtime_error("missing NVFP4 tensor: " + std::string(key));
    return *it;
}

const Nvfp4Record& Nvfp4Artifact::at(std::size_t index) const {
    if (records_ == nullptr || index >= header_->tensor_count) throw std::out_of_range("NVFP4 record index");
    return records_[index];
}

std::string_view Nvfp4Artifact::name(const Nvfp4Record& record) const { return record_name(record); }

const std::byte* Nvfp4Artifact::weight(const Nvfp4Record& record) const {
    return static_cast<const std::byte*>(weights_.base) + record.weight_offset;
}

const std::byte* Nvfp4Artifact::scales(const Nvfp4Record& record) const {
    return static_cast<const std::byte*>(scales_.base) + record.scale_offset;
}

std::size_t Nvfp4Artifact::tensor_count() const noexcept { return header_ == nullptr ? 0 : header_->tensor_count; }
std::uint64_t Nvfp4Artifact::weight_bytes() const noexcept { return header_ == nullptr ? 0 : header_->weight_bytes; }
std::uint64_t Nvfp4Artifact::scale_bytes() const noexcept { return header_ == nullptr ? 0 : header_->scale_bytes; }
std::uint64_t Nvfp4Artifact::source_payload_bytes() const noexcept {
    return header_ == nullptr ? 0 : header_->source_payload_bytes;
}

}  // namespace p92

