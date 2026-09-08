#include "p92/catalog.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace p92 {
namespace {

constexpr std::size_t kShardCount = 50;

std::uint64_t read_u64_le(const std::array<unsigned char, 8>& bytes) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

std::size_t dtype_bytes(DType dtype) {
    switch (dtype) {
        case DType::bf16:
        case DType::f16:
        case DType::i16:
        case DType::u16: return 2;
        case DType::f32:
        case DType::i32:
        case DType::u32: return 4;
        case DType::f64:
        case DType::i64:
        case DType::u64: return 8;
        case DType::i8:
        case DType::u8:
        case DType::boolean: return 1;
    }
    throw std::runtime_error("invalid dtype");
}

DType parse_dtype(std::string_view text) {
    if (text == "BF16") return DType::bf16;
    if (text == "F16") return DType::f16;
    if (text == "F32") return DType::f32;
    if (text == "F64") return DType::f64;
    if (text == "I8") return DType::i8;
    if (text == "U8") return DType::u8;
    if (text == "I16") return DType::i16;
    if (text == "U16") return DType::u16;
    if (text == "I32") return DType::i32;
    if (text == "U32") return DType::u32;
    if (text == "I64") return DType::i64;
    if (text == "U64") return DType::u64;
    if (text == "BOOL") return DType::boolean;
    throw std::runtime_error("unsupported safetensors dtype: " + std::string(text));
}

class JsonCursor final {
public:
    explicit JsonCursor(std::string_view text) : current_(text.data()), end_(text.data() + text.size()) {}

    bool consume(char wanted) {
        space();
        if (current_ == end_ || *current_ != wanted) return false;
        ++current_;
        return true;
    }

    void expect(char wanted) {
        if (!consume(wanted)) {
            throw std::runtime_error(std::string("expected JSON '") + wanted + "'");
        }
    }

    std::string string() {
        space();
        if (current_ == end_ || *current_++ != '"') throw std::runtime_error("expected JSON string");
        std::string value;
        while (current_ != end_) {
            const unsigned char c = static_cast<unsigned char>(*current_++);
            if (c == '"') return value;
            if (c < 0x20) throw std::runtime_error("control byte in JSON string");
            if (c != '\\') {
                value.push_back(static_cast<char>(c));
                continue;
            }
            if (current_ == end_) throw std::runtime_error("truncated JSON escape");
            const char escaped = *current_++;
            switch (escaped) {
                case '"':
                case '\\':
                case '/': value.push_back(escaped); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                default: throw std::runtime_error("unsupported JSON string escape");
            }
        }
        throw std::runtime_error("unterminated JSON string");
    }

    std::uint64_t uint64() {
        space();
        if (current_ == end_ || *current_ < '0' || *current_ > '9') {
            throw std::runtime_error("expected unsigned JSON integer");
        }
        std::uint64_t value = 0;
        do {
            const unsigned digit = static_cast<unsigned>(*current_++ - '0');
            if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
                throw std::runtime_error("JSON integer overflow");
            }
            value = value * 10 + digit;
        } while (current_ != end_ && *current_ >= '0' && *current_ <= '9');
        return value;
    }

    std::vector<std::uint64_t> uint_array() {
        expect('[');
        std::vector<std::uint64_t> values;
        if (consume(']')) return values;
        for (;;) {
            values.push_back(uint64());
            if (consume(']')) return values;
            expect(',');
        }
    }

    void skip_value() {
        space();
        if (current_ == end_) throw std::runtime_error("truncated JSON value");
        if (*current_ == '"') {
            static_cast<void>(string());
            return;
        }
        if (*current_ == '{') {
            ++current_;
            if (consume('}')) return;
            for (;;) {
                static_cast<void>(string());
                expect(':');
                skip_value();
                if (consume('}')) return;
                expect(',');
            }
        }
        if (*current_ == '[') {
            ++current_;
            if (consume(']')) return;
            for (;;) {
                skip_value();
                if (consume(']')) return;
                expect(',');
            }
        }
        while (current_ != end_ && *current_ != ',' && *current_ != '}' && *current_ != ']' &&
               *current_ != ' ' && *current_ != '\t' && *current_ != '\n' && *current_ != '\r') {
            ++current_;
        }
    }

private:
    void space() {
        while (current_ != end_ &&
               (*current_ == ' ' || *current_ == '\t' || *current_ == '\n' || *current_ == '\r')) {
            ++current_;
        }
    }

    const char* current_;
    const char* end_;
};

std::string shard_name(std::size_t index) {
    std::array<char, 40> buffer {};
    const int count = std::snprintf(buffer.data(), buffer.size(), "model-%05zu.safetensors", index + 1);
    if (count <= 0 || static_cast<std::size_t>(count) >= buffer.size()) {
        throw std::runtime_error("cannot format shard name");
    }
    return std::string(buffer.data(), static_cast<std::size_t>(count));
}

std::vector<Tensor> parse_header(std::string_view header,
                                 std::uint32_t shard,
                                 std::uint64_t payload_base,
                                 std::uint64_t shard_bytes) {
    JsonCursor json(header);
    json.expect('{');
    std::vector<Tensor> tensors;
    if (json.consume('}')) return tensors;
    for (;;) {
        std::string name = json.string();
        json.expect(':');
        if (name == "__metadata__") {
            json.skip_value();
        } else {
            Tensor tensor;
            tensor.name = std::move(name);
            tensor.shard = shard;
            bool have_dtype = false;
            bool have_shape = false;
            bool have_offsets = false;
            std::array<std::uint64_t, 2> offsets {};
            json.expect('{');
            if (!json.consume('}')) {
                for (;;) {
                    const std::string field = json.string();
                    json.expect(':');
                    if (field == "dtype") {
                        tensor.dtype = parse_dtype(json.string());
                        have_dtype = true;
                    } else if (field == "shape") {
                        tensor.shape = json.uint_array();
                        have_shape = true;
                    } else if (field == "data_offsets") {
                        const std::vector<std::uint64_t> parsed = json.uint_array();
                        if (parsed.size() != 2 || parsed[1] < parsed[0]) {
                            throw std::runtime_error("invalid data_offsets for " + tensor.name);
                        }
                        offsets = {parsed[0], parsed[1]};
                        have_offsets = true;
                    } else {
                        json.skip_value();
                    }
                    if (json.consume('}')) break;
                    json.expect(',');
                }
            }
            if (!have_dtype || !have_shape || !have_offsets) {
                throw std::runtime_error("incomplete tensor metadata for " + tensor.name);
            }
            std::uint64_t elements = 1;
            for (const std::uint64_t dimension : tensor.shape) {
                if (dimension != 0 && elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
                    throw std::runtime_error("shape overflow for " + tensor.name);
                }
                elements *= dimension;
            }
            const std::uint64_t expected = elements * dtype_bytes(tensor.dtype);
            tensor.nbytes = offsets[1] - offsets[0];
            if (tensor.nbytes != expected) {
                throw std::runtime_error("tensor byte count mismatch for " + tensor.name);
            }
            tensor.offset = payload_base + offsets[0];
            if (tensor.offset > shard_bytes || tensor.nbytes > shard_bytes - tensor.offset) {
                throw std::runtime_error("tensor lies outside shard: " + tensor.name);
            }
            tensors.push_back(std::move(tensor));
        }
        if (json.consume('}')) break;
        json.expect(',');
    }
    return tensors;
}

}  // namespace

Catalog::~Catalog() {
    close();
}

void Catalog::close() noexcept {
    for (const Mapping& shard : shards_) {
        if (shard.base != nullptr) ::munmap(shard.base, shard.bytes);
    }
    shards_.clear();
    tensors_.clear();
    payload_bytes_ = 0;
}

void Catalog::open(const std::filesystem::path& checkpoint) {
    close();
    try {
        for (std::size_t i = 0; i < kShardCount; ++i) {
            const std::filesystem::path path = checkpoint / shard_name(i);
            const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) throw std::system_error(errno, std::generic_category(), "open " + path.string());
            struct stat info {};
            if (::fstat(fd, &info) != 0 || info.st_size <= 8) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "fstat " + path.string());
            }
            std::array<unsigned char, 8> prefix {};
            if (::pread(fd, prefix.data(), prefix.size(), 0) != static_cast<ssize_t>(prefix.size())) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "read header length " + path.string());
            }
            const std::uint64_t header_bytes = read_u64_le(prefix);
            if (header_bytes == 0 || header_bytes > static_cast<std::uint64_t>(info.st_size) - 8 ||
                header_bytes > std::numeric_limits<std::size_t>::max()) {
                ::close(fd);
                throw std::runtime_error("invalid safetensors header length: " + path.string());
            }
            std::string header(static_cast<std::size_t>(header_bytes), '\0');
            if (::pread(fd, header.data(), header.size(), 8) != static_cast<ssize_t>(header.size())) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "read header " + path.string());
            }
            void* base = ::mmap(nullptr, static_cast<std::size_t>(info.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
            const int map_error = errno;
            ::close(fd);
            if (base == MAP_FAILED) {
                throw std::system_error(map_error, std::generic_category(), "mmap " + path.string());
            }
            shards_.push_back({base, static_cast<std::size_t>(info.st_size), path});
            std::vector<Tensor> parsed = parse_header(
                header, static_cast<std::uint32_t>(i), 8 + header_bytes,
                static_cast<std::uint64_t>(info.st_size));
            for (Tensor& tensor : parsed) {
                payload_bytes_ += tensor.nbytes;
                tensors_.push_back(std::move(tensor));
            }
        }
        std::sort(tensors_.begin(), tensors_.end(),
                  [](const Tensor& a, const Tensor& b) { return a.name < b.name; });
        for (std::size_t i = 1; i < tensors_.size(); ++i) {
            if (tensors_[i - 1].name == tensors_[i].name) {
                throw std::runtime_error("duplicate tensor: " + tensors_[i].name);
            }
        }
    } catch (...) {
        close();
        throw;
    }
}

const Tensor& Catalog::find(std::string_view name) const {
    const auto it = std::lower_bound(
        tensors_.begin(), tensors_.end(), name,
        [](const Tensor& tensor, std::string_view key) { return tensor.name < key; });
    if (it == tensors_.end() || it->name != name) {
        throw std::runtime_error("missing tensor: " + std::string(name));
    }
    return *it;
}

const Tensor& Catalog::at(std::size_t index) const {
    if (index >= tensors_.size()) throw std::out_of_range("catalog tensor index");
    return tensors_[index];
}

const std::byte* Catalog::data(const Tensor& tensor) const {
    if (tensor.shard >= shards_.size()) throw std::out_of_range("catalog tensor shard");
    return static_cast<const std::byte*>(shards_[tensor.shard].base) + tensor.offset;
}

std::string_view dtype_name(DType dtype) noexcept {
    switch (dtype) {
        case DType::bf16: return "BF16";
        case DType::f16: return "F16";
        case DType::f32: return "F32";
        case DType::f64: return "F64";
        case DType::i8: return "I8";
        case DType::u8: return "U8";
        case DType::i16: return "I16";
        case DType::u16: return "U16";
        case DType::i32: return "I32";
        case DType::u32: return "U32";
        case DType::i64: return "I64";
        case DType::u64: return "U64";
        case DType::boolean: return "BOOL";
    }
    return "INVALID";
}

}  // namespace p92

