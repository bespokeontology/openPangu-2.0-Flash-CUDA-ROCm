#include "hf_tokenizer.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hf {
namespace {

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0x10ffff) {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        throw std::runtime_error("invalid Unicode code point");
    }
}

std::pair<std::uint32_t, std::size_t> read_utf8(std::string_view text, std::size_t offset) {
    if (offset >= text.size()) throw std::runtime_error("truncated UTF-8");
    const auto first = static_cast<unsigned char>(text[offset]);
    if (first < 0x80) return {first, 1};
    int length = 0;
    std::uint32_t value = 0;
    if ((first & 0xe0) == 0xc0) { length = 2; value = first & 0x1f; }
    else if ((first & 0xf0) == 0xe0) { length = 3; value = first & 0x0f; }
    else if ((first & 0xf8) == 0xf0) { length = 4; value = first & 0x07; }
    else throw std::runtime_error("invalid UTF-8 lead byte");
    if (offset + static_cast<std::size_t>(length) > text.size()) {
        throw std::runtime_error("truncated UTF-8 sequence");
    }
    for (int i = 1; i < length; ++i) {
        const auto byte = static_cast<unsigned char>(text[offset + i]);
        if ((byte & 0xc0) != 0x80) throw std::runtime_error("invalid UTF-8 continuation");
        value = (value << 6) | (byte & 0x3f);
    }
    return {value, static_cast<std::size_t>(length)};
}

class JsonCursor final {
public:
    JsonCursor(std::string_view text, std::size_t offset) : text_(text), position_(offset) {}

    void expect(char wanted) {
        skip_space();
        if (position_ >= text_.size() || text_[position_] != wanted) {
            throw std::runtime_error(std::string("expected JSON '") + wanted + "'");
        }
        ++position_;
    }

    bool consume(char wanted) {
        skip_space();
        if (position_ >= text_.size() || text_[position_] != wanted) return false;
        ++position_;
        return true;
    }

    std::string string() {
        skip_space();
        if (position_ >= text_.size() || text_[position_++] != '"') {
            throw std::runtime_error("expected JSON string");
        }
        std::string output;
        while (position_ < text_.size()) {
            const unsigned char byte = static_cast<unsigned char>(text_[position_++]);
            if (byte == '"') return output;
            if (byte < 0x20) throw std::runtime_error("control character in JSON string");
            if (byte != '\\') {
                output.push_back(static_cast<char>(byte));
                continue;
            }
            if (position_ >= text_.size()) throw std::runtime_error("truncated JSON escape");
            const char escaped = text_[position_++];
            switch (escaped) {
                case '"': case '\\': case '/': output.push_back(escaped); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': {
                    std::uint32_t codepoint = hex4();
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (position_ + 2 > text_.size() || text_[position_] != '\\' ||
                            text_[position_ + 1] != 'u') {
                            throw std::runtime_error("missing low JSON surrogate");
                        }
                        position_ += 2;
                        const std::uint32_t low = hex4();
                        if (low < 0xdc00 || low > 0xdfff) throw std::runtime_error("invalid low JSON surrogate");
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                    }
                    append_utf8(output, codepoint);
                    break;
                }
                default: throw std::runtime_error("unsupported JSON escape");
            }
        }
        throw std::runtime_error("unterminated JSON string");
    }

    std::uint32_t uint32() {
        skip_space();
        const char* begin = text_.data() + position_;
        const char* end = text_.data() + text_.size();
        std::uint32_t value = 0;
        const auto result = std::from_chars(begin, end, value);
        if (result.ec != std::errc()) throw std::runtime_error("expected JSON uint32");
        position_ = static_cast<std::size_t>(result.ptr - text_.data());
        return value;
    }

    void skip_value() {
        skip_space();
        if (position_ >= text_.size()) throw std::runtime_error("truncated JSON value");
        if (text_[position_] == '"') { static_cast<void>(string()); return; }
        if (text_[position_] == '{') {
            ++position_;
            if (consume('}')) return;
            for (;;) {
                static_cast<void>(string());
                expect(':');
                skip_value();
                if (consume('}')) return;
                expect(',');
            }
        }
        if (text_[position_] == '[') {
            ++position_;
            if (consume(']')) return;
            for (;;) {
                skip_value();
                if (consume(']')) return;
                expect(',');
            }
        }
        while (position_ < text_.size() && text_[position_] != ',' &&
               text_[position_] != '}' && text_[position_] != ']' &&
               text_[position_] != ' ' && text_[position_] != '\n' &&
               text_[position_] != '\r' && text_[position_] != '\t') {
            ++position_;
        }
    }

private:
    std::uint32_t hex4() {
        if (position_ + 4 > text_.size()) throw std::runtime_error("truncated JSON Unicode escape");
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[position_++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= c - '0';
            else if (c >= 'a' && c <= 'f') value |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') value |= c - 'A' + 10;
            else throw std::runtime_error("invalid JSON Unicode escape");
        }
        return value;
    }

    void skip_space() {
        while (position_ < text_.size() &&
               (text_[position_] == ' ' || text_[position_] == '\n' ||
                text_[position_] == '\r' || text_[position_] == '\t')) {
            ++position_;
        }
    }

    std::string_view text_;
    std::size_t position_ = 0;
};

std::size_t value_offset(std::string_view json, std::string_view field) {
    const std::string marker = "\"" + std::string(field) + "\"";
    const std::size_t found = json.find(marker);
    if (found == std::string_view::npos) throw std::runtime_error("tokenizer field missing: " + std::string(field));
    const std::size_t colon = json.find(':', found + marker.size());
    if (colon == std::string_view::npos) throw std::runtime_error("tokenizer field has no value");
    return colon + 1;
}

std::string pair_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key.append(left);
    key.push_back('\0');
    key.append(right);
    return key;
}

}  // namespace

class Tokenizer::Impl final {
public:
    explicit Impl(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open tokenizer: " + path.string());
        input.seekg(0, std::ios::end);
        const std::streamoff bytes = input.tellg();
        input.seekg(0, std::ios::beg);
        if (bytes <= 0) throw std::runtime_error("empty tokenizer file");
        json_.resize(static_cast<std::size_t>(bytes));
        input.read(json_.data(), bytes);
        if (!input) throw std::runtime_error("cannot read tokenizer file");
        make_byte_map();
        parse_vocab();
        parse_merges();
        parse_added();
        compile_regex();
        // Identity is REPORTED, never enforced: the same loader serves any HF
        // tokenizer.json. A caller needing an exact model identity checks these
        // numbers (Pangu: vocab 148899, merges 148643, added 701, reverse 151552).
        std::fprintf(stderr, "[tokenizer] vocab=%zu merges=%zu added=%zu reverse=%zu\n",
                     vocab_.size(), merge_ranks_.size(), added_.size(), reverse_.size());
    }

    ~Impl() {
        if (regex_ != nullptr) pcre2_code_free(regex_);
    }

    std::vector<std::int32_t> encode(std::string_view text) const {
        std::vector<std::int32_t> output;
        std::size_t offset = 0;
        while (offset < text.size()) {
            std::size_t added_position = std::string_view::npos;
            const Added* selected = nullptr;
            for (const Added& token : added_) {
                const std::size_t found = text.find(token.text, offset);
                if (found < added_position ||
                    (found == added_position && selected != nullptr && token.text.size() > selected->text.size())) {
                    added_position = found;
                    selected = &token;
                }
            }
            if (selected == nullptr) {
                encode_normal(text.substr(offset), output);
                break;
            }
            if (added_position > offset) {
                encode_normal(text.substr(offset, added_position - offset), output);
            }
            output.push_back(selected->id);
            offset = added_position + selected->text.size();
        }
        return output;
    }

    std::string decode(const std::vector<std::int32_t>& tokens) const {
        std::string output;
        std::string byte_encoded;
        const auto flush = [&]() {
            std::size_t offset = 0;
            while (offset < byte_encoded.size()) {
                const auto [codepoint, bytes] = read_utf8(byte_encoded, offset);
                const auto found = byte_decoder_.find(codepoint);
                if (found == byte_decoder_.end()) throw std::runtime_error("vocabulary token is not byte-level");
                output.push_back(static_cast<char>(found->second));
                offset += bytes;
            }
            byte_encoded.clear();
        };
        for (const std::int32_t token : tokens) {
            if (token < 0 || static_cast<std::size_t>(token) >= reverse_.size()) {
                throw std::invalid_argument("token outside tokenizer vocabulary");
            }
            if (is_added_.at(token)) {
                flush();
                output += reverse_.at(token);
            } else {
                byte_encoded += reverse_.at(token);
            }
        }
        flush();
        return output;
    }

private:
    struct Added {
        std::string text;
        std::int32_t id = -1;
    };

    void make_byte_map() {
        std::array<bool, 256> present {};
        std::vector<int> bytes;
        for (int value = 33; value <= 126; ++value) { bytes.push_back(value); present[value] = true; }
        for (int value = 161; value <= 172; ++value) { bytes.push_back(value); present[value] = true; }
        for (int value = 174; value <= 255; ++value) { bytes.push_back(value); present[value] = true; }
        int extra = 0;
        for (int value = 0; value < 256; ++value) {
            if (!present[value]) {
                bytes.push_back(value);
                ++extra;
            }
        }
        int extension = 0;
        for (const int byte : bytes) {
            const std::uint32_t codepoint = present[byte] ? static_cast<std::uint32_t>(byte)
                                                          : static_cast<std::uint32_t>(256 + extension++);
            append_utf8(byte_encoder_[byte], codepoint);
            byte_decoder_.emplace(codepoint, static_cast<std::uint8_t>(byte));
        }
        if (extra != extension) throw std::runtime_error("byte-level map construction failed");
    }

    void parse_vocab() {
        JsonCursor cursor(json_, value_offset(json_, "vocab"));
        cursor.expect('{');
        // Sized from the file, not from a constant: a padded head (Pangu 151552 rows)
        // is not the same number as another model's vocabulary, and hardcoding either
        // one rejects the other model at load.
        reverse_.clear();
        is_added_.clear();
        if (!cursor.consume('}')) {
            for (;;) {
                std::string token = cursor.string();
                cursor.expect(':');
                const std::uint32_t id = cursor.uint32();
                if (id >= reverse_.size()) { reverse_.resize(static_cast<std::size_t>(id) + 1); is_added_.resize(static_cast<std::size_t>(id) + 1, false); }
                vocab_.emplace(token, static_cast<std::int32_t>(id));
                reverse_[id] = std::move(token);
                if (cursor.consume('}')) break;
                cursor.expect(',');
            }
        }
    }

    void parse_merges() {
        JsonCursor cursor(json_, value_offset(json_, "merges"));
        cursor.expect('[');
        std::uint32_t rank = 0;
        if (!cursor.consume(']')) {
            for (;;) {
                // Two encodings are in the wild for the same information:
                //   [["left","right"], ...]   (Pangu)
                //   ["left right", ...]       (Qwen, newer HF tokenizers)
                // A byte-level BPE token never contains a literal space, so the
                // flat form splits safely on the first one.
                std::string left, right;
                if (cursor.consume('[')) {
                    left = cursor.string();
                    cursor.expect(',');
                    right = cursor.string();
                    cursor.expect(']');
                } else {
                    const std::string merged = cursor.string();
                    const std::size_t sp = merged.find(' ');
                    if (sp == std::string::npos) throw std::runtime_error("malformed merge entry");
                    left = merged.substr(0, sp);
                    right = merged.substr(sp + 1);
                }
                merge_ranks_.emplace(pair_key(left, right), rank++);
                if (cursor.consume(']')) break;
                cursor.expect(',');
            }
        }
    }

    void parse_added() {
        JsonCursor cursor(json_, value_offset(json_, "added_tokens"));
        cursor.expect('[');
        if (!cursor.consume(']')) {
            for (;;) {
                cursor.expect('{');
                std::string content;
                std::uint32_t id = std::numeric_limits<std::uint32_t>::max();
                if (!cursor.consume('}')) {
                    for (;;) {
                        const std::string field = cursor.string();
                        cursor.expect(':');
                        if (field == "content") content = cursor.string();
                        else if (field == "id") id = cursor.uint32();
                        else cursor.skip_value();
                        if (cursor.consume('}')) break;
                        cursor.expect(',');
                    }
                }
                if (content.empty()) throw std::runtime_error("empty added token");
                if (id >= reverse_.size()) { reverse_.resize(static_cast<std::size_t>(id) + 1); is_added_.resize(static_cast<std::size_t>(id) + 1, false); }
                reverse_[id] = content;
                is_added_[id] = true;
                added_.push_back({std::move(content), static_cast<std::int32_t>(id)});
                if (cursor.consume(']')) break;
                cursor.expect(',');
            }
        }
        std::sort(added_.begin(), added_.end(), [](const Added& left, const Added& right) {
            return left.text.size() > right.text.size();
        });
    }

    // The pre-tokenizer pattern is read from the JSON, not compiled into this file:
    // HF tokenizer.json carries it as pre_tokenizer -> ... -> {"type":"Split",
    // "pattern":{"Regex":"..."}}. Pangu and Qwen files both use exactly that shape.
    std::string extract_regex_pattern() const {
        const std::size_t pt = json_.find("\"pre_tokenizer\"");
        if (pt == std::string::npos) return std::string();
        const std::size_t key = json_.find("\"Regex\"", pt);
        if (key == std::string::npos) return std::string();
        const std::size_t open = json_.find('"', key + 7);
        if (open == std::string::npos) return std::string();
        std::string out;
        for (std::size_t i = open + 1; i < json_.size(); ++i) {
            const char c = json_[i];
            if (c == '"') break;
            if (c != '\\') { out.push_back(c); continue; }
            if (i + 1 >= json_.size()) break;
            const char esc = json_[++i];
            switch (esc) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'u': {
                    if (i + 4 >= json_.size()) break;
                    unsigned cp = 0;
                    for (int k = 1; k <= 4; ++k) {
                        const char h = json_[i + static_cast<std::size_t>(k)];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= static_cast<unsigned>(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
                    }
                    i += 4;
                    append_utf8(out, cp);
                    break;
                }
                default: out.push_back(esc); break;
            }
        }
        return out;
    }

    void compile_regex() {
        const std::string pattern = extract_regex_pattern();
        if (pattern.empty()) throw std::runtime_error("tokenizer json declares no pre_tokenizer Regex");
        int error = 0;
        PCRE2_SIZE offset = 0;
        regex_ = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.c_str()), PCRE2_ZERO_TERMINATED,
                               PCRE2_UTF | PCRE2_UCP, &error, &offset, nullptr);
        if (regex_ == nullptr) throw std::runtime_error("cannot compile tokenizer regex");
    }

    void encode_normal(std::string_view text, std::vector<std::int32_t>& output) const {
        if (text.empty()) return;
        pcre2_match_data* match = pcre2_match_data_create_from_pattern(regex_, nullptr);
        if (match == nullptr) throw std::runtime_error("cannot allocate tokenizer regex match data");
        try {
            std::size_t cursor = 0;
            while (cursor < text.size()) {
                const int result = pcre2_match(regex_, reinterpret_cast<PCRE2_SPTR>(text.data()),
                                               text.size(), cursor, 0, match, nullptr);
                if (result == PCRE2_ERROR_NOMATCH) {
                    encode_piece(text.substr(cursor), output);
                    break;
                }
                if (result < 0) throw std::runtime_error("tokenizer regex match failed");
                PCRE2_SIZE* positions = pcre2_get_ovector_pointer(match);
                if (positions[0] > cursor) encode_piece(text.substr(cursor, positions[0] - cursor), output);
                if (positions[1] <= positions[0]) throw std::runtime_error("zero-width tokenizer match");
                encode_piece(text.substr(positions[0], positions[1] - positions[0]), output);
                cursor = positions[1];
            }
        } catch (...) {
            pcre2_match_data_free(match);
            throw;
        }
        pcre2_match_data_free(match);
    }

    void encode_piece(std::string_view piece, std::vector<std::int32_t>& output) const {
        std::string encoded;
        for (const unsigned char byte : piece) encoded += byte_encoder_[byte];
        std::vector<std::string> symbols;
        for (std::size_t offset = 0; offset < encoded.size();) {
            const auto [codepoint, bytes] = read_utf8(encoded, offset);
            static_cast<void>(codepoint);
            symbols.emplace_back(encoded.substr(offset, bytes));
            offset += bytes;
        }
        while (symbols.size() > 1) {
            std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
            std::string best_left;
            std::string best_right;
            for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
                const auto found = merge_ranks_.find(pair_key(symbols[i], symbols[i + 1]));
                if (found != merge_ranks_.end() && found->second < best_rank) {
                    best_rank = found->second;
                    best_left = symbols[i];
                    best_right = symbols[i + 1];
                }
            }
            if (best_rank == std::numeric_limits<std::uint32_t>::max()) break;
            std::vector<std::string> merged;
            for (std::size_t i = 0; i < symbols.size();) {
                if (i + 1 < symbols.size() && symbols[i] == best_left && symbols[i + 1] == best_right) {
                    merged.push_back(symbols[i] + symbols[i + 1]);
                    i += 2;
                } else {
                    merged.push_back(std::move(symbols[i]));
                    ++i;
                }
            }
            symbols = std::move(merged);
        }
        for (const std::string& symbol : symbols) {
            const auto found = vocab_.find(symbol);
            if (found == vocab_.end()) throw std::runtime_error("BPE symbol missing from vocabulary");
            output.push_back(found->second);
        }
    }

    std::string json_;
    std::unordered_map<std::string, std::int32_t> vocab_;
    std::vector<std::string> reverse_;
    std::vector<bool> is_added_;
    std::unordered_map<std::string, std::uint32_t> merge_ranks_;
    std::vector<Added> added_;
    std::array<std::string, 256> byte_encoder_;
    std::unordered_map<std::uint32_t, std::uint8_t> byte_decoder_;
    pcre2_code* regex_ = nullptr;
};

Tokenizer::Tokenizer(const std::filesystem::path& tokenizer_json)
    : impl_(std::make_unique<Impl>(tokenizer_json)) {}
Tokenizer::~Tokenizer() = default;
std::vector<std::int32_t> Tokenizer::encode(std::string_view text) const { return impl_->encode(text); }
std::string Tokenizer::decode(const std::vector<std::int32_t>& tokens) const { return impl_->decode(tokens); }
std::string Tokenizer::decode_token(std::int32_t token) const { return impl_->decode({token}); }

std::string Tokenizer::pangu_chat_prompt(std::string_view user,
                                   std::string_view system,
                                   bool thinking) {
    std::string prompt = "<|pangu_text_start|><|message_start|>system\n";
    prompt.append(system);
    prompt += "<|message_end|><|message_start|>user\n";
    prompt.append(user);
    prompt += "<|message_end|><|message_start|>assistant\n";
    prompt += thinking ? "<think>" : "</think>";
    return prompt;
}

}  // namespace hf
