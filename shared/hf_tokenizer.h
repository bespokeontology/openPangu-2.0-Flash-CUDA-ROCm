#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace hf {

class Tokenizer final {
public:
    explicit Tokenizer(const std::filesystem::path& tokenizer_json);
    ~Tokenizer();

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) = delete;
    Tokenizer& operator=(Tokenizer&&) = delete;

    std::vector<std::int32_t> encode(std::string_view text) const;
    std::string decode(const std::vector<std::int32_t>& tokens) const;
    std::string decode_token(std::int32_t token) const;

    static std::string pangu_chat_prompt(std::string_view user,
                                   std::string_view system = "You are a helpful assistant.",
                                   bool thinking = true);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hf
