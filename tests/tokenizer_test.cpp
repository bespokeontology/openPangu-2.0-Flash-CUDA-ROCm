#include "p92/tokenizer.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: p92_tokenizer_test TOKENIZER_JSON\n";
        return 2;
    }
    try {
        p92::Tokenizer tokenizer(argv[1]);
        const std::vector<std::pair<std::string, std::vector<std::int32_t>>> cases {
            {"Hello, world!", {14518, 11, 3532, 0}},
            {"define probability", {1972, 18527}},
            {"你好，世界！", {27968, 305, 6028, 2481}},
            {"[unused11] {\"name\":\"bash\"} [unused12]",
             {148923, 6774, 717, 2848, 33588, 7208, 220, 148924}},
        };
        for (const auto& [text, expected] : cases) {
            const std::vector<std::int32_t> actual = tokenizer.encode(text);
            if (actual != expected) throw std::runtime_error("tokenizer encode parity failure: " + text);
            if (tokenizer.decode(actual) != text) throw std::runtime_error("tokenizer round-trip failure: " + text);
        }
        const std::string chat = p92::Tokenizer::chat_prompt("Explain Kolmogorov complexity.");
        const std::vector<std::int32_t> chat_expected {
            148899, 148901, 11585, 198, 4418, 632, 280, 20058, 30982, 13,
            148902, 148901, 1344, 198, 112065, 36482, 85587, 127877, 23794, 13,
            148902, 148901, 98390, 198, 148905,
        };
        if (tokenizer.encode(chat) != chat_expected || tokenizer.decode(chat_expected) != chat) {
            throw std::runtime_error("official chat-template tokenizer parity failure");
        }
        std::cout << "P92_TOKENIZER_OK cases=" << cases.size() + 1
                  << " vocab=151552 merges=148643 added=701\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_TOKENIZER_ERROR " << error.what() << '\n';
        return 1;
    }
}
