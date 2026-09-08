#include "p92/model.h"
#include "p92/tokenizer.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::int32_t kMessageEnd = 148902;
constexpr std::int32_t kTextEnd = 148900;

int base64_value(char value) {
    if (value >= 'A' && value <= 'Z') return value - 'A';
    if (value >= 'a' && value <= 'z') return value - 'a' + 26;
    if (value >= '0' && value <= '9') return value - '0' + 52;
    if (value == '+') return 62;
    if (value == '/') return 63;
    return -1;
}

std::string decode_base64(std::string_view encoded) {
    if (encoded.empty() || encoded.size() % 4 != 0) {
        throw std::invalid_argument("invalid base64 request payload");
    }
    std::string decoded;
    decoded.reserve(encoded.size() / 4 * 3);
    for (std::size_t offset = 0; offset < encoded.size(); offset += 4) {
        const bool final = offset + 4 == encoded.size();
        const int first = base64_value(encoded[offset]);
        const int second = base64_value(encoded[offset + 1]);
        const int third = encoded[offset + 2] == '=' ? 0 : base64_value(encoded[offset + 2]);
        const int fourth = encoded[offset + 3] == '=' ? 0 : base64_value(encoded[offset + 3]);
        if (first < 0 || second < 0 || third < 0 || fourth < 0 ||
            (!final && (encoded[offset + 2] == '=' || encoded[offset + 3] == '=')) ||
            (encoded[offset + 2] == '=' && encoded[offset + 3] != '=')) {
            throw std::invalid_argument("invalid base64 request payload");
        }
        const std::uint32_t bits = static_cast<std::uint32_t>(first << 18) |
            static_cast<std::uint32_t>(second << 12) |
            static_cast<std::uint32_t>(third << 6) |
            static_cast<std::uint32_t>(fourth);
        decoded.push_back(static_cast<char>((bits >> 16) & 0xffU));
        if (encoded[offset + 2] != '=') {
            decoded.push_back(static_cast<char>((bits >> 8) & 0xffU));
        }
        if (encoded[offset + 3] != '=') decoded.push_back(static_cast<char>(bits & 0xffU));
    }
    return decoded;
}

struct GenerateResult {
    int emitted = 0;
    int accepted = 0;
    int proposed = 0;
    bool length_limited = false;
    double seconds = 0.0;
};

GenerateResult generate(p92::Model& model,
                        const p92::Tokenizer& tokenizer,
                        std::int32_t prediction,
                        int max_new) {
    GenerateResult metrics;
    std::vector<std::int32_t> generated;
    std::string rendered;
    const auto started = std::chrono::steady_clock::now();
    while (metrics.emitted < max_new) {
        if (prediction == kMessageEnd || prediction == kTextEnd) {
            static_cast<void>(model.forward(prediction));
            break;
        }
        if (max_new - metrics.emitted >= 4) {
            const p92::Model::SpeculativeResult result = model.speculative_step(prediction);
            metrics.proposed += 3;
            metrics.accepted += static_cast<int>(result.accepted_drafts);
            bool reached_end = false;
            for (std::uint32_t index = 0; index < result.emitted_count; ++index) {
                const std::int32_t token = result.emitted[index];
                if (token == kMessageEnd || token == kTextEnd) {
                    reached_end = true;
                    break;
                }
                generated.push_back(token);
                ++metrics.emitted;
                const std::string decoded = tokenizer.decode(generated);
                if (decoded.starts_with(rendered)) {
                    std::cout << decoded.substr(rendered.size()) << std::flush;
                    rendered = decoded;
                }
            }
            prediction = result.next_token;
            if (reached_end) break;
            continue;
        }
        generated.push_back(prediction);
        ++metrics.emitted;
        const std::string decoded = tokenizer.decode(generated);
        if (decoded.starts_with(rendered)) {
            std::cout << decoded.substr(rendered.size()) << std::flush;
            rendered = decoded;
        }
        prediction = model.forward(prediction);
    }
    if (metrics.emitted == max_new) {
        metrics.length_limited = true;
        static_cast<void>(model.forward(kMessageEnd));
    }
    metrics.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return metrics;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: p92_serve_mtp CHECKPOINT NVFP4_ARTIFACT "
                     "[MAX_CONTEXT] [MAX_NEW]\n";
        return 2;
    }
    try {
        const int max_context = argc >= 4 ? std::stoi(argv[3]) : 524288;
        const int default_max_new = argc >= 5 ? std::stoi(argv[4]) : 8192;
        if (max_context <= 0 || default_max_new <= 0 || default_max_new > max_context) {
            throw std::invalid_argument("invalid serve context/output limits");
        }
        p92::Tokenizer tokenizer(std::string(argv[1]) + "/tokenizer.json");
        const auto load_started = std::chrono::steady_clock::now();
        p92::Model model(argv[1], argv[2], static_cast<std::uint32_t>(max_context),
                         p92::Model::WeightMode::fully_resident, true);
        const double load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - load_started).count();
        std::cerr << "P92_LOAD_DONE seconds=" << load_seconds
                  << " projection_bytes=" << model.projection_bytes()
                  << " auxiliary_bytes=" << model.auxiliary_bytes() << '\n';
        std::cout << "P92_READY max_context=" << max_context
                  << " max_new_tokens=" << default_max_new << '\n' << std::flush;

        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                if (line == "QUIT") break;
                if (line == "RESET") {
                    model.reset();
                    std::cout << "P92_RESET\n" << std::flush;
                    continue;
                }
                const std::size_t first_space = line.find(' ');
                const std::size_t second_space = first_space == std::string::npos
                    ? std::string::npos : line.find(' ', first_space + 1);
                if (first_space == std::string::npos || second_space == std::string::npos) {
                    throw std::invalid_argument("malformed serve request");
                }
                const std::string operation = line.substr(0, first_space);
                const int requested_max = std::stoi(
                    line.substr(first_space + 1, second_space - first_space - 1));
                if ((operation != "RUN" && operation != "APPEND") ||
                    requested_max <= 0 || requested_max > default_max_new) {
                    throw std::invalid_argument("invalid serve request operation or max-new");
                }
                const std::string prompt = decode_base64(
                    std::string_view(line).substr(second_space + 1));
                const std::vector<std::int32_t> tokens = tokenizer.encode(prompt);
                if (tokens.empty()) throw std::invalid_argument("empty serve prompt");
                if (operation == "RUN") model.reset();
                const auto prefill_started = std::chrono::steady_clock::now();
                const std::int32_t prediction = model.prefill(tokens);
                const double prefill_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - prefill_started).count();
                std::cout << "P92_BEGIN\n" << std::flush;
                const GenerateResult result = generate(
                    model, tokenizer, prediction, requested_max);
                std::cout << "\nP92_END position=" << model.position()
                          << " emitted=" << result.emitted
                          << " accepted=" << result.accepted
                          << " proposed=" << result.proposed
                          << " finish=" << (result.length_limited ? "length" : "stop")
                          << '\n' << std::flush;
                std::cerr << "P92_REQUEST prefill_tokens=" << tokens.size()
                          << " prefill_s=" << prefill_seconds
                          << " decode_s=" << result.seconds
                          << " tok_s="
                          << (result.seconds > 0.0 ? result.emitted / result.seconds : 0.0)
                          << '\n';
            } catch (const std::exception& error) {
                std::cout << "P92_ERROR " << error.what() << '\n' << std::flush;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_SERVE_ERROR " << error.what() << '\n';
        return 1;
    }
}
