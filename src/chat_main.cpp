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

#ifndef P92_CHAT_MTP
#define P92_CHAT_MTP 0
#endif

constexpr bool kUseMtp = P92_CHAT_MTP != 0;

constexpr std::int32_t kMessageEnd = 148902;
constexpr std::int32_t kTextEnd = 148900;

std::vector<std::int32_t> continuation_prompt(const p92::Tokenizer& tokenizer,
                                               std::string_view user,
                                               bool thinking) {
    std::string text = "<|message_start|>user\n";
    text.append(user);
    text += "<|message_end|><|message_start|>assistant\n";
    text += thinking ? "<think>" : "</think>";
    return tokenizer.encode(text);
}

void feed(p92::Model& model, const std::vector<std::int32_t>& tokens,
          std::int32_t& prediction) {
    if (tokens.empty()) throw std::runtime_error("empty prompt");
    prediction = model.prefill(tokens);
}

void generate(p92::Model& model, const p92::Tokenizer& tokenizer,
              std::int32_t prediction, int max_new) {
    std::vector<std::int32_t> generated;
    std::string rendered;
    const auto started = std::chrono::steady_clock::now();
    int emitted = 0;
    int accepted_drafts = 0;
    int proposed_drafts = 0;
    while (emitted < max_new) {
        if (prediction == kMessageEnd || prediction == kTextEnd) {
            static_cast<void>(model.forward(prediction));
            break;
        }
        if constexpr (kUseMtp) {
            if (max_new - emitted >= 4) {
                const p92::Model::SpeculativeResult result =
                    model.speculative_step(prediction);
                proposed_drafts += 3;
                accepted_drafts += static_cast<int>(result.accepted_drafts);
                bool reached_end = false;
                for (std::uint32_t index = 0; index < result.emitted_count; ++index) {
                    const std::int32_t token = result.emitted[index];
                    if (token == kMessageEnd || token == kTextEnd) {
                        reached_end = true;
                        break;
                    }
                    generated.push_back(token);
                    ++emitted;
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
        }
        generated.push_back(prediction);
        ++emitted;
        const std::string decoded = tokenizer.decode(generated);
        if (decoded.starts_with(rendered)) {
            std::cout << decoded.substr(rendered.size()) << std::flush;
            rendered = decoded;
        }
        prediction = model.forward(prediction);
    }
    if (emitted == max_new) static_cast<void>(model.forward(kMessageEnd));
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "\n[" << emitted << " tokens, " << seconds << "s, "
              << (seconds > 0.0 ? emitted / seconds : 0.0) << " tok/s, context="
              << model.position();
    if constexpr (kUseMtp) {
        std::cout << ", mtp_accept=" << accepted_drafts << '/' << proposed_drafts;
    }
    std::cout << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: " << (kUseMtp ? "p92_chat_mtp" : "p92_chat")
                  << " CHECKPOINT NVFP4_ARTIFACT [MAX_CONTEXT] [MAX_NEW]\n";
        return 2;
    }
    try {
        const int max_context = argc >= 4 ? std::stoi(argv[3]) : 524288;
        const int max_new = argc >= 5 ? std::stoi(argv[4]) : 512;
        if (max_new <= 0) throw std::invalid_argument("max-new must be positive");
        p92::Tokenizer tokenizer(std::string(argv[1]) + "/tokenizer.json");
        std::cerr << "P92_LOAD begin mode=fully_resident context=" << max_context << '\n';
        const auto load_started = std::chrono::steady_clock::now();
        p92::Model model(argv[1], argv[2], static_cast<std::uint32_t>(max_context),
                         p92::Model::WeightMode::fully_resident, kUseMtp);
        const double load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - load_started).count();
        std::cerr << "P92_READY load_s=" << load_seconds
                  << " mtp=" << kUseMtp
                  << " projection_bytes=" << model.projection_bytes()
                  << " auxiliary_bytes=" << model.auxiliary_bytes() << '\n';
        std::cerr << "Commands: /reset, /quit\n";

        bool first_turn = true;
        for (;;) {
            std::cout << "\nYou> " << std::flush;
            std::string user;
            if (!std::getline(std::cin, user) || user == "/quit") break;
            if (user == "/reset") {
                model.reset();
                first_turn = true;
                std::cout << "context reset\n";
                continue;
            }
            if (user.empty()) continue;
            const std::vector<std::int32_t> prompt = first_turn
                ? tokenizer.encode(p92::Tokenizer::chat_prompt(user))
                : continuation_prompt(tokenizer, user, true);
            first_turn = false;
            std::int32_t prediction = -1;
            const auto prefill_started = std::chrono::steady_clock::now();
            feed(model, prompt, prediction);
            const double prefill_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - prefill_started).count();
            std::cout << "Pangu> " << std::flush;
            generate(model, tokenizer, prediction, max_new);
            std::cerr << "prefill_tokens=" << prompt.size()
                      << " prefill_s=" << prefill_seconds << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_CHAT_ERROR " << error.what() << '\n';
        return 1;
    }
}
