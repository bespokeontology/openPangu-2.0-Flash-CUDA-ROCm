#include "p92/model.h"
#include "p92/tokenizer.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_profiler_api.h>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: p92_mtp_test CHECKPOINT NVFP4_ARTIFACT [PROMPT] [MAX_NEW]\n";
        return 2;
    }
    try {
        constexpr std::int32_t bos = 148899;
        const auto load_started = std::chrono::steady_clock::now();
        p92::Model model(argv[1], argv[2], 512,
                         p92::Model::WeightMode::fully_resident, true);
        const double load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - load_started).count();

        p92::Tokenizer tokenizer(std::string(argv[1]) + "/tokenizer.json");
        const std::int32_t seed = argc >= 4
            ? model.prefill(tokenizer.encode(p92::Tokenizer::chat_prompt(argv[3])))
            : model.forward(bos);
        const auto draft_started = std::chrono::steady_clock::now();
        const std::array<std::int32_t, 3> drafts = model.mtp_propose(seed);
        const double draft_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - draft_started).count();
        const std::array<std::int32_t, 3> repeated = model.mtp_propose(seed);
        if (drafts != repeated) {
            throw std::runtime_error("MTP proposal is not deterministic after state restore");
        }

        std::array<std::int32_t, 3> target_predictions {-1, -1, -1};
        std::int32_t target_input = seed;
        int accepted = 0;
        double target_seconds = 0.0;
        for (int stage = 0; stage < 3; ++stage) {
            const auto target_started = std::chrono::steady_clock::now();
            target_predictions[stage] = model.forward(target_input);
            target_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - target_started).count();
            if (target_predictions[stage] != drafts[stage]) break;
            ++accepted;
            target_input = drafts[stage];
        }

        std::vector<std::int32_t> generated;
        int accepted_total = 0;
        int proposed_total = 0;
        int blocks = 0;
        double language_seconds = 0.0;
        double proposal_seconds = 0.0;
        double verification_seconds = 0.0;
        double commit_seconds = 0.0;
        std::int32_t prediction = target_predictions[0];
        bool reached_end = false;
        if (argc >= 4) {
            model.reset();
            prediction = model.prefill(
                tokenizer.encode(p92::Tokenizer::chat_prompt(argv[3])));
            const int maximum_generated = argc == 5 ? std::stoi(argv[4]) : 32;
            if (maximum_generated <= 0) throw std::invalid_argument("MAX_NEW must be positive");
            const bool profile_cuda = std::getenv("P92_CUDA_PROFILE") != nullptr;
            if (profile_cuda && cudaProfilerStart() != cudaSuccess) {
                throw std::runtime_error("cannot start CUDA profiler range");
            }
            const auto language_started = std::chrono::steady_clock::now();
            while (static_cast<int>(generated.size()) < maximum_generated) {
                const p92::Model::SpeculativeResult result =
                    model.speculative_step(prediction);
                ++blocks;
                proposed_total += 3;
                accepted_total += static_cast<int>(result.accepted_drafts);
                proposal_seconds += result.proposal_seconds;
                verification_seconds += result.verification_seconds;
                commit_seconds += result.commit_seconds;
                for (std::uint32_t index = 0; index < result.emitted_count; ++index) {
                    const std::int32_t token = result.emitted[index];
                    if (token == 148900 || token == 148902) {
                        reached_end = true;
                        break;
                    }
                    generated.push_back(token);
                }
                prediction = result.next_token;
                if (reached_end) break;
            }
            language_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - language_started).count();
            if (profile_cuda && cudaProfilerStop() != cudaSuccess) {
                throw std::runtime_error("cannot stop CUDA profiler range");
            }
        }

        std::cout << "P92_MTP_OK seed=" << seed
                  << " drafts=" << drafts[0] << ',' << drafts[1] << ',' << drafts[2]
                  << " target=" << target_predictions[0] << ','
                  << target_predictions[1] << ',' << target_predictions[2]
                  << " accepted=" << accepted
                  << " load_s=" << load_seconds
                  << " draft_ms=" << draft_seconds * 1000.0
                  << " target_serial_ms=" << target_seconds * 1000.0
                  << " language_blocks=" << blocks
                  << " language_accept=" << accepted_total << '/' << proposed_total
                  << " language_s=" << language_seconds
                  << " language_tok_s=" << (language_seconds > 0.0
                      ? generated.size() / language_seconds : 0.0)
                  << " proposal_ms=" << proposal_seconds * 1000.0
                  << " verify_ms=" << verification_seconds * 1000.0
                  << " commit_ms=" << commit_seconds * 1000.0
                  << " reached_end=" << reached_end
                  << " text=" << (argc >= 4 ? tokenizer.decode(generated) : std::string())
                  << " projection_bytes=" << model.projection_bytes()
                  << " auxiliary_bytes=" << model.auxiliary_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_MTP_ERROR " << error.what() << '\n';
        return 1;
    }
}
