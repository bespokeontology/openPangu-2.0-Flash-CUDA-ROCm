#include "p92/model.h"
#include "p92/tokenizer.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double seconds_since(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
}

bool same(const p92::Model::SpeculativeResult& left,
          const p92::Model::SpeculativeResult& right) {
    if (left.emitted_count != right.emitted_count ||
        left.accepted_drafts != right.accepted_drafts ||
        left.next_token != right.next_token) {
        return false;
    }
    for (std::uint32_t index = 0; index < left.emitted_count; ++index) {
        if (left.emitted[index] != right.emitted[index]) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: p92_state_snapshot_test CHECKPOINT NVFP4_ARTIFACT\n";
        return 2;
    }
    try {
        const std::filesystem::path checkpoint(argv[1]);
        p92::Tokenizer tokenizer(checkpoint / "tokenizer.json");
        p92::Model model(checkpoint, argv[2], 4096,
                         p92::Model::WeightMode::fully_resident, true);
        const std::vector<std::int32_t> prompt = tokenizer.encode(
            p92::Tokenizer::chat_prompt(
                "Use one sentence to explain why reproducible state matters."));
        const std::int32_t prediction = model.prefill(prompt);

        const auto capture_started = std::chrono::steady_clock::now();
        model.capture_state("fixture");
        const double capture_seconds = seconds_since(capture_started);
        const p92::Model::SpeculativeResult first = model.speculative_step(prediction);

        const auto restore_started = std::chrono::steady_clock::now();
        model.restore_state("fixture");
        const double restore_seconds = seconds_since(restore_started);
        const p92::Model::SpeculativeResult second = model.speculative_step(prediction);
        model.release_state("fixture");

        if (!same(first, second)) {
            throw std::runtime_error("restored MTP branch changed token predictions");
        }
        std::cout << "P92_STATE_SNAPSHOT_OK position=" << prompt.size()
                  << " emitted=" << first.emitted_count
                  << " accepted=" << first.accepted_drafts
                  << " capture_s=" << capture_seconds
                  << " restore_s=" << restore_seconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_STATE_SNAPSHOT_ERROR " << error.what() << '\n';
        return 1;
    }
}
