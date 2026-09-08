#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

namespace p92 {

class Model final {
public:
    struct SpeculativeResult {
        std::array<std::int32_t, 4> emitted {};
        std::uint32_t emitted_count = 0;
        std::uint32_t accepted_drafts = 0;
        std::int32_t next_token = -1;
        double proposal_seconds = 0.0;
        double verification_seconds = 0.0;
        double commit_seconds = 0.0;
    };

    enum class WeightMode {
        selective,
        fully_resident,
    };

    Model(const std::filesystem::path& checkpoint,
          const std::filesystem::path& nvfp4_artifact,
          std::uint32_t max_context,
          WeightMode mode,
          bool enable_mtp = false);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;

    std::int32_t forward(std::int32_t token);
    std::int32_t prefill(std::span<const std::int32_t> tokens);
    std::array<std::int32_t, 3> mtp_propose(std::int32_t next_token);
    SpeculativeResult speculative_step(std::int32_t next_token);
    void capture_state(std::string_view key);
    void restore_state(std::string_view key);
    void release_state(std::string_view key);
    void reset();
    std::uint32_t position() const noexcept;
    std::uint64_t projection_bytes() const noexcept;
    std::uint64_t auxiliary_bytes() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace p92
