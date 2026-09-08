#include "p92/model.h"

#include "p92/auxiliary.h"
#include "p92/executor.h"
#include "p92/kernels.h"
#include "p92/resident.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace p92 {
namespace {

using BFloat16 = cuda::BFloat16;

constexpr int kTargetLayers = 46;
constexpr int kMtpLayers = 3;
constexpr int kAllLayers = kTargetLayers + kMtpLayers;
constexpr int kHidden = 2560;
constexpr int kStreams = 4;
constexpr int kFlat = kHidden * kStreams;
constexpr int kQueryLora = 1024;
constexpr int kKvLora = 512;
constexpr int kRope = 64;
constexpr int kQueryUp = 48 * (128 + 64);
constexpr int kLatentHeads = 48 * kKvLora;
constexpr int kAttentionOutput = 48 * 128;
constexpr int kDenseIntermediate = 9216;
constexpr int kExpertIntermediate = 1024;
constexpr int kSinks = 128;
constexpr int kExperts = 256;
constexpr int kSelectedExperts = 8;
constexpr int kVocabulary = 151552;
constexpr int kDsaHeads = 24;
constexpr int kDsaHead = 128;
constexpr int kDsaTopK = 2048;
constexpr int kSlidingWindow = 512;
constexpr int kMtpSlidingWindow = 2048;
constexpr int kMaximumBlockRows = 64;
constexpr int kMaximumSpecRows = kMtpLayers + 1;
constexpr std::uint32_t kMaximumContext = 524288;
constexpr int kMomeStatePerLayer = 2 * (kQueryLora + kKvLora + kAttentionOutput);
constexpr float kNormEpsilon = 1.0e-5F;

void check(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
    }
}

bool has_block_post_norm(int layer) {
    constexpr std::array<int, 9> layers {0, 4, 9, 14, 19, 24, 29, 34, 39};
    return std::find(layers.begin(), layers.end(), layer) != layers.end();
}

struct MhcWeights {
    const BFloat16* phi = nullptr;
    const BFloat16* gamma = nullptr;
    const BFloat16* alpha = nullptr;
    const BFloat16* beta = nullptr;
};

struct ExpertWeights {
    ResidentProjection gate;
    ResidentProjection up;
    ResidentProjection down;
};

struct LayerWeights {
    bool sliding = false;
    std::uint32_t sliding_window = 0;
    bool dense = false;
    const BFloat16* input_norm = nullptr;
    const BFloat16* post_attention_norm = nullptr;
    const BFloat16* pre_mlp_norm = nullptr;
    const BFloat16* post_mlp_norm = nullptr;
    const BFloat16* block_post_norm = nullptr;
    MhcWeights attention_mhc;
    MhcWeights mlp_mhc;

    ResidentProjection q_a;
    ResidentProjection q_b;
    ResidentProjection o;
    const BFloat16* kv_a = nullptr;
    const BFloat16* kv_b = nullptr;
    const BFloat16* q_norm = nullptr;
    const BFloat16* kv_norm = nullptr;
    const BFloat16* q_conv = nullptr;
    const BFloat16* kv_conv = nullptr;
    const BFloat16* o_conv = nullptr;
    const BFloat16* sink_rope = nullptr;
    ResidentProjection index_query;
    ResidentProjection index_key;
    const BFloat16* index_key_norm = nullptr;
    const BFloat16* index_weights = nullptr;

    ExpertWeights dense_mlp;
    const BFloat16* router = nullptr;
    const float* router_correction = nullptr;
    ExpertWeights shared;
    std::array<std::optional<ExpertWeights>, kExperts> experts;
    cuda::Nvfp4ExpertProjection* expert_bank = nullptr;
};

struct MtpWeights {
    const BFloat16* embedding_norm = nullptr;
    const BFloat16* hidden_norm = nullptr;
    ResidentProjection eh_projection;
    const BFloat16* head_norm = nullptr;
};

}  // namespace

class Model::Impl final {
public:
    struct StateSnapshot {
        std::uint32_t position = 0;
        std::vector<std::uint32_t> cache_rows;
        std::vector<std::size_t> cache_snapshot_offsets;
        std::vector<std::size_t> index_snapshot_offsets;
        BFloat16* cache_kv = nullptr;
        BFloat16* cache_rope = nullptr;
        BFloat16* index_cache = nullptr;
        BFloat16* mome_state = nullptr;
        BFloat16* last_target_hidden = nullptr;
        std::uint64_t bytes = 0;
    };

    Impl(const std::filesystem::path& checkpoint,
         const std::filesystem::path& artifact,
         std::uint32_t max_context,
         WeightMode mode,
         bool enable_mtp)
        : max_context_(max_context), mode_(mode), mtp_enabled_(enable_mtp) {
        if (mtp_enabled_ && mode_ != WeightMode::fully_resident) {
            throw std::invalid_argument("MTP currently requires fully resident weights");
        }
        if (max_context_ == 0 || max_context_ > kMaximumContext) {
            throw std::invalid_argument("native runtime max_context must be in [1, 524288]");
        }
        auxiliary_.open(checkpoint);
        if (mode_ == WeightMode::fully_resident) resident_.open(artifact);
        else selective_.open(artifact);
        bind_weights();
        allocate_state();
        check(cuda::bf16_mla_plan_create(&bf16_mla_plan_),
              "create batched BF16 MLA plan");
        prepare_sinks();
        reset();
    }

    ~Impl() {
        for (auto& entry : state_snapshots_) free_snapshot(entry.second);
        cuda::bf16_mla_plan_destroy(bf16_mla_plan_);
        for (void* allocation : allocations_) cudaFree(allocation);
    }

    std::int32_t forward(std::int32_t token) {
        if (token < 0 || token >= kVocabulary) throw std::invalid_argument("token outside vocabulary");
        if (position_ >= max_context_) throw std::runtime_error("native context is full");
        const cudaStream_t stream = executor_.stream();
        const Tensor& embedding = auxiliary_.catalog().find("model.embed_tokens.weight");
        const auto* embedding_host = reinterpret_cast<const BFloat16*>(auxiliary_.catalog().data(embedding));
        check(cudaMemcpyAsync(embedding_, embedding_host + static_cast<std::size_t>(token) * kHidden,
                              kHidden * sizeof(BFloat16), cudaMemcpyHostToDevice, stream),
              "upload token embedding");
        check(cuda::repeat_streams(embedding_, streams_a_, 1, kHidden, kStreams, stream),
              "repeat mHC streams");
        BFloat16* current_streams = streams_a_;
        BFloat16* other_streams = streams_b_;

        for (int layer_index = 0; layer_index < kTargetLayers; ++layer_index) {
            LayerWeights& layer = layers_.at(layer_index);
            BFloat16* const residual_attention = current_streams;
            check(cuda::mhc_pre_parallel(residual_attention, layer.attention_mhc.phi,
                                         layer.attention_mhc.gamma, layer.attention_mhc.alpha,
                                         layer.attention_mhc.beta, mixed_, h_post_, h_res_,
                                         mhc_inverse_, mhc_mixes_, 1, kNormEpsilon, stream),
                  "attention mHC pre");
            check(cuda::rmsnorm(mixed_, layer.input_norm, normalized_, 1, kHidden,
                                kNormEpsilon, stream),
                  "input RMSNorm");
            run_attention(layer, layer_index);
            check(cuda::rmsnorm(hidden_, layer.post_attention_norm, normalized_, 1,
                                kHidden, kNormEpsilon, stream),
                  "post-attention RMSNorm");
            check(cuda::mhc_post(normalized_, residual_attention, h_post_, h_res_,
                                 other_streams, 1, stream),
                  "attention mHC post");

            BFloat16* const residual_mlp = other_streams;
            check(cuda::mhc_pre_parallel(residual_mlp, layer.mlp_mhc.phi, layer.mlp_mhc.gamma,
                                         layer.mlp_mhc.alpha, layer.mlp_mhc.beta,
                                         mixed_, h_post_, h_res_, mhc_inverse_, mhc_mixes_,
                                         1, kNormEpsilon, stream),
                  "MLP mHC pre");
            check(cuda::rmsnorm(mixed_, layer.pre_mlp_norm, normalized_, 1, kHidden,
                                kNormEpsilon, stream),
                  "pre-MLP RMSNorm");
            if (layer.dense) run_dense(layer.dense_mlp);
            else run_moe(layer, layer_index);
            check(cuda::rmsnorm(hidden_, layer.post_mlp_norm, normalized_, 1, kHidden,
                                kNormEpsilon, stream),
                  "post-MLP RMSNorm");
            check(cuda::mhc_post(normalized_, residual_mlp, h_post_, h_res_,
                                 current_streams, 1, stream),
                  "MLP mHC post");
            if (layer.block_post_norm != nullptr) {
                check(cuda::rmsnorm(current_streams, layer.block_post_norm, other_streams,
                                    1, kFlat, kNormEpsilon, stream),
                      "block-post RMSNorm");
                std::swap(current_streams, other_streams);
            }
        }

        check(cuda::mhc_merge(current_streams, merge_.phi, merge_.gamma,
                              merge_.alpha, merge_.beta, hidden_, 1,
                              kNormEpsilon, stream),
              "final mHC merge");
        const BFloat16* output_hidden = hidden_;
        if (mtp_enabled_) {
            check(cudaMemcpyAsync(target_hidden_rows_, hidden_, kHidden * sizeof(BFloat16),
                                  cudaMemcpyDeviceToDevice, stream),
                  "retain target hidden row for MTP");
            const std::array<std::int32_t, 1> input_token {token};
            catch_up_mtp(input_token, target_hidden_rows_, position_);
            output_hidden = target_hidden_rows_;
        }
        check(cuda::rmsnorm(output_hidden, output_norm_, normalized_, 1, kHidden,
                            kNormEpsilon, stream),
              "output RMSNorm");
        executor_.project(language_head_, normalized_, logits_);
        check(cuda::argmax(logits_, kVocabulary, argmax_, stream), "greedy argmax");
        std::int32_t next = -1;
        check(cudaMemcpyAsync(&next, argmax_, sizeof(next), cudaMemcpyDeviceToHost, stream),
              "read greedy token");
        executor_.synchronize();
        if (next < 0 || next >= kVocabulary) throw std::runtime_error("invalid greedy token");
        ++position_;
        return next;
    }

    std::int32_t prefill(std::span<const std::int32_t> tokens) {
        if (tokens.empty()) throw std::invalid_argument("prefill requires at least one token");
        if (static_cast<std::uint64_t>(position_) + tokens.size() > max_context_) {
            throw std::runtime_error("native context is full");
        }
        std::int32_t prediction = -1;
        std::size_t offset = 0;
        while (offset < tokens.size()) {
            const std::size_t count = std::min<std::size_t>(
                kMaximumBlockRows, tokens.size() - offset);
            prediction = prefill_block(tokens.subspan(offset, count), {}, true);
            offset += count;
        }
        return prediction;
    }

    std::array<std::int32_t, kMtpLayers> mtp_propose(std::int32_t next_token) {
        if (!mtp_enabled_) throw std::logic_error("MTP predictor is not enabled");
        if (position_ == 0) throw std::logic_error("MTP requires target hidden state");
        if (next_token < 0 || next_token >= kVocabulary) {
            throw std::invalid_argument("MTP seed token outside vocabulary");
        }
        if (static_cast<std::uint64_t>(position_) + kMtpLayers > max_context_) {
            throw std::runtime_error("insufficient context for MTP proposal");
        }
        const cudaStream_t stream = executor_.stream();
        BFloat16* const mtp_state = mome_state_ +
            static_cast<std::size_t>(kTargetLayers) * kMomeStatePerLayer;
        const std::size_t state_bytes =
            static_cast<std::size_t>(kMtpLayers) * kMomeStatePerLayer * sizeof(BFloat16);
        check(cudaMemcpyAsync(mtp_state_snapshot_, mtp_state, state_bytes,
                              cudaMemcpyDeviceToDevice, stream),
              "snapshot MTP recurrent state");
        transfer_sliding_cache_rows(
            false, kTargetLayers, kMtpLayers, position_, kMtpLayers, 0, kMtpLayers,
            mtp_cache_kv_snapshot_, mtp_cache_rope_snapshot_);
        check(cudaMemcpyAsync(mtp_hidden_chain_, last_target_hidden_,
                              kHidden * sizeof(BFloat16), cudaMemcpyDeviceToDevice, stream),
              "seed MTP hidden chain");

        std::array<std::int32_t, kMtpLayers> drafts {};
        std::array<std::int32_t, kMtpLayers> stage_tokens {};
        stage_tokens[0] = next_token;
        try {
            for (int stage = 0; stage < kMtpLayers; ++stage) {
                const int rows = stage + 1;
                drafts[stage] = run_mtp_stage_rows(
                    stage,
                    std::span<const std::int32_t>(stage_tokens.data(),
                                                  static_cast<std::size_t>(rows)),
                    mtp_hidden_chain_, position_, true);
                check(cudaMemcpyAsync(
                          mtp_hidden_chain_ + static_cast<std::size_t>(stage + 1) * kHidden,
                          mtp_residual_ + static_cast<std::size_t>(rows - 1) * kHidden,
                          kHidden * sizeof(BFloat16), cudaMemcpyDeviceToDevice, stream),
                      "extend MTP hidden chain");
                if (stage + 1 < kMtpLayers) stage_tokens[stage + 1] = drafts[stage];
            }
        } catch (...) {
            cudaMemcpyAsync(mtp_state, mtp_state_snapshot_, state_bytes,
                            cudaMemcpyDeviceToDevice, stream);
            try {
                transfer_sliding_cache_rows(
                    true, kTargetLayers, kMtpLayers, position_,
                    kMtpLayers, 0, kMtpLayers,
                    mtp_cache_kv_snapshot_, mtp_cache_rope_snapshot_);
            } catch (...) {
            }
            cudaStreamSynchronize(stream);
            throw;
        }
        check(cudaMemcpyAsync(mtp_state, mtp_state_snapshot_, state_bytes,
                              cudaMemcpyDeviceToDevice, stream),
              "restore MTP recurrent state");
        transfer_sliding_cache_rows(
            true, kTargetLayers, kMtpLayers, position_, kMtpLayers, 0, kMtpLayers,
            mtp_cache_kv_snapshot_, mtp_cache_rope_snapshot_);
        executor_.synchronize();
        return drafts;
    }

    Model::SpeculativeResult speculative_step(std::int32_t next_token) {
        if (!mtp_enabled_) throw std::logic_error("MTP predictor is not enabled");
        if (static_cast<std::uint64_t>(position_) + kMaximumSpecRows > max_context_) {
            throw std::runtime_error("insufficient context for speculative block");
        }
        const auto proposal_started = std::chrono::steady_clock::now();
        const std::array<std::int32_t, kMtpLayers> drafts = mtp_propose(next_token);
        const auto proposal_finished = std::chrono::steady_clock::now();
        std::array<std::int32_t, kMaximumSpecRows> block_tokens {
            next_token, drafts[0], drafts[1], drafts[2],
        };
        std::array<std::int32_t, kMaximumSpecRows> target_predictions {};
        const cudaStream_t stream = executor_.stream();
        const std::size_t target_state_bytes =
            static_cast<std::size_t>(kTargetLayers) *
            kMomeStatePerLayer * sizeof(BFloat16);
        check(cudaMemcpyAsync(target_state_snapshot_, mome_state_, target_state_bytes,
                              cudaMemcpyDeviceToDevice, stream),
              "snapshot target recurrent state");
        transfer_sliding_cache_rows(
            false, 0, kTargetLayers, position_,
            kMaximumSpecRows, 0, kMaximumSpecRows,
            target_cache_kv_snapshot_, target_cache_rope_snapshot_);
        const std::uint32_t start_position = position_;
        try {
            static_cast<void>(prefill_block(block_tokens, target_predictions, false));
        } catch (...) {
            position_ = start_position;
            cudaMemcpyAsync(mome_state_, target_state_snapshot_, target_state_bytes,
                            cudaMemcpyDeviceToDevice, stream);
            try {
                transfer_sliding_cache_rows(
                    true, 0, kTargetLayers, start_position,
                    kMaximumSpecRows, 0, kMaximumSpecRows,
                    target_cache_kv_snapshot_, target_cache_rope_snapshot_);
            } catch (...) {
            }
            cudaStreamSynchronize(stream);
            throw;
        }
        const auto verification_finished = std::chrono::steady_clock::now();

        std::uint32_t accepted = 0;
        while (accepted < kMtpLayers &&
               target_predictions[accepted] == drafts[accepted]) {
            ++accepted;
        }
        std::uint32_t committed_rows = accepted + 1;
        for (std::uint32_t row = 0; row < committed_rows; ++row) {
            if (block_tokens[row] == 148900 || block_tokens[row] == 148902) {
                committed_rows = row + 1;
                accepted = row;
                break;
            }
        }
        restore_target_mome_row(static_cast<int>(committed_rows - 1));
        if (committed_rows < kMaximumSpecRows) {
            transfer_sliding_cache_rows(
                true, 0, kTargetLayers, start_position,
                kMaximumSpecRows, static_cast<int>(committed_rows),
                kMaximumSpecRows - static_cast<int>(committed_rows),
                target_cache_kv_snapshot_, target_cache_rope_snapshot_);
        }
        position_ = start_position + committed_rows;
        catch_up_mtp(
            std::span<const std::int32_t>(block_tokens.data(), committed_rows),
            target_hidden_rows_, start_position);
        executor_.synchronize();
        const auto commit_finished = std::chrono::steady_clock::now();

        Model::SpeculativeResult result;
        result.emitted[0] = next_token;
        for (std::uint32_t index = 0; index < accepted; ++index) {
            result.emitted[index + 1] = drafts[index];
        }
        result.emitted_count = committed_rows;
        result.accepted_drafts = accepted;
        result.next_token = target_predictions[accepted];
        result.proposal_seconds = std::chrono::duration<double>(
            proposal_finished - proposal_started).count();
        result.verification_seconds = std::chrono::duration<double>(
            verification_finished - proposal_finished).count();
        result.commit_seconds = std::chrono::duration<double>(
            commit_finished - verification_finished).count();
        return result;
    }

    void capture_state(std::string_view key_view) {
        if (key_view.empty()) throw std::invalid_argument("state snapshot key is empty");
        const std::string key(key_view);
        if (auto existing = state_snapshots_.find(key); existing != state_snapshots_.end()) {
            free_snapshot(existing->second);
            state_snapshots_.erase(existing);
        }

        StateSnapshot snapshot;
        snapshot.position = position_;
        snapshot.cache_rows.resize(active_layer_count());
        snapshot.cache_snapshot_offsets.resize(active_layer_count());
        snapshot.index_snapshot_offsets.assign(
            active_layer_count(), std::numeric_limits<std::size_t>::max());

        std::size_t cache_rows = 0;
        std::size_t index_rows = 0;
        for (int layer = 0; layer < active_layer_count(); ++layer) {
            const std::uint32_t rows = std::min(position_, cache_capacities_[layer]);
            snapshot.cache_rows[layer] = rows;
            snapshot.cache_snapshot_offsets[layer] = cache_rows;
            cache_rows += rows;
            if (index_cache_offsets_[layer] != std::numeric_limits<std::size_t>::max()) {
                snapshot.index_snapshot_offsets[layer] = index_rows;
                index_rows += position_;
            }
        }

        const auto allocate_snapshot = [&](BFloat16** pointer,
                                           std::size_t values,
                                           const char* label) {
            check(cudaMalloc(reinterpret_cast<void**>(pointer),
                             std::max<std::size_t>(values, 1) * sizeof(BFloat16)),
                  std::string("allocate state snapshot ") + label);
            snapshot.bytes += values * sizeof(BFloat16);
        };
        try {
            allocate_snapshot(&snapshot.cache_kv, cache_rows * kKvLora, "latent cache");
            allocate_snapshot(&snapshot.cache_rope, cache_rows * kRope, "RoPE cache");
            allocate_snapshot(&snapshot.index_cache, index_rows * kDsaHead, "DSA cache");
            allocate_snapshot(
                &snapshot.mome_state,
                static_cast<std::size_t>(active_layer_count()) * kMomeStatePerLayer,
                "MoME state");
            if (mtp_enabled_) {
                allocate_snapshot(&snapshot.last_target_hidden, kHidden,
                                  "last target hidden");
            }

            const cudaStream_t stream = executor_.stream();
            for (int layer = 0; layer < active_layer_count(); ++layer) {
                const std::size_t rows = snapshot.cache_rows[layer];
                if (rows != 0) {
                    const std::size_t source = cache_offsets_[layer];
                    const std::size_t target = snapshot.cache_snapshot_offsets[layer];
                    check(cudaMemcpyAsync(
                              snapshot.cache_kv + target * kKvLora,
                              cache_kv_ + source * kKvLora,
                              rows * kKvLora * sizeof(BFloat16),
                              cudaMemcpyDeviceToDevice, stream),
                          "snapshot latent cache rows");
                    check(cudaMemcpyAsync(
                              snapshot.cache_rope + target * kRope,
                              cache_rope_ + source * kRope,
                              rows * kRope * sizeof(BFloat16),
                              cudaMemcpyDeviceToDevice, stream),
                          "snapshot RoPE cache rows");
                }
                const std::size_t index_target = snapshot.index_snapshot_offsets[layer];
                if (index_target != std::numeric_limits<std::size_t>::max() &&
                    position_ != 0) {
                    check(cudaMemcpyAsync(
                              snapshot.index_cache + index_target * kDsaHead,
                              index_cache_ + index_cache_offsets_[layer] * kDsaHead,
                              static_cast<std::size_t>(position_) * kDsaHead *
                                  sizeof(BFloat16),
                              cudaMemcpyDeviceToDevice, stream),
                          "snapshot DSA cache rows");
                }
            }
            check(cudaMemcpyAsync(
                      snapshot.mome_state, mome_state_,
                      static_cast<std::size_t>(active_layer_count()) *
                          kMomeStatePerLayer * sizeof(BFloat16),
                      cudaMemcpyDeviceToDevice, stream),
                  "snapshot MoME state");
            if (mtp_enabled_) {
                check(cudaMemcpyAsync(snapshot.last_target_hidden, last_target_hidden_,
                                      kHidden * sizeof(BFloat16),
                                      cudaMemcpyDeviceToDevice, stream),
                      "snapshot last target hidden");
            }
            executor_.synchronize();
            state_snapshots_.emplace(key, std::move(snapshot));
        } catch (...) {
            free_snapshot(snapshot);
            throw;
        }
    }

    void restore_state(std::string_view key_view) {
        const auto found = state_snapshots_.find(std::string(key_view));
        if (found == state_snapshots_.end()) {
            throw std::invalid_argument("state snapshot not found");
        }
        const StateSnapshot& snapshot = found->second;
        if (snapshot.cache_rows.size() != static_cast<std::size_t>(active_layer_count()) ||
            snapshot.position > max_context_) {
            throw std::runtime_error("state snapshot shape mismatch");
        }
        const cudaStream_t stream = executor_.stream();
        for (int layer = 0; layer < active_layer_count(); ++layer) {
            const std::size_t rows = snapshot.cache_rows[layer];
            if (rows != 0) {
                const std::size_t source = snapshot.cache_snapshot_offsets[layer];
                const std::size_t target = cache_offsets_[layer];
                check(cudaMemcpyAsync(
                          cache_kv_ + target * kKvLora,
                          snapshot.cache_kv + source * kKvLora,
                          rows * kKvLora * sizeof(BFloat16),
                          cudaMemcpyDeviceToDevice, stream),
                      "restore latent cache rows");
                check(cudaMemcpyAsync(
                          cache_rope_ + target * kRope,
                          snapshot.cache_rope + source * kRope,
                          rows * kRope * sizeof(BFloat16),
                          cudaMemcpyDeviceToDevice, stream),
                      "restore RoPE cache rows");
            }
            const std::size_t index_source = snapshot.index_snapshot_offsets[layer];
            if (index_source != std::numeric_limits<std::size_t>::max() &&
                snapshot.position != 0) {
                check(cudaMemcpyAsync(
                          index_cache_ + index_cache_offsets_[layer] * kDsaHead,
                          snapshot.index_cache + index_source * kDsaHead,
                          static_cast<std::size_t>(snapshot.position) * kDsaHead *
                              sizeof(BFloat16),
                          cudaMemcpyDeviceToDevice, stream),
                      "restore DSA cache rows");
            }
        }
        check(cudaMemcpyAsync(
                  mome_state_, snapshot.mome_state,
                  static_cast<std::size_t>(active_layer_count()) *
                      kMomeStatePerLayer * sizeof(BFloat16),
                  cudaMemcpyDeviceToDevice, stream),
              "restore MoME state");
        if (mtp_enabled_) {
            check(cudaMemcpyAsync(last_target_hidden_, snapshot.last_target_hidden,
                                  kHidden * sizeof(BFloat16),
                                  cudaMemcpyDeviceToDevice, stream),
                  "restore last target hidden");
        }
        position_ = snapshot.position;
        executor_.synchronize();
    }

    void release_state(std::string_view key_view) {
        const auto found = state_snapshots_.find(std::string(key_view));
        if (found == state_snapshots_.end()) return;
        free_snapshot(found->second);
        state_snapshots_.erase(found);
    }

    void reset() {
        position_ = 0;
        const cudaStream_t stream = executor_.stream();
        check(cudaMemsetAsync(cache_kv_, 0,
                              cache_slots_ * kKvLora * sizeof(BFloat16),
                              stream),
              "clear MLA latent cache");
        check(cudaMemsetAsync(cache_rope_, 0,
                              cache_slots_ * kRope * sizeof(BFloat16),
                              stream),
              "clear MLA RoPE cache");
        check(cudaMemsetAsync(index_cache_, 0,
                              index_cache_slots_ * kDsaHead * sizeof(BFloat16),
                              stream),
              "clear DSA key cache");
        check(cudaMemsetAsync(mome_state_, 0,
                              static_cast<std::size_t>(active_layer_count()) *
                                  kMomeStatePerLayer * sizeof(BFloat16),
                              stream),
              "clear MoME state");
        if (mtp_enabled_) {
            check(cudaMemsetAsync(last_target_hidden_, 0, kHidden * sizeof(BFloat16), stream),
                  "clear previous target hidden state");
        }
        executor_.synchronize();
    }

    std::uint32_t position() const noexcept { return position_; }
    std::uint64_t projection_bytes() const noexcept {
        return mode_ == WeightMode::fully_resident ? resident_.resident_bytes()
                                                   : selective_.resident_bytes();
    }
    std::uint64_t auxiliary_bytes() const noexcept { return auxiliary_.resident_bytes(); }

private:
    static void free_snapshot(StateSnapshot& snapshot) noexcept {
        cudaFree(snapshot.cache_kv);
        cudaFree(snapshot.cache_rope);
        cudaFree(snapshot.index_cache);
        cudaFree(snapshot.mome_state);
        cudaFree(snapshot.last_target_hidden);
        snapshot = {};
    }

    int active_layer_count() const noexcept {
        return mtp_enabled_ ? kAllLayers : kTargetLayers;
    }

    template <class T>
    T* allocate(std::size_t count, const char* label) {
        T* pointer = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), std::max<std::size_t>(count, 1) * sizeof(T)),
              std::string("allocate ") + label);
        allocations_.push_back(pointer);
        return pointer;
    }

    const BFloat16* bf16(const std::string& name) {
        return auxiliary_.load_bf16(name).data;
    }

    const float* f32(const std::string& name) {
        return auxiliary_.load_f32(name).data;
    }

    ResidentProjection projection(const std::string& name) {
        return mode_ == WeightMode::fully_resident ? resident_.find(name) : selective_.load(name);
    }

    MhcWeights bind_mhc(const std::string& prefix) {
        return {
            bf16(prefix + ".phi.weight"),
            bf16(prefix + ".norm_gamma"),
            bf16(prefix + ".branch_alpha"),
            bf16(prefix + ".branch_beta"),
        };
    }

    ExpertWeights bind_expert(const std::string& prefix) {
        return {
            projection(prefix + ".gate_proj.weight"),
            projection(prefix + ".up_proj.weight"),
            projection(prefix + ".down_proj.weight"),
        };
    }

    ExpertWeights& expert(LayerWeights& layer, int layer_index, int expert_id) {
        std::optional<ExpertWeights>& cached = layer.experts.at(expert_id);
        if (!cached.has_value()) {
            cached = bind_expert("model.layers." + std::to_string(layer_index) +
                                 ".mlp.experts." + std::to_string(expert_id));
        }
        return *cached;
    }

    void bind_weights() {
        layers_.resize(active_layer_count());
        for (int index = 0; index < kTargetLayers; ++index) {
            LayerWeights& layer = layers_[index];
            const std::string prefix = "model.layers." + std::to_string(index) + ".";
            const std::string attention = prefix + "self_attn.";
            layer.sliding = index % 3 != 0;
            layer.sliding_window = layer.sliding ? kSlidingWindow : 0;
            layer.dense = index < 2;
            layer.input_norm = bf16(prefix + "input_layernorm.weight");
            layer.post_attention_norm = bf16(prefix + "post_attention_layernorm.weight");
            layer.pre_mlp_norm = bf16(prefix + "pre_mlp_layernorm.weight");
            layer.post_mlp_norm = bf16(prefix + "post_mlp_layernorm.weight");
            if (has_block_post_norm(index)) {
                layer.block_post_norm = bf16(prefix + "block_post_layernorm.weight");
            }
            layer.attention_mhc = bind_mhc(prefix + "attn_mhc_module");
            layer.mlp_mhc = bind_mhc(prefix + "mlp_mhc_module");
            layer.q_a = projection(attention + "q_a_proj.weight");
            layer.q_b = projection(attention + "q_b_proj.weight");
            layer.o = projection(attention + "o_proj.weight");
            layer.kv_a = bf16(attention + "kv_a_proj_with_mqa.weight");
            layer.kv_b = bf16(attention + "kv_b_proj.weight");
            layer.q_norm = bf16(attention + "q_a_layernorm.weight");
            layer.kv_norm = bf16(attention + "kv_a_layernorm.weight");
            layer.q_conv = bf16(attention + "qa_conv.weight");
            layer.kv_conv = bf16(attention + "compresskv_conv.weight");
            layer.o_conv = bf16(attention + "o_conv.weight");
            layer.sink_rope = bf16(attention + "param_sink_k_pe");
            if (!layer.sliding) {
                layer.index_query = projection(attention + "indexer.wq_b.weight");
                layer.index_key = projection(attention + "indexer.wk.weight");
                layer.index_key_norm = bf16(attention + "indexer.k_norm.weight");
                layer.index_weights = bf16(attention + "indexer.weights_proj.weight");
            }
            if (layer.dense) {
                layer.dense_mlp = bind_expert(prefix + "mlp");
            } else {
                layer.router = bf16(prefix + "mlp.gate.weight");
                layer.router_correction = f32(prefix + "mlp.e_score_correction_bias");
                layer.shared = bind_expert(prefix + "mlp.shared_experts");
            }
        }
        if (mtp_enabled_) {
            for (int stage = 0; stage < kMtpLayers; ++stage) {
                const int index = kTargetLayers + stage;
                LayerWeights& layer = layers_[index];
                const std::string prefix =
                    "model.layers." + std::to_string(index) + ".";
                const std::string attention = prefix + "self_attn.";
                layer.sliding = true;
                layer.sliding_window = kMtpSlidingWindow;
                layer.dense = false;
                layer.input_norm = bf16(prefix + "input_layernorm.weight");
                layer.post_attention_norm = bf16(
                    prefix + "post_attention_layernorm.weight");
                layer.pre_mlp_norm = bf16(prefix + "pre_mlp_layernorm.weight");
                layer.post_mlp_norm = bf16(prefix + "post_mlp_layernorm.weight");
                layer.q_a = projection(attention + "q_a_proj.weight");
                layer.q_b = projection(attention + "q_b_proj.weight");
                layer.o = projection(attention + "o_proj.weight");
                layer.kv_a = bf16(attention + "kv_a_proj_with_mqa.weight");
                layer.kv_b = bf16(attention + "kv_b_proj.weight");
                layer.q_norm = bf16(attention + "q_a_layernorm.weight");
                layer.kv_norm = bf16(attention + "kv_a_layernorm.weight");
                layer.q_conv = bf16(attention + "qa_conv.weight");
                layer.kv_conv = bf16(attention + "compresskv_conv.weight");
                layer.o_conv = bf16(attention + "o_conv.weight");
                layer.sink_rope = bf16(attention + "param_sink_k_pe");
                layer.router = bf16(prefix + "mlp.gate.weight");
                layer.router_correction = f32(
                    prefix + "mlp.e_score_correction_bias");
                layer.shared = bind_expert(prefix + "mlp.shared_experts");

                mtp_[stage].embedding_norm = bf16(prefix + "enorm.weight");
                mtp_[stage].hidden_norm = bf16(prefix + "hnorm.weight");
                mtp_[stage].eh_projection = projection(prefix + "eh_proj.weight");
                mtp_[stage].head_norm = bf16(prefix + "shared_head.norm.weight");
            }
        }
        merge_ = {
            bf16("model.merge_mhc_module.phi.weight"),
            bf16("model.merge_mhc_module.norm_gamma"),
            bf16("model.merge_mhc_module.branch_alpha_pre"),
            bf16("model.merge_mhc_module.branch_beta_pre"),
        };
        output_norm_ = bf16("model.norm.weight");
        language_head_ = projection("lm_head.weight");
        if (mode_ == WeightMode::fully_resident) {
            constexpr std::size_t kEntriesPerLayer =
                static_cast<std::size_t>(kExperts) * 3;
            std::vector<cuda::Nvfp4ExpertProjection> host_banks(
                static_cast<std::size_t>(active_layer_count()) * kEntriesPerLayer);
            for (int layer_index = 2; layer_index < active_layer_count(); ++layer_index) {
                LayerWeights& layer = layers_[layer_index];
                for (int expert_id = 0; expert_id < kExperts; ++expert_id) {
                    ExpertWeights bound = bind_expert(
                        "model.layers." + std::to_string(layer_index) +
                        ".mlp.experts." + std::to_string(expert_id));
                    layer.experts[expert_id] = bound;
                    const std::size_t base =
                        static_cast<std::size_t>(layer_index) * kEntriesPerLayer +
                        static_cast<std::size_t>(expert_id) * 3;
                    host_banks[base] = {
                        bound.gate.weight, bound.gate.scales_swizzled, bound.gate.alpha};
                    host_banks[base + 1] = {
                        bound.up.weight, bound.up.scales_swizzled, bound.up.alpha};
                    host_banks[base + 2] = {
                        bound.down.weight, bound.down.scales_swizzled, bound.down.alpha};
                }
            }
            expert_banks_ = allocate<cuda::Nvfp4ExpertProjection>(
                host_banks.size(), "resident routed expert table");
            check(cudaMemcpy(expert_banks_, host_banks.data(),
                             host_banks.size() * sizeof(cuda::Nvfp4ExpertProjection),
                             cudaMemcpyHostToDevice),
                  "upload resident routed expert table");
            for (int layer_index = 2; layer_index < active_layer_count(); ++layer_index) {
                layers_[layer_index].expert_bank =
                    expert_banks_ + static_cast<std::size_t>(layer_index) * kEntriesPerLayer;
            }
        }
    }

    void allocate_state() {
        cache_offsets_.resize(active_layer_count());
        cache_capacities_.resize(active_layer_count());
        index_cache_offsets_.assign(active_layer_count(), std::numeric_limits<std::size_t>::max());
        for (int index = 0; index < active_layer_count(); ++index) {
            const std::uint32_t capacity = layers_[index].sliding
                ? std::min<std::uint32_t>(max_context_, layers_[index].sliding_window)
                : max_context_;
            cache_offsets_[index] = cache_slots_;
            cache_capacities_[index] = capacity;
            cache_slots_ += capacity;
            if (!layers_[index].sliding) {
                index_cache_offsets_[index] = index_cache_slots_;
                index_cache_slots_ += max_context_;
            }
        }
        embedding_ = allocate<BFloat16>(static_cast<std::size_t>(kMaximumBlockRows) * kHidden,
                                        "embedding scratch");
        streams_a_ = allocate<BFloat16>(static_cast<std::size_t>(kMaximumBlockRows) * kFlat,
                                        "mHC streams A");
        streams_b_ = allocate<BFloat16>(static_cast<std::size_t>(kMaximumBlockRows) * kFlat,
                                        "mHC streams B");
        mixed_ = allocate<BFloat16>(static_cast<std::size_t>(kMaximumBlockRows) * kHidden,
                                    "mixed hidden");
        normalized_ = allocate<BFloat16>(static_cast<std::size_t>(kMaximumBlockRows) * kHidden,
                                         "normalized hidden");
        hidden_ = allocate<BFloat16>(static_cast<std::size_t>(kMaximumBlockRows) * kHidden,
                                     "hidden output");
        h_post_ = allocate<float>(static_cast<std::size_t>(kMaximumBlockRows) * kStreams,
                                  "mHC post coefficients");
        h_res_ = allocate<float>(static_cast<std::size_t>(kMaximumBlockRows) * kStreams * kStreams,
                                 "mHC residual coefficients");
        mhc_inverse_ = allocate<float>(kMaximumBlockRows, "mHC inverse norms");
        mhc_mixes_ = allocate<float>(static_cast<std::size_t>(kMaximumBlockRows) * 24,
                                     "mHC projected coefficients");
        query_lora_ = allocate<BFloat16>(static_cast<std::size_t>(kMaximumBlockRows) * kQueryLora,
                                         "query lora");
        query_norm_ = allocate<BFloat16>(static_cast<std::size_t>(kMaximumBlockRows) * kQueryLora,
                                         "normalized query lora");
        query_up_ = allocate<BFloat16>(static_cast<std::size_t>(kMaximumBlockRows) * kQueryUp,
                                       "query up");
        kv_down_ = allocate<BFloat16>(static_cast<std::size_t>(kMaximumBlockRows) *
                                          (kKvLora + kRope),
                                      "compressed KV");
        block_cache_kv_ = allocate<BFloat16>(
            static_cast<std::size_t>(kMaximumBlockRows) * kKvLora, "block latent KV");
        block_cache_rope_ = allocate<BFloat16>(
            static_cast<std::size_t>(kMaximumBlockRows) * kRope, "block RoPE keys");
        q_absorbed_ = allocate<BFloat16>(static_cast<std::size_t>(kMaximumBlockRows) * kLatentHeads,
                                         "absorbed query");
        q_rope_ = allocate<BFloat16>(static_cast<std::size_t>(kMaximumBlockRows) * 48 * kRope,
                                     "query rope");
        latent_output_ = allocate<BFloat16>(
            static_cast<std::size_t>(kMaximumBlockRows) * kLatentHeads,
            "latent attention output");
        attention_output_ = allocate<BFloat16>(
            static_cast<std::size_t>(kMaximumBlockRows) * kAttentionOutput,
            "attention output");
        mla_bf16_input_ = allocate<BFloat16>(
            static_cast<std::size_t>(kMaximumBlockRows) * kLatentHeads,
            "batched BF16 MLA input");
        mla_bf16_output_ = allocate<BFloat16>(
            static_cast<std::size_t>(kMaximumBlockRows) * kLatentHeads,
            "batched BF16 MLA output");
        gate_output_ = allocate<BFloat16>(
            static_cast<std::size_t>(kMaximumBlockRows) * kDenseIntermediate,
            "gate output");
        up_output_ = allocate<BFloat16>(
            static_cast<std::size_t>(kMaximumBlockRows) * kDenseIntermediate,
            "up output");
        router_logits_ = allocate<float>(static_cast<std::size_t>(kMaximumBlockRows) * kExperts,
                                         "router logits");
        expert_ids_ = allocate<std::int32_t>(
            static_cast<std::size_t>(kMaximumBlockRows) * kSelectedExperts, "expert ids");
        expert_weights_ = allocate<float>(
            static_cast<std::size_t>(kMaximumBlockRows) * kSelectedExperts, "expert weights");
        group_rows_ = allocate<std::int32_t>(kMaximumBlockRows, "grouped expert rows");
        group_scales_ = allocate<float>(kMaximumBlockRows, "grouped expert scales");
        argmax_ = allocate<std::int32_t>(1, "argmax");
        logits_ = allocate<BFloat16>(kVocabulary, "logits");
        sink_kv_ = allocate<BFloat16>(
            static_cast<std::size_t>(active_layer_count()) * kSinks * kKvLora,
                                      "normalized sink KV");
        cache_kv_ = allocate<BFloat16>(cache_slots_ * kKvLora,
                                       "latent KV cache");
        cache_rope_ = allocate<BFloat16>(cache_slots_ * kRope,
                                         "RoPE cache");
        index_query_ = allocate<BFloat16>(
            static_cast<std::size_t>(kMaximumBlockRows) * kDsaHeads * kDsaHead,
            "DSA query");
        index_key_ = allocate<BFloat16>(
            static_cast<std::size_t>(kMaximumBlockRows) * kDsaHead, "DSA key");
        index_weights_ = allocate<BFloat16>(
            static_cast<std::size_t>(kMaximumBlockRows) * kDsaHeads,
            "DSA head weights");
        index_cache_ = allocate<BFloat16>(index_cache_slots_ * kDsaHead, "DSA key cache");
        dsa_scores_in_ = allocate<float>(max_context_, "DSA input scores");
        dsa_scores_out_ = allocate<float>(max_context_, "DSA sorted scores");
        dsa_indices_in_ = allocate<std::int32_t>(max_context_, "DSA input indices");
        dsa_indices_out_ = allocate<std::int32_t>(max_context_, "DSA selected indices");
        dsa_sort_workspace_bytes_ = cuda::dsa_sort_workspace_bytes(static_cast<int>(max_context_));
        if (dsa_sort_workspace_bytes_ == 0) {
            throw std::runtime_error("cannot size DSA radix-sort workspace");
        }
        dsa_sort_workspace_ = allocate<std::uint8_t>(dsa_sort_workspace_bytes_,
                                                      "DSA radix-sort workspace");
        mome_state_ = allocate<BFloat16>(
            static_cast<std::size_t>(active_layer_count()) * kMomeStatePerLayer,
                                         "MoME recurrent state");
        if (mtp_enabled_) {
            target_hidden_rows_ = allocate<BFloat16>(
                static_cast<std::size_t>(kMaximumBlockRows) * kHidden,
                "target hidden rows for MTP");
            last_target_hidden_ = allocate<BFloat16>(kHidden, "last target hidden for MTP");
            previous_hidden_rows_ = allocate<BFloat16>(
                static_cast<std::size_t>(kMaximumBlockRows) * kHidden,
                "previous target hidden rows for MTP");
            mtp_embedding_norm_ = allocate<BFloat16>(
                static_cast<std::size_t>(kMaximumBlockRows) * kHidden,
                "MTP normalized embeddings");
            mtp_hidden_norm_ = allocate<BFloat16>(
                static_cast<std::size_t>(kMaximumBlockRows) * kHidden,
                "MTP normalized hidden rows");
            mtp_eh_input_ = allocate<BFloat16>(
                static_cast<std::size_t>(kMaximumBlockRows) * 2 * kHidden,
                "MTP concatenated input");
            mtp_residual_ = allocate<BFloat16>(
                static_cast<std::size_t>(kMaximumBlockRows) * kHidden,
                "MTP residual rows");
            mtp_hidden_chain_ = allocate<BFloat16>(
                static_cast<std::size_t>(kMtpLayers + 1) * kHidden,
                "MTP chained hidden rows");
            mtp_state_snapshot_ = allocate<BFloat16>(
                static_cast<std::size_t>(kMtpLayers) * kMomeStatePerLayer,
                "MTP recurrent-state snapshot");
            mtp_cache_kv_snapshot_ = allocate<BFloat16>(
                static_cast<std::size_t>(kMtpLayers) * kMtpLayers * kKvLora,
                "MTP cache snapshot");
            mtp_cache_rope_snapshot_ = allocate<BFloat16>(
                static_cast<std::size_t>(kMtpLayers) * kMtpLayers * kRope,
                "MTP RoPE snapshot");
            target_state_snapshot_ = allocate<BFloat16>(
                static_cast<std::size_t>(kTargetLayers) * kMomeStatePerLayer,
                "target recurrent-state snapshot");
            verify_mome_rows_ = allocate<BFloat16>(
                static_cast<std::size_t>(kTargetLayers) * kMaximumSpecRows *
                    kMomeStatePerLayer,
                "target verifier row states");
            target_cache_kv_snapshot_ = allocate<BFloat16>(
                static_cast<std::size_t>(kTargetLayers) * kMaximumSpecRows * kKvLora,
                "target cache snapshot");
            target_cache_rope_snapshot_ = allocate<BFloat16>(
                static_cast<std::size_t>(kTargetLayers) * kMaximumSpecRows * kRope,
                "target RoPE snapshot");
            block_logits_ = allocate<BFloat16>(
                static_cast<std::size_t>(kMaximumSpecRows) * kVocabulary,
                "speculative block logits");
            block_argmax_ = allocate<std::int32_t>(kMaximumSpecRows,
                                                   "speculative block argmax");
        }
    }

    void load_embedding_rows(std::span<const std::int32_t> tokens) {
        const cudaStream_t stream = executor_.stream();
        const Tensor& tensor = auxiliary_.catalog().find("model.embed_tokens.weight");
        const auto* host = reinterpret_cast<const BFloat16*>(
            auxiliary_.catalog().data(tensor));
        for (std::size_t row = 0; row < tokens.size(); ++row) {
            const std::int32_t token = tokens[row];
            if (token < 0 || token >= kVocabulary) {
                throw std::invalid_argument("MTP token outside vocabulary");
            }
            check(cudaMemcpyAsync(
                      embedding_ + row * kHidden,
                      host + static_cast<std::size_t>(token) * kHidden,
                      kHidden * sizeof(BFloat16), cudaMemcpyHostToDevice, stream),
                  "upload MTP token embedding");
        }
    }

    void transfer_sliding_cache_rows(bool restore,
                                     int first_layer,
                                     int layer_count,
                                     std::uint32_t start_position,
                                     int snapshot_rows,
                                     int first_row,
                                     int row_count,
                                     BFloat16* kv_snapshot,
                                     BFloat16* rope_snapshot) {
        if (first_layer < 0 || layer_count <= 0 || snapshot_rows <= 0 ||
            first_row < 0 || row_count <= 0 ||
            first_row + row_count > snapshot_rows ||
            first_layer + layer_count > active_layer_count() ||
            kv_snapshot == nullptr || rope_snapshot == nullptr) {
            throw std::invalid_argument("invalid sliding-cache snapshot range");
        }
        const cudaStream_t stream = executor_.stream();
        for (int local_layer = 0; local_layer < layer_count; ++local_layer) {
            const int layer_index = first_layer + local_layer;
            if (!layers_[layer_index].sliding) continue;
            const std::uint32_t capacity = cache_capacities_[layer_index];
            BFloat16* layer_kv = cache_kv_ + cache_offsets_[layer_index] * kKvLora;
            BFloat16* layer_rope = cache_rope_ + cache_offsets_[layer_index] * kRope;
            BFloat16* saved_kv = kv_snapshot +
                static_cast<std::size_t>(local_layer) * snapshot_rows * kKvLora;
            BFloat16* saved_rope = rope_snapshot +
                static_cast<std::size_t>(local_layer) * snapshot_rows * kRope;
            check(cuda::mla_transfer_cache_rows(
                      restore, layer_kv, layer_rope, saved_kv, saved_rope,
                      static_cast<int>(start_position), snapshot_rows,
                      first_row, row_count, static_cast<int>(capacity), stream),
                  restore ? "restore sliding cache rows" :
                            "snapshot sliding cache rows");
        }
    }

    void restore_target_mome_row(int row) {
        if (row < 0 || row >= kMaximumSpecRows) {
            throw std::invalid_argument("invalid target MoME verifier row");
        }
        check(cuda::mome_commit_captured_row(
                  mome_state_, verify_mome_rows_, kTargetLayers,
                  kMaximumSpecRows, row, kQueryLora, kKvLora,
                  kAttentionOutput, executor_.stream()),
              "commit target MoME row state");
    }

    std::int32_t run_mtp_stage_rows(int stage,
                                    std::span<const std::int32_t> tokens,
                                    const BFloat16* previous_hidden,
                                    std::uint32_t start_position,
                                    bool compute_token) {
        if (stage < 0 || stage >= kMtpLayers || tokens.empty() ||
            tokens.size() > kMaximumBlockRows || previous_hidden == nullptr) {
            throw std::invalid_argument("invalid MTP stage rows");
        }
        const int rows = static_cast<int>(tokens.size());
        const int layer_index = kTargetLayers + stage;
        LayerWeights& layer = layers_.at(layer_index);
        const MtpWeights& weights = mtp_.at(stage);
        const cudaStream_t stream = executor_.stream();

        load_embedding_rows(tokens);
        check(cuda::rmsnorm(embedding_, weights.embedding_norm, mtp_embedding_norm_,
                            rows, kHidden, kNormEpsilon, stream),
              "MTP embedding RMSNorm");
        check(cuda::rmsnorm(previous_hidden, weights.hidden_norm, mtp_hidden_norm_,
                            rows, kHidden, kNormEpsilon, stream),
              "MTP hidden RMSNorm");
        const std::size_t half_bytes = kHidden * sizeof(BFloat16);
        const std::size_t row_bytes = 2 * half_bytes;
        check(cudaMemcpy2DAsync(mtp_eh_input_, row_bytes,
                                mtp_embedding_norm_, half_bytes,
                                half_bytes, rows, cudaMemcpyDeviceToDevice, stream),
              "pack normalized MTP embeddings");
        check(cudaMemcpy2DAsync(mtp_eh_input_ + kHidden, row_bytes,
                                mtp_hidden_norm_, half_bytes,
                                half_bytes, rows, cudaMemcpyDeviceToDevice, stream),
              "pack normalized MTP hidden rows");
        executor_.project_rows(weights.eh_projection, mtp_eh_input_, hidden_, rows);

        check(cudaMemcpyAsync(mtp_residual_, hidden_,
                              static_cast<std::size_t>(rows) * half_bytes,
                              cudaMemcpyDeviceToDevice, stream),
              "save MTP attention residual");
        check(cuda::rmsnorm(hidden_, layer.input_norm, normalized_, rows, kHidden,
                            kNormEpsilon, stream),
              "MTP input RMSNorm");
        run_attention_rows(layer, layer_index, rows, start_position, false);
        check(cuda::rmsnorm(hidden_, layer.post_attention_norm, normalized_, rows,
                            kHidden, kNormEpsilon, stream),
              "MTP post-attention RMSNorm");
        check(cuda::add_inplace(mtp_residual_, normalized_, rows * kHidden, stream),
              "MTP attention residual add");

        check(cuda::rmsnorm(mtp_residual_, layer.pre_mlp_norm, normalized_, rows,
                            kHidden, kNormEpsilon, stream),
              "MTP pre-MLP RMSNorm");
        if (rows == 1) run_moe(layer, layer_index);
        else run_moe_rows(layer, layer_index, rows);
        check(cuda::rmsnorm(hidden_, layer.post_mlp_norm, normalized_, rows,
                            kHidden, kNormEpsilon, stream),
              "MTP post-MLP RMSNorm");
        check(cuda::add_inplace(mtp_residual_, normalized_, rows * kHidden, stream),
              "MTP MLP residual add");

        if (!compute_token) return -1;
        const BFloat16* final_hidden = mtp_residual_ +
            static_cast<std::size_t>(rows - 1) * kHidden;
        check(cuda::rmsnorm(final_hidden, weights.head_norm, normalized_, 1, kHidden,
                            kNormEpsilon, stream),
              "MTP shared-head RMSNorm");
        executor_.project(language_head_, normalized_, logits_);
        check(cuda::argmax(logits_, kVocabulary, argmax_, stream), "MTP greedy argmax");
        std::int32_t token = -1;
        check(cudaMemcpyAsync(&token, argmax_, sizeof(token), cudaMemcpyDeviceToHost, stream),
              "read MTP greedy token");
        executor_.synchronize();
        if (token < 0 || token >= kVocabulary) {
            throw std::runtime_error("invalid MTP greedy token");
        }
        return token;
    }

    void catch_up_mtp(std::span<const std::int32_t> tokens,
                      const BFloat16* target_hidden_rows,
                      std::uint32_t start_position) {
        if (!mtp_enabled_ || tokens.empty() || target_hidden_rows == nullptr) return;
        const int rows = static_cast<int>(tokens.size());
        const cudaStream_t stream = executor_.stream();
        check(cudaMemcpyAsync(previous_hidden_rows_, last_target_hidden_,
                              kHidden * sizeof(BFloat16), cudaMemcpyDeviceToDevice, stream),
              "seed shifted MTP hidden rows");
        if (rows > 1) {
            check(cudaMemcpyAsync(
                      previous_hidden_rows_ + kHidden, target_hidden_rows,
                      static_cast<std::size_t>(rows - 1) * kHidden * sizeof(BFloat16),
                      cudaMemcpyDeviceToDevice, stream),
                  "shift target hidden rows for MTP");
        }
        for (int stage = 0; stage < kMtpLayers; ++stage) {
            static_cast<void>(run_mtp_stage_rows(
                stage, tokens, previous_hidden_rows_, start_position, false));
        }
        check(cudaMemcpyAsync(
                  last_target_hidden_,
                  target_hidden_rows + static_cast<std::size_t>(rows - 1) * kHidden,
                  kHidden * sizeof(BFloat16), cudaMemcpyDeviceToDevice, stream),
              "retain last target hidden row");
    }

    void prepare_sinks() {
        const cudaStream_t stream = executor_.stream();
        for (int index = 0; index < active_layer_count(); ++index) {
            const std::string name = "model.layers." + std::to_string(index) +
                                     ".self_attn.param_sink_compressed_kv";
            const BFloat16* source = bf16(name);
            BFloat16* destination = sink_kv_ +
                static_cast<std::size_t>(index) * kSinks * kKvLora;
            check(cuda::rmsnorm(source, layers_[index].kv_norm, destination,
                                kSinks, kKvLora, kNormEpsilon, stream),
                  "normalize learned attention sinks");
        }
        executor_.synchronize();
    }

    std::int32_t prefill_block(std::span<const std::int32_t> tokens,
                               std::span<std::int32_t> row_predictions,
                               bool update_mtp) {
        const int rows = static_cast<int>(tokens.size());
        if (rows <= 0 || rows > kMaximumBlockRows) {
            throw std::invalid_argument("prefill block rows must be in [1, 64]");
        }
        if (!row_predictions.empty() && row_predictions.size() != tokens.size()) {
            throw std::invalid_argument("target row-prediction count does not match block");
        }
        const cudaStream_t stream = executor_.stream();
        const Tensor& embedding = auxiliary_.catalog().find("model.embed_tokens.weight");
        const auto* embedding_host = reinterpret_cast<const BFloat16*>(
            auxiliary_.catalog().data(embedding));
        for (int row = 0; row < rows; ++row) {
            const std::int32_t token = tokens[static_cast<std::size_t>(row)];
            if (token < 0 || token >= kVocabulary) {
                throw std::invalid_argument("prefill token outside vocabulary");
            }
            check(cudaMemcpyAsync(
                      embedding_ + static_cast<std::size_t>(row) * kHidden,
                      embedding_host + static_cast<std::size_t>(token) * kHidden,
                      kHidden * sizeof(BFloat16), cudaMemcpyHostToDevice, stream),
                  "upload prefill token embedding");
        }
        check(cuda::repeat_streams(embedding_, streams_a_, rows, kHidden, kStreams, stream),
              "repeat prefill mHC streams");
        BFloat16* current_streams = streams_a_;
        BFloat16* other_streams = streams_b_;
        for (int layer_index = 0; layer_index < kTargetLayers; ++layer_index) {
            LayerWeights& layer = layers_.at(layer_index);
            BFloat16* const residual_attention = current_streams;
            check(cuda::mhc_pre_parallel(residual_attention, layer.attention_mhc.phi,
                                         layer.attention_mhc.gamma, layer.attention_mhc.alpha,
                                         layer.attention_mhc.beta, mixed_, h_post_, h_res_,
                                         mhc_inverse_, mhc_mixes_, rows, kNormEpsilon, stream),
                  "prefill attention mHC pre");
            check(cuda::rmsnorm(mixed_, layer.input_norm, normalized_, rows, kHidden,
                                kNormEpsilon, stream),
                  "prefill input RMSNorm");
            run_attention_rows(
                layer, layer_index, rows, position_,
                !update_mtp && !row_predictions.empty());
            check(cuda::rmsnorm(hidden_, layer.post_attention_norm, normalized_, rows,
                                kHidden, kNormEpsilon, stream),
                  "prefill post-attention RMSNorm");
            check(cuda::mhc_post(normalized_, residual_attention, h_post_, h_res_,
                                 other_streams, rows, stream),
                  "prefill attention mHC post");

            BFloat16* const residual_mlp = other_streams;
            check(cuda::mhc_pre_parallel(residual_mlp, layer.mlp_mhc.phi, layer.mlp_mhc.gamma,
                                         layer.mlp_mhc.alpha, layer.mlp_mhc.beta,
                                         mixed_, h_post_, h_res_, mhc_inverse_, mhc_mixes_,
                                         rows, kNormEpsilon, stream),
                  "prefill MLP mHC pre");
            check(cuda::rmsnorm(mixed_, layer.pre_mlp_norm, normalized_, rows,
                                kHidden, kNormEpsilon, stream),
                  "prefill pre-MLP RMSNorm");
            if (layer.dense) run_dense_rows(layer.dense_mlp, rows);
            else run_moe_rows(layer, layer_index, rows);
            check(cuda::rmsnorm(hidden_, layer.post_mlp_norm, normalized_, rows,
                                kHidden, kNormEpsilon, stream),
                  "prefill post-MLP RMSNorm");
            check(cuda::mhc_post(normalized_, residual_mlp, h_post_, h_res_,
                                 current_streams, rows, stream),
                  "prefill MLP mHC post");
            if (layer.block_post_norm != nullptr) {
                check(cuda::rmsnorm(current_streams, layer.block_post_norm, other_streams,
                                    rows, kFlat, kNormEpsilon, stream),
                      "prefill block-post RMSNorm");
                std::swap(current_streams, other_streams);
            }
        }
        check(cuda::mhc_merge(current_streams, merge_.phi, merge_.gamma,
                              merge_.alpha, merge_.beta, hidden_, rows,
                              kNormEpsilon, stream),
              "prefill final mHC merge");
        const BFloat16* target_rows = hidden_;
        if (mtp_enabled_) {
            check(cudaMemcpyAsync(
                      target_hidden_rows_, hidden_,
                      static_cast<std::size_t>(rows) * kHidden * sizeof(BFloat16),
                      cudaMemcpyDeviceToDevice, stream),
                  "retain prefill target hidden rows for MTP");
            target_rows = target_hidden_rows_;
            if (update_mtp) catch_up_mtp(tokens, target_hidden_rows_, position_);
        }
        std::int32_t prediction = -1;
        if (row_predictions.empty()) {
            const BFloat16* last_hidden = target_rows +
                static_cast<std::size_t>(rows - 1) * kHidden;
            check(cuda::rmsnorm(last_hidden, output_norm_, normalized_, 1, kHidden,
                                kNormEpsilon, stream),
                  "prefill final RMSNorm");
            executor_.project(language_head_, normalized_, logits_);
            check(cuda::argmax(logits_, kVocabulary, argmax_, stream),
                  "prefill greedy argmax");
            check(cudaMemcpyAsync(&prediction, argmax_, sizeof(prediction),
                                  cudaMemcpyDeviceToHost, stream),
                  "read prefill greedy token");
        } else {
            check(cuda::rmsnorm(target_rows, output_norm_, normalized_, rows, kHidden,
                                kNormEpsilon, stream),
                  "verify-block output RMSNorm");
            executor_.project_rows(language_head_, normalized_, block_logits_, rows);
            check(cuda::argmax_rows(block_logits_, rows, kVocabulary, block_argmax_, stream),
                  "verify-block greedy argmax");
            check(cudaMemcpyAsync(row_predictions.data(), block_argmax_,
                                  static_cast<std::size_t>(rows) * sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost, stream),
                  "read verify-block greedy tokens");
        }
        executor_.synchronize();
        if (!row_predictions.empty()) prediction = row_predictions.back();
        if (prediction < 0 || prediction >= kVocabulary) {
            throw std::runtime_error("invalid prefill greedy token");
        }
        position_ += static_cast<std::uint32_t>(rows);
        return prediction;
    }

    void run_attention_rows(LayerWeights& layer,
                            int layer_index,
                            int rows,
                            std::uint32_t start_position,
                            bool capture_mome) {
        const cudaStream_t stream = executor_.stream();
        BFloat16* state = mome_state_ +
            static_cast<std::size_t>(layer_index) * kMomeStatePerLayer;
        BFloat16* q_state = state;
        BFloat16* kv_state = q_state + 2 * kQueryLora;
        BFloat16* o_state = kv_state + 2 * kKvLora;
        BFloat16* q_state_rows = nullptr;
        BFloat16* kv_state_rows = nullptr;
        BFloat16* o_state_rows = nullptr;
        if (capture_mome) {
            if (layer_index < 0 || layer_index >= kTargetLayers ||
                rows > kMaximumSpecRows) {
                throw std::invalid_argument("invalid verifier MoME capture");
            }
            BFloat16* layer_rows = verify_mome_rows_ +
                static_cast<std::size_t>(layer_index) * kMaximumSpecRows *
                    kMomeStatePerLayer;
            q_state_rows = layer_rows;
            kv_state_rows = q_state_rows +
                static_cast<std::size_t>(kMaximumSpecRows) * 2 * kQueryLora;
            o_state_rows = kv_state_rows +
                static_cast<std::size_t>(kMaximumSpecRows) * 2 * kKvLora;
        }
        executor_.project_rows(layer.q_a, normalized_, query_lora_, rows);
        check(capture_mome
                  ? cuda::mome_conv3_rows_strided_capture(
                        query_lora_, kQueryLora, layer.q_conv, q_state,
                        q_state_rows, rows, kQueryLora, stream)
                  : cuda::mome_conv3_rows(
                        query_lora_, layer.q_conv, q_state, rows, kQueryLora, stream),
              "prefill q MoME");
        check(cuda::rmsnorm(query_lora_, layer.q_norm, query_norm_, rows,
                            kQueryLora, kNormEpsilon, stream),
              "prefill q RMSNorm");
        executor_.project_rows(layer.q_b, query_norm_, query_up_, rows);
        check(cuda::bf16_gemm(normalized_, layer.kv_a, kv_down_, rows,
                              kKvLora + kRope, kHidden, stream),
              "prefill BF16 kv-a projection");
        check(capture_mome
                  ? cuda::mome_conv3_rows_strided_capture(
                        kv_down_, kKvLora + kRope, layer.kv_conv, kv_state,
                        kv_state_rows, rows, kKvLora, stream)
                  : cuda::mome_conv3_rows_strided(
                        kv_down_, kKvLora + kRope, layer.kv_conv, kv_state,
                        rows, kKvLora, stream),
              "prefill kv MoME");
        check(cuda::rmsnorm_strided(kv_down_, kKvLora + kRope, layer.kv_norm,
                                    block_cache_kv_, kKvLora, rows, kKvLora,
                                    kNormEpsilon, stream),
              "prefill kv RMSNorm");
        check(cuda::mla_prepare_query_rows(
                  query_up_, layer.kv_b, q_absorbed_, q_rope_, rows,
                  static_cast<int>(start_position), 6400000.0F, stream),
              "prepare prefill MLA queries");
        check(cuda::mla_prepare_key_rope_rows(kv_down_, block_cache_rope_, rows,
                                               static_cast<int>(start_position),
                                               6400000.0F, stream),
              "prepare prefill MLA key RoPE");

        const std::size_t cache_offset = cache_offsets_.at(layer_index);
        const std::uint32_t cache_capacity = cache_capacities_.at(layer_index);
        BFloat16* layer_cache_kv = cache_kv_ + cache_offset * kKvLora;
        BFloat16* layer_cache_rope = cache_rope_ + cache_offset * kRope;
        const BFloat16* layer_sinks = sink_kv_ +
            static_cast<std::size_t>(layer_index) * kSinks * kKvLora;
        const float attention_scale = 1.0F / std::sqrt(192.0F);
        if (layer.sliding) {
            if (start_position + static_cast<std::uint32_t>(rows) <= cache_capacity) {
                check(cuda::mla_store_cache_rows(
                          layer_cache_kv, layer_cache_rope,
                          block_cache_kv_, block_cache_rope_,
                          static_cast<int>(start_position), rows,
                          static_cast<int>(cache_capacity), stream),
                      "store prefill sliding cache rows");
                check(cuda::mla_attention_prefill_rows_tiled(
                          q_absorbed_, q_rope_, layer_sinks, layer.sink_rope,
                          layer_cache_kv, layer_cache_rope,
                          static_cast<int>(start_position), rows,
                          latent_output_, attention_scale, stream),
                      "prefill fused-row sliding-window MLA attention");
            } else {
                for (int row = 0; row < rows; ++row) {
                    const std::uint32_t absolute =
                        start_position + static_cast<std::uint32_t>(row);
                    const std::uint32_t slot = absolute % cache_capacity;
                    check(cudaMemcpyAsync(
                              layer_cache_kv + static_cast<std::size_t>(slot) * kKvLora,
                              block_cache_kv_ + static_cast<std::size_t>(row) * kKvLora,
                              kKvLora * sizeof(BFloat16), cudaMemcpyDeviceToDevice, stream),
                          "store wrapped sliding KV");
                    check(cudaMemcpyAsync(
                              layer_cache_rope + static_cast<std::size_t>(slot) * kRope,
                              block_cache_rope_ + static_cast<std::size_t>(row) * kRope,
                              kRope * sizeof(BFloat16), cudaMemcpyDeviceToDevice, stream),
                          "store wrapped sliding RoPE");
                    const std::uint32_t count = std::min<std::uint32_t>(
                        absolute + 1, cache_capacity);
                    const std::uint32_t start =
                        (absolute + 1 - count) % cache_capacity;
                    check(cuda::mla_attention_decode_circular_tiled(
                              q_absorbed_ + static_cast<std::size_t>(row) * kLatentHeads,
                              q_rope_ + static_cast<std::size_t>(row) * 48 * kRope,
                              layer_sinks, layer.sink_rope, layer_cache_kv, layer_cache_rope,
                              static_cast<int>(start), static_cast<int>(count),
                              static_cast<int>(cache_capacity),
                              latent_output_ + static_cast<std::size_t>(row) * kLatentHeads,
                              attention_scale, stream),
                          "prefill sliding-window MLA attention");
                }
            }
        } else {
            check(cuda::mla_store_cache_rows(
                      layer_cache_kv, layer_cache_rope,
                      block_cache_kv_, block_cache_rope_,
                      static_cast<int>(start_position), rows,
                      static_cast<int>(cache_capacity), stream),
                  "store prefill DSA cache rows");
            executor_.project_rows(layer.index_query, query_norm_, index_query_, rows);
            executor_.project_rows(layer.index_key, normalized_, index_key_, rows);
            check(cuda::rmsnorm(index_key_, layer.index_key_norm, index_key_, rows,
                                kDsaHead, kNormEpsilon, stream),
                  "prefill DSA key RMSNorm");
            check(cuda::bf16_gemm(normalized_, layer.index_weights, index_weights_,
                                  rows, kDsaHeads, kHidden, stream),
                  "prefill DSA head-weight projection");
            check(cuda::dsa_rope_rows(index_query_, index_key_, rows,
                                      static_cast<int>(start_position), 6400000.0F, stream),
                  "prefill DSA query/key RoPE");
            BFloat16* layer_index_cache = index_cache_ +
                index_cache_offsets_.at(layer_index) * kDsaHead;
            check(cudaMemcpyAsync(
                      layer_index_cache + static_cast<std::size_t>(start_position) * kDsaHead,
                      index_key_, static_cast<std::size_t>(rows) * kDsaHead * sizeof(BFloat16),
                      cudaMemcpyDeviceToDevice, stream),
                  "store prefill DSA index keys");
            if (start_position + static_cast<std::uint32_t>(rows) <= kDsaTopK) {
                check(cuda::mla_attention_prefill_rows_tiled(
                          q_absorbed_, q_rope_, layer_sinks, layer.sink_rope,
                          layer_cache_kv, layer_cache_rope, static_cast<int>(start_position),
                          rows, latent_output_, attention_scale, stream),
                      "prefill dense MLA before DSA threshold");
            } else {
                for (int row = 0; row < rows; ++row) {
                    const int positions = static_cast<int>(start_position) + row + 1;
                    BFloat16* row_output = latent_output_ +
                        static_cast<std::size_t>(row) * kLatentHeads;
                    if (positions <= kDsaTopK) {
                        check(cuda::mla_attention_decode_tiled(
                                  q_absorbed_ + static_cast<std::size_t>(row) * kLatentHeads,
                                  q_rope_ + static_cast<std::size_t>(row) * 48 * kRope,
                                  layer_sinks, layer.sink_rope, layer_cache_kv, layer_cache_rope,
                                  0, positions, row_output, attention_scale, stream),
                              "prefill dense MLA at DSA boundary");
                    } else {
                        check(cuda::dsa_select(
                                  index_query_ +
                                      static_cast<std::size_t>(row) * kDsaHeads * kDsaHead,
                                  index_weights_ + static_cast<std::size_t>(row) * kDsaHeads,
                                  layer_index_cache, positions,
                                  dsa_scores_in_, dsa_scores_out_,
                                  dsa_indices_in_, dsa_indices_out_,
                                  dsa_sort_workspace_, dsa_sort_workspace_bytes_, stream),
                              "prefill DSA top-k selection");
                        check(cuda::mla_attention_decode_indexed_tiled(
                                  q_absorbed_ + static_cast<std::size_t>(row) * kLatentHeads,
                                  q_rope_ + static_cast<std::size_t>(row) * 48 * kRope,
                                  layer_sinks, layer.sink_rope, layer_cache_kv, layer_cache_rope,
                                  dsa_indices_out_, kDsaTopK, row_output,
                                  attention_scale, stream),
                              "prefill indexed DSA latent MLA attention");
                    }
                }
            }
        }
        if (rows >= 4) {
            check(cuda::mla_value_up_rows_tensorcore(
                      bf16_mla_plan_, latent_output_, layer.kv_b,
                      mla_bf16_input_, mla_bf16_output_, attention_output_, rows, stream),
                  "tensor-core prefill MLA value up");
        } else {
            check(cuda::mla_value_up_rows(
                      latent_output_, layer.kv_b, attention_output_, rows, stream),
                  "prefill MLA value up");
        }
        check(capture_mome
                  ? cuda::mome_conv3_rows_strided_capture(
                        attention_output_, kAttentionOutput, layer.o_conv, o_state,
                        o_state_rows, rows, kAttentionOutput, stream)
                  : cuda::mome_conv3_rows(
                        attention_output_, layer.o_conv, o_state,
                        rows, kAttentionOutput, stream),
              "prefill output MoME");
        executor_.project_rows(layer.o, attention_output_, hidden_, rows);
    }

    void run_dense_rows(const ExpertWeights& weights, int rows) {
        const cudaStream_t stream = executor_.stream();
        executor_.project_rows(weights.gate, normalized_, gate_output_, rows);
        executor_.project_rows(weights.up, normalized_, up_output_, rows);
        check(cuda::swiglu(gate_output_, up_output_, gate_output_,
                           rows * kDenseIntermediate, stream),
              "prefill dense SwiGLU");
        executor_.project_rows(weights.down, gate_output_, hidden_, rows);
    }

    void run_moe_rows(LayerWeights& layer, int layer_index, int rows) {
        const cudaStream_t stream = executor_.stream();
        check(cuda::bf16_gemm_f32(normalized_, layer.router, router_logits_,
                                  rows, kExperts, kHidden, stream),
              "prefill MoE router projection");
        check(cuda::router_top8_rows(router_logits_, layer.router_correction,
                                     expert_ids_, expert_weights_, rows, 2.5F, stream),
              "prefill MoE router top8");
        if (layer.expert_bank != nullptr && rows <= kMaximumSpecRows) {
            executor_.project_expert_gate_up(
                layer.expert_bank, expert_ids_, normalized_, gate_output_, up_output_,
                kHidden, kExpertIntermediate, rows);
            check(cuda::swiglu(
                      gate_output_, up_output_, gate_output_,
                      rows * kSelectedExperts * kExpertIntermediate, stream),
                  "device-grouped prefill expert SwiGLU");
            executor_.project_expert_down(
                layer.expert_bank, expert_ids_, gate_output_, mixed_,
                kHidden, kExpertIntermediate, rows);
            check(cuda::expert_combine_rows(
                      mixed_, expert_weights_, hidden_, rows,
                      kSelectedExperts, kHidden, stream),
                  "device-grouped prefill expert combine");
        } else {
            std::array<std::int32_t, kMaximumBlockRows * kSelectedExperts> host_ids {};
            std::array<float, kMaximumBlockRows * kSelectedExperts> host_weights {};
            check(cudaMemcpyAsync(host_ids.data(), expert_ids_,
                                  static_cast<std::size_t>(rows) * kSelectedExperts *
                                      sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost, stream),
                  "read prefill routed expert ids");
            check(cudaMemcpyAsync(host_weights.data(), expert_weights_,
                                  static_cast<std::size_t>(rows) * kSelectedExperts * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream),
                  "read prefill routed expert weights");
            executor_.synchronize();
            check(cudaMemsetAsync(hidden_, 0,
                                  static_cast<std::size_t>(rows) * kHidden * sizeof(BFloat16),
                                  stream),
                  "clear prefill MoE output");
            std::array<std::int32_t, kMaximumBlockRows> selected_rows {};
            std::array<float, kMaximumBlockRows> selected_scales {};
            for (int expert_id = 0; expert_id < kExperts; ++expert_id) {
                int selected_count = 0;
                for (int row = 0; row < rows; ++row) {
                    for (int slot = 0; slot < kSelectedExperts; ++slot) {
                        const int index = row * kSelectedExperts + slot;
                        if (host_ids[static_cast<std::size_t>(index)] == expert_id) {
                            selected_rows[static_cast<std::size_t>(selected_count)] = row;
                            selected_scales[static_cast<std::size_t>(selected_count)] =
                                host_weights[static_cast<std::size_t>(index)];
                            ++selected_count;
                        }
                    }
                }
                if (selected_count == 0) continue;
                check(cudaMemcpyAsync(group_rows_, selected_rows.data(),
                                      static_cast<std::size_t>(selected_count) * sizeof(std::int32_t),
                                      cudaMemcpyHostToDevice, stream),
                      "upload grouped expert rows");
                check(cudaMemcpyAsync(group_scales_, selected_scales.data(),
                                      static_cast<std::size_t>(selected_count) * sizeof(float),
                                      cudaMemcpyHostToDevice, stream),
                      "upload grouped expert scales");
                check(cuda::gather_rows(normalized_, group_rows_, mixed_,
                                        selected_count, kHidden, stream),
                      "gather grouped expert input");
                ExpertWeights& selected = expert(layer, layer_index, expert_id);
                executor_.project_rows(selected.gate, mixed_, gate_output_, selected_count);
                executor_.project_rows(selected.up, mixed_, up_output_, selected_count);
                check(cuda::swiglu(gate_output_, up_output_, gate_output_,
                                   selected_count * kExpertIntermediate, stream),
                      "prefill expert SwiGLU");
                executor_.project_rows(selected.down, gate_output_, mixed_, selected_count);
                check(cuda::scatter_add_scaled_rows(hidden_, mixed_, group_rows_, group_scales_,
                                                    selected_count, kHidden, stream),
                      "scatter grouped expert output");
            }
        }
        executor_.project_rows(layer.shared.gate, normalized_, gate_output_, rows);
        executor_.project_rows(layer.shared.up, normalized_, up_output_, rows);
        check(cuda::swiglu(gate_output_, up_output_, gate_output_,
                           rows * kExpertIntermediate, stream),
              "prefill shared expert SwiGLU");
        executor_.project_rows(layer.shared.down, gate_output_, mixed_, rows);
        check(cuda::add_inplace(hidden_, mixed_, rows * kHidden, stream),
              "combine prefill shared expert");
    }

    void run_attention(LayerWeights& layer, int layer_index) {
        const cudaStream_t stream = executor_.stream();
        BFloat16* state = mome_state_ + static_cast<std::size_t>(layer_index) * kMomeStatePerLayer;
        BFloat16* q_state = state;
        BFloat16* kv_state = q_state + 2 * kQueryLora;
        BFloat16* o_state = kv_state + 2 * kKvLora;
        executor_.project(layer.q_a, normalized_, query_lora_);
        check(cuda::mome_conv3_step(query_lora_, layer.q_conv, q_state, kQueryLora, stream),
              "q MoME");
        check(cuda::rmsnorm(query_lora_, layer.q_norm, query_norm_, 1, kQueryLora,
                            kNormEpsilon, stream),
              "q RMSNorm");
        executor_.project(layer.q_b, query_norm_, query_up_);
        check(cuda::bf16_gemv(normalized_, layer.kv_a, kv_down_, kKvLora + kRope,
                              kHidden, stream),
              "BF16 kv-a projection");
        check(cuda::mome_conv3_step(kv_down_, layer.kv_conv, kv_state, kKvLora, stream),
              "kv MoME");
        const std::size_t cache_offset = cache_offsets_.at(layer_index);
        const std::uint32_t cache_capacity = cache_capacities_.at(layer_index);
        const std::uint32_t cache_slot = layer.sliding ? position_ % cache_capacity : position_;
        BFloat16* layer_cache_kv = cache_kv_ + cache_offset * kKvLora;
        BFloat16* layer_cache_rope = cache_rope_ + cache_offset * kRope;
        BFloat16* current_kv = layer_cache_kv + static_cast<std::size_t>(cache_slot) * kKvLora;
        BFloat16* current_rope = layer_cache_rope + static_cast<std::size_t>(cache_slot) * kRope;
        check(cuda::rmsnorm(kv_down_, layer.kv_norm, current_kv, 1, kKvLora,
                            kNormEpsilon, stream),
              "kv RMSNorm");
        check(cuda::mla_prepare_query(query_up_, layer.kv_b, q_absorbed_, q_rope_,
                                      static_cast<int>(position_), 6400000.0F, stream),
              "prepare MLA query");
        check(cuda::mla_prepare_key_rope(kv_down_, current_rope,
                                         static_cast<int>(position_), 6400000.0F, stream),
              "prepare MLA key rope");
        const BFloat16* layer_sinks = sink_kv_ +
            static_cast<std::size_t>(layer_index) * kSinks * kKvLora;
        const float attention_scale = 1.0F / std::sqrt(192.0F);
        if (layer.sliding) {
            const std::uint32_t cache_count = std::min<std::uint32_t>(position_ + 1,
                                                                      cache_capacity);
            const std::uint32_t cache_start = (position_ + 1 - cache_count) % cache_capacity;
            check(cuda::mla_attention_decode_circular_tiled(
                      q_absorbed_, q_rope_, layer_sinks, layer.sink_rope,
                      layer_cache_kv, layer_cache_rope,
                      static_cast<int>(cache_start), static_cast<int>(cache_count),
                      static_cast<int>(cache_capacity), latent_output_, attention_scale, stream),
                  "sliding-window latent MLA attention");
        } else {
            executor_.project(layer.index_query, query_norm_, index_query_);
            executor_.project(layer.index_key, normalized_, index_key_);
            check(cuda::rmsnorm(index_key_, layer.index_key_norm, index_key_, 1,
                                kDsaHead, kNormEpsilon, stream),
                  "DSA key RMSNorm");
            check(cuda::bf16_gemv(normalized_, layer.index_weights, index_weights_,
                                  kDsaHeads, kHidden, stream),
                  "DSA head-weight projection");
            check(cuda::dsa_rope(index_query_, index_key_, static_cast<int>(position_),
                                 6400000.0F, stream),
                  "DSA query/key RoPE");
            BFloat16* layer_index_cache = index_cache_ +
                index_cache_offsets_.at(layer_index) * kDsaHead;
            check(cudaMemcpyAsync(layer_index_cache + static_cast<std::size_t>(position_) * kDsaHead,
                                  index_key_, kDsaHead * sizeof(BFloat16),
                                  cudaMemcpyDeviceToDevice, stream),
                  "store DSA key");
            const int positions = static_cast<int>(position_ + 1);
            if (positions <= kDsaTopK) {
                check(cuda::mla_attention_decode_tiled(
                          q_absorbed_, q_rope_, layer_sinks, layer.sink_rope,
                          layer_cache_kv, layer_cache_rope, 0, positions,
                          latent_output_, attention_scale, stream),
                      "dense latent MLA attention before DSA threshold");
            } else {
                check(cuda::dsa_select(
                          index_query_, index_weights_, layer_index_cache, positions,
                          dsa_scores_in_, dsa_scores_out_, dsa_indices_in_, dsa_indices_out_,
                          dsa_sort_workspace_, dsa_sort_workspace_bytes_, stream),
                      "DSA top-k selection");
                check(cuda::mla_attention_decode_indexed_tiled(
                          q_absorbed_, q_rope_, layer_sinks, layer.sink_rope,
                          layer_cache_kv, layer_cache_rope, dsa_indices_out_, kDsaTopK,
                          latent_output_, attention_scale, stream),
                      "indexed DSA latent MLA attention");
            }
        }
        check(cuda::mla_value_up(latent_output_, layer.kv_b, attention_output_, stream),
              "MLA value up");
        check(cuda::mome_conv3_step(attention_output_, layer.o_conv, o_state,
                                    kAttentionOutput, stream),
              "output MoME");
        executor_.project(layer.o, attention_output_, hidden_);
    }

    void run_dense(const ExpertWeights& weights) {
        const cudaStream_t stream = executor_.stream();
        executor_.project(weights.gate, normalized_, gate_output_);
        executor_.project(weights.up, normalized_, up_output_);
        check(cuda::swiglu(gate_output_, up_output_, gate_output_, kDenseIntermediate, stream),
              "dense SwiGLU");
        executor_.project(weights.down, gate_output_, hidden_);
    }

    void run_moe(LayerWeights& layer, int layer_index) {
        const cudaStream_t stream = executor_.stream();
        check(cuda::bf16_gemv_f32(normalized_, layer.router, router_logits_,
                                  kExperts, kHidden, stream),
              "MoE router projection");
        check(cuda::router_top8(router_logits_, layer.router_correction, expert_ids_,
                                expert_weights_, 2.5F, stream),
              "MoE router top8");
        if (layer.expert_bank != nullptr) {
            executor_.project_expert_gate_up(
                layer.expert_bank, expert_ids_, normalized_, gate_output_, up_output_,
                kHidden, kExpertIntermediate);
            check(cuda::swiglu(
                      gate_output_, up_output_, gate_output_,
                      kSelectedExperts * kExpertIntermediate, stream),
                  "resident grouped expert SwiGLU");
            executor_.project_expert_down(
                layer.expert_bank, expert_ids_, gate_output_, mixed_,
                kHidden, kExpertIntermediate);
            check(cuda::expert_combine(
                      mixed_, expert_weights_, hidden_, kSelectedExperts, kHidden, stream),
                  "resident grouped expert combine");
        } else {
            std::array<std::int32_t, kSelectedExperts> ids {};
            check(cudaMemcpyAsync(ids.data(), expert_ids_, sizeof(ids),
                                  cudaMemcpyDeviceToHost, stream),
                  "read routed expert ids");
            executor_.synchronize();
            check(cudaMemsetAsync(hidden_, 0, kHidden * sizeof(BFloat16), stream),
                  "clear MoE output");
            for (int slot = 0; slot < kSelectedExperts; ++slot) {
                if (ids[slot] < 0 || ids[slot] >= kExperts) {
                    throw std::runtime_error("router produced invalid expert id");
                }
                ExpertWeights& selected = expert(layer, layer_index, ids[slot]);
                executor_.project(selected.gate, normalized_, gate_output_);
                executor_.project(selected.up, normalized_, up_output_);
                check(cuda::swiglu(
                          gate_output_, up_output_, gate_output_, kExpertIntermediate, stream),
                      "expert SwiGLU");
                executor_.project(selected.down, gate_output_, mixed_);
                check(cuda::add_scaled_inplace(hidden_, mixed_, expert_weights_ + slot,
                                               kHidden, stream),
                      "combine routed expert");
            }
        }
        executor_.project(layer.shared.gate, normalized_, gate_output_);
        executor_.project(layer.shared.up, normalized_, up_output_);
        check(cuda::swiglu(gate_output_, up_output_, gate_output_, kExpertIntermediate, stream),
              "shared expert SwiGLU");
        executor_.project(layer.shared.down, gate_output_, mixed_);
        check(cuda::add_inplace(hidden_, mixed_, kHidden, stream), "combine shared expert");
    }

    std::uint32_t max_context_ = 0;
    WeightMode mode_ = WeightMode::selective;
    bool mtp_enabled_ = false;
    std::uint32_t position_ = 0;
    AuxiliaryWeights auxiliary_;
    ResidentWeights resident_;
    SelectiveWeights selective_;
    ProjectionExecutor executor_;
    std::vector<LayerWeights> layers_;
    std::array<MtpWeights, kMtpLayers> mtp_;
    MhcWeights merge_;
    const BFloat16* output_norm_ = nullptr;
    ResidentProjection language_head_;
    cuda::Nvfp4ExpertProjection* expert_banks_ = nullptr;
    std::vector<void*> allocations_;
    std::unordered_map<std::string, StateSnapshot> state_snapshots_;
    std::vector<std::size_t> cache_offsets_;
    std::vector<std::uint32_t> cache_capacities_;
    std::vector<std::size_t> index_cache_offsets_;
    std::size_t cache_slots_ = 0;
    std::size_t index_cache_slots_ = 0;
    std::size_t dsa_sort_workspace_bytes_ = 0;

    BFloat16* embedding_ = nullptr;
    BFloat16* streams_a_ = nullptr;
    BFloat16* streams_b_ = nullptr;
    BFloat16* mixed_ = nullptr;
    BFloat16* normalized_ = nullptr;
    BFloat16* hidden_ = nullptr;
    float* h_post_ = nullptr;
    float* h_res_ = nullptr;
    float* mhc_inverse_ = nullptr;
    float* mhc_mixes_ = nullptr;
    BFloat16* query_lora_ = nullptr;
    BFloat16* query_norm_ = nullptr;
    BFloat16* query_up_ = nullptr;
    BFloat16* kv_down_ = nullptr;
    BFloat16* block_cache_kv_ = nullptr;
    BFloat16* block_cache_rope_ = nullptr;
    BFloat16* q_absorbed_ = nullptr;
    BFloat16* q_rope_ = nullptr;
    BFloat16* latent_output_ = nullptr;
    BFloat16* attention_output_ = nullptr;
    BFloat16* gate_output_ = nullptr;
    BFloat16* up_output_ = nullptr;
    float* router_logits_ = nullptr;
    std::int32_t* expert_ids_ = nullptr;
    float* expert_weights_ = nullptr;
    std::int32_t* group_rows_ = nullptr;
    float* group_scales_ = nullptr;
    std::int32_t* argmax_ = nullptr;
    BFloat16* logits_ = nullptr;
    BFloat16* sink_kv_ = nullptr;
    BFloat16* cache_kv_ = nullptr;
    BFloat16* cache_rope_ = nullptr;
    BFloat16* index_query_ = nullptr;
    BFloat16* index_key_ = nullptr;
    BFloat16* index_weights_ = nullptr;
    BFloat16* index_cache_ = nullptr;
    float* dsa_scores_in_ = nullptr;
    float* dsa_scores_out_ = nullptr;
    std::int32_t* dsa_indices_in_ = nullptr;
    std::int32_t* dsa_indices_out_ = nullptr;
    std::uint8_t* dsa_sort_workspace_ = nullptr;
    cuda::Bf16MlaPlan* bf16_mla_plan_ = nullptr;
    BFloat16* mla_bf16_input_ = nullptr;
    BFloat16* mla_bf16_output_ = nullptr;
    BFloat16* mome_state_ = nullptr;
    BFloat16* target_hidden_rows_ = nullptr;
    BFloat16* last_target_hidden_ = nullptr;
    BFloat16* previous_hidden_rows_ = nullptr;
    BFloat16* mtp_embedding_norm_ = nullptr;
    BFloat16* mtp_hidden_norm_ = nullptr;
    BFloat16* mtp_eh_input_ = nullptr;
    BFloat16* mtp_residual_ = nullptr;
    BFloat16* mtp_hidden_chain_ = nullptr;
    BFloat16* mtp_state_snapshot_ = nullptr;
    BFloat16* mtp_cache_kv_snapshot_ = nullptr;
    BFloat16* mtp_cache_rope_snapshot_ = nullptr;
    BFloat16* target_state_snapshot_ = nullptr;
    BFloat16* verify_mome_rows_ = nullptr;
    BFloat16* target_cache_kv_snapshot_ = nullptr;
    BFloat16* target_cache_rope_snapshot_ = nullptr;
    BFloat16* block_logits_ = nullptr;
    std::int32_t* block_argmax_ = nullptr;
};

Model::Model(const std::filesystem::path& checkpoint,
             const std::filesystem::path& nvfp4_artifact,
             std::uint32_t max_context,
             WeightMode mode,
             bool enable_mtp)
    : impl_(std::make_unique<Impl>(
          checkpoint, nvfp4_artifact, max_context, mode, enable_mtp)) {}

Model::~Model() = default;

std::int32_t Model::forward(std::int32_t token) { return impl_->forward(token); }
std::int32_t Model::prefill(std::span<const std::int32_t> tokens) {
    return impl_->prefill(tokens);
}
std::array<std::int32_t, 3> Model::mtp_propose(std::int32_t next_token) {
    return impl_->mtp_propose(next_token);
}
Model::SpeculativeResult Model::speculative_step(std::int32_t next_token) {
    return impl_->speculative_step(next_token);
}
void Model::capture_state(std::string_view key) { impl_->capture_state(key); }
void Model::restore_state(std::string_view key) { impl_->restore_state(key); }
void Model::release_state(std::string_view key) { impl_->release_state(key); }
void Model::reset() { impl_->reset(); }
std::uint32_t Model::position() const noexcept { return impl_->position(); }
std::uint64_t Model::projection_bytes() const noexcept { return impl_->projection_bytes(); }
std::uint64_t Model::auxiliary_bytes() const noexcept { return impl_->auxiliary_bytes(); }

}  // namespace p92
