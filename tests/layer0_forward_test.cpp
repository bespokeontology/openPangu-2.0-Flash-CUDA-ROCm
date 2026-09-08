#include "p92/auxiliary.h"
#include "p92/executor.h"
#include "p92/kernels.h"
#include "p92/resident.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kHidden = 2560;
constexpr int kStreams = 4;
constexpr int kFlat = kHidden * kStreams;
constexpr int kQueryLora = 1024;
constexpr int kKvLora = 512;
constexpr int kRope = 64;
constexpr int kQueryUp = 48 * (128 + 64);
constexpr int kLatentHeads = 48 * kKvLora;
constexpr int kAttentionOutput = 48 * 128;
constexpr int kIntermediate = 9216;
constexpr int kSinks = 128;
constexpr int kBosToken = 148899;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <class T>
class DeviceBuffer final {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)), "allocate layer buffer");
    }
    ~DeviceBuffer() { if (pointer_ != nullptr) cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const noexcept { return pointer_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }
private:
    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};

const std::uint16_t* bf16(p92::AuxiliaryWeights& weights, const std::string& name) {
    return weights.load_bf16(name).data;
}

void kernel_check(cudaError_t status, const char* operation) {
    check(status, operation);
}

std::uint64_t fnv1a(const std::vector<std::uint16_t>& values) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::uint16_t value : values) {
        hash ^= value & 0xffU;
        hash *= 1099511628211ULL;
        hash ^= value >> 8;
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: p92_layer0_forward_test CHECKPOINT NVFP4_ARTIFACT\n";
        return 2;
    }
    try {
        p92::AuxiliaryWeights auxiliary;
        auxiliary.open(argv[1]);
        p92::SelectiveWeights projections;
        projections.open(argv[2]);
        p92::ProjectionExecutor executor;
        const cudaStream_t stream = executor.stream();

        const p92::ResidentProjection q_a = projections.load(
            "model.layers.0.self_attn.q_a_proj.weight");
        const p92::ResidentProjection q_b = projections.load(
            "model.layers.0.self_attn.q_b_proj.weight");
        const p92::ResidentProjection o_proj = projections.load(
            "model.layers.0.self_attn.o_proj.weight");
        const p92::ResidentProjection gate = projections.load(
            "model.layers.0.mlp.gate_proj.weight");
        const p92::ResidentProjection up = projections.load(
            "model.layers.0.mlp.up_proj.weight");
        const p92::ResidentProjection down = projections.load(
            "model.layers.0.mlp.down_proj.weight");

        const std::uint16_t* kv_b_weight = bf16(auxiliary,
            "model.layers.0.self_attn.kv_b_proj.weight");
        const std::uint16_t* kv_a_weight = bf16(auxiliary,
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight");
        const std::uint16_t* sink_kv = bf16(auxiliary,
            "model.layers.0.self_attn.param_sink_compressed_kv");
        const std::uint16_t* sink_rope = bf16(auxiliary,
            "model.layers.0.self_attn.param_sink_k_pe");
        const std::uint16_t* kv_norm = bf16(auxiliary,
            "model.layers.0.self_attn.kv_a_layernorm.weight");

        const p92::Catalog& catalog = auxiliary.catalog();
        const p92::Tensor& embedding = catalog.find("model.embed_tokens.weight");
        const auto* embedding_host = reinterpret_cast<const std::uint16_t*>(catalog.data(embedding));
        if (embedding.shape.size() != 2 || embedding.shape[0] <= kBosToken || embedding.shape[1] != kHidden) {
            throw std::runtime_error("unexpected embedding contract");
        }

        const std::string layer = "model.layers.0.";
        const std::uint16_t* attn_phi = bf16(auxiliary, layer + "attn_mhc_module.phi.weight");
        const std::uint16_t* attn_gamma = bf16(auxiliary, layer + "attn_mhc_module.norm_gamma");
        const std::uint16_t* attn_alpha = bf16(auxiliary, layer + "attn_mhc_module.branch_alpha");
        const std::uint16_t* attn_beta = bf16(auxiliary, layer + "attn_mhc_module.branch_beta");
        const std::uint16_t* mlp_phi = bf16(auxiliary, layer + "mlp_mhc_module.phi.weight");
        const std::uint16_t* mlp_gamma = bf16(auxiliary, layer + "mlp_mhc_module.norm_gamma");
        const std::uint16_t* mlp_alpha = bf16(auxiliary, layer + "mlp_mhc_module.branch_alpha");
        const std::uint16_t* mlp_beta = bf16(auxiliary, layer + "mlp_mhc_module.branch_beta");
        const std::uint16_t* input_norm = bf16(auxiliary, layer + "input_layernorm.weight");
        const std::uint16_t* post_attention_norm = bf16(auxiliary, layer + "post_attention_layernorm.weight");
        const std::uint16_t* pre_mlp_norm = bf16(auxiliary, layer + "pre_mlp_layernorm.weight");
        const std::uint16_t* post_mlp_norm = bf16(auxiliary, layer + "post_mlp_layernorm.weight");
        const std::uint16_t* block_post_norm = bf16(auxiliary, layer + "block_post_layernorm.weight");
        const std::uint16_t* q_norm = bf16(auxiliary, layer + "self_attn.q_a_layernorm.weight");
        const std::uint16_t* q_conv = bf16(auxiliary, layer + "self_attn.qa_conv.weight");
        const std::uint16_t* kv_conv = bf16(auxiliary, layer + "self_attn.compresskv_conv.weight");
        const std::uint16_t* o_conv = bf16(auxiliary, layer + "self_attn.o_conv.weight");

        DeviceBuffer<std::uint16_t> d_embedding(kHidden);
        DeviceBuffer<std::uint16_t> streams_a(kFlat);
        DeviceBuffer<std::uint16_t> streams_b(kFlat);
        DeviceBuffer<std::uint16_t> mixed(kHidden);
        DeviceBuffer<std::uint16_t> normalized(kHidden);
        DeviceBuffer<std::uint16_t> hidden(kHidden);
        DeviceBuffer<float> h_post(kStreams);
        DeviceBuffer<float> h_res(kStreams * kStreams);
        DeviceBuffer<std::uint16_t> query_lora(kQueryLora);
        DeviceBuffer<std::uint16_t> query_norm(kQueryLora);
        DeviceBuffer<std::uint16_t> query_up(kQueryUp);
        DeviceBuffer<std::uint16_t> kv_down(kKvLora + kRope);
        DeviceBuffer<std::uint16_t> q_absorbed(kLatentHeads);
        DeviceBuffer<std::uint16_t> q_rope(48 * kRope);
        DeviceBuffer<std::uint16_t> key_rope(kRope);
        DeviceBuffer<std::uint16_t> cache_kv(kKvLora);
        DeviceBuffer<std::uint16_t> sink_kv_normalized(kSinks * kKvLora);
        DeviceBuffer<std::uint16_t> latent_output(kLatentHeads);
        DeviceBuffer<std::uint16_t> attention_output(kAttentionOutput);
        DeviceBuffer<std::uint16_t> q_state(kQueryLora * 2);
        DeviceBuffer<std::uint16_t> kv_state(kKvLora * 2);
        DeviceBuffer<std::uint16_t> o_state(kAttentionOutput * 2);
        DeviceBuffer<std::uint16_t> gate_output(kIntermediate);
        DeviceBuffer<std::uint16_t> up_output(kIntermediate);

        const auto run = [&]() {
            check(cudaMemcpyAsync(d_embedding.get(), embedding_host +
                                      static_cast<std::size_t>(kBosToken) * kHidden,
                                  d_embedding.bytes(), cudaMemcpyHostToDevice, stream),
                  "upload BOS embedding");
            check(cudaMemsetAsync(q_state.get(), 0, q_state.bytes(), stream), "clear q MoME state");
            check(cudaMemsetAsync(kv_state.get(), 0, kv_state.bytes(), stream), "clear kv MoME state");
            check(cudaMemsetAsync(o_state.get(), 0, o_state.bytes(), stream), "clear o MoME state");
            kernel_check(p92::cuda::repeat_streams(d_embedding.get(), streams_a.get(), 1,
                                                    kHidden, kStreams, stream),
                         "repeat mHC streams");

            kernel_check(p92::cuda::mhc_pre(streams_a.get(), attn_phi, attn_gamma,
                                             attn_alpha, attn_beta, mixed.get(),
                                             h_post.get(), h_res.get(), 1, 1.0e-5F, stream),
                         "attention mHC pre");
            kernel_check(p92::cuda::rmsnorm(mixed.get(), input_norm, normalized.get(),
                                             1, kHidden, 1.0e-5F, stream),
                         "input RMSNorm");
            executor.project(q_a, normalized.get(), query_lora.get());
            kernel_check(p92::cuda::mome_conv3_step(query_lora.get(), q_conv, q_state.get(),
                                                     kQueryLora, stream),
                         "q MoME");
            kernel_check(p92::cuda::rmsnorm(query_lora.get(), q_norm, query_norm.get(),
                                             1, kQueryLora, 1.0e-5F, stream),
                         "q RMSNorm");
            executor.project(q_b, query_norm.get(), query_up.get());
            kernel_check(p92::cuda::bf16_gemv(normalized.get(), kv_a_weight, kv_down.get(),
                                               kKvLora + kRope, kHidden, stream),
                         "BF16 kv-a projection");
            kernel_check(p92::cuda::mome_conv3_step(kv_down.get(), kv_conv, kv_state.get(),
                                                     kKvLora, stream),
                         "kv MoME");
            kernel_check(p92::cuda::rmsnorm(kv_down.get(), kv_norm, cache_kv.get(),
                                             1, kKvLora, 1.0e-5F, stream),
                         "kv RMSNorm");
            kernel_check(p92::cuda::mla_prepare_query(query_up.get(), kv_b_weight,
                                                       q_absorbed.get(), q_rope.get(),
                                                       0, 6400000.0F, stream),
                         "MLA query prepare");
            kernel_check(p92::cuda::mla_prepare_key_rope(kv_down.get(), key_rope.get(),
                                                          0, 6400000.0F, stream),
                         "MLA key prepare");
            kernel_check(p92::cuda::rmsnorm(sink_kv, kv_norm, sink_kv_normalized.get(),
                                             kSinks, kKvLora, 1.0e-5F, stream),
                         "sink RMSNorm");
            kernel_check(p92::cuda::mla_attention_decode(
                             q_absorbed.get(), q_rope.get(), sink_kv_normalized.get(), sink_rope,
                             cache_kv.get(), key_rope.get(), 0, 1, latent_output.get(),
                             1.0F / std::sqrt(192.0F), stream),
                         "latent attention");
            kernel_check(p92::cuda::mla_value_up(latent_output.get(), kv_b_weight,
                                                  attention_output.get(), stream),
                         "MLA value up");
            kernel_check(p92::cuda::mome_conv3_step(attention_output.get(), o_conv, o_state.get(),
                                                     kAttentionOutput, stream),
                         "output MoME");
            executor.project(o_proj, attention_output.get(), hidden.get());
            kernel_check(p92::cuda::rmsnorm(hidden.get(), post_attention_norm, normalized.get(),
                                             1, kHidden, 1.0e-5F, stream),
                         "post-attention RMSNorm");
            kernel_check(p92::cuda::mhc_post(normalized.get(), streams_a.get(), h_post.get(),
                                              h_res.get(), streams_b.get(), 1, stream),
                         "attention mHC post");

            kernel_check(p92::cuda::mhc_pre(streams_b.get(), mlp_phi, mlp_gamma,
                                             mlp_alpha, mlp_beta, mixed.get(),
                                             h_post.get(), h_res.get(), 1, 1.0e-5F, stream),
                         "MLP mHC pre");
            kernel_check(p92::cuda::rmsnorm(mixed.get(), pre_mlp_norm, normalized.get(),
                                             1, kHidden, 1.0e-5F, stream),
                         "pre-MLP RMSNorm");
            executor.project(gate, normalized.get(), gate_output.get());
            executor.project(up, normalized.get(), up_output.get());
            kernel_check(p92::cuda::swiglu(gate_output.get(), up_output.get(), gate_output.get(),
                                            kIntermediate, stream),
                         "dense SwiGLU");
            executor.project(down, gate_output.get(), hidden.get());
            kernel_check(p92::cuda::rmsnorm(hidden.get(), post_mlp_norm, normalized.get(),
                                             1, kHidden, 1.0e-5F, stream),
                         "post-MLP RMSNorm");
            kernel_check(p92::cuda::mhc_post(normalized.get(), streams_b.get(), h_post.get(),
                                              h_res.get(), streams_a.get(), 1, stream),
                         "MLP mHC post");
            kernel_check(p92::cuda::rmsnorm(streams_a.get(), block_post_norm, streams_b.get(),
                                             1, kFlat, 1.0e-5F, stream),
                         "block-post RMSNorm");
            executor.synchronize();
            std::vector<std::uint16_t> output(kFlat);
            check(cudaMemcpy(output.data(), streams_b.get(), output.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost),
                  "read layer-0 output");
            return output;
        };

        const std::vector<std::uint16_t> first = run();
        const std::vector<std::uint16_t> second = run();
        if (first != second) throw std::runtime_error("layer-0 backend is not deterministic");
        double square_sum = 0.0;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        for (const std::uint16_t bits : first) {
            __nv_bfloat16_raw raw;
            raw.x = bits;
            const double value = __bfloat162float(static_cast<__nv_bfloat16>(raw));
            if (!std::isfinite(value)) throw std::runtime_error("non-finite layer-0 state");
            square_sum += value * value;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        const double rms = std::sqrt(square_sum / first.size());
        if (!(rms > 0.01 && rms < 100.0)) throw std::runtime_error("layer-0 state RMS is implausible");
        std::cout << "P92_LAYER0_FORWARD_OK"
                  << " token=" << kBosToken
                  << " nvfp4_bytes=" << projections.resident_bytes()
                  << " auxiliary_bytes=" << auxiliary.resident_bytes()
                  << " rms=" << rms
                  << " min=" << minimum
                  << " max=" << maximum
                  << " hash=" << std::hex << fnv1a(first) << std::dec << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_LAYER0_FORWARD_ERROR " << error.what() << '\n';
        return 1;
    }
}
