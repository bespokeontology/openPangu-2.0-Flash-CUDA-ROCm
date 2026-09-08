#include "p92/auxiliary.h"
#include "p92/executor.h"
#include "p92/kernels.h"
#include "p92/resident.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kHidden = 2560;
constexpr int kExperts = 256;
constexpr int kSelected = 8;
constexpr int kIntermediate = 1024;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <class T>
class DeviceBuffer final {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)), "allocate MoE buffer");
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

std::uint16_t float_to_bf16(float value) {
    return static_cast<__nv_bfloat16_raw>(__float2bfloat16_rn(value)).x;
}

float bf16_to_float(std::uint16_t bits) {
    __nv_bfloat16_raw raw;
    raw.x = bits;
    return __bfloat162float(static_cast<__nv_bfloat16>(raw));
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

struct Expert {
    p92::ResidentProjection gate;
    p92::ResidentProjection up;
    p92::ResidentProjection down;
};

Expert load_expert(p92::SelectiveWeights& weights, int expert) {
    const std::string prefix = "model.layers.2.mlp.experts." + std::to_string(expert) + ".";
    return {
        weights.load(prefix + "gate_proj.weight"),
        weights.load(prefix + "up_proj.weight"),
        weights.load(prefix + "down_proj.weight"),
    };
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: p92_moe_forward_test CHECKPOINT NVFP4_ARTIFACT\n";
        return 2;
    }
    try {
        p92::AuxiliaryWeights auxiliary;
        auxiliary.open(argv[1]);
        p92::SelectiveWeights projections;
        projections.open(argv[2]);
        p92::ProjectionExecutor executor;
        const cudaStream_t stream = executor.stream();

        const std::uint16_t* router_weight = auxiliary.load_bf16(
            "model.layers.2.mlp.gate.weight").data;
        const float* correction = auxiliary.load_f32(
            "model.layers.2.mlp.e_score_correction_bias").data;
        const Expert shared {
            projections.load("model.layers.2.mlp.shared_experts.gate_proj.weight"),
            projections.load("model.layers.2.mlp.shared_experts.up_proj.weight"),
            projections.load("model.layers.2.mlp.shared_experts.down_proj.weight"),
        };

        std::vector<std::uint16_t> input(kHidden);
        for (int i = 0; i < kHidden; ++i) {
            input[i] = float_to_bf16(0.35F * std::sin(i * 0.017F) +
                                      0.22F * std::cos(i * 0.0061F));
        }
        DeviceBuffer<std::uint16_t> d_input(kHidden);
        DeviceBuffer<float> router_logits(kExperts);
        DeviceBuffer<std::int32_t> expert_ids(kSelected);
        DeviceBuffer<float> expert_weights(kSelected);
        DeviceBuffer<std::uint16_t> gate_output(kIntermediate);
        DeviceBuffer<std::uint16_t> up_output(kIntermediate);
        DeviceBuffer<std::uint16_t> expert_output(kHidden);
        DeviceBuffer<std::uint16_t> result(kHidden);
        check(cudaMemcpyAsync(d_input.get(), input.data(), d_input.bytes(),
                              cudaMemcpyHostToDevice, stream),
              "upload MoE input");
        check(p92::cuda::bf16_gemv_f32(d_input.get(), router_weight, router_logits.get(),
                                        kExperts, kHidden, stream),
              "router projection");
        check(p92::cuda::router_top8(router_logits.get(), correction, expert_ids.get(),
                                     expert_weights.get(), 2.5F, stream),
              "router top8");
        std::array<std::int32_t, kSelected> ids {};
        std::array<float, kSelected> weights {};
        check(cudaMemcpyAsync(ids.data(), expert_ids.get(), sizeof(ids),
                              cudaMemcpyDeviceToHost, stream),
              "read routed expert ids");
        check(cudaMemcpyAsync(weights.data(), expert_weights.get(), sizeof(weights),
                              cudaMemcpyDeviceToHost, stream),
              "read routed expert weights");
        executor.synchronize();
        if (std::set<std::int32_t>(ids.begin(), ids.end()).size() != kSelected) {
            throw std::runtime_error("router selected duplicate experts");
        }
        float weight_sum = 0.0F;
        std::array<Expert, kSelected> selected;
        for (int slot = 0; slot < kSelected; ++slot) {
            if (ids[slot] < 0 || ids[slot] >= kExperts || !(weights[slot] > 0.0F)) {
                throw std::runtime_error("invalid routed expert result");
            }
            weight_sum += weights[slot];
            selected[slot] = load_expert(projections, ids[slot]);
        }
        if (std::abs(weight_sum - 2.5F) > 1.0e-5F) {
            throw std::runtime_error("routed expert weights do not sum to scaling factor");
        }

        std::vector<p92::cuda::Nvfp4ExpertProjection> host_bank(
            static_cast<std::size_t>(kExperts) * 3);
        for (int slot = 0; slot < kSelected; ++slot) {
            const std::size_t base = static_cast<std::size_t>(ids[slot]) * 3;
            host_bank[base] = {selected[slot].gate.weight,
                               selected[slot].gate.scales_swizzled,
                               selected[slot].gate.alpha};
            host_bank[base + 1] = {selected[slot].up.weight,
                                   selected[slot].up.scales_swizzled,
                                   selected[slot].up.alpha};
            host_bank[base + 2] = {selected[slot].down.weight,
                                   selected[slot].down.scales_swizzled,
                                   selected[slot].down.alpha};
        }
        DeviceBuffer<p92::cuda::Nvfp4ExpertProjection> expert_bank(host_bank.size());
        DeviceBuffer<std::uint16_t> grouped_gate(kSelected * kIntermediate);
        DeviceBuffer<std::uint16_t> grouped_up(kSelected * kIntermediate);
        DeviceBuffer<std::uint16_t> grouped_down(kSelected * kHidden);
        DeviceBuffer<std::uint16_t> grouped_result(kHidden);
        check(cudaMemcpyAsync(expert_bank.get(), host_bank.data(), expert_bank.bytes(),
                              cudaMemcpyHostToDevice, stream),
              "upload grouped expert bank");

        const auto read = [&](DeviceBuffer<std::uint16_t>& source, const char* operation) {
            std::vector<std::uint16_t> host(kHidden);
            check(cudaMemcpy(host.data(), source.get(), source.bytes(), cudaMemcpyDeviceToHost),
                  operation);
            return host;
        };
        const auto dispatch_routed_serial = [&]() {
            check(cudaMemsetAsync(result.get(), 0, result.bytes(), stream),
                  "clear serial routed output");
            for (int slot = 0; slot < kSelected; ++slot) {
                executor.project(selected[slot].gate, d_input.get(), gate_output.get());
                executor.project(selected[slot].up, d_input.get(), up_output.get());
                check(p92::cuda::swiglu(gate_output.get(), up_output.get(), gate_output.get(),
                                         kIntermediate, stream),
                      "serial routed SwiGLU");
                executor.project(selected[slot].down, gate_output.get(), expert_output.get());
                check(p92::cuda::add_scaled_inplace(result.get(), expert_output.get(),
                                                     expert_weights.get() + slot, kHidden, stream),
                      "serial routed combine");
            }
        };
        const auto run_routed_serial = [&]() {
            dispatch_routed_serial();
            executor.synchronize();
            return read(result, "read serial routed output");
        };
        const auto dispatch_routed_grouped = [&]() {
            executor.project_expert_gate_up(
                expert_bank.get(), expert_ids.get(), d_input.get(),
                grouped_gate.get(), grouped_up.get(), kHidden, kIntermediate);
            check(p92::cuda::swiglu(grouped_gate.get(), grouped_up.get(), grouped_gate.get(),
                                     kSelected * kIntermediate, stream),
                  "grouped routed SwiGLU");
            executor.project_expert_down(
                expert_bank.get(), expert_ids.get(), grouped_gate.get(), grouped_down.get(),
                kHidden, kIntermediate);
            check(p92::cuda::expert_combine(
                      grouped_down.get(), expert_weights.get(), grouped_result.get(),
                      kSelected, kHidden, stream),
                  "grouped routed combine");
        };
        const auto run_routed_grouped = [&]() {
            dispatch_routed_grouped();
            executor.synchronize();
            return read(grouped_result, "read grouped routed output");
        };

        const std::vector<std::uint16_t> serial_routed = run_routed_serial();
        const std::vector<std::uint16_t> grouped_routed = run_routed_grouped();
        constexpr int kVerifierRows = 4;
        DeviceBuffer<std::uint16_t> verifier_input(kVerifierRows * kHidden);
        DeviceBuffer<std::int32_t> verifier_ids(kVerifierRows * kSelected);
        DeviceBuffer<float> verifier_weights(kVerifierRows * kSelected);
        DeviceBuffer<std::uint16_t> verifier_gate(
            kVerifierRows * kSelected * kIntermediate);
        DeviceBuffer<std::uint16_t> verifier_up(
            kVerifierRows * kSelected * kIntermediate);
        DeviceBuffer<std::uint16_t> verifier_down(
            kVerifierRows * kSelected * kHidden);
        DeviceBuffer<std::uint16_t> verifier_result(kVerifierRows * kHidden);
        for (int row = 0; row < kVerifierRows; ++row) {
            check(cudaMemcpyAsync(
                      verifier_input.get() + static_cast<std::size_t>(row) * kHidden,
                      d_input.get(), d_input.bytes(), cudaMemcpyDeviceToDevice, stream),
                  "repeat verifier input");
            check(cudaMemcpyAsync(
                      verifier_ids.get() + static_cast<std::size_t>(row) * kSelected,
                      expert_ids.get(), expert_ids.bytes(), cudaMemcpyDeviceToDevice, stream),
                  "repeat verifier expert ids");
            check(cudaMemcpyAsync(
                      verifier_weights.get() + static_cast<std::size_t>(row) * kSelected,
                      expert_weights.get(), expert_weights.bytes(), cudaMemcpyDeviceToDevice,
                      stream),
                  "repeat verifier expert weights");
        }
        const auto dispatch_verifier = [&]() {
            executor.project_expert_gate_up(
                expert_bank.get(), verifier_ids.get(), verifier_input.get(),
                verifier_gate.get(), verifier_up.get(), kHidden, kIntermediate,
                kVerifierRows);
            check(p92::cuda::swiglu(
                      verifier_gate.get(), verifier_up.get(), verifier_gate.get(),
                      kVerifierRows * kSelected * kIntermediate, stream),
                  "verifier grouped routed SwiGLU");
            executor.project_expert_down(
                expert_bank.get(), verifier_ids.get(), verifier_gate.get(),
                verifier_down.get(), kHidden, kIntermediate, kVerifierRows);
            check(p92::cuda::expert_combine_rows(
                      verifier_down.get(), verifier_weights.get(), verifier_result.get(),
                      kVerifierRows, kSelected, kHidden, stream),
                  "verifier grouped routed combine");
        };
        dispatch_verifier();
        executor.synchronize();
        std::vector<std::uint16_t> verifier_host(kVerifierRows * kHidden);
        check(cudaMemcpy(verifier_host.data(), verifier_result.get(), verifier_result.bytes(),
                         cudaMemcpyDeviceToHost),
              "read verifier grouped output");
        for (int row = 0; row < kVerifierRows; ++row) {
            if (!std::equal(grouped_routed.begin(), grouped_routed.end(),
                            verifier_host.begin() + static_cast<std::size_t>(row) * kHidden)) {
                throw std::runtime_error("M=4 grouped routed output differs from M=1");
            }
        }
        double grouped_dot = 0.0;
        double grouped_serial_square = 0.0;
        double grouped_result_square = 0.0;
        double grouped_maximum = 0.0;
        std::size_t grouped_mismatches = 0;
        for (int element = 0; element < kHidden; ++element) {
            const double expected = bf16_to_float(serial_routed[element]);
            const double actual = bf16_to_float(grouped_routed[element]);
            grouped_dot += expected * actual;
            grouped_serial_square += expected * expected;
            grouped_result_square += actual * actual;
            grouped_maximum = std::max(grouped_maximum, std::abs(expected - actual));
            grouped_mismatches += serial_routed[element] != grouped_routed[element];
        }
        const double grouped_cosine = grouped_dot /
            std::sqrt(grouped_serial_square * grouped_result_square);
        if (grouped_cosine < 0.9999) {
            throw std::runtime_error("grouped routed expert numerical gate failed");
        }

        const auto time_dispatch = [&](const auto& dispatch, int iterations) {
            cudaEvent_t begin = nullptr;
            cudaEvent_t end = nullptr;
            check(cudaEventCreate(&begin), "create MoE timing begin event");
            check(cudaEventCreate(&end), "create MoE timing end event");
            dispatch();
            executor.synchronize();
            check(cudaEventRecord(begin, stream), "record MoE timing begin event");
            for (int iteration = 0; iteration < iterations; ++iteration) dispatch();
            check(cudaEventRecord(end, stream), "record MoE timing end event");
            check(cudaEventSynchronize(end), "synchronize MoE timing end event");
            float elapsed_ms = 0.0F;
            check(cudaEventElapsedTime(&elapsed_ms, begin, end), "measure MoE timing");
            check(cudaEventDestroy(begin), "destroy MoE timing begin event");
            check(cudaEventDestroy(end), "destroy MoE timing end event");
            return elapsed_ms / iterations;
        };
        constexpr int kTimingIterations = 20;
        const float serial_routed_ms = time_dispatch(dispatch_routed_serial, kTimingIterations);
        const float grouped_routed_ms = time_dispatch(dispatch_routed_grouped, kTimingIterations);
        const float verifier_routed_ms = time_dispatch(dispatch_verifier, kTimingIterations);

        const auto run = [&]() {
            check(cudaMemsetAsync(result.get(), 0, result.bytes(), stream), "clear MoE result");
            for (int slot = 0; slot < kSelected; ++slot) {
                executor.project(selected[slot].gate, d_input.get(), gate_output.get());
                executor.project(selected[slot].up, d_input.get(), up_output.get());
                check(p92::cuda::swiglu(gate_output.get(), up_output.get(), gate_output.get(),
                                         kIntermediate, stream),
                      "expert SwiGLU");
                executor.project(selected[slot].down, gate_output.get(), expert_output.get());
                check(p92::cuda::add_scaled_inplace(result.get(), expert_output.get(),
                                                     expert_weights.get() + slot, kHidden, stream),
                      "combine routed expert");
            }
            executor.project(shared.gate, d_input.get(), gate_output.get());
            executor.project(shared.up, d_input.get(), up_output.get());
            check(p92::cuda::swiglu(gate_output.get(), up_output.get(), gate_output.get(),
                                     kIntermediate, stream),
                  "shared SwiGLU");
            executor.project(shared.down, gate_output.get(), expert_output.get());
            check(p92::cuda::add_inplace(result.get(), expert_output.get(), kHidden, stream),
                  "combine shared expert");
            executor.synchronize();
            std::vector<std::uint16_t> host(kHidden);
            check(cudaMemcpy(host.data(), result.get(), result.bytes(), cudaMemcpyDeviceToHost),
                  "read MoE output");
            return host;
        };

        const std::vector<std::uint16_t> first = run();
        const std::vector<std::uint16_t> second = run();
        if (first != second) throw std::runtime_error("MoE backend is not deterministic");
        double square_sum = 0.0;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        for (const std::uint16_t bits : first) {
            const double value = bf16_to_float(bits);
            if (!std::isfinite(value)) throw std::runtime_error("non-finite MoE output");
            square_sum += value * value;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        const double rms = std::sqrt(square_sum / first.size());
        if (!(rms > 1.0e-6 && rms < 100.0)) throw std::runtime_error("MoE output RMS is implausible");
        std::cout << "P92_MOE_FORWARD_OK ids=";
        for (int slot = 0; slot < kSelected; ++slot) {
            if (slot != 0) std::cout << ',';
            std::cout << ids[slot];
        }
        std::cout << " weight_sum=" << weight_sum
                  << " nvfp4_bytes=" << projections.resident_bytes()
                  << " rms=" << rms
                  << " min=" << minimum
                  << " max=" << maximum
                  << " grouped_cos=" << grouped_cosine
                  << " grouped_max_abs=" << grouped_maximum
                  << " grouped_mismatch=" << grouped_mismatches
                  << " serial_routed_ms=" << serial_routed_ms
                  << " grouped_routed_ms=" << grouped_routed_ms
                  << " grouped_speedup=" << serial_routed_ms / grouped_routed_ms
                  << " verifier_rows=" << kVerifierRows
                  << " verifier_routed_ms=" << verifier_routed_ms
                  << " hash=" << std::hex << fnv1a(first) << std::dec << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_MOE_FORWARD_ERROR " << error.what() << '\n';
        return 1;
    }
}
