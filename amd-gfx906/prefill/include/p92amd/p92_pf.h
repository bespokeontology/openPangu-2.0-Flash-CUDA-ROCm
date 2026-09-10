// openPangu prefill pipeline: contract and variant selection.
//
// ACCEPTANCE METRIC. Composed prefill throughput in tokens/second on the full
// four-GPU pipeline. Nothing else decides a variant. Kernel timings exist as
// diagnosis underneath and never as a verdict.
//
//     prompt length / chunk size  ->  prefill tok/s  ->  TTFT  ->  correctness
//
// This matters more here than in decode because the prefill kernels deliberately
// TRADE OCCUPANCY FOR REUSE. k_pf_attn<TQ> holds TQ queries resident so each key
// is loaded once for TQ of them: TQ=2 is VGPR 64 at 4 waves/SIMD, TQ=4 is VGPR
// 122 at 2. k_pf_gateup holds an expert's weight row resident across its routed
// tokens at VGPR 80 and 3 waves. A microbenchmark reports one side of that trade
// while the composed pipeline pays or collects on the other - the mla_g16 result
// is the standing proof, where VGPR 76->50 and occupancy 3->4 both improved and
// the kernel got 6.2 percent SLOWER.
//
// Therefore every variant is selectable AT RUNTIME, not compiled in. The
// pipeline runs end to end with each and is judged on:
//
//     TQ=2   XXXX prefill tok/s
//     TQ=4   XXXX prefill tok/s
//     delta  +X.X percent
//
// Do not benchmark a kernel in isolation, pick a winner, and then compose it.
#ifndef P92AMD_PF_H
#define P92AMD_PF_H

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

// PLACEMENT POLICY. Prefill and decode have different economics - decode is
// weight-bandwidth bound and wants aggregate HBM, prefill is compute-heavy and
// can lose to synchronisation at modest prompt lengths - so prefill placement is
// a first-class runtime policy, not a hardwired assumption. P<n>/D4 means n
// cards prefill and hand state to four-card decode.
//
// ON THIS BOX THE POLICY IS PINNED. The model is 51.22 GB (50.02 expert arena
// measured + 1.20 non-expert) against 17.16 GB a card:
//
//     P1  51.22 GB/card   does not fit
//     P2  25.61 GB/card   does not fit
//     P3  17.07 GB/card   does not fit, and leaves nothing for KV
//     P4  12.81 GB/card   fits, 4.35 GB spare
//
// So P4 is forced by memory, not chosen by measurement. The escape - one card
// computing while reading weights from peers - is dead on arrival: peer READ
// measured 3.85 GB/s, so streaming 50 GB per chunk costs 13 s.
//
// THE HANDOFF IS NOT THE OBSTACLE, which matters for the 32-card box. Moving
// three quarters of the KV at the measured 23.1 GB/s peer-push rate costs
// 0.88 ms at a 512-token prompt, 3.52 ms at 2048, 14.09 ms at 8192 - cheap
// against prefilling those prompts. Where 32 cards hold eight independent full
// copies, placement freedom is real and dedicated prefill workers feeding decode
// groups becomes the interesting design. The policy exists here so that box
// inherits it rather than rediscovering it.
//
// WHAT IS ACTUALLY FREE ON FOUR CARDS is pipeline depth: every card must compute
// the layers whose weights it holds, but how many chunks are in flight across
// those stages is a real choice. depth 1 is no overlap; depth 4 fills the
// pipeline at C/(C+3) utilisation.
struct PfConfig {
    int chunk;        // tokens per chunk: 256 / 512 / 1024
    int attn_tq;      // 2, 4 or 8
    int attn_kernel;  // 1 = original single-key loop, 2 = DPP/exp2/gated/2-in-flight
    int dsa_scorer;   // 1 = LDS-staged query, 3 = dim-major query in SGPRs
    int dsa_sel;      // real DSA selector + union-tiled sparse attention.
                      // ON by default: without it the engine attends the full
                      // prefix on DSA layers and is NOT the model above 2048.
    int dsa_win;      // CEILING PROBE ONLY: bound DSA layers to a trailing window.
                      // A window is NOT top-k selection and is NOT correct. It
                      // measures the upper bound on what real DSA could recover.
    int sp_pack;      // 1 = packed-union sparse attention (k_pf_attn_spk):
                      // the union builder also emits contiguous per-tile key
                      // streams so the sweep has no index-load chain.
    int sp_g2;        // 1 = G=2 head-shared sparse attention (k_pf_attn_spg2):
                      // two heads of one query per block, selection-packed
                      // streams, no union and no masks. Cuts the 48x head
                      // re-read factor in half at TQ=2's register footprint.
    int gateup_var;   // 0 weight-resident (default)
    int prefill_cards;// P<n>. Pinned to 4 here; the knob exists for 32 cards.
    int pipe_depth;   // chunks in flight across the layer stages
    int verbose;
};

// Bytes the model needs per card at a given prefill placement. Measured, not
// estimated: the arena is 257 slots x 44 MoE layers x 4,423,680 B.
static inline double pf_bytes_per_card(int n) {
    return (50.02e9 + 1.20e9) / (double)(n > 0 ? n : 1);
}
static inline int pf_placement_feasible(int n, double vram_bytes) {
    return pf_bytes_per_card(n) < vram_bytes * 0.92;   // leave KV + activations
}

// Every knob is an environment override so a sweep is a shell loop over the
// SAME binary. Nothing is compile-time.
static inline PfConfig pf_config(void) {
    PfConfig c;
    const char *e;
    c.chunk      = (e = getenv("PF_CHUNK"))   ? atoi(e) : 512;
    c.attn_tq    = (e = getenv("PF_ATTN_TQ")) ? atoi(e) : 2;   // 2 promoted 09-09 (3 waves)
    c.attn_kernel = (e = getenv("PF_ATTN_KERNEL")) ? atoi(e) : 2;   // 2 promoted 09-09
    c.dsa_win     = (e = getenv("PF_DSA_WIN")) ? atoi(e) : 0;
    c.dsa_sel     = (e = getenv("PF_DSA_SEL")) ? atoi(e) : 1;
    c.sp_pack     = (e = getenv("PF_ATTN_SP_PACK")) ? atoi(e) : 0;
    c.sp_g2       = (e = getenv("PF_ATTN_SP_G2")) ? atoi(e) : 1;   // shipped 09-09
    c.dsa_scorer  = (e = getenv("PF_DSA_SCORER")) ? atoi(e) : 3;   // shipped 09-09
    c.gateup_var = (e = getenv("PF_GATEUP"))  ? atoi(e) : 0;
    c.prefill_cards = (e = getenv("PF_PREFILL_CARDS")) ? atoi(e) : 4;
    c.pipe_depth    = (e = getenv("PF_PIPE_DEPTH"))    ? atoi(e) : 4;
    // Fail closed rather than silently thrashing: an infeasible placement is a
    // configuration error, not something to discover from a slow run.
    if (!pf_placement_feasible(c.prefill_cards, 17.16e9)) {
        fprintf(stderr,
            "PF_PREFILL_CARDS=%d needs %.2f GB a card against 17.16 available; "
            "P4 is the only feasible placement on this box\n",
            c.prefill_cards, pf_bytes_per_card(c.prefill_cards) / 1e9);
        c.prefill_cards = 4;
    }
    c.verbose    = (e = getenv("PF_VERBOSE")) ? atoi(e) : 0;
    return c;
}

// Runtime dispatch over the templated attention tiles.
//
// MUST BE INLINE, IN THE CALLER'S TU. HIP defaults to non-RDC, so every
// translation unit gets its own device image: a hipLaunchKernelGGL in a TU that
// does not DEFINE the kernel binds to nothing and silently does nothing - no
// error, no launch, an untouched output buffer. This cost a first-light debug
// cycle, with an unconditional store in the kernel failing to appear. The GLM
// donor documents the same constraint on its own kernels.
// PF_ATTN_DISPATCH is emitted by whichever TU includes p92_pf_attn.hip.
#define PF_ATTN_DISPATCH                                                      \
static inline hipError_t pf_attn_launch(int kern, int tq, const float *qlat,   \
        const float *qpe, const uint16_t *lat, const uint16_t *kpe,            \
        const uint16_t *sink_lat, const uint16_t *sink_pe,                     \
        int chunk_base, int T, int win, float *out, hipStream_t st) {          \
    const dim3 blk(64);                                                        \
    switch (tq) {                                                              \
    case 2: { const dim3 g((T+1)/2, 48);                                       \
        if (kern == 2)                                                         \
            hipLaunchKernelGGL((k_pf_attn2<2>), g, blk, 0, st, qlat, qpe, lat, \
                               kpe, sink_lat, sink_pe, chunk_base, T, win, out);\
        else                                                                    \
            hipLaunchKernelGGL((k_pf_attn<2>), g, blk, 0, st, qlat, qpe, lat,  \
                               kpe, sink_lat, sink_pe, chunk_base, T, win, out);\
        } break;                                                                \
    case 4: { const dim3 g((T+3)/4, 48);                                       \
        if (kern == 2)                                                         \
            hipLaunchKernelGGL((k_pf_attn2<4>), g, blk, 0, st, qlat, qpe, lat, \
                               kpe, sink_lat, sink_pe, chunk_base, T, win, out);\
        else                                                                    \
            hipLaunchKernelGGL((k_pf_attn<4>), g, blk, 0, st, qlat, qpe, lat,  \
                               kpe, sink_lat, sink_pe, chunk_base, T, win, out);\
        } break;                                                                \
    case 8: { const dim3 g((T+7)/8, 48);                                       \
        if (kern == 2)                                                         \
            hipLaunchKernelGGL((k_pf_attn2<8>), g, blk, 0, st, qlat, qpe, lat, \
                               kpe, sink_lat, sink_pe, chunk_base, T, win, out);\
        else                                                                    \
            hipLaunchKernelGGL((k_pf_attn<8>), g, blk, 0, st, qlat, qpe, lat,  \
                               kpe, sink_lat, sink_pe, chunk_base, T, win, out);\
        } break;                                                                \
    default: return hipErrorInvalidValue;                                      \
    }                                                                          \
    return hipGetLastError();                                                   \
}

// What the harness must print, and the only thing that decides a variant.
struct PfResult {
    int    prompt_tokens;
    int    chunk;
    double prefill_ms;      // whole prompt, all chunks
    double prefill_toks;    // prompt_tokens / prefill_s      <-- ACCEPTANCE
    double handoff_ms;      // KV/state redistribution, P<n> -> D4
    double decode_toks;     // steady decode after the handoff
    double e2e_toks;        // (prompt + generated) / total_s  <-- the real number
    double ttft_ms;         // to the first generated token
    double stage_ms[8];     // diagnosis only
    int    cards_busy;      // observed concurrency
};

#endif
