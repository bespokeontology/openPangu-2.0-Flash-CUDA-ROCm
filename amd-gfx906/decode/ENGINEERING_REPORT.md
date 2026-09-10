# openPangu-2.0-Flash on 4x AMD MI50 — engineering report

92B parameters, ~6B active, running natively on four gfx906 cards at
**69.9 tok/s** with coherent generation. Every hot-path operation is an
openPangu-shape-specialised gfx906 kernel. There is no generic GEMM, no rocBLAS
and no fallback path in the engine.

For NVFP4-specific results and traps see `receipts/09_NVFP4_TRAPS.md`, which is
maintained as a first-class document. Rejected optimisations with their numbers
are in `receipts/08_REJECTED.md`.

---

## 1. Result

    first token        21.29 ms
    steady decode      16.20 ms/token
    throughput         63.18 tok/s   clean runs 63.00 to 63.34, twelve
                                     interleaved pairs against a 62.50-62.76
                                     control; distributions disjoint
    ownership          12/12/11/11 across four MI50s, executed sequentially
    transport          three host-staged crossings a token, 20,480 B each
    residency          all-NVFP4 expert arena, 47 GiB, 257 slots a layer

Reference points, same model: the DGX Spark CUDA engine measured 21.01 tok/s
native trunk decode and 52.10 tok/s with its three-stage MTP predictor. This
engine has no MTP.

## 2. What the model actually is

46 trunk layers, of which 0 and 1 are dense and 2-45 are MoE with 256 routed
experts top-8 plus one shared expert. Attention is MLA with a 512-dim latent, 48
heads, 128 nope plus 64 rope, and 128 parameter sinks a layer. Every third layer
(index % 3 == 0) uses DSA sparse attention with a 24-head indexer selecting
2048 positions; the rest use a 512 sliding window. Four-stream mHC hyper-
connections bracket every sublayer, with a 4x4 Sinkhorn over 20 iterations.
Three MoME depthwise causal convolutions sit inside each attention block.

## 3. Chronology

**Mechanism 1-3, expert path.** NVFP4 expert GEMV specialised for openPangu
shapes; layer ownership corrected from a Qwen truncation that silently dropped
two layers (46 % 4 = 2); representation chosen by measurement.

**Mechanism 4, W8A8 recovered not invented.** Huawei ships
`openPangu-2.0-Flash-Int8`. Reading it, plus the preserved Windows CUDA
`pangu_under8_patch`, gave the contract: int8 weights row-major with one
symmetric bf16 scale per output row, int32 accumulation, requantisation after
SwiGLU. Their tool `jointfix` was then found in the reference tree and settled
the method: a per-input-channel smoothing scale grid-searched against a BF16
output-reconstruction objective, absorbed into `pre_mlp_layernorm`, with GPTQ
weight quantisation. **This is why their one-scale-per-token activation rule
does not transfer to our unsmoothed artifact.**

**Mechanism 5, NVFP4 on dot4, 21.6x.** See NVFP4 traps item 2. This inverted the
residency design: the two representations now cost the same per expert.

**Mechanism 6-7, arena and router.** A 47 GiB per-device arena addressed by
`slot = local_moe_index * 257 + expert`; the router's sigmoid/bias/top-8/
renormalise/x2.5 contract verified against real weights.

**Mechanism 8, MLA decode.** Ported from the GLM gfx906 tiled online-softmax
kernel, then rebuilt twice. Register pressure, not parallelism or access
pattern, was the cap.

**Mechanism 9, DSA indexer.** A transposed [128][cap] key cache removes every
cross-lane reduction from the scorer. Two defects were found by measurement: a
single-pass radix select that returned 2048 entries differing from the true
top-2048 by 636 while looking plausible, and a serial 2048-bin walk that was the
entire fixed cost of selection.

**Mechanism 10-15.** mHC from the GLM donor; the shared expert as arena slot 256
with a 9-wide dispatch; the dense MLP; MoME, which the port map described as one
convolution and is three; lm_head with a real quality defect (see section 5);
and transport.

**First coherent token**, then five measured optimisation passes.

## 4. Optimisation ladder, all measured on the composed engine

    34.60  first coherent token
    43.78  MLA sinks balanced across splits
    47.31  bf16 conversion glue removed
    51.48  expert down lane width 8 -> 32
    53.30  gate/up unroll, occupancy 4 -> 6 waves
    53.95  single-wave router
    56.31  NVFP4 permutation moved into its producer
    59.50  activation staging removed from the projections
    60.59  Sinkhorn overlapped with the collapse
    61.60  RMSNorm fused with the quantiser
    62.61  vectorised mHC logits loads
    63.18  DPP cross-lane argmax in route_top8 (replaces ds_bpermute)
    63.51  mHC Sinkhorn: 82 -> 20 ds_bpermute, order-preserving
    64.24  dead 10 KB memset before every MoE block removed
    66.08  downacc reduction hoisted, 45 ds_bpermute -> 5 (reassociates)
    66.72  mla_g16 32-lane reduction, exact DPP, 5 DS ops -> 2
    69.92  mla_g16 branchless loads, 6 vmcnt(0) barriers -> 2

28.90 -> 15.97 ms/token, 1.81x. Every step is a named physical mechanism, not a
rewrite: an unbalanced split, redundant format conversion, a 128-byte memory
transaction, register pressure, cross-wave traffic, redundant LDS staging.

## 5. Findings that changed the design

**The Spark tensor capture is invalid.** Its harness dumps with an
unsynchronised `cudaMemcpy` against kernels on another stream, and every dumped
buffer is reused several times a pass. Each file holds what that buffer last
held in the PREVIOUS pass. Proved by an invariant the oracle's own source forces:
`repeat_streams` replicates the embedding into four identical streams and
`mhc_pre` collapses them with a positive weighted sum, so `L0_INPUT_NORM` must
equal `rmsnorm(embedding) * gamma` for any mHC weights. It does not - cosine
-0.178 - and inverting it matches no row in the vocabulary. Corroborated
exactly: `cache_kv` is the one dumped buffer that is NOT reused, and its capture
rms matches this engine's at 1.46862 against 1.46888. Layer 0 is therefore gated
against a scalar CPU reference transliterated from the oracle SOURCE.

**A single int8 activation is not good enough at the head.** Over 96 random
activations it flipped the emitted token on 3. Two int8 components fix it but
cost 483 -> 1620 us applied everywhere, so they run only on the top-256
candidates: 565 us at 96/96 argmax agreement. Numerical precision matters where
it changes an output, not as a general principle.

**P2P is absent because of a missing kernel option, not hardware.** All 12
ordered pairs report `canAccessPeer` 1, accept `enablePeerAccess`, and return
success from `hipMemcpyPeer` while moving no bytes; a kernel on A dereferencing
B's allocation silently aliases A's own memory. The running kernel lacks
`CONFIG_PCI_P2PDMA`, `CONFIG_DMABUF_MOVE_NOTIFY` and `CONFIG_HSA_AMD_P2P`.
Worth almost nothing here - three crossings cost 0.048 ms/token serial and about
0.009 once overlapped - but it would cripple a tensor-parallel model.

## 6. Method, and its cost

Eleven optimisation hypotheses have been tested; six were wrong, several of
which would pass a code review. Grouping MLA heads to share latent rows made it
worse. The access-pattern theory was flat. Removing 23 barriers from the router
made it slower because on gfx906 a shuffle is `ds_bpermute`, the same unit a
barrier uses.

Two sweeps produced numbers that were wrong rather than merely disappointing,
both recorded so they cannot become folklore: one left the launch grid fixed
while shrinking rows per block, so the kernel wrote a fraction of its output and
looked faster; another used an unscoped substitution that rewrote a neighbouring
kernel's lane math.

What survived as method:

- Gate every kernel against an independent scalar reference, on relative error,
  never on bit-exactness against another implementation.
- The output is the real gate. Cosine is a bring-up localiser, not a quality bar.
- Measure the COMPOSED path. Component microbenchmarks do not add up, and their
  rates are conditions rather than ceilings.
- Median of three fixed-length runs; the noise floor here is about 3 percent.
- Retrieve before inventing. The Huawei checkpoint, the `jointfix` tool, the CUDA
  oracle source, and the GLM and Qwen gfx906 kernels each removed work that would
  otherwise have been rediscovered.


> **CORRECTION 2026-09-09.** Any statement in this document that MI50
> P2P is unavailable/dead at the driver is FALSE. It was true of Linux
> 5.15.0-190 only. Under Linux 6.8.0-138 the same hostile gate moves from
> 0/12 to 12/12 directed pairs with no collateral, because KFD creates
> `p2p_links` under 6.8 and created none under 5.15. Same cards, same ROCm
> 5.7.1 userspace, no ACS override. See receipts/07_P2P_DIAGNOSIS.md.
> The failure MODE described - peer pointers silently aliasing local memory
> instead of faulting - was real and is why a P2P path must be gated by a
> data-integrity check, never a capability bit.
