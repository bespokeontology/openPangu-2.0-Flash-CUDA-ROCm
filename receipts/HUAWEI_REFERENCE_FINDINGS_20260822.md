# Huawei openPangu reference findings — 2026-08-22

## Material present locally

There is no official openPangu-2.0-Flash architecture paper PDF in the local
checkpoint or `openPangu-2.0-Infer` trees. The authoritative local material for
this CUDA port is the Huawei model card, checkpoint configuration, and shipped
Omni inference implementation:

- `<OFFICIAL_CHECKPOINT_DIR>/README_EN.md`
- `<OFFICIAL_CHECKPOINT_DIR>/configuration_openpangu_v2.py`
- `<OPENPANGU_INFER_REFERENCE>/README_EN.md`
- `components/omni-npu/src/omni_npu/v1/models/pangu/pangu_v2_moe.py`
- `components/omni-npu/src/omni_npu/v1/models/pangu/pangu_v2_moe_mtp.py`
- `components/omni-npu/src/omni_npu/attention/backends/{mla,dsa,mome}.py`
- `components/omni-cache/omni_cache/cache/prefill/mome_state_utils.py`

The unrelated Pangu audit/ISO PDFs copied from the older Dragon workspace do
not define Flash-92 tensor shapes or runtime semantics and are not treated as
model implementation authority.

## Architecture contract

- 92B total parameters, approximately 6B active parameters, 512K context.
- MLA is retained. Full-context DSA and local SWA are interleaved in a 1:2
  ratio. DSA supplies sparse global context; SWA supplies local context.
- Residual topology is four-stream mHC.
- Three shipped MTP heads draft three additional tokens per target step.
- The MTP stage is not a shallow logits head: it RMS-normalizes the embedding
  and previous target hidden state, concatenates them, applies `eh_proj`, runs
  a complete decoder layer, then applies that stage's shared head.

## Attention and state semantics relevant to CUDA

- Huawei stores compressed latent KV and RoPE separately. The learned sink
  compressed-KV bank and learned sink-RoPE bank are shared by query heads;
  `update_sink_kv` inserts a singleton KV-head dimension. A CUDA kernel must not
  advance the sink-RoPE pointer once per query head.
- Query absorption is a head-batched matrix multiply into latent space.
- Value-up is a head-batched matrix multiply from latent output to the
  per-head value width.
- Sink-aware MLA attention is one fused operation covering score, causal/window
  masking, softmax, sinks, and latent-value accumulation. Huawei pads the 48
  query heads to the next power of two for this fused operator and discards the
  padded heads afterward.
- DSA prepends the learned sink indices to the selected sparse global indices,
  applies sparse fused attention over the compressed cache, then performs the
  same value-up projection.
- MoME recurrent state storage explicitly includes the speculative-token
  count. Accepted-token count selects which speculative state becomes live;
  rejected tail state cannot be published wholesale.
- The backend advertises uniform speculative rows as decode and pads dynamic
  inputs for graph execution. The model overlaps mHC Sinkhorn work with the MLA
  value-up/output epilog on a side stream.

## Consequences for this engine

- The native M=4 verifier, per-row MoME capture, accepted-row publication, and
  causal cache-row restore match the Huawei state contract.
- The accepted strided-batched BF16 value-up cut matches Huawei's explicit
  head-batched value-up structure and improves the measured engine.
- A probe that decomposed attention into four cuBLAS score calls, a softmax
  kernel, and two cuBLAS value calls was not promoted. Besides an initial
  sink-RoPE batch-stride bug found by compute-sanitizer, that execution shape
  conflicts with Huawei's fused sink-aware attention strategy and adds six
  library submissions per layer.
- The next attention implementation should therefore be a fused SM121 kernel:
  keep the current shared latent/sink banks, compute causal scores and softmax
  in one resident tile schedule, and use tensor cores inside that fused kernel
  without materializing a full head-by-token score bank.
- CUDA-graph/persistent submission and overlap of mHC side work with the MLA
  epilog are subsequent measured cuts. They should be judged after the fused
  attention kernel, not used to hide an inefficient attention decomposition.
- DFlash2 is not a generic replacement for the shipped predictor. It requires
  predictor weights trained for this exact model/tokenizer. Flash-92 already
  ships its own three-stage MTP predictor and verifier contract.

## Measured CUDA experiments after the reference review

Two Huawei-shaped ideas were implemented as isolated probes and rejected from
the product after full-model measurements:

- A head-resident M=4 fused BF16 tensor-core attention kernel kept QK, online
  softmax, and latent-value accumulation inside one launch. The first version
  used tensor cores for QK and PV and had local cosine 0.999999 with maximum
  BF16 delta 0.0078125; the narrower QK-only version retained FP32 softmax/PV,
  reached cosine 1.0 with maximum delta 0.000244141, and reduced the fixture
  from 0.1503 ms to 0.1215 ms. Both changed target decisions enough to reduce
  live MTP economics. The QK-only run required 32 blocks with 66/96 accepted
  drafts and reached 39.23 tok/s, versus the frozen 28 blocks and 69/84. The
  probe and dispatch were removed.
- Huawei overlaps mHC coefficient work with the branch epilog. A split CUDA
  implementation reproduced mixed activations, h_post, h_res, and the final
  merge hash exactly. On this GB10, however, alternating frozen/new full-model
  runs showed the event/side-stream cost was larger than the hidden Sinkhorn
  work. Frozen verifier walls were 1751.9, 1762.3, and 1834.7 ms; overlap was
  1791.9, 1828.5, and 1886.5 ms at the same 69/84 acceptance. The scheduling
  code was removed.
- The always-on shared expert was appended as a ninth problem to the grouped
  routed gate/up and down projections for M=1, eliminating three standalone
  projections plus duplicate input quantization. The fixture remained
  coherent (cosine 0.999992, maximum BF16 delta 0.000732422), but alternating
  full-model runs rejected the cut: the frozen baseline reached 45.10 tok/s
  while the fused path reached 44.32 tok/s with identical 69/84 acceptance and
  28 verification blocks. Extending the scheme to M>1 also exposed an invalid
  row-scale layout in the grouped SM121 kernel. All fusion code was removed.
- The real M=4 route trace contained 21,528 `(row, slot)` assignments but only
  13,886 unique experts, a 1.55x cross-row reuse factor. Reordering the existing
  N=1 CUTLASS problems by `(projection, expert)` on device keeps identical
  destinations and arithmetic while putting repeated weights next to each
  other in the scheduler. The maximal-reuse fixture fell from about 0.285 ms
  to 0.227-0.259 ms, and real verifier wall fell from about 1.81 s to
  1.71-1.74 s before the attention cut. True dynamic N=2-4 expert batching was
  also implemented and produced exact fixture output, but its compact
  quantize/scatter and dynamic scheduler path regressed the same fixture to
  0.496 ms, so that larger mechanism was removed.
- The prior fused tensor-core attention probe was not the only way to expose
  parallelism. The accepted MLA cut retains the existing single-row tiled
  reduction exactly and launches the four causal verifier rows concurrently
  in the grid. The rows share sink/cache traffic through L2 rather than one
  78-KiB block serializing all four queries. The 64-row A/B gate reports zero
  attention mismatches, the real 96-token output and 69/84 acceptance are
  unchanged, verifier wall falls to 1.45-1.51 s, and effective throughput is
  53.19-55.33 tok/s.

The accepted value-up freeze remains the immutable comparison baseline. The
expert-major scheduling plus row-parallel MLA candidate is the first measured
Flash-92 build to clear the 50 tok/s mixed-acceptance goal; it must pass the
full state, prefill, and persistent-chat gates before its own freeze.
