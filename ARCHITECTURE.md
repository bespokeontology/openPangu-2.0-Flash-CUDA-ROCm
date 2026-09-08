# Native model contract

Sources of truth:

- Official checkpoint `config.json` and all 50 Huawei safetensors shards.
- Huawei `openPangu-2.0-Infer` implementation at commit
  `2c16b67fb408ddd1ddfd8da855e2bedd5fc23e15`.
- The transferred Windows report is a behavioral/architecture reference only.

## Fixed dimensions

- Vocabulary 151,552; hidden width 2,560; BF16 source weights.
- 46 target layers plus MTP layers 46, 47, and 48.
- Four mHC streams, 20 Sinkhorn normalization iterations.
- MLA: 48 query heads, q-LoRA 1,024, kv-LoRA 512, qk-nope 128,
  qk-rope 64, value width 128.
- DSA on layers 0, 3, ..., 45; 512-token SWA on the remaining target layers.
- 128 learned attention sinks per layer; 524,288-token maximum context.
- Dense FFN on target layers 0 and 1, width 9,216.
- Routed MoE from layer 2 onward: 256 experts, top 8, expert width 1,024,
  one shared expert, sigmoid routing with correction bias, normalized top-k,
  routed scale 2.5.
- MoME is a width-3 causal depthwise convolution on q-LoRA, compressed KV,
  and attention output. Its recurrent state is persistent and participates in
  speculative snapshot/restore.

## Target-layer execution

The semantic reference order is:

1. mHC attention pre-mix and 4x4 Sinkhorn residual matrix.
2. input RMSNorm.
3. q/kv low-rank projections, MoME causal updates, RoPE, DSA or SWA attention,
   value up-projection, output MoME, and output projection.
4. post-attention RMSNorm and mHC post-mix/residual update.
5. pre-MLP RMSNorm.
6. dense SwiGLU or routed plus shared MoE.
7. post-MLP RMSNorm and mHC post-mix/residual update.
8. optional block-post RMSNorm on layers 0, 4, 9, 14, 19, 24, 29, 34, 39.

The optimized CUDA implementation may fuse these boundaries and reorder
independent work. It may not drop the MoME state update, mHC matrix state,
shared expert, router correction, attention sinks, or sandwich norms.

## Long-context state

- DSA layers retain full compressed MLA latent/RoPE history and one 128-value
  indexer key per token. Their first 2,048 tokens use dense causal attention;
  later tokens score full history and attend to the selected top 2,048.
- SWA layers retain only a 512-token circular compressed MLA cache.
- The DSA indexer uses 24 query heads of width 128. The first 64 values receive
  the same half-rotation RoPE convention as Huawei's implementation.
- Learned sinks are not indexer candidates and remain present in every DSA and
  SWA attention call.

## MTP contract

Each of the three shipped predictor layers performs:

1. embed the proposed token;
2. RMSNorm embedding and previous target hidden state separately;
3. concatenate to width 5,120 and project back to 2,560 with `eh_proj`;
4. execute the corresponding full decoder block;
5. apply that stage's shared-head norm and language-model head.

MTP is enabled only after serial target generation is coherent. Its verifier
must snapshot and restore KV, MoME convolution state, position metadata, and
mHC state together.

## Numerical policy

Windows bitwise identity is not required. Gates use finite-state checks,
cosine/error tolerances at selected projections, deterministic fixed-backend
generation, and coherent token behavior. NVFP4 E2M1 plus E4M3 block scales,
BF16 outputs, F32 reductions where stability requires them, fused kernels, and
different reduction pairing are valid.

## Weight policy

- Embedding lookup remains BF16 initially.
- Eligible 2D projection weights are packed offline to NVFP4 with 16-value
  blocks and per-tensor global scaling.
- The 576x2560 `kv_a_proj_with_mqa` matrices are deliberately BF16 in the
  first runtime because 576 is not compatible with the current 128-row SM121
  NVFP4 tactic. Their combined decode traffic is only about 136 MB/token;
  padding or a 64-row NVFP4 tactic can replace this after coherent generation.
- Norms, mHC coefficients, router correction biases, sinks, and MoME width-3
  kernels remain BF16/F32 as shipped until separately measured.
- The native runtime reads Huawei safetensors plus the generated NVFP4 artifact;
  it does not read GGUF or link the Q4 baseline runtime.
