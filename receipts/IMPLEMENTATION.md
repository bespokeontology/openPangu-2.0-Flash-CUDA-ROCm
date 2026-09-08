# Implementation record

## 2026-08-22

- Official checkpoint: 50 safetensors shards, 37,587 tensors,
  200,272,648,050 payload bytes.
- Model contract: 46 trunk layers, three shipped MTP layers, 524,288-token
  context, 256 routed experts with top-8 selection, approximately 6B active
  parameters per token.
- The reference Q4 CUDA baseline is coherent at approximately 11.3 decode
  token/s. Cold 8.5K-token prefill measured 243-274 token/s.
- Native target is NVIDIA NVFP4 E2M1 with E4M3 block scales and SM121
  block-scaled tensor cores. Q4_K_M and NVFP4 are different formats.
- Windows floating-point identity is not required. Windows and Huawei sources
  define architecture and behavioral semantics only.
- Final decode target: at least 50 token/s, reported only after measurement.

### Native gates

`p92_catalog_inspect` completed against the official checkpoint:

- `P92_CATALOG_OK tensors=37587 shards=50 payload_bytes=200272648050`
- Startup wall including 50 header reads and read-only shard mmap: about 4.1 s.

`p92_nvfp4_projection_test` packed and executed the real Huawei tensor
`model.layers.0.self_attn.q_a_proj.weight` (1024x2560):

- BF16 bytes: 5,242,880
- NVFP4 packed bytes: 1,310,720
- E4M3 block-scale bytes: 163,840
- SM121 tensor-core vs software NVFP4 cosine: 1.000000
- SM121 NVFP4 vs original BF16 cosine: 0.991788
- Maximum absolute difference vs BF16 for the deterministic probe: 0.215007
- Tensor-core workspace: 131,200 bytes

This is the first real official-checkpoint NVFP4 projection in the native tree,
not a synthetic matrix and not a Q4/llama execution.

`p92_mhc_state_test` then exercised layer-0 mHC pre/Sinkhorn/post and the final
model merge using the official checkpoint tensors:

- Sinkhorn maximum row/column stochastic error: `9.53674e-07`
- Final merged state range: `[-5.53125, 5.40625]`
- Deterministic merged BF16 FNV-1a hash: `97e7641f3e5c39c`
- All intermediate and final values finite.

The local repository now owns the SM121 kernel instantiations. It no longer
links the Qwen kernel archive; it reuses only the checked CUTLASS/FlashInfer
headers while compiling `p92_kernels` itself.

### Full NVFP4 artifact

The C++/CUDA packer completed all 36,528 eligible official tensors in
1,479.4 seconds. Completed files were atomically renamed only after both data
files and the manifest were synced.

- Directory: `<NVFP4_ARTIFACT_DIR>`
- `manifest.bin`: 4,675,632 bytes,
  SHA-256 `40b50bc4431fc0bac518215cba70a323f316eaec9299288447ccc30e6518f2ff`
- `weights.nvfp4`: 49,823,023,104 bytes,
  SHA-256 `e071070f6240aa4198b47997f78c0b6bf8e29d690635c0daa3e95ab8cb4ea8de`
- `scales.e4m3`: 6,227,877,888 bytes,
  SHA-256 `832dad78a767f1539949b3fdd5cf40f729d31c2f63e9a982602d54084da81856`

`p92_nvfp4_artifact_inspect` validated every record and file boundary.
`p92_nvfp4_projection_test` then executed layer-0 q-a directly from the
completed artifact and reproduced the live-pack tensor-core/software and BF16
quality measurements.

### Resident projection warehouse

`p92_resident_test` loaded the completed artifact into one immutable CUDA
weight bank, converted every tensor's linear E4M3 block-scale array to the
SM121 tensor-core swizzle, and bound all 36,528 projections by checkpoint
name:

- `P92_RESIDENT_OK tensors=36528 resident_bytes=56051047104 load_s=184.248`
- Resident packed weights: 49,823,023,104 bytes.
- Resident swizzled scales: 6,227,877,888 bytes.
- Resident per-projection alpha values: 146,112 bytes.
- The gate ran while the separate DeepDiver process occupied about 17.8 GiB,
  so the 56.05 GB projection bank fits concurrently on this UMA system.

The 184-second startup is a one-time persistent-process cost, not decode
latency. It is dominated by 36,528 individual scale uploads and swizzle
launches. Persisting a pre-swizzled scale file can remove that launch storm,
but it is deliberately deferred until the coherent model forward exists.

### Native latent MLA and first transformer block

`p92_mla_decode_test` exercises the real Huawei layer-0 `kv_b` matrix and
learned sink tensors with the CUDA latent-cache implementation. Against a
direct CPU reference:

- absorbed-query cosine: `0.999999`
- latent-attention cosine: `0.999999`
- value-up cosine: `0.999999`
- maximum absolute differences: `0.000946283`, `0.00780511`, `0.00720954`

The implementation keeps the 512-value compressed KV plus 64-value RoPE key;
it does not expand and store 48 copies of K/V.

`p92_layer0_forward_test` then ran the official BOS embedding through the full
first target block twice:

- attention mHC pre/post and 20-iteration Sinkhorn
- input/post-attention sandwich norms
- q projection, all three stateful MoME width-3 convolutions, latent MLA with
  128 learned sinks, value-up, and output projection
- MLP mHC pre/post, dense NVFP4 gate/up/down, SwiGLU, and block-post norm
- `P92_LAYER0_FORWARD_OK token=148899 nvfp4_bytes=55443480 auxiliary_bytes=16793708 rms=0.534404 min=-11.125 max=23.75 hash=2b82bfebeef2ac6f`

The two runs were byte-identical and all outputs were finite. The checkpoint's
576x2560 `kv_a_proj_with_mqa` is the one large projection not present in the
128-row-aligned NVFP4 artifact, so this first runtime executes it with a native
BF16 GEMV. That is approximately 136 MB of aggregate weight traffic per token
across 46 target layers and is not the decode bandwidth bottleneck.

### Routed MoE and complete target pass

`p92_moe_forward_test` executes the shipped layer-2 router in F32, applies the
F32 correction bias, selects top-8 by corrected sigmoid score, normalizes the
raw selected scores to the model's 2.5 routed scale, runs all three NVFP4
projections for each selected expert, and adds the independent shared expert.

- Selected experts: `123,204,232,142,203,11,178,3`
- Routed weight sum: `2.5`
- Output RMS/range: `0.0291035`, `[-0.0961914, 0.101562]`
- Deterministic output hash: `5157e26474aa63f6`

`p92_full_forward_test` then used the low-memory selective projection loader to
run official BOS token 148899 through all 46 target layers, the final mHC
stream merge, output RMSNorm, the real 151,552-row NVFP4 language head, and
greedy argmax:

- `P92_FULL_FORWARD_OK input=148899 next=2772 position=1`
- Active NVFP4 projections loaded for the pass: 2,768,639,188 bytes.
- Auxiliary BF16/F32 tensors: 829,577,074 bytes.
- Core-load wall: 9.35 seconds; first forward including first-use expert
  admissions: 5.44 seconds.

This proves the direct Huawei-to-CUDA target graph reaches a valid vocabulary
token without GGUF or llama. The separate full-resident timing run was stopped
at the user's request during its one-time load; no steady resident decode
number is claimed yet. The llama baseline service remains inactive.

### Native tokenizer and terminal boundary

`p92_tokenizer` is an all-C++ implementation of the checkpoint's exact
PCRE2 Unicode split, GPT-style byte mapping, BPE merge table, and AddedToken
handling. It reads Huawei's original `tokenizer.json`; it does not use Python
or a llama tokenizer at runtime.

- `P92_TOKENIZER_OK cases=5 vocab=151552 merges=148643 added=701`
- Exact encode/decode parity covers English, Chinese, the official Pangu chat
  prompt, and `[unused11]` / `[unused12]` tool-envelope text.

`p92_chat` is built as a persistent multi-turn CLI around the fully resident
native model. It supports `/reset` and `/quit`, preserves conversation state,
and applies the official Pangu message boundary tokens. It links CUDA runtime
and PCRE2 only; `ldd` contains neither Python nor llama. It has intentionally
not been launched yet because doing so allocates the 56.05 GB projection bank.

### Pre-swizzled startup path

`p92_nvfp4_swizzle` converted the existing 6,227,877,888-byte linear E4M3
scale artifact once, using only one tensor-sized CUDA scratch allocation, and
atomically published `scales.swizzled.e4m3` in 29.7 seconds.

- Size: 6,227,877,888 bytes.
- SHA-256: `c2c8cc2c0170fe6abfac71f79392ecec433b71763b9a62ffc78ab460c367e1f2`.
- `ResidentWeights` now validates the exact extent and bulk-uploads this file;
  the proven in-memory swizzle remains as a fallback.

This removes 36,528 scale upload and kernel-launch pairs from future
persistent-engine startup without changing projection data or decoding
settings.

### DSA and official long-context cache layout

The native target now implements Huawei's mixed attention schedule instead of
stopping at 2,048 tokens:

- DSA layers 0, 3, ..., 45 retain full MLA latent/RoPE history and a separate
  128-value indexer-key history.
- The other 30 layers use a true 512-token circular MLA cache.
- The shipped DSA projections are bound by their original checkpoint names:
  `indexer.wq_b`, `indexer.wk`, `indexer.k_norm`, and
  `indexer.weights_proj`.
- DSA query comes from normalized q-LoRA; key and 24 head weights come from the
  normalized attention input. RoPE applies to the first 64 values of every
  128-value indexer head/key.
- Up through 2,048 positions, DSA layers attend densely. Beyond that threshold,
  the CUDA indexer scores every causal key as
  `sum_h weight[h] * relu(dot(query[h], key))`, selects the shipped top 2,048,
  and passes their indices to latent MLA. All 128 learned sinks remain visible.

The standalone CUDA gate matched a direct CPU scorer for its first 64 selected
indices:

- `P92_DSA_OK positions=257 workspace_bytes=1 maximum_workspace_bytes=4260351 top=45,31,38`
- The 524,288-position radix-sort workspace is 4,260,351 bytes.
- Position-zero indexer RoPE is exact.

The terminal runtime accepts the checkpoint's official 524,288-token context
limit by default. At that limit the mixed cache allocation is approximately
9.7 GB for MLA latent/RoPE state plus approximately 2.15 GB for DSA index keys,
rather than allocating full history to all 46 layers. The full 56.05 GB model
was not launched for this gate; resident semantic generation remains a
separate, explicitly announced test.

### M=64 prefill projection foundation

`ProjectionExecutor::project_rows` now exposes the already-owned SM121 NVFP4
tensor-core GEMM for 1 through 64 activation rows. It quantizes and swizzles the
entire row bank once, then executes tactic 21 for M>=16; M=1 decode retains its
existing tactics 6/20.

The low-memory gate used the real packed Huawei layer-0 q-a projection
(1024x2560) and compared one M=64 call with 64 production M=1 calls:

- `P92_PROJECTION_ROWS_OK rows=64 shape=1024x2560 cosine=1 max_abs=0`
- Repeated M=64 timing: 0.024-0.030 ms; 64 serial calls: 1.33-1.34 ms;
  measured speedup: 44.9-55.7x.
- The gate loaded only this projection, not the resident 56.05 GB model.

`mome_conv3_rows` preserves the causal width-3 recurrence across a row bank.
Its exact-output gate matched 64 serial state updates byte-for-byte and took
0.132-0.138 ms versus 0.141-0.145 ms for the already-warm serial sequence. Thus prefill can
batch the weight-heavy projections without changing the shipped MoME state
machine or adding a recurrence penalty.

### Complete native prefill block

The model now consumes prompt tokens in chunks of up to 64 and executes the
complete target graph as a causal row bank:

- M=64 NVFP4 q/q-up/o, dense FFN, shared-expert, DSA-indexer, and grouped
  routed-expert projections.
- BF16 M-row KV-a and router projections, stride-aware KV normalization, and
  causal MoME updates.
- Batched MLA query/key preparation and value-up. DSA layers populate full
  caches and use dense attention through 2,048 positions, then select top
  2,048 per causal row. SWA layers insert and attend each row in ring order so
  a block cannot overwrite keys needed by an earlier row.
- Routed rows are grouped by expert. Each selected expert is admitted/read once
  for its group, then its weighted outputs are scattered to the original rows;
  the shared expert runs once over the whole row bank.
- The language head runs only on the final prompt row.

The real Huawei M=64 MLA component gate passed:

- `P92_MLA_ROWS_OK rows=64 kv_cos=1 norm_cos=1`
- Key RoPE, absorbed query, query RoPE, causal latent attention, and value-up
  each had zero BF16 mismatches against 64 serial calls.
- The DSA row-RoPE gate likewise matched eight serial positions exactly.

The first complete multi-token target gate used the selective loader, not the
56.05 GB resident bank. Serial BOS processing produced token 2772 and then
5015; the two-row prefill block produced the same token 5015, and reset/replay
was deterministic:

- `P92_PREFILL_OK rows=2 first=2772 prediction=5015`
- Warm serial: 0.1828 s; warm M=2 block: 0.1275 s; speedup: 1.43x.
- Cold times (including expert admission): 3.02 s serial and 3.25 s block.
- Selectively admitted NVFP4 projections: 4,135,854,972 bytes; auxiliary
  BF16/F32 state: 831,547,250 bytes.

This is a semantic integration gate, not the final prefill benchmark. The
separate low-memory M=64 q-a gate remains 44.9-55.7x faster than 64 M=1 calls;
a full M=64 graph benchmark can admit most routed experts and is therefore
deferred to the explicitly announced resident-model run.

### First complete resident native engine

After the prefill integration was committed, the bounded context-64 resident
gate loaded the complete immutable NVFP4 warehouse and executed two target
tokens. The process exited normally and released all GPU-visible memory:

- `P92_FULL_FORWARD_OK input=148899 next=2772 second=5015 position=2`
- Resident projection bytes: 56,051,047,104.
- Auxiliary BF16/F32 bytes: 831,547,250.
- Bulk pre-swizzled startup: 43.2512 seconds (down from the original
  launch-per-tensor 184.248-second loader).
- First token: 114.612 ms; second token: 87.809 ms, or 11.388 tok/s.

The resident and selective engines therefore agree on the first two greedy
tokens, and the row-bank prefill path agrees with both. This 11.39 tok/s result
is the frozen working baseline, not the 50 tok/s target. Its 87.8 ms decode wall
still contains roughly 1,300 tiny M=1 projection launches per token, dominated
by 24 routed-expert projections per MoE layer; grouped/device-dispatched expert
execution and the three shipped MTP stages are the next leverage points.
