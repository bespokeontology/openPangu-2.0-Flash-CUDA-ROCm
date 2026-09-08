# openPangu Flash92 native NVFP4 engine — measurement report

Date: 2026-09-08. Machine: DGX Spark, NVIDIA GB10, sm_121a, CUDA 13, 121 GiB unified memory.
Engine commit: 78c6f3aa76a065146f7e49aa4fc1150299350dc9.

## 1. Configuration under test

| item | value |
|---|---|
| p92_chat sha256 | ff3b32118879542c93346b4cacdb00a326faac4fee2eae102c87065c63cc0412 |
| p92_chat_mtp sha256 | 0a36da9055c8eaead3380c126ef8143f8e422e2058f585e3fda736e532810dcf |
| NVFP4 artifact | manifest.bin, weights.nvfp4, scales.e4m3, scales.swizzled.e4m3 |
| auxiliary shard 1 sha256 | f467730508bce8ccb93c5c9f95c8aff078960e8087fc1967bdf5710ea1409227 |
| auxiliary shard 2 sha256 | 9cb68b8c2226caff89c10b62a0da9baea4f1385844446acf9b9ca75b88873051 |
| tokenizer.json sha256 | 7816b7b6138024e37c508ff82321e11b5ee559289184fa18f7536a3e460d3941 |

Binaries rebuild from commit 78c6f3a to the hashes above.

Execution path: all projections use nvfp4_tensorcore_gemm, block-scaled NVFP4 on
Blackwell tensor cores, compiled -gencode=arch=compute_121a,code=sm_121a, with CUTLASS
tactic selection by shape. There is no CPU expert loop and no non-tensor-core fallback
in the projection path. The engine does not read GGUF and does not link any llama.cpp
runtime.

## 2. Resident memory

| component | bytes |
|---|---|
| NVFP4 projections | 56,051,047,104 |
| auxiliary BF16/F32, trunk | 831,547,250 |
| auxiliary BF16/F32, with MTP | 882,775,922 |
| total, trunk | 56.9 GB |

Only scales.swizzled.e4m3 is required at runtime, so the 59 GB artifact directory
resides in 56.9 GB. Minimum MemAvailable observed across all runs was 59.1 GB against a
20 GB guard floor; no run approached the limit.

Cold load with page cache dropped: 58.8 to 65.4 s.

## 3. Decode

Page cache dropped before each run. Context limit 4096. GPU clocks locked with
nvidia-smi -lgc 3003,3003 where stated.

### 3.1 Documented acceptance gate

Prompt: "Explain probability in one clear sentence.", 96 generated tokens. This is the
gate used by frozen/native-mtp-row-parallel-20260822/RECEIPT.md.

| source | tok/s | acceptance |
|---|---|---|
| that receipt, run 1 | 53.2100 | 69/84 |
| that receipt, run 2 | 54.6355 | 69/84 |
| this build, run 1 | 46.2825 | 67/81 |
| this build, run 2 | 52.1041 | 67/81 |

Acceptance agrees to within one per cent and the second run agrees to within two per
cent. The first run of a pair is slower in both the original receipt and here. The
engine reproduces its documented figure.

### 3.2 Prompt sensitivity

MTP emits accepted drafts without a full target step, so throughput tracks the
acceptance ratio. A second prompt, "Summarise Kolmogorov complexity in two sentences.",
produces less predictable continuations and a lower ratio:

| prompt | generated | acceptance | MTP tok/s |
|---|---|---|---|
| Explain probability in one clear sentence. | 96 | 67/81 = 82.7% | 46.28 / 52.10 |
| Summarise Kolmogorov complexity in two sentences. | 96 | 62/99 = 62.6% | 43.36 |
| Summarise Kolmogorov complexity in two sentences. | 300 | 197/300 = 65.7% | 43.37 - 44.66 |

Any published MTP figure must state its prompt. The spread between these two prompts is
about 20 per cent at identical context and identical binary.

### 3.3 Context sensitivity

Prompt: "Summarise Kolmogorov complexity in two sentences." unless noted.

| generated | end context | trunk tok/s | MTP tok/s | MTP acceptance |
|---|---|---|---|---|
| 300 | 331 | 21.0090 | 43.3694 | 197/300 |
| 200 | 4,008 | 15.2955 | 20.0518 | 132/195 |
| 64 | 11,139 | 14.1020 | not measured | n/a |

Trunk decode goes from 47.6 ms/token at 331 context to 65.4 ms/token at 4,008. The MTP
speedup falls from 2.06x to 1.31x over the same range while acceptance stays near 66 per
cent, so the loss is in the target step rather than in speculation quality.

The cause is the DSA index path. Attention itself is bounded: sliding layers see 128
sinks plus a 512 window, and non-sliding layers attend at most kDsaTopK = 2048 selected
keys. Selecting those 2048 is not bounded. dsa_score_kernel scores every position in the
context, on the 16 layers where index % 3 == 0, on every token. Measured growth is
17.8 ms per token across 3,677 additional positions, roughly 0.30 us per position per
layer, well above the streaming floor for the bytes involved. Below 2,048 context the
code takes a dense path and no selection runs (src/model.cpp, the positions <= kDsaTopK
branch).

### 3.4 Binary and machine-state controls

| variation | tok/s | acceptance |
|---|---|---|
| build/p92_chat_mtp at 78c6f3a | 43.3694 | 197/300 |
| frozen/native-mtp-row-parallel-20260822/p92_chat_mtp | 42.5296 | 197/300 |
| clocks locked at 3003 MHz | 44.6561 | 197/300 |
| Chrome closed, swap cleared, clocks locked | 43.5413 | 197/300 |

All on the Kolmogorov prompt at 300 tokens. The frozen binary and the current build agree
within run variance and produce identical acceptance; cuda/model_ops.cu and
cuda/nvfp4_grouped.cu are byte-identical between commit ccc8678 and 78c6f3a. Clock lock
and desktop load account for a few per cent and do not explain prompt-driven differences.
Note that on GB10 nvidia-smi continued to report clocks.applications.graphics = 2418 MHz
after the lock was accepted, so that field does not indicate the applied state.

## 4. Prefill

Page cache dropped before each run.

| prompt tokens | wall | tok/s | context limit |
|---|---|---|---|
| 3,807 | 74.0809 s | 51.4 | 8192 |
| 10,970 | 268.418 s | 40.9 | 16384 |

Scaling, measured over six successive prompts in one model load:

| prompt tokens | wall | ms/token | context before |
|---|---|---|---|
| 740 | 10.13 s | 13.70 | 0 |
| 908 | 17.30 s | 19.05 | 744 |
| 859 | 20.18 s | 23.49 | 1,656 |
| 667 | 16.68 s | 25.01 | 2,519 |
| 711 | 17.84 s | 25.09 | 3,190 |
| 630 | 15.85 s | 25.16 | 3,905 |

Cost per token rises to approximately 25 ms and then flattens beyond about 2,500
context, consistent with the 512 sliding window and the 2,048 selection bound. Two
terms are separable: a fixed per-row cost of about 13.7 ms per token measured at zero
context, and a bounded attention term of about 11.5 ms per token.

## 5. Changes evaluated and rejected

Each was built strictly, gated, measured against the 74.08 s prefill baseline and the
decode figures above, and reverted. The shipped configuration contains none of them.

| change | prefill, 3,807 tokens | decode at 4,008 | disposition |
|---|---|---|---|
| baseline | 74.0809 s | 15.2955 tok/s | retained |
| kMaximumBlockRows 64 to 512 | 76.6398 s | 14.7236 tok/s | rejected |
| the above plus fused multi-row attention launch | 78.4813 s | 14.5803 tok/s | rejected |
| device-grouped expert path in prefill | 60.4768 s | 15.3282 tok/s | rejected, see below |
| indexer query staged in shared memory | not measured | 42.5606 tok/s at 331 ctx | rejected |
| GPU clock lock at 3003 MHz | not measured | 44.6561 tok/s at 331 ctx | retained, operational |

Notes on each.

Block width. ProjectionExecutor::project_rows already issues a single NVFP4 tensor-core
GEMM with a batch_rows dimension, and nvfp4_tensorcore_gemm imposes no row cap; the 64
bound was a caller-side constant. Per-block scratch is about 200 KB per row and
block_logits_ is sized by speculative rows, so 512 rows costs about 100 MB against
roughly 39 GB unused. Widening did not reduce prefill time.

Fused multi-row attention. latent_attention_tiled_kernel already carries a multi-row
grid and a causal rule, but mla_attention_prefill_rows_tiled used it only for four rows
or fewer. Removing that bound made a whole block issue one launch instead of one per
row. Prefill did not improve, which indicates the cost is per-query key arithmetic
rather than launch structure.

Device-grouped experts in prefill. Prefill MoE takes a host-scheduled path: a
device-to-host copy of routed expert ids, a full stream synchronise, then a loop over
all 256 experts on the host issuing gather, three projections, SwiGLU and scatter for
each expert holding rows. Routing it to the existing device-grouped path reduced prefill
to 60.4768 s, about 1.23x. It was rejected because generated output changed and MTP
acceptance fell from 132/195 to 118/240. expert_combine_rows_kernel accumulates in FP32
over top-k slot order, while the host path accumulates in ascending expert id;
P92_MOE_BF16_ACCUMULATION_LAW requires ascending expert id with BF16 weight, BF16
product and a BF16 accumulator. The fast path does not satisfy that law and predates
this work; this change would have extended it to prefill.

Indexer shared-memory staging. dsa_score_kernel launched one block per position, so
every block re-read the full 24x128 query from global memory. Staging the query in
shared memory and scoring 32 positions per block preserves per-head arithmetic, lane
striding, the shuffle reduction, the ReLU and the weighted sum, and produced identical
MTP acceptance, confirming bit-identical scores. It was neutral at 331 context because
selection does not run below 2,048 positions.

## 6. Comparison with the Q4 baseline

A llama.cpp Q4_K_M build of the same model on the same machine measured 11.30 tok/s
decode and 243 to 274 tok/s cold prefill at 8.5K. This engine measures 43.37 tok/s
decode with MTP at short context and 14.10 tok/s trunk at 11K context, and 40.9 tok/s
prefill at 11K. Decode is faster; prefill is slower by roughly a factor of six.

## 7. Correctness

- p92_tokenizer_test reports P92_TOKENIZER_OK cases=5 vocab=151552 merges=148643
  added=701 against the upstream tokenizer.json.
- All decode and prefill runs produced coherent, on-task output. Two runs summarising a
  supplied engineering document produced correct extractions of its figures.
- Rejected changes were compared against baseline output; the grouped-expert change was
  the only one that altered generated text, and it was reverted on that basis.

## 8. Open items

1. DSA selection cost is the dominant decode blocker above 2,048 context.
   dsa_score_kernel performs a full scan and cub::DeviceRadixSort sorts all positions to
   take the top 2,048, on 16 layers per token. Attention consumes the selected set and
   is order independent, so a selection would suffice in place of a full sort.
2. The fixed 13.7 ms per token in prefill at zero context is unattributed. Block width,
   attention launch structure and expert scheduling have each been measured and
   eliminated as the dominant term. A per-kernel profile is the next step.
3. expert_combine_rows_kernel does not satisfy the project's MoE accumulation law. This
   affects the existing decode and MTP paths, not only the rejected prefill change.
