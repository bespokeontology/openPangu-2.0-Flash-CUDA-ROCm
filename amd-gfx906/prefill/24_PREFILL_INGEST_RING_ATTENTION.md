# 24 — Prefill ingest in the server, ring ownership, blocked attention, determinism (2026-09-12)

Every number below is from a run under the machine lock on the four-MI50 box, interleaved with another
lane's runs on the same cards; raw outputs are in the freeze directory of this release
(`p92-serve-ingest-20260912`). Real prompts throughout: the model's own tokenizer (`p92_encode chat`) over
this project's engineering documents plus a question — 1,117 / 4,446 / 9,405 / 38,716 tokens.

## 1. Defect: the frozen prefill pipeline never delivered chunk state to cards 1-3

`p92_pf_bench.hip` pushed each chunk's four-stream state into the next card's peer landing buffer, but no
consumer copied the landing buffer into the buffer its kernels read, and nothing waited on the arrival
flag. Cards 1-3 computed every chunk on uninitialised state. Same kernels, same launches, same traffic —
the 710-732 tok/s ladder measured the cost of the correct computation — but no output past card 0 was
meaningful, consistent with every equivalence receipt (L0-L2) having been taken on card 0. Fixed in the
bench and the server: the consumer copies its landing slot locally, several slots per card, a position
counter for back-pressure. Cost: none (synthetic 4K 707 tok/s on the fixed build, 684-693 for the frozen
binary the same hour).

## 2. What the shipped product did, and the new ingest path

`p92_serve` / `p92_chat` walked the prompt one decode step per token: 64 tok/s, 18.4 s to the first
generated token on the 1,117-token prompt. The prefill bench was never wired to the decoder (receipts 21-23
named this build). Now the chunk pipeline runs in-process (`P92_PREFILL=1`, default): the decoder continues
from the same caches (every layer cap = MAXPOS, absolute slots; the arena's expert scales are flipped
between the prefill and decode layouts at the phase boundary and verified as a permutation; the prefill
covers all but the last prompt token).

| real 1,117-token prompt, contiguous arena | token-walk (shipped) | prefill ingest |
|---|---:|---:|
| ingest | 18.4 s (64 tok/s) | 1.86-2.06 s (541-601 tok/s) |
| time to first generated token | 18.4 s | 1.9-2.1 s |
| greedy continuation, 33 tokens | reference | identical 33/33 |
| decode at the same positions | 16.62 ms/token | 16.66 ms/token |

The in-server prefill equals the standalone bench on the same prompt (586-593 vs 593-595 tok/s, two
interleaved pairs).

## 3. Ring ownership (the short-prompt lever)

`p92_pack_arena ART OUT 4` packs an arena where card `b % 4` owns layer block `b` (4 layers); the arena
header records the block size and the engine reads its ownership map from the arena it loads. The prefill
is a block schedule: per-card job lists in readiness order, landing slots by job position, per-job done
flags, a position counter for back-pressure (a first version deadlocked because seed jobs did not advance
it). Decode crosses cards 11 times a token instead of 3.

Same binary, contiguous vs ring arena (per-head-pair attention, atomic down projection):

| prompt | contiguous (overlap of 4) | ring (overlap) | gain | decode after, contig / ring |
|---|---:|---:|---:|---:|
| 1K | 541-601 tok/s (2.2x) | 805-851 (3.14x) | +36-49 % | 16.9 / 17.3 ms |
| 4K | 673 (3.13x) | 769 (3.55x) | +14 % | 20.2 / 20.1 ms |
| 8K | 711 (3.47x) | 755 (3.70x) | +6 % | 20.2 / 20.3 ms |

## 4. Determinism

The engine was nondeterministic run to run: `k_pf_down` scattered expert contributions with fp32
`atomicAdd`, so the summation order across a token's nine experts varied and a near-tied first token
flipped ("The user wants me to…" vs "First, I need to…", both coherent, same prompt, same configuration).
`PF_DOWN_DET=1` (default): each grouped row writes its own partial, a combine adds a token's nine partials
in a fixed order, no atomics. Gate: two contiguous runs identical (33/33 greedy tokens), the ring run
identical to them; the down bucket also fell 590 -> 512 ms at 1K (-13 %).

## 5. Blocked attention

`k_pf_attn_blk` (`PF_ATTN_BLK=1`): one query x 24 heads x 32 lanes per block, key tiles staged once in LDS
and applied to all heads (2 x 1152 B per query-key instead of 24 x); window and selection modes (the DSA
layers gather the selection from the cache, no packed streams). 83 VGPR / 0 spills / 3 waves per SIMD.
Gate (`tests/p92_pf_blk_test.hip`): worst relative 2.2e-6 vs an independent host reference in all modes,
0 of 983,040 outputs differ > 1e-3 from `k_pf_attn2<2>` / `k_pf_attn_spg2`, negative control moves
983,030 of 983,040. In the engine (contiguous arena): attention bucket -11 % at 1K, 4K and 8K; greedy
tokens identical at 1K and 8K.

Why only 11 %: the production kernels are not traffic-bound at these contexts (the window sits in L2);
they are issue-bound — every bf16 element costs an unpack before its FMA, and the cross-lane reductions ride
the LDS path inside a dependent chain. v1 has the same instruction mix per MAC.

`k_pf_attn_blk2` (`PF_ATTN_BLK=2`): the tile is converted bf16 -> fp16 once while it is staged, the score is
18 `v_dot2_f32_f16` per lane (no unpack), the 16-lane head row is summed by four DPP row-rotates (no LDS-path
op), fp32 accumulation; 24 heads x 16 lanes a block, 106 VGPR, no spills. Gate: worst relative 1.7e-6 vs
the host reference, 1.3e-6 vs the production kernels, negative control moves 983,030 of 983,040. Engine
A/B: see §8.

## 6. Projections on rocBLAS (`PF_PROJ_RB=1`, default after §8)

Load-time int8 per-row mirrors of the NVFP4 q_a / q_b / o_proj / indexer wk, wq_b; fp32 copies of the bf16
kv_a, router gate and weights_proj on sgemm (which also retires the per-token GEMV loops that re-read the
whole weight once per token). rocBLAS loads Tensile code objects lazily on the first call of a shape, ~1 s
a device across the projection shapes — measured inside the first prompt (1K: +1055 ms) — so every
production shape is now run once at init. Warm, at 8K: projection bucket 6262 -> 4589 ms (-27 %), +7.7 %
prefill tok/s, identical tokens; at 4K unchanged (the warm-up was inside the measurement). Per-row weights
and per-token activations are coarser than the hand-written path's per-16 groups; the token streams in §8 decided
the default.

## 7. Device-side decode chain (`P92_CHAIN=1`, default)

Each crossing was a host synchronisation plus a pinned-memory round trip. Now the producer card stores the
20 KB state straight into the consumer's buffer and records an event the consumer's stream waits on; the
host does not synchronise until the head. Ring arena, 128 generated tokens: 17.40 -> 16.89 ms/token;
contiguous: 16.82 -> 16.81; tokens identical in both. The ring's decode cost is recovered.

## 8. Engine A/B on the ring arena (deterministic down, chain on; build 877a157f)

| prompt | per-head-pair kernels | blocked v1 | blocked v2 | v1 + rocBLAS projections (warm) |
|---|---:|---:|---:|---:|
| 1K | 848-851 tok/s | 712 | **871** | **1006** |
| 8K | 755 (earlier build) | 826 | **900** | **942** |

Tokens: v2 and the rocBLAS arm reproduce the reference stream at 1K (33/33). At 8K the three arms agree on the
first 20 (v1 vs v2), 20 (v1 vs rocBLAS) and 23 (v2 vs rocBLAS) of 33 greedy tokens and then diverge at a
near-tie; every continuation is coherent and on task. The v1 stream of this build equals the v1 stream of the
previous build 33/33 (deterministic across builds). v1's 12-wave lockstep blocks fit
the ring's 4-layer jobs badly (712 at 1K, and the earlier release-candidate run's 4K overlap fell from 3.55x to
2.87x); v2's 6-wave blocks do not. Defaults set from this table: `PF_ATTN_BLK=2`, `PF_PROJ_RB=1`.

Release-candidate run of the previous build (ring, v1, deterministic down, chain): 8K 812 tok/s, 32K 852.5 tok/s
(45.4 s), continuations coherent and on task.

## 9. Frozen binary (`p92-serve-ingest-20260912`, sha256 b7815f48…) — confirmation runs on its defaults

Ring arena, no environment switches (`P92_TEMP=0` for greedy), real prompts, 32 generated tokens:

| prompt tokens | prefill | wall | time to first generated token | decode after | old attention (`PF_ATTN_BLK=0`) |
|---:|---:|---:|---:|---:|---:|
| 1,116 | **988.9 tok/s** | 1.13 s | 1.15 s | 16.74 ms/token | — |
| 4,445 | **1037.7 tok/s** | 4.28 s | 4.30 s | 20.29 ms/token | 848.8 tok/s, identical tokens |
| 9,404 | **1013.2 tok/s** | 9.28 s | 9.30 s | 19.84 ms/token | 865.5 tok/s, identical tokens |
| 38,715 | **963.8 tok/s** | 40.2 s | 40.2 s | 20.54 ms/token | — |

Continuations: the 1K reference stream ("The user wants me to summarize the three most important measured
results from the document, each with its number and the condition it was measured under. Let me analyze the
document…"); on task and coherent at 4K, 8K and 32K. Against the v1.1 product: 64 tok/s ingest and 18.4 s to the
first token on the 1K prompt; against the v1.1 bench ladder (710-732 tok/s, synthetic, cards 1-3 uninitialised).

Build, run and switches: `amd-gfx906/README.md`. The freeze directory holds the binary, md5/sha256, the source
archive at the commit, the build line, the arena packing command and the raw receipt outputs.
