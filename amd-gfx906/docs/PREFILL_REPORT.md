# Prefill report — frozen SPG2 authority (AMD gfx906)

## 1. Authority

| item | value |
|---|---|
| tag | `freeze-p92-prefill-spg2-20260909` -> `17d0fa9` |
| binary | `p92_pf_bench` sha256 `d5ad29f368d33d67bdcbae7dd7d12f2a53379bd55df0ddcc084c69fca503ae72` |
| receipt | `prefill/23_SPG2_SHIPS.md` (in this tree) |
| config | chunk 256, pipeline depth 4, 4 cards, TQ=2 sliding attention, SPG2 DSA attention |

## 2. Context ladder (frozen binary, production config)

| prompt tokens | tok/s | attention share | DSA pack ms (summed) | TTFT |
|---:|---:|---:|---:|---:|
| 4,096 | 710.0 | 52.4% | 194 | 0.65 s |
| 8,192 | 730.7 | 53.8% | 577 | 0.65 s |
| 16,384 | 732.4 | 53.8% | 1,363 | 0.65 s |
| 32,768 | 724.9 | 53.2% | 2,990 | 0.65 s |
| 65,536 | 698.2 | 51.2% | 6,380 | 0.65 s |

Near-flat through 32K, -3.5% at 64K. Overlap factor 3.73-3.80x of 4 cards.

## 3. SPG2 architecture

The DSA selection belongs to the QUERY: all 48 attention heads of one query
attend the same top-2048 key set, so the previous grid `(tiles, 48)` re-read
each selected key 48 times. SPG2 assigns a block TWO heads of ONE query
(TQ=1, G=2). Named register state equals the TQ=2 kernel's (2x8 query latent +
2x8 accumulator + 2 m/l pairs), measured 75 VGPR / 32 SGPR / 3 waves per SIMD /
0 spills, so the third resident wave survives while the DSA key traffic falls
~34%.

The union and per-query mask machinery is deleted for this path: with TQ=1 the
union of a selection IS the selection and every selected key attends. The
producer emits contiguous per-query key streams (latent 1024 B + rope 128 B per
slot) so the consumer is a pure pointer walk with constant strides - no index
loads, no per-key address math, no scalar-load dependency chain. The pack copy
runs ~780 GB/s with a coalesced (key, float4) mapping.

Sliding-window layers keep the previous TQ=2 dense path unchanged.

## 4. Measured improvement

Composed 64K, three interleaved pairs, warmup discarded:

| pair | index path (TQ=2) | SPG2 |
|---|---:|---:|
| 1 | 641.3 | 698.2 |
| 2 | 642.7 | 698.2 |
| 3 | 639.9 | 697.6 |
| median | 641.3 | **698.2** |

+8.9% composed; attention 220.5 s -> 180.5 s summed (-18.2%); the selection
pack adds 6.4 s summed.

Gate: G2 head-shared attention vs independent host reference 7.1e-07 worst
relative error with a variable selection count per query (exercises the walk
clamp); negative control moves outputs. Tolerance gate, not bit-exactness.

## 5. Memory / residency envelope

| context | verdict |
|---:|---|
| 4K-64K | runs; ~5.5 KB of context storage per position per card |
| 131,072 | fails: the 5.24 MB peer-landing allocation returns out of memory after the MAXPOS-scaled DSA caches; the wrapper's message ("peer transport refused") is misleading, the cause is allocation |
| 262,144 | fails: HIP out of memory during the per-layer DSA cache allocation |

No kernel, selector, union-LDS or index-capacity assumption breaks at either
length; the ceiling is VRAM residency on 4x16 GB cards. Recorded as
characterization only.

## 6. Negative results (banked, not deleted)

| experiment | result | verdict |
|---|---|---|
| TQ=8 sparse attention (1 wave/SIMD, ~0.6x key traffic) | 414.8 / 422.5 tok/s at 64K vs TQ=2 669.2 / 658.9 | rejected: occupancy dominates traffic |
| TQ=4 (2 waves) | 599.7 / 593.8 tok/s | superseded by TQ=2 |
| packed-union TQ=2 with uncoalesced copy | attention -2.7% but dsauni 0.99 -> 20.4 s; net -3.9% | rejected as-is; coalescing fix retained for SPG2 |
| union-index software pipeline (compiler) | compiler re-serialised the rotation; no gain | rejected |
| cooperative multi-wave attention | refuted on occupancy arithmetic, then again by the TQ=8 result | rejected |

## 7. Large-M prefill work continues

Additional large-M prefill work (batching more tokens per attention tile and per
expert group, and revisiting sliding-layer traffic, which is now the dominant
attention component at 12.3 TB/card) may continue after this release. It is not
part of the frozen authority above.
