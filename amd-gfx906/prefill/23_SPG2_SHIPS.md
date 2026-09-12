# 23 — SPG2 SHIPS (2026-09-09/10)

G=2 head-shared sparse attention, banked on the composed pipeline at 64K.

## The chain of reasoning (all measured)

1. TQ discriminator at 64K (3 interleaved pairs... 2 pairs + warmup):
   TQ=2 669.2/658.9, TQ=4 599.7/593.8, TQ=8 414.8/422.5 tok/s.
   OCCUPANCY DOMINATES TRAFFIC. TQ=2 (3 waves/SIMD) is the knee; TQ=8 (1 wave)
   is catastrophic. The 589-vs-602 scorer discrepancy resolves to ~597 honest.
2. Attention at 64K already runs at ~75-80% of its byte floor (37 TB/card
   moved against a 41 s floor; measured 51-55 s). Removing the union-index
   scalar-load chain (packed-union walk, bit-exact) moved attention only
   -2.7% and the uncoalesced pack copy cost +19.4 s: packed-TQ2 AS-IS
   REJECTED (-3.9%). The index chain was not the big remaining cost.
3. The remaining lever is traffic WITHOUT losing the third wave. The DSA
   selection is per QUERY - all 48 heads attend the same top-2048. A block
   serving TWO heads of ONE query (TQ=1, G=2) holds the same named state as
   TQ=2 (measured 75 VGPR / 32 SGPR / 3 waves/SIMD / 0 spills) and halves the
   48x head re-read factor: DSA key traffic -34%.

## The build

- k_pf_attn_spg2: 2 heads x 1 query per block over selection-packed streams.
  No union, no masks (every selected key attends), two running pointers with
  constant strides, one dependent load per block (the nsel clamp). The ISA
  walk is pure pointer arithmetic: v_add_co with 0x80/0x400 immediates, staged
  vmcnt waits, zero scalar-load chain.
- k_pf_sel_pack: coalesced (key, float4) copy - thread (w=t>>6, f=t&63) so a
  wave writes four CONSECUTIVE keys (4 KB contiguous per store instruction).
  The first mapping (one whole key per thread) was 12.5% sector efficiency /
  26 GB/s; the fixed copy runs ~780 GB/s. (Same fix in k_pf_union_pack.)
- Sliding-window layers stay on k_pf_attn2<2>. Unchanged.
- MoE fixed-grid launch (dead-slot order contract) removes the blocking D2H
  live-group read per MoE layer. No regression.

## Gates

sparse gate (tests/p92_pf_sparse_test.hip), all on the same binary:
  index path vs host reference:   7.328e-07 worst relative
  packed vs index path:           bit-exact (free; NOT a requirement)
  spg2 vs host reference:         7.114e-07 worst relative (variable nsel,
                                  exercises the walk clamp)
  negative control:               mask bit cleared -> outputs move
Numerical gates use local tolerance + host reference; bit identity is not the
authority (banned as a gate).

## Composed result, 64K, interleaved, warmup discarded

| pair | INDEX-TQ2 | SPG2 |
|---|---|---|
| 1 | 641.3 | 698.2 |
| 2 | 642.7 | 698.2 |
| 3 | 639.9 | 697.6 |

+8.9% median. attention 220.5 -> 180.5 s summed (-18.2%); dsauni 0.99 -> 6.4 s
(the pack copy - its own traffic, measured, not a defect).

## Frozen

tag freeze-p92-prefill-spg2-20260909 -> 17d0fa9
<data>/frozen/p92-prefill-spg2-20260909/p92_pf_bench
sha256 d5ad29f368d33d67bdcbae7dd7d12f2a53379bd55df0ddcc084c69fca503ae72

## Open

- Context ladder (4K-32K) to confirm the win is not 64K-specific.
- The 6.4 s pack copy is the next dsauni cost; the selection could be packed
  once per position instead of per (chunk, layer) if the DSA layers shared a
  packed cache across chunks - open question, not yet a build.
- G=2 leaves the sliding layers untouched: 12.3 TB/card of their traffic is
  now the dominant attention component. Revisit only if the profile says so.

## Context ladder (frozen binary, production config)

| context | tok/s | attn share | dsauni ms |
|---|---|---|---|
| 4096 | 710.0 | 52.4% | 194 |
| 8192 | 730.7 | 53.8% | 577 |
| 16384 | 732.4 | 53.8% | 1363 |
| 32768 | 724.9 | 53.2% | 2990 |
| 65536 | 698.2 | 51.2% | 6380 |

Near-flat through 32K; -3.5% at 64K. Prefill rung FROZEN.

## Envelope characterization (frozen binary, beyond 64K)

- 131072: fails at pf_pipeline_init ("peer transport refused") - actually VRAM
  exhaustion: the 5.24 MB p2p landing alloc fails after the MAXPOS-scaled DSA
  caches (cache_kv 1024 B/pos/DSA-layer, idx_cache 512 B/pos, scores ~1 KB/pos)
  push the tightest card past 17.16 GB. Misleading message, allocation cause.
- 262144: HIP out of memory at the per-layer DSA cache alloc (@404).
- No kernel, selector, union-LDS (spg2 has none), or index assumption broke.
  The engine would run at 128K+ if the DSA cache residency were reduced (e.g.
  INT8 index keys or on-demand cache regions). Characterization only - not
  a build.
