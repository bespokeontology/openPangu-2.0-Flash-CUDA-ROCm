# 21. DSA PREFILL SHIPS — sparse attention is flat in context, dense is not

## Result

Context ladder, 4 x MI50, PF_CHUNK=256, PF_PIPE_DEPTH=4, interleaved, first pair
discarded as warmup. Every pair disjoint.

| context | dense DSA | sparse DSA | gain |
|---|---|---|---|
| 4096  | 662.1, 662.0 | 697.7, 699.9 | +5.5% |
| 8192  | 566.1, 571.3 | 699.5, 699.9 | +23.1% |
| 16384 | 425.6, 428.3 | 678.5, 679.6 | +58.9% |

**The slope is the result, not any single point.** Dense attends the full causal
prefix on the 16 DSA layers, so it collapses as the prompt grows: 662 -> 566 -> 426.
Sparse caps attended positions at 2048 after selection and is nearly flat:
699 -> 700 -> 679. Attention card time at 16384 goes 103990 -> 50544 ms.

The selector becomes the growing term, exactly as intended: its stage goes
110 ms at 4096 (key cache only, no selection below the threshold) to 3581 ms at
16384, where it is 4.1% of card time. That is the optimisation problem the
architecture now poses - make selection cheap enough that sparse attention wins -
and it is a far better problem than making full-prefix attention faster.

## This is also a correctness fix

Above 2048 tokens the engine now reproduces the model's sparse attention instead
of attending to the full prefix. Every equivalence claim before this commit was
valid only below 2048 tokens.

## Why it beat its own budget, and why it did not reach the ceiling

Budget was ~63.5 tok/s from the window ceiling. The selector costs 7.3 tok/s at
4096, leaving ~56. Sparse delivered +35.7 tok/s there, not 56, and the reason is
measurable rather than mysterious: the window probe shared ONE contiguous 2048-key
range perfectly across a TQ=4 tile, while real per-query selections do not. Four
adjacent queries select 8192 slots that collapse to 3794 unique - high overlap,
but 3794/4 = 949 keys a query against dense's 4096/4 = 1024, only 7% better at
4096 context. At 16384 dense is 4096 a query against the same ~949, which is where
the 58.9% comes from.

727.8 was never the target. It was the performance of an attention kernel that
magically already knew a 2048-position window.

## What was built

- `k_pf_ix_key` / `k_pf_ix_query` - indexer key and query, model RoPE rule, key
  normed and query not, key written at EVERY position.
- `k_pf_idx_scores` - prefill-native: the 24x128 query staged once per block and
  reused across 256 positions, where decode re-read it per position. Query
  batching was rejected on arithmetic: 3072 FMA against 256 B is 12 FMA/byte
  against a machine balance near 8.8, so it is compute-bound at one query.
- `k_pf_idx_select` - decode's 3-pass radix, batched one block per query.
- `k_pf_union` - the mechanism that makes sparse attention viable at all. 16 VGPRs.
- `k_pf_attn_sp` - k_pf_attn2's inner mathematics with the causal range replaced
  by the union list and the causal mask by a per-slot membership mask. 119 VGPRs
  against the dense kernel's 123, so the mask costs no occupancy.

## Gates

- Selector: scores 3.388e-07 vs an independent host reference; selected set
  EXACTLY the host top-2048 on every checked query; negative control moves it.
- Sparse attention: 7.328e-07 vs a host reference attending exactly the masked
  subset, with irregular masks; clearing ONE mask bit moves 24384 outputs.
- Both reject non-finite before any reduction and range-check generated data.
