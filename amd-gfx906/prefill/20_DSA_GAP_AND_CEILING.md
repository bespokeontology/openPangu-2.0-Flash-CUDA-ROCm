# 20. PREFILL SKIPS DSA SELECTION ENTIRELY — correctness gap and a 9.5% ceiling

## The gap

openPangu runs sparse attention on the 16 layers where `index % 3 == 0`: the indexer
scores every cached position and attention consumes the **top 2048**. The decode engine
implements this — `if (w.dsa && npos > P92_IDX_TOPK)` runs `k_p92_idx_scores` then a
select, and passes `nsel = P92_IDX_TOPK`. Below 2048 positions it takes a dense path.

**The prefill body launches zero indexer kernels.** `k_pf_attn` sweeps the full causal
prefix on DSA layers. Above 2048 tokens this is a different computation from the
reference engine, not an approximation of it.

The layer-by-layer equivalence check in receipt 19 could not see this: it ran at 8
tokens, far below the threshold, where both engines take the dense path. Every
equivalence claim made so far is therefore valid only for prompts under 2048 tokens.

## The ceiling

`PF_DSA_WIN` bounds DSA layers with a trailing window. A window is **not** a top-k
selection and is **not** a correctness fix; it exists solely to bound what a correct
implementation could recover. 4096 tokens, chunk 256, four cards, kernel 2:

| DSA bound | tok/s | attn card ms | wall ms | overlap |
|---|---|---|---|---|
| none (full prefix, current) | 664.4 | 10832.13 | 6165.22 | 3.09x |
| 2048 | **727.8** | 9560.16 | 5628.15 | 3.16x |

**+9.5%.** For comparison, the whole k_pf_attn2 rewrite — DPP butterfly, exp2, gated
rescale, two keys in flight, six disjoint A/B pairs — was worth +2.07%. Doing less work
is worth 4.6x doing the same work faster.

And it grows. Uncapped DSA cost is O(n^2) in prompt length; capped is O(n * 2048). At
4096 the key-visit count falls 1.93e8 -> 1.60e8 (-17.4%); at 8192 the same cap removes
roughly half. The model's native context is 524288.

## What this does NOT say

It does not say a correct DSA prefill gains 9.5%. The indexer is not free: it scores
every position for every query. Per query on a DSA layer the indexer is 24 heads x 128
dims x npos against attention's 48 heads x 576 dims x npos, so roughly 11% of the
attention cost it is trying to cut by 25% — before the top-2048 selection itself, which
is the dominant decode cost at long context. The net could be anywhere from clearly
positive to negative, and prefill has T queries per chunk where decode has one.

The honest statement is: capping DSA work is worth 9.5% at 4096 and more at longer
contexts, an indexer costs some fraction of that back, and the engine is currently not
equivalent to the reference above 2048 tokens either way.
