# Two correctness defects on the unreachable DSA path

Found 2026-09-09 by a dispatch census, verified by hand against the CUDA oracle.
**Neither has ever affected a measurement or an output**, because both live on a
path the production workload cannot reach. Both would produce silently wrong
output at long context. Not regressions - pre-existing, and never exercised.

## Defect 1: the DSA index cache is populated only after position 2048

`w.ix_cache` is written at `tests/p92_generate.hip:431` and read at `:438`.
**Both are inside branch A**, guarded at `:427` by `w.dsa && npos > P92_IDX_TOPK`.
Nothing else in the engine touches it (grep-verified: lines 117, 305, 431, 438).

So the first time `npos` reaches 2049, `k_p92_idx_scores` reads 2049 rows of the
index cache of which exactly one - the current position, written moments before
at :431 - was ever initialised. The other 2048 are raw `hipMalloc` contents.

The oracle does not do this. In BOTH its paths the key is stored
unconditionally, and the top-k test comes AFTER the store:

    decode   model.cpp:1624 project -> 1625 rmsnorm -> 1631 dsa_rope -> 1637 store
    prefill  model.cpp:1399 project_rows -> 1400 rmsnorm -> 1406 dsa_rope_rows
                            -> 1411 store -> 1416 if (start+rows <= kDsaTopK)

`kDsaTopK` at :1416 and :1427 selects which ATTENTION path runs. It does not
gate the cache write.

**Failure mode is silent.** No fault, no NaN. Uninitialised VRAM that happens to
be zero gives every position an identical score, the whole array lands in the
tie band, and the ascending gather returns positions 0..2047 - the model attends
to the oldest 2048 tokens and ignores everything since, fluently.

**Fix**: hoist the index-key projection, the norm and the cache write out of
branch A so they run at every position, matching model.cpp:1624-1637.

## Defect 2: the indexer k_norm is loaded and never applied

`w.ix_knorm` is assigned at `tests/p92_generate.hip:282` from
`model.layers.%d.self_attn.indexer.k_norm.weight` and referenced nowhere else in
the engine (grep-verified: lines 115 and 282 only).

The oracle applies it between the key projection and the RoPE, in both paths:
`cuda::rmsnorm(index_key_, layer.index_key_norm, index_key_, ...)` at
model.cpp:1400 and :1625.

**Fix**: insert `k_p92_rmsnorm<128>` with `w.ix_knorm` between the projection
and `k_p92_ix_key`.

## Why neither has ever fired

`npos = step + 1`, and the production command generates 220 tokens with no
prefill, so `npos <= 220 < 2049`. Structurally impossible to reach, not merely
unlikely. Independently, at `MAXPOS=512` the caches are sized `cap=512`, so a
run long enough to reach 2049 would write past the allocation before the branch
fired: reaching branch A requires raising BOTH `NTOK` and `MAXPOS`.

## Correction to a claim made earlier today

`receipts/10_GFX906_PLAYBOOK.md` section 10 said the whole of
`src/p92_dsa_index.hip` is unreachable. **That was wrong**, and this is the
second correction to the same claim.

Six of its kernels - `idx_hist`, `idx_pick`, `idx_count`, `idx_scan`,
`idx_write`, `idx_pad` - run on EVERY token, in the head's top-256 refine at
`tests/p92_generate.hip:608-616`. Only `k_p92_idx_scores` and
`k_p92_idx_select` are DSA-exclusive and unreachable.

The precise statement, which is more useful than the one it replaces: those six
kernels are exercised at exactly ONE shape, `n = 151552, want = 256,
nchunk = 37`. The DSA shape - `n = npos, want = 2048` - is never instantiated.
By playbook section 9, being exercised at one shape grants no authority at
another. So the selector code is *tested*, but nothing about its behaviour at
the DSA shape is established.

## Measurement note: disk I/O inside the timed region

`tests/p92_generate.hip:548` starts the per-token clock; `:549` calls
`ck.embed_row(token)`, which `::open()`s the safetensors shard, parses its
header and `pread`s the row - **per token, inside the timing**.

This is inside every tok/s figure this project has banked, 34.60 through 69.92.
It does not invalidate any A/B comparison, since both arms pay it identically
and the file is page-cached after the first token, but the absolute figures
include per-token file I/O whose cost depends on page-cache state - a runtime
variable inside the measured region.

It matters much more for prefill, where a T-token chunk would pay T opens.
Hoist the embedding lookup out of the loop, or out of the timed region, before
prefill numbers are taken.
