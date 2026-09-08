# Prefill: measurements and two rejected changes

Prefill is the weakest component of this engine. This document records what was measured
and what was tried, so the next attempt does not repeat it.

## Scaling

One model load, six successive prompts of roughly 700 tokens each, so each prompt is
prefilled at a larger starting context. DGX Spark GB10, page cache dropped before the
run, context limit 8192.

| prompt tokens | wall | ms/token | context before |
|---|---|---|---|
| 740 | 10.13 s | 13.70 | 0 |
| 908 | 17.30 s | 19.05 | 744 |
| 859 | 20.18 s | 23.49 | 1,656 |
| 667 | 16.68 s | 25.01 | 2,519 |
| 711 | 17.84 s | 25.09 | 3,190 |
| 630 | 15.85 s | 25.16 | 3,905 |

Cost per token rises with context and then flattens at approximately 25 ms per token
beyond about 2,500 context. The plateau is consistent with the 512-token sliding window
on layers where `index % 3 != 0` and the DSA indexer's top-2048 selection: once context
exceeds both, the number of keys attended per query stops growing.

Two components are therefore visible. A fixed per-row cost of about 13.7 ms per token,
measured at zero context before any cache exists, and an attention term that grows with
context to about 11.5 ms per token and then stops.

## Structure

`Model::prefill` (`src/model.cpp`) splits the prompt into blocks of
`kMaximumBlockRows = 64` and calls `prefill_block` for each. Within a block, projections
go through `ProjectionExecutor::project_rows`, which issues a single NVFP4 tensor-core
GEMM with a `batch_rows` dimension.

Attention is selected per layer in `run_attention_rows`:

```
if (start_position + rows <= cache_capacity)
    mla_attention_prefill_rows_tiled(...)
else
    for (int row = 0; row < rows; ++row)
        mla_attention_decode_circular_tiled(...)
```

`cache_capacity` is the sliding window on sliding layers and the context limit on
non-sliding layers. Both branches call the same device kernel,
`latent_attention_tiled_kernel`, which carries a multi-row grid (`row = blockIdx.y`,
bounded by `query_rows`) and a causal rule (`cache_count += row * cache_count_step`).
`mla_attention_prefill_rows_tiled` uses that multi-row grid only when
`batch_rows <= 4`; above that it also loops rows.

## Change 1: block width 64 to 512 — rejected

The tensor-core GEMM has no row cap; it rejects only `batch_rows <= 0`. The 64 bound
was a caller-side constant in `src/model.cpp` and a guard in `src/executor.cpp`. All
per-block scratch scales linearly with the row count at roughly 200 KB per row, and
`block_logits_` is sized by speculative rows rather than block rows, so 512 rows costs
about 100 MB against roughly 39 GB of unused budget.

| configuration | prefill, 3,807 tokens | decode at 4,008 context |
|---|---|---|
| 64 rows (baseline) | 74.08 s | 15.2955 tok/s |
| 512 rows | 76.64 s | 14.7236 tok/s |

Rejected. Wider blocks did not reduce prefill time.

## Change 2: fused multi-row attention launch — rejected

Because change 1 left total work unchanged, the row loop inside
`mla_attention_prefill_rows_tiled` was removed so that any block issues one launch over
the existing multi-row grid rather than one launch per row. This used capability
already present in the kernel; no kernel was written. Wrapped sliding-window layers kept
the existing per-row circular path.

| configuration | prefill, 3,807 tokens | decode at 4,008 context |
|---|---|---|
| 64 rows (baseline) | 74.08 s | 15.2955 tok/s |
| 512 rows + fused multi-row launch | 78.48 s | 14.5803 tok/s |

Rejected. Collapsing 64 launches into one did not reduce prefill time, which indicates
the cost is the per-query key arithmetic rather than launch structure or block width.

## What the evidence rules out

- Block width. Measured directly, twice, at 8x the baseline width.
- Launch count for attention. Measured by fusing the launches.
- Expert scheduling as the dominant term. A separate experiment moved 64-row prefill MoE
  from the host-scheduled per-expert loop onto the existing device-grouped path and
  measured 74.08 s to 60.48 s, about 1.23x. That change also altered generated output
  and reduced MTP acceptance from 132/195 to 118/240, because
  `expert_combine_rows_kernel` accumulates in FP32 over top-k slot order while the host
  path accumulates in ascending expert id, and the project's MoE accumulation law
  requires the latter. It was not retained.

## What remains

The fixed 13.7 ms per token at zero context is unexplained by the above and is the
component to attribute next. A per-kernel profile of one prefill block, rather than
further structural changes, is the appropriate next step.
