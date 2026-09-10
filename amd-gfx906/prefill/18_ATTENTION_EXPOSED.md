# 18. ATTENTION IS FULLY EXPOSED — wall-sensitivity pair, 2026-09-09

Composed openPangu prefill, 4096 tokens, PF_CHUNK=256, PF_PIPE_DEPTH=4, four MI50s.
Binary /tmp/pfbench_sens at 64eca56. Each run under the machine-wide q27bench lock.

## The question

`attn` is 57.7% of summed card time. That is a profiler percentage, and profiled
milliseconds are not recoverable milliseconds: work sitting before a rendezvous
can be improved to no effect at all. Before spending a kernel campaign there, the
bucket has to be shown to convert into WALL.

## Method

Duplicate a known quantity of the exact work and watch the wall. The probe names
a KERNEL, not a stage, because duplicating a stage measures what you intend only
if re-running it is a no-op — and several stages here are not: `k_pf_mhc_scale`
multiplies the mixing logits in place, `k_pf_mome_st` advances a chunk carry.
`k_pf_attn` writes `c.ctx` from unchanged sources and repeats cleanly.

- **mode 1** duplicates it WHERE IT RUNS.
- **mode 2** duplicates the SAME kernel, the SAME number of times, at the tail of
  each card's work, where nothing can hide it. This is the positive control.
  Position is the only variable between the two.

Both count launches and print the count, because a flat wall and a probe that
never issued are otherwise indistinguishable.

## Result

| run | wall ms | d(wall) | attn card ms | SUM ms | overlap | launches |
|---|---|---|---|---|---|---|
| baseline        | 6326.98 | —       | 11238.38 | 19491.29 | 3.08x | 0   |
| mode 1 in-place | 9076.99 | +2750.01| 19557.77 | 27769.81 | 3.06x | 736 |
| mode 2 tail     | 9054.56 | +2727.58| 11187.32 | 19386.43 | 2.14x | 736 |

**Instrument validated.** The tail control moved the wall by 2727.58 ms for
8279 ms of added card work at an effective parallelism near 3 — nearly
proportional, as required. Had it not, every number here would be void.

**Exposure ratio = 2750.01 / 2727.58 = 1.008.** Attention in its real position
converts to wall at the same rate as work in a position where it cannot be
hidden. The bucket is fully exposed.

Mode 2's `attn` bucket correctly does NOT rise: its extra launches are outside
the stage timer. Its wall rises anyway, so the overlap factor collapses from
3.08x to 2.14x. That divergence between a flat SUM and a growing wall is the
signature of work that nothing is hiding, and it is why overlap alone cannot be
read as a health metric.

## Consequence

Conversion rate: 2750.01 ms wall per 8319.39 ms of attention card time = 0.331.
Halving attention recovers 11238.38/2 * 0.331 = 1860 ms of a 6326.98 ms wall,
i.e. 647 -> ~916 tok/s. The cooperative multi-wave kernel campaign is justified
on measurement rather than on the profiler percentage.

## What this does NOT say

It does not say attention can be halved. It says that if it is, the wall follows.
