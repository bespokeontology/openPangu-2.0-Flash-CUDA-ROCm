# 20 — Decode profile: where the 14.9 ms actually goes

Measured on 4x gfx906 (16 GB, 1700 MHz), ROCm, single stream, MTP OFF, greedy.
This receipt records a **profile**, not an optimization: it establishes which wall
the decode engine is actually against, and it corrects two earlier diagnoses.

## Artifacts

| artifact | sha256 |
|---|---|
| `p92_gen_v2` (326,840 B) | `59a25c51e8467875b3d41f9202f88935a95ccb43e6cbd385a7cfa84ce08510d0` |
| `src/p92_dsa_scores2.hip` | `9cb42cf17416322337f3e48e2af5831501eeef493827b530be56016b03a46c7f` |
| `tests/p92_generate.hip` | `ee5258a9e9c8c6155481c4de84c1fba692ce53fca0741753744795caee07b688` |

Build is a single translation unit (the harness includes every kernel):

```
hipcc --offload-arch=gfx906 -O3 -std=c++17 -Iinclude -I. tests/p92_generate.hip -o p92_gen_v2
```

Profile method (`rocprof` 5.7, per-dispatch counters; the timestamped variant was
used for the device-overlap question):

```
rocprof --stats -o stats.csv ./p92_gen_v2 <ckpt> <artifact> <arena> 500 4096 148899 <prompt>
```

## 1. Kernel time is ~100% of the wall (512 context)

500-token run. Sum of all kernel durations over all devices: **7.457 s =
14.91 ms/token**. Wall clock measured in the same run: **14.82 ms/token**
(69.9 tok/s). Ratio 1.005.

There is no hidden host wall at short context. The host rendezvous in the token
loop (the per-token staging barriers) costs a fraction of a millisecond here, not
milliseconds: it is not the thing to attack first.

## 2. Cost distribution per token (46 trunk layers)

`calls/token` = dispatches per generated token; `ms/token` derived from the
profile's share of the 14.91 ms total.

| kernel | calls/token | ms/token | share |
|---|---|---|---|
| `k_p92_f4_gateup` (MoE gate+up) | 44 | 2.094 | 14.0% |
| `k_p92_mla_g16` | 46 | 1.616 | 10.8% |
| `k_p92_f4_downacc` (MoE down) | 44 | 1.225 | 8.2% |
| `k_p92_proj<6144,64>` | 46 | 1.010 | 6.8% |
| `k_p92_mhc_collapse` | 92 | 0.782 | 5.2% |
| `k_p92_mla_absorb` | 46 | 0.712 | 4.8% |
| `k_p92_proj<1024,32>` | 46 | 0.599 | 4.0% |
| `k_p92_mla_value_up` | 46 | 0.592 | 4.0% |
| `k_p92_route_top8` | 44 | 0.547 | 3.7% |
| `k_p92_rmsnorm_quant<2560>` | 92 | 0.494 | 3.3% |
| `k_p92_mhc_logits` | 93 | 0.436 | 2.9% |
| `k_p92_head_logits` | 1 | 0.430 | 2.9% |
| `k_p92_proj<2560,32>` | 46 | 0.413 | 2.8% |
| `k_p92_rmsnorm<2560>` | 93 | 0.366 | 2.5% |
| `k_p92_bf16_gemv<2560>` | 46 | 0.330 | 2.2% |
| `k_p92_mhc_reexpand` | 92 | 0.291 | 2.0% |
| `k_p92_route_gemv` | 44 | 0.250 | 1.7% |
| `k_p92_quant<6144>` | 46 | 0.238 | 1.6% |
| `k_p92_mla_combine` | 46 | 0.233 | 1.6% |
| `k_p92_rmsnorm_quant<1024>` | 46 | 0.209 | 1.4% |

Top 20 = 85.5%. The 46 trunk layers are **44 MoE + 2 dense**
(`k_p92_dense_gateup`/`k_p92_dense_down` at 2 calls/token), and the DSA index
path is active on **16 of 46** layers (`k_p92_ix_key` = 16 calls/token).

`k_p92_head_logits` is a single dispatch per token at **430 us**: it streams the
whole vocabulary head and is bandwidth-bound, ~0.43 ms is close to the floor for
that weight volume at achievable HBM rate.

## 3. The DSA index/select machinery is negligible

`idx_pick` + `idx_hist` + `idx_write` + `idx_count` + `idx_scan` + `idx_pad`
together are **0.52% of token time = 0.078 ms/token**. Whatever else is true,
rewriting the index selector cannot pay for itself. This is the number that
retires a line of work.

## 4. The four devices do not overlap during decode

Layer ownership is contiguous (12/12/11/11) and the token crosses blocks through
3 peer handoffs, so the block chain is a strict dependency chain. Two independent
measurements agree with that structure:

- aggregate kernel time per token (14.91 ms) ~= wall per token (14.82 ms);
- in the timestamped 20-token trace, per-device busy time over the whole trace
  (load + prefill + 20 decode steps) was **74.4 / 80.6 / 81.7 / 86.1 ms**,
  i.e. ~3.7 ms of kernel work per device per decode token.

So at any instant roughly one card is executing and three are waiting for their
stage of the chain. The 4 cards buy **capacity** (weights resident), not speed:
per-token wall = sum over cards of that card's stage time.

Occupancy of the hot kernels is not the problem either: `k_p92_mla_g16` 80 VGPR
(3 waves/SIMD), `k_p92_proj<6144,64>` 56 (4), `k_p92_proj<2560,32>` 48 (5),
`k_p92_bf16_gemv<2560>` 44 (5).

## 5. Context scaling: the step is at 2048, not a linear scan

Same binary, greedy, single stream:

| context | ms/token | tok/s |
|---|---|---|
| 512 | 14.82 | 67.5 |
| 5,669 | 19.83 | 50.4 |
| 43,135 | 20.46 | 48.9 |

512 -> 5,669 costs +5.0 ms; 5,669 -> 43,135 costs **+0.63 ms for +37,466
positions**. The decode cost is therefore dominated by the one-time regime switch
into sparse attention at 2,048 positions (selector + sparse read of the 2,048
selected slots), not by an O(context) scan. An earlier claim in receipt 19's
follow-up that the index scan makes long-context decode expensive was **wrong**;
this table is the correction.

## 6. Second scorer rewrite: measured, small, kept

`k_p92_idx_scores_v2` (`P92_IDX_SCORER=2`) restructures the DSA score kernel to
64 positions x 4 dim-chunks per 256-thread block with a 24 KB LDS partial array,
collapsing the serial chain from 128 to 32 steps. Same arithmetic as v1.

| 4,096 context | ms/token | tok/s |
|---|---|---|
| v1 (one thread/position) | 17.87 | 55.94 |
| v2 | 17.54 | 57.01 |

**+1.9%**, and the greedy token stream is byte-identical between arms (discrete
contract: identical selection -> identical tokens). Kept, and it is also the
ceiling: with the whole index path at 0.5% of token time, no further scorer work
can return more than that.

## What this implies for the next build

1. **Latency**: the only way to use the three idle cards is to stop partitioning by
   layer and start partitioning *within* a layer (tensor parallel on the hotspots:
   MoE 22.4%, MLA 19.6%, projections ~13.5%). Ceiling is bounded below by the
   collective cost, which the tensor-parallel Qwen engine already measures at
   1.7-3.2 ms/token for 129 collectives on the same hardware.
2. **Throughput**: because each card is idle ~75% of every token, running several
   sequences staggered through the existing block chain multiplies throughput
   without touching weights or kernels. This is the cheap version of the same
   insight and it serves the server/ingest path first.
3. **Not** worth doing: more index-selector work (0.5% ceiling), and host-rendezvous
   removal as a first move (kernel time is ~100% of the wall at short context).
