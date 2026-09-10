# OPENPANGU PREFILL — HANDOFF 2026-09-09 night

Tree: `/srv/pangu/p92-prefill` on **amd-host** (4x MI50, gfx906).
Frozen decode engine `/srv/pangu/p92-amd` — **never modify; copy to /tmp to patch.**

## 1. WHERE IT IS

Composed native prefill, all 46 layers, 4 cards, WITH the model's sparse DSA.

| context | tok/s | vs dense-DSA |
|---|---|---|
| 4096  | 698 | 662 |
| 8192  | 700 | 568 |
| 16384 | 684 | 427 |
| 32768 | 636 | — |
| 65536 | 602 | — |

Dense collapses with prompt length; sparse is near-flat. That is the whole result.

## 2. FREEZES (binaries archived + SHA256 + RECEIPT, in `/srv/pangu/frozen/`)

`/tmp` is wiped at boot on both boxes. Never freeze there.

| tag | commit | sha256 | what |
|---|---|---|---|
| `freeze-p92-prefill-664-20260909` | 4e2f7c4 | af426da8… | dense baseline, comparison arm |
| `freeze-p92-prefill-dsa-20260909` | de6414b | 8e4b6ddd… | sparse DSA shipped |
| `freeze-p92-prefill-scorer-v3-20260909` | ac8cac9 | 093bf90f… | **current authority** |

## 3. CORRECTNESS — verified BOTH sides of the 2048 DSA threshold

- **Below 2048:** layer 0 reproduces decode EXACTLY on every intermediate.
- **Above 2048:** at absolute position 2100 (npos=2101, both engines on the sparse
  path) L0 decode 0.4517 vs prefill 0.4533, +0.35%, identical max 26. L0 is decisive:
  first layer, no accumulated drift, DSA layer with selection active.
- Later layers drift to ~15% by L8 via MoE expert-flip (a top-8 router on sub-percent
  input differences flips 3 of 8 experts). Expected, not a defect.
- **Method:** patch a COPY of the decode harness (`/tmp/p92dbg`) so it consumes a fixed
  token sequence (`P92_FIXED_SEQ=1`) — it otherwise generates its own continuation and
  the engines differ on INPUTS, making any comparison meaningless. Trace an absolute
  prefill position with `PF_TRACE_POS`.

## 4. SCALING LAW — the thing to reason from

**The union is BOUNDED.** Four adjacent queries' top-2048 selections collapse to:

| context | union mean | union/2048 | keys/query | dense keys/query |
|---|---|---|---|---|
| 4096  | 2324 | 1.135 | 581 | 512 |
| 65536 | 3088 | 1.508 | 772 | 8192 |

16x context inflates the union only 33%; per-doubling increments decay geometrically
(0.130, 0.105, 0.079, 0.059, ratio ~0.79) toward a limit near **3550**. `union/context`
collapses 0.568 → 0.047. Attention work per query SATURATES.

**Every stage is linear in context EXCEPT DSA score generation, which is n^1.98.**

## 5. RANKING at 64K (current authority, `SUM 410104 ms`, overlap 3.69x of 4)

| stage | ms | share |
|---|---|---|
| **attn** | **243194** | **59.3%** |
| proj | 43998 | 10.7% |
| gateup | 40639 | 9.9% |
| down | 32169 | 7.8% |
| dsascore | 27679 | 6.7% |
| mhc / route / dsasel / dsa / dsauni / norm / rope / requant / handoff | 22124 | 5.4% |

## 6. IN FLIGHT — the open decision

**Attention is memory-bound, and that reopens a mechanism I previously refuted.**

Per key a block does 4 x 1088 = 4352 FMA against a 1152-byte load = **3.8 FMA/byte**
against a machine balance near 8.8. The grid is `(tiles, 48)` and MLA shares ONE KV
head across 48 query heads, so **every key is read 48 times** — ~64 TB at 64K, running
at ~35% of streaming peak.

Cooperative multi-wave was refuted earlier on OCCUPANCY grounds (splitting the
accumulator cannot reach 3 waves/SIMD: 118 VGPR, need <=84, named state is only 72).
**That refutation still stands.** But if attention is TRAFFIC-bound, the value of
splitting the accumulator across waves is that it lets more queries share one key
load. Different justification, same mechanism.

**DISCRIMINATOR READY, NOT YET RUN:** `/tmp/disc.sh`, binary `/tmp/pfbench_tq`.
Runs TQ=2/4/8 in the sparse path at 64K, interleaved, warmup discarded.

| TQ | VGPR | waves/SIMD | key traffic |
|---|---|---|---|
| 2 | 81 | 3 | ~2x |
| 4 | 119 | 2 | shipped |
| 8 | 199 | 1 | ~0.6x |

- **TQ=8 wins** → traffic-bound → BUILD the cooperative multi-wave kernel
  (block = 4 waves, each wave owns 128 of the 512 latent dims, 2 dims/lane;
   TQ=12 then costs acc 24 + qv 24 + m 12 + l 12 = 72 + ~46 overhead = ~118 VGPR,
   still 2 waves/SIMD, and cuts traffic 3x. Score needs a cross-wave LDS reduction —
   amortise the barrier over a GROUP of keys, not per key).
- **TQ=2 wins** → occupancy-bound → cooperative is refuted twice; attack `proj`
  (10.7%) or `gateup` (9.9%) instead.

## 7. OPEN ITEM

Freeze verify of scorer-v3 gave **589.2 tok/s** at 64K where the A/B gave 602.2/602.3
consistently. 2.2% gap, outside the A/B's band. A 3-run re-verify was launched
(`/tmp/reverify.sh`, task output in the session tasks dir) and had completed run 1 when
this handoff was written. **Resolve before quoting 602 as the authority number.**

## 8. LAWS BANKED TODAY

- **A numerical gate must count non-finite BEFORE any reduction.** `fmax(m, fabs(NaN))`
  returns the other operand, so all-NaN output scores as a perfect match. Cost a whole
  debugging cycle.
- **A gate that builds its own input cannot catch a producer mismatch.** The MoE gate
  preshuffled its own scales, so it tested the consumer against a layout the production
  producer never emitted.
- **Range-check generated test data.** `r()%2000-1000` is unsigned: half the draws
  wrapped to ~4.29e9, range 0 to 4.6e15 against an intended [-0.25, 0.25].
- **One event pair per LAUNCH SITE, not per stage.** A per-stage timer re-recorded every
  layer measured only the last one, under-reporting by the layer count and showing an
  idle machine (0.13x overlap) that was in fact busy (1.51x).
- **Uniformity without contiguity can look WORSE.** Moving a wave-uniform operand from
  LDS to a uniform global address (→ SGPRs) cut ds_read 12→0 and waits 12→1 but RAISED
  instruction count 58→61. Only storing the operand contiguously so the scalar loads
  vectorise gave the real kernel: 39 instructions, 1.62/FMA vs 2.42. **Both halves are
  needed and the intermediate step looks like failure.**
- **Generalisable check:** in any hot loop, count `ds_read` against `v_fma` in the
  innermost backward branch. One ds_read per FMA with a wait behind it, on a value every
  lane agrees on, is the signature.
- **Profiled ms are not recoverable ms** — but the sensitivity probe must name a KERNEL,
  not a stage, because several stages are not idempotent (`k_pf_mhc_scale` multiplies the
  logits in place, `k_pf_mome_st` advances a chunk carry). Mode 1 duplicates in place;
  mode 2 duplicates the SAME kernel at the tail as a positive control. Both count
  launches so a flat wall cannot be confused with a probe that never fired.
  Attention measured **fully exposed**, ratio 1.008.
- **Cold-run penalty on this box is ~4%.** Discard the first run and say so in receipts.
  Cause is page cache: the other lane's model load evicts ours (their cold load 118.4 s
  vs a few seconds warm, 2.3M major faults with 62 GB MemAvailable).

## 9. OPERATIONAL

- **openPangu owns the cards.** Operator settled this; the Qwen lane stands down until
  they explicitly change priorities. Do NOT yield windows on request.
- **`~/q27bench` is the machine-wide lock. Never run unlocked.** Two defects fixed today:
  it truncated the lock file at open BEFORE flock (destroying the holder record on every
  refusal), and its stray check `pgrep -x q27_gen` was blind to any other engine — a
  resident p92 pushed the peer's job into GTT host memory at ~1/100th speed and presented
  as a hang. It now asks the driver via `rocm-smi --showpids`. Backup at `~/q27bench.bak-20260909`.
- Peer session address: `<peer session socket>`.

## 10. ENV FLAGS

`PF_CHUNK` (256 is production) · `PF_PIPE_DEPTH` (4) · `PF_ATTN_KERNEL` (2 default) ·
`PF_ATTN_TQ` (4) · `PF_DSA_SEL` (1) · `PF_DSA_SCORER` (3) · `PF_DSA_WIN` (ceiling probe
ONLY, not a correctness path) · `PF_TRACE` + `PF_TRACE_POS` · `PF_UNION_STATS` ·
`PF_SENS_STAGE` (1 = in place, 2 = tail control) + `PF_SENS_REPS` · `PF_VERBOSE`

## 11. ADDENDUM — a launch-overhead lead worth one cheap measurement

The Qwen lane measured, on this box, that a bare kernel launch submitted at a DRAINED
post-barrier point costs **10.83 us of wall against 1.245 us nominal** — found by
launching empty kernels 512 times per token and watching throughput fall 24%. Moving
submission before the barrier, with the GPU waiting on a flag via `hipStreamWaitValue32`
rather than the host waiting, recovered 0.545 ms at one site.

**Prefill has exactly that shape in one place.** `tests/p92_pf_bench.hip` does a BLOCKING
`hipMemcpy(&ng, c.ngroups, 4, D2H)` on every MoE layer to read the live-group count
before sizing the gate/up and down grids. That drains the queue and then the host
submits the next kernel — the precise edge described above. At 64K with chunk 256 that
is roughly 2816 forced syncs per card.

It is very likely removable outright rather than needing a flag: `k_pf_gateup` and
`k_pf_down` already early-out with `if (grp.nrows <= 0) return;`, so the grid can simply
be launched at a fixed `PF_GROUP_MAX` tiles and the sync deleted. With T=256 tokens and
top-8 routing nearly all 257 slots are live anyway, so the wasted blocks should be few.

Bulk launch exposure in prefill is otherwise small — ~294K launches at 64K, ~73.6K a
card, so ~0.8 s against a 115 s wall (0.7%) — but the blocking sync is a different and
much more expensive class than ordinary submission cost. Measure before and after with
the stage timers; do not assume.
