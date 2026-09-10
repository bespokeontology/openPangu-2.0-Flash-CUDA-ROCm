# Rejected optimisations, and what measurement said

Kept because a rejected experiment is evidence. Conditions change - fusion,
batch width, MTP, launch structure, occupancy pressure from a neighbouring
kernel - and something that lost today can win later. Rejected means measured,
understood and removed from today's hot path. It does not mean erased.

Every entry records the mechanism, the correctness result, the numbers, and the
reason it lost.

---

## R1. Router top-8 by wave-local top-8 plus a 32-candidate merge

**Mechanism.** The baseline selects 8 of 256 with eight sequential passes over
all 256 scores, each a block-wide reduction plus an LDS fold plus a serial
thread-0 section: 24 barriers a call, 44 calls a token. The rewrite gave each of
the four waves its own top-8 with no barrier, since a wave is already
synchronous, then merged the 32 candidates in one wave. One barrier total.

**Correctness: PASS.** Expert ids identical to the scalar reference, tie rule
identical (equal score selects the lower id), selected-weight sum exactly
2.500000, generation coherent.

**Performance: LOST.**

    baseline route_top8      25.00 us/call   1.100 ms/token
    rewrite  route_top8      28.19 us/call   1.240 ms/token
    baseline whole engine    53.30 tok/s median
    rewrite  whole engine    52.88 tok/s median

**Why.** It trades barriers for shuffles, and on gfx906 a shuffle is
ds_bpermute - the same LDS path a barrier uses. Eight rounds of seven shuffles
in every wave is 56 shuffles a wave against the baseline's 48 plus a cheap fold,
and three of the four waves then idle through the merge. Removing barriers is
not free when the replacement is more traffic on the same unit.

**Do not read the profiler totals as a win.** The rewrite's total kernel time
read 20.00 ms/token against 20.71 for the binary profiled before it. That
comparison is invalid: the 20.71 profile was taken on the 51.48 binary, which
still had the full-unroll gate/up. Wall-clock median did not corroborate any
gain and the router itself regressed. Treated as attribution drift.

**Superseded by** the single-wave form, which uses ONE wave with four experts a
lane: no barrier, no LDS, no cross-wave traffic. 19.93 us/call, and that one is
in production.

**Resurrect if:** routing is fused with a neighbouring op so the merge phase has
company; batch width grows so the four waves have real work each; or a future
gfx906 path makes cross-lane traffic cheaper than it is today.

---

## R2. MLA head grouping, four heads a block to share latent rows

**Mechanism.** All 48 heads read the same latent rows, so a block was made four
HEADS at one split rather than one head at four splits, to fetch each row once
for four consumers.

**Correctness: PASS.** relative_l2 8.618e-07.

**Performance: LOST.** 512.46 -> 546.40 us/layer. Reverted.

**Why.** The sharing is real but the output writes scatter across four head
planes instead of one, and that cost more than the reads saved.

---

## R3. MLA access-pattern hypothesis

**Mechanism.** Suspected the random DSA selection order was the cost, so
contiguous and strided selections were measured against random.

**Result: NO EFFECT.** random 512.48, contiguous 526.21, strided 590.72 us.
Contiguous is marginally *worse*. The hypothesis was simply wrong; the real cap
was register pressure, which the lane-width sweep found.

---

## R4. Two int8 activation components across the whole head

**Mechanism.** The head needs about 14 bits of activation precision, which two
int8 components give. Applied to all 151,552 rows.

**Correctness: PASS, and necessary** - one component flips the emitted token on
3 of 96 trials.

**Performance: LOST as a whole-vocabulary policy.** 483 -> 1620 us. The
prediction that it would be nearly free because the nibble decode is shared was
wrong: the second LDS staging and the halved occupancy dominate.

**Superseded by** the fast full-vocabulary pass plus a top-256 refine, 565 us at
identical token agreement. The two-component form is still in production, just
only where it can change an answer.

---

## R5. Dense down narrower lane groups

**Mechanism.** The dense down kernel pays a six-shuffle reduction per row with a
full wave, so narrower lane groups were swept.

**Result: LOST at every width.** 64 lanes 41.16 us, 32 lanes 66.20, 16 lanes
66.75, 8 lanes 186.10. Coalescing dominates, not shuffle count: at 64 lanes a
step is one 512-byte transaction, at 8 lanes it is 64 bytes. The original
geometry was already right.

**Note the opposite result for the EXPERT down**, where 8 lanes was the
baseline and 32 won. The two kernels have different K - 9216 against 1024 - so
the same reasoning gives different answers. Neither result transfers.

---

## R6. HIP graphs for launch overhead

**Mechanism.** Suspected per-layer launch count was the cost of composition.

**Result: NOT THE PROBLEM.** Measured directly: an empty kernel launch is 1.245
us at one block and 1.916 us at 24x256, a dependent chain 2.024 us, and 92
launches cost 207.3 us loose against 175.8 us captured in a graph. Graphs buy
15 percent of launch cost, which is a small fraction of a small number.

---

## The pattern

Eleven optimisation hypotheses have now been tested on this engine and six were
wrong, including several that would pass a code review. Two of the wrong ones -
head grouping and the access-pattern theory - were about MLA, whose real cap
turned out to be register pressure, found only by sweeping.

The rule that survives: measure the dominant term in the COMPOSED path, form
one hypothesis, test it, and believe the number over the reasoning.

---

## route_top8 rung (2026-09-07) — two falsified hypotheses

Baseline: single wave, 64 threads, 4 experts/lane, 8 serialized argmax rounds.
Cost structure measured with `tests/router_probe.hip` round-count ladder:
**2.23 us fixed + 0.84 us/round**, 8 rounds = 8.96 us.

### Rejected: 4-wave rewrite

Hypothesis: one wave underuses the CU; spread 256 experts over 4 waves and
reduce across waves through LDS.

Measured **28.19 us against 25.00 us** for the equivalent single-wave probe
configuration — 13% *slower*. The cross-wave LDS barrier per round costs more
than the extra parallelism returns at this size. 256 experts is simply too small
a reduction to pay for inter-wave synchronization 8 times.

### Rejected: xor-shuffle + deferred weight pass

Hypothesis: `__shfl_xor` butterfly reduces round latency versus the existing
pattern, and deferring the weight gather out of the selection loop shortens the
critical path.

Measured **8.97 us against 8.96 us** — no change. This one is instructive: the
deferred weight pass was retained (it is in the accepted DPP version) because it
is structurally cleaner, but on its own it bought nothing. The butterfly did not
help because `__shfl_xor` lowers to the same `ds_bpermute` as `__shfl`; changing
the *pattern* of cross-lane communication cannot help when the *path* is the cost.

That negative result is what pointed at the transport rather than the algorithm,
and led to the DPP finding recorded in `10_GFX906_PLAYBOOK.md`.

---

## Expert path: the order-preserving avenue is exhausted (2026-09-08)

Baseline freeze-p92-amd-64.2. Nine order-preserving mechanisms tried on
k_p92_f4_gateup and k_p92_f4_downacc. ONE paid (the dead memset, banked as the
64.24 rung). Eight did not. All measured standalone at 256 resident experts,
9 dispatch slots, best of 3 x 300 iterations, realistic UE4M3 scale bytes.

    gate/up production                                        46.0 us
      branch-free UE4M3 decode                                56.7   WORSE
      + explicit one-iteration prefetch                       51.9   WORSE
      + unroll 2                                              51.7   WORSE
      dot4 chain split into two independent 2-chains          51.1   WORSE (bit-exact)
      dot4 chain split into four independent, balanced tree    52.7   WORSE (bit-exact)
      SADDR wave-uniform base + 32-bit offset                 51.5   WORSE (bit-exact)
      non-temporal weight loads                               49.2   WORSE (bit-exact)
      DPP row_shl reduction, 10 ds_bpermute -> 2              50.5   WORSE (bit-exact)
      __launch_bounds__/waves_per_eu pressure                  n/a   NO EFFECT

    down production                                           34.0 us
      DPP row_shl reduction, 45 ds_bpermute -> 9              33.6   +1.2%, bit-exact
                                                                     = 18 us/token = 0.11%,
                                                                     below composed noise

### Why the obvious things fail here

**The branches are an optimization.** p92_ue4m3_d's two conditionals compile to
exec-mask regions. e==0 and b==127 are rare in real scale data, so
s_cbranch_execz skips them for the whole wave at near-zero cost. Computing both
paths unconditionally and selecting is strictly more VALU. Measuring this with
synthetic scale bytes drawn uniformly from 0..127 makes the branch fire ~39% of
the time and gives a completely different answer - use realistic exponents.

**ILP is not the constraint; occupancy is, and it is pinned.** VGPR 35 gives 7
waves/SIMD, and the compiler will not go below 35 under any launch_bounds or
waves_per_eu pressure. Every mechanism that adds live state - prefetch registers,
split accumulators, SADDR's extra base pointers - raises VGPR and costs a wave.
SADDR went to VGPR 38 / occupancy 6, the opposite of its predicted register saving.

**dot4 chains are not the bottleneck.** Splitting p92_group_dot4's 4-deep
accumulator chain is bit-identical (integer addition is associative, and wrapping
addition is associative mod 2^32) and still loses, because the added live
accumulators cost more than the shortened chain returns.

**DS ops here are already hidden.** The router rung got 33% from replacing
ds_bpermute because it was a single wave with nothing to hide latency. At 7-8
waves/SIMD the expert kernels hide it, so the same substitution loses. This is the
playbook rule holding in both directions.

### What the counters say

    kernel    fetch MB/call  computed  VALUBusy  MemUnitBusy  MemUnitStalled
    gate/up      25.855       26.542     56.7%      55.1%          1.9%
    downacc      12.077       13.271     43.6%      45.0%          0.4%

Actual DRAM fetch is BELOW the computed weight budget - L2 absorbs some reuse and
there is no over-fetch anywhere. Both units sit near half busy with essentially no
memory backpressure. Independently, the ISA gives a VALU floor of 25.52 us for
gate/up against 47.0 measured = 54.3% utilization, matching the counter's 56.7%.
Two independent methods agree.

That combination - neither unit saturated, no stall, occupancy pinned by register
count, and every restructuring losing - is what an already well-scheduled
latency-bound kernel looks like. The remaining gap is not obviously recoverable
without changing the arithmetic.

### Two methodological corrections made here

**A fair roof must be measured at the target kernel's occupancy.** The 34.85 us
"memory roof" for gate/up came from a ~10-VGPR streaming probe. Pinned to
gate/up's actual 7 waves/SIMD the same 4-stream read costs 38.07 us. The gap was
7.9 us, not 11.2.

**Source-level ablation does not decompose kernel cost additively.** A variant
with a component removed is a different program with different register
allocation and scheduling. In one decomposition attempt a variant doing strictly
MORE work ran 5.5 us FASTER than its own subset, which proves the method invalid
rather than the kernel odd. Use hardware counters, not differences of programs.

**And validate DPP lane semantics every time.** row_shl matches __shfl_down;
row_shr is the shfl_up direction. An external analysis asserted the opposite, it
was propagated without checking, and the result was a non-bit-exact kernel. The
standalone check takes one minute. This is the second time this rule has earned
its place.

---

## Runtime modulo elimination in mla_g16 (2026-09-08) — NEUTRAL

**Mechanism.** `tok = win_cap ? ((win_first + b) % win_cap) : sel[b]` compiled to
a full runtime integer division per entry: `v_rcp_iflag_f32`, two `v_mul_lo_u32`,
two `v_mul_hi_u32` and their dependents, roughly 22 instructions. On the
220-token benchmark this executes on every entry of every one of the 46 layers.

**Replacement.** Reduce `win_first` modulo `win_cap` on the HOST, after which
`win_first < win_cap` and `b < win_cap` give `win_first + b < 2*win_cap`, so
`t >= win_cap ? t - win_cap : t` is exact.

**Correctness: PASS.** Integer-division ops 3 -> 0. 220-token output
bit-identical.

**Performance: NEUTRAL.** 66.89 / 66.35 / 66.56 against a 66.25-66.86 baseline.
No measurable change.

**Why it matters anyway.** This is the control that makes the branchless-load
result interpretable. Removing ~22 instructions per entry from a latency-bound
kernel bought exactly nothing, while ADDING 139 instructions to the same loop
(283 -> 422) by restructuring its dependencies bought 5.36 percent. Static
instruction count failed as a proxy in both directions on the same kernel on the
same day. See receipts/10_GFX906_PLAYBOOK.md section 8.

Not committed. Worth reapplying purely for clarity if that file is touched
again, since a compare-subtract is simpler than a division and provably exact.
