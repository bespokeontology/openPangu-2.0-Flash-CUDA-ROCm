# gfx906 playbook

Architecture-specific mechanisms discovered while porting openPangu-2.0-Flash.
These are properties of gfx906/CDNA1 and wave64, not of Pangu. They are expected
to transfer to any other model we put on these cards (Qwen3.8-27B next).

Each entry: the abstraction, what it actually lowers to, the measurement, the rule.

---

## 1. HIP abstraction is not the hardware primitive: __shfl -> ds_bpermute

**Discovered** 2026-09-07, openPangu route_top8.

`__shfl`, `__shfl_xor`, `__shfl_down` and friends lower on gfx906 to
`ds_bpermute_b32` — the LDS/DS cross-lane path, not an ALU operation. It does not
touch LDS *storage*, but it is issued on the DS pipe and carries DS-path latency.
In a kernel with enough occupancy that latency hides. In a single-wave kernel with
nothing else in flight, it is exposed and it dominates.

route_top8 is one wave of 64 threads doing 8 serialized argmax rounds over 256
experts. Round-count ladder (`tests/router_probe.hip`):

| variant, 8 rounds | us | slope us/round |
|---|---|---|
| production, `__shfl` (= ds_bpermute) | 8.96 | 0.84 |
| no cross-lane at all (invalid, control) | 5.90 | 0.44 |
| DPP row_shr/row_bcast | 6.04 | 0.46 |

Fixed floor 2.23 us. **Roughly half the serialized loop was cross-lane transport,
not comparison work.** DPP recovers essentially all of it — 6.04 against a 5.90
no-communication bound.

The replacement is `__builtin_amdgcn_update_dpp`, which is a DPP modifier on a
normal VALU instruction: the crossbar runs in the vector ALU's operand path, so a
reduction step costs about what the `v_max`/compare itself costs.

```cpp
#define P92_DPP(x, ctrl) __builtin_amdgcn_update_dpp((x), (x), (ctrl), 0xF, 0xF, false)
// row_shr:1,2,4,8 reduce within each row of 16 lanes
// row_bcast:15 (0x142) then row_bcast:31 (0x143) carry across the row boundaries
// after all six, the wave-wide result is in lane 63 -> readlane 63 (SALU, free)
```

An argmax carries a value AND an index. Both must move through DPP, and the tie
rule has to be applied on the *reduced* pair, not the local one, or the value is
right while the index is wrong — a failure the value-only check cannot see. The
router gate compares ids against a CPU reference and checks a deliberate
three-way tie; both are required.

### The rule

> HIP abstraction is not the hardware primitive. On gfx906, inspect generated ISA
> for wave operations: `__shfl*` may lower to the DS-path `ds_bpermute`; DPP can
> provide an ALU-path alternative, but its exact lane semantics must be validated
> independently before benchmarking.

Validate semantics in a standalone kernel first (`row_shr` boundaries, `row_bcast`
lane coverage, identity behaviour, index propagation). Only then touch production.

### Where else this applies

Any single-block or single-wave kernel with no occupancy to hide DS latency:

| kernel | us | cross-lane |
|---|---|---|
| route_top8 | 8.96 -> 6.04 | DONE |
| mhc_collapse | 10.06 | Sinkhorn `__shfl_xor` within 16 lanes — row_shr covers it exactly |
| rmsnorm_quant | 5.83 | 6 shuffles for the sum reduction, 1 block |

Not applicable to the high-occupancy kernels (experts, MLA, proj) — there the DS
latency is already hidden and DPP buys nothing.

---

## 2. Dead-code elimination masquerading as a speedup

**Discovered** 2026-09-07, same rung.

The first DPP attempt measured 6.07 us and emitted `0xBEBEBEBE` expert ids. My
patch had consumed the `chosen[k] = bi;` assignment, so the top-8 loop had no
output dependency and the compiler removed it entirely.

Two independent alarms fired: garbage output, and a faster kernel beating the
measured no-communication floor. Either alone is sufficient to invalidate.

### The rule

> A kernel that beats its own measured lower bound has stopped doing the work.
> Every optimization measurement is invalid until its correctness gate passes on
> the same binary that produced the timing.

This is the third instance of the same class on this port; the other two are in
`08_REJECTED.md` (a sweep that left the launch grid fixed while shrinking
rows/block, so the kernel wrote a fraction of its rows; and an unscoped
`re.sub` that rewrote a neighbouring kernel's lane math). Always time and gate
the same build.

---

## 3. DPP exactness: what is expressible, and what reassociates

**Discovered** 2026-09-08, mhc_collapse Sinkhorn (4x4, 16 active lanes, 20 iterations).

DPP offers quad_perm, row_shl/shr/ror, row_mirror, row_half_mirror, row_bcast.
Which of these are exact xor substitutes matters, because a reduction that moves
the same SET of lanes in a different PAIRING reassociates the float arithmetic.

| need | DPP | order-preserving? |
|---|---|---|
| xor 1 over a quad | quad_perm [1,0,3,2] = 0xB1 | YES |
| xor 2 over a quad | quad_perm [2,3,0,1] = 0x4E | YES |
| xor 7 | row_half_mirror 0x141 (i -> i^7 within each 8) | YES |
| xor 15 | row_mirror 0x140 (i -> i^15 within each 16) | YES |
| **xor 8** | **mirror then half_mirror: (l^15)^7 = l^8** | **YES, two ALU ops** |
| **xor 4** | **none** | **NO substitute exists** |
| 64-lane associative reduction | row_shr 1/2/4/8 + row_bcast 15/31 | YES |

row_ror:4 / row_ror:8 sum the same SET of lanes as xor 4 / xor 8, so the result
is mathematically equal, but the pairing differs. For lane 4:

    xor:  (v4 + v0) + (v12 + v8)
    ror:  (v4 + v8) + (v12 + v0)

Lanes 0-3 coincide, which is why a spot check of the first quad passes while the
kernel still changes its output. **Check the pairing at a lane >= 4, never only
lane 0.**

The xor 8 identity is the useful discovery: two chained ALU-path DPP ops beat one
DS-path ds_bpermute AND preserve the source lane exactly. Validated bit-identical
against __shfl_xor(v,8,64) on non-trivial float data, both for the move and for
the full column-sum total.

### Four variants, measured on gfx906

Layer-5 weights, real checkpoint, isolated kernel plus full-engine.

| variant | ds_bpermute/call | collapse us | mHC site us | comb rel_l2 | Sinkhorn row/col dev |
|---|---|---|---|---|---|
| A accepted (all __shfl_xor) | 82 | 8.47 | 14.79 | 9.677e-08 | 9.91e-07 / 9.69e-07 |
| B quad_perm only | 40 | 7.87 | 14.19 | 9.677e-08 | 9.91e-07 / 9.69e-07 |
| C full DPP (ror4/ror8) | 0 | 7.83 | 14.15 | 1.113e-07 | 1.01e-06 / 1.01e-06 |
| **D exact-max** | **20** | **7.83** | **14.15** | **9.677e-08** | **9.91e-07 / 9.69e-07** |

**D matches C's speed exactly and A's numerics exactly.** Removing the last 20
ds_bpermute (C) buys nothing measurable, and C is marginally FURTHER from the
scalar CPU reference on every column. So on this hardware there is no
speed-versus-exactness trade to make: take D.

D generates a 220-token sequence identical to freeze-p92-amd-63.1.

### DO NOT "finish the job" by turning D back into C

D leaves 20 ds_bpermute per call in the disassembly. That is deliberate and
measured, not an unfinished optimization.

The four-variant ladder shows where the cost actually sits:

    A -> B   remove 42 ds_bpermute   8.47 -> 7.87 us   pays
    B -> D   remove 20 more          7.87 -> 7.83 us   pays a little
    D -> C   remove the last 20      7.83 -> 7.83 us   pays NOTHING

The last 20 are not on the limiting dependency path, so removing them buys zero
— and the only way to remove them is row_ror, which reassociates the arithmetic
and measurably degrades agreement with the scalar reference.

> **Rule.** ISA-count minimization is not the objective. Replace DS-path
> operations where measurement shows they are on the critical path. Never distort
> numerics to drive a ds_bpermute count to zero. The ISA explains, the
> microbenchmark diagnoses, the composed engine decides.

This complements the router finding rather than contradicting it: in route_top8
the cross-lane ops WERE the critical path (0.84 us/round of which 0.44 was
transport), so removing all of them paid 33%. Same instruction, opposite
conclusion, because the measurement differed.

### On the CUDA precedent — a warning, not an inherited constraint

engine/openpangu-flash92-native/receipts/DECODE_OPTIMIZATION.md records that a
256-thread mHC reduction was rejected on CUDA because "its local cosine was 1.0
but the full target's serial BOS branch changed", and that the shipped one-warp
kernel deliberately "retains the original lane-stride accumulation and shuffle
order".

That is a CUDA implementation decision on different reduction primitives,
different scheduling and a different engine. It is a reason to TEST reassociation
on gfx906, not a reason to forbid it here. Our own sequence already legitimately
diverges from the oracle after token 2 (oracle 148899 -> 2772 -> 5015, ours
148899 -> 2772 -> 8175), so cross-backend token identity is not our contract.

We tested it independently. C was built, gated and timed on AMD. It is rejected
because it is not faster than D while being slightly less accurate — an
AMD-measured result. Had C been meaningfully faster, its trajectory quality would
have been the question to answer, not its ancestry.

> **Rule.** Model contracts transfer between backends. Implementation accidents
> and hardware-specific optimization decisions do not. Bank the other backend's
> finding as a hypothesis and a test case, then confirm or falsify it on the
> hardware in front of you.

The Sinkhorn overlap is the standing example in the other direction: Huawei
overlaps mHC coefficient work with the MLA epilog on a side stream; the CUDA port
reproduced it exactly and REJECTED it (frozen 1751.9 / 1762.3 / 1834.7 ms against
overlapped 1791.9 / 1828.5 / 1886.5 ms at identical 69/84 acceptance) because the
event cost exceeded the hidden work. Our rung 60.59 achieves the same overlap
INTRA-KERNEL, Sinkhorn on wave 0 while waves 1-3 collapse, with no event or
stream cost — and it paid. Same principle, different mechanism, opposite verdict.

### Why the win was small, and what that teaches

Predicted 2.2% from mhc_collapse's 5.5% profile share; measured 0.7%.

Because of rung 60.59 the Sinkhorn already runs concurrently with the collapse,
so it was largely off the critical path before this rung started.

> **Rule.** A kernel's profile share is not its addressable time when part of its
> work is already hidden behind other waves in the same kernel. Profile share
> ranks candidates; it does not predict the win.

## 4. mHC quality reference

IMPLEMENTATION.md records the oracle's figure: Sinkhorn maximum row/column
stochastic error 9.53674e-07, with merged BF16 FNV-1a hash 97e7641f3e5c39c and
merged state range [-5.53125, 5.40625].

9.53674e-07 is exactly 2^-20, i.e. one ulp at that scale rather than a tolerance.
Our kernel measures 9.91e-07 / 9.69e-07 — same order, 1.04x the oracle figure —
so it cannot be used as a literal pass threshold. The mHC gate's doubly-stochastic
assertion has been tightened from 1e-4 to 2e-6, which is ~2x the oracle figure and
still 50x tighter than before.

---

## 5. Cross-lane semantics: prove them, never accept them asserted

**Two failures in the same category, 2026-09-07 and 2026-09-08.**

    row_shl:N   lane n receives lane n+N   ==  __shfl_down(v,N)   VERIFIED
    row_shr:N   lane n receives lane n-N   ==  __shfl_up(v,N)     VERIFIED

Both verified on gfx906 by a 12-line standalone kernel comparing against
__shfl_down directly. Takes one minute. Do it every time.

### Failure 1: the spectacular one

The first DPP router kernel returned expert ids 0xBEBEBEBE and measured FASTER
than the no-communication floor. A patch had consumed the `chosen[k] = bi;`
assignment, so the loop had no output dependency and the compiler deleted it.
Two independent alarms — garbage output, and a time below a measured lower
bound. Easy to catch.

### Failure 2: the dangerous one

An external analysis asserted that `row_shr` was the __shfl_down substitute,
with a confident correction saying it was NOT row_shl. That was backwards, and
it was integrated into two kernels without an independent check.

The result did not look broken. It produced plausible float output, at a
plausible speed, and only the bit-exactness comparison against production
revealed 9216/9216 and 2560/2560 rows differing. Had bit-exactness not been
asserted at that moment, a silently wrong reduction would have entered the
engine and been attributed later to the reassociation rung.

> **Rule.** A cross-lane operation's lane mapping is proven by a standalone
> kernel that compares it against the operation it claims to replace, on
> non-trivial data, before it goes anywhere near production. An assertion from
> any source — documentation, an analysis agent, a prior receipt, or your own
> recollection of a previous kernel — is a hypothesis, not a fact. The check
> costs a minute; the failure is silent and produces plausible output.

Note the asymmetry that makes this worth a standing rule: failure 1 was obvious
because the output was garbage and the timing was impossible. Failure 2 was
dangerous precisely because everything looked reasonable. Prefer checks that
fail loudly, and never rely on a wrong answer being visibly wrong.

---

## 6. Occupancy is a proxy, not the objective

**Measured 2026-09-08, mla_g16, gfx906.**

    variant              VGPR   waves/SIMD   time
    LANES=32 production    76        3       46.75 us
    LANES=64               50        4       49.64 us

The intended mechanism WORKED. Registers fell 76 -> 50, occupancy rose 3 -> 4,
and the kernel got 6.2 percent SLOWER. Both variants verified bit-identical, so
this is not a correctness artefact.

The cause is that changing the lane width changed the economics of the
cross-lane reduction. At LANES=32 a single `__shfl_xor` instruction reduces BOTH
half-wave groups simultaneously (lanes 0-31 and 32-63 are independent groups),
so 5 shuffles serve 2 entries = 2.5 per entry. At LANES=64 there is one group,
so the same reduction is 6 shuffles for 1 entry. **2.4x the DS traffic per unit
of work**, which outweighed a whole occupancy level.

> **Rule.** Occupancy is a proxy for latency hiding, not a thing worth having.
> A change that improves the occupancy metric can worsen wall time by altering
> the per-unit cost of something else. Never promote on a resource metric;
> promote on measured time. And when a geometry parameter changes, re-derive the
> cost of EVERY mechanism that depends on it, not just the one being targeted.

The corollary is that occupancy also is not the diagnosis. gate/up sits at 7
waves and could not be improved by any of nine mechanisms; mla_g16 sits at 3 and
got worse when raised to 4. In both cases the number was a symptom of the live
state the algorithm genuinely needs, not a lever.

### The lane-width curve is now closed at both ends

    LANES=16   G16_DPL 32, ~96 live floats, 1.0 shuffle/entry   244.3 GB/s  rejected earlier
    LANES=32   G16_DPL 16,  48 live floats, 2.5 shuffle/entry   383.0 GB/s  optimum
    LANES=64   G16_DPL  8,  24 live floats, 6.0 shuffle/entry   slower      rejected 2026-09-08

A register/DS trade with an interior optimum. Do not re-sweep this axis without
a new mechanism that changes the shape of the curve.

---

## 7. Three lessons from accounting for a kernel that turned out to be at its optimum

From mla_g16, 2026-09-08. The kernel was 11 percent of the engine and every
structural attack on it failed. The accounting is worth more than the 3.5
percent that was eventually taken.

### 7a. Bound the maximum prize before restructuring

Before designing an ABI change to eliminate a 18-shuffle group merge, the merge
was simply DELETED - producing mathematically wrong output, clearly labelled as
a bound - to see what it was worth. Answer: 0.4 us of a 45 us kernel.

    production, 5 ds_bpermute per entry     46.77 us
    DPP, 2 ds_bpermute + 4 ALU per entry    45.14 us
    BOUND: DPP + merge deleted entirely     44.72 us

Knowing the ceiling was 0.4 us made an ABI redesign (32 partials instead of 16,
doubling pacc traffic, changing the producer/consumer contract) economically
irrational in one measurement. A deliberately incorrect variant is a legitimate
and cheap instrument for bounding a prize - label it loudly and never let it
near a correctness comparison.

### 7b. Idle execution units can be a dependency symptom, not headroom

    mla_g16: VALUBusy 26.9 percent, MemUnitBusy 35.6 percent, MemUnitStalled 0.8

The naive reading is that the VALU has almost 4x headroom. It does not. The
machine has capacity; the WAVE cannot expose enough independent work to use it,
because each online-softmax step depends on the previous one. Low utilization
plus low stalls means the critical dependency chain is the constraint.

Distinguish the two cases by trying to ADD independent work. Here that was tried
three ways and all three lost: wave64 (more occupancy), dropping the v[] array
(fewer registers), and computing two entries before either softmax update (more
ILP). When adding independence makes it worse, the chain is the answer.

> **Rule.** Never infer headroom in an execution unit from its utilization
> alone. Low busy with low stalled is a dependency signature.

### 7c. Register provenance matters more than register count

receipts/06_LEDGER.md records an earlier MLA rung reaching "~48 VGPRs". The
kernel now compiles to 76 at the same geometry. The obvious suspect was the
v[16] latent array - 16 registers held only so the values survive the reduction.

Removing it entirely recovered 6, not 16: VGPR 76 -> 70, still 3 waves/SIMD,
and 34 percent slower from the extra read. The growth is address and control
state from functionality added later (sinks/body joint-run balancing, the
sliding-window modulo, the two-candidate base pointer select), not the array
that looks expensive in the source.

And 70 crosses no occupancy boundary anyway - gfx906 needs <=64 for 4 waves - so
"fix the VGPR count" had no demonstrated prize even if it had worked.

> **Rule.** Before spending effort on register pressure, establish WHICH live
> ranges hold the registers and WHICH occupancy threshold the reduction would
> cross. A register count with no threshold behind it is not a target.

---

## 8. Static instruction count can have the WRONG SIGN as a performance proxy

**Two experiments on the same kernel, same day, same production path, both
bit-identical. They point in opposite directions.**

    experiment A - delete instructions
      replace a per-entry runtime integer division with an exact host-reduced
      compare-subtract
        integer-division ops   3 -> 0
        instructions/entry     ~22 removed
        measured               NEUTRAL  (66.89/66.35/66.56 vs 66.25-66.86)

    experiment B - restructure dependencies
      make five exec-masked loads unconditional by giving inactive lanes a valid
      address instead of nullptr
        loop instructions     283 -> 422   (+49 percent)
        s_waitcnt vmcnt(0)      6 -> 2
        saveexec               12 -> 7
        VGPR                   77 -> 77
        occupancy               3 -> 3
        measured               +5.36 percent paired median, 12/12 pairs positive

Shortening the instruction stream bought nothing. LENGTHENING it by half, at
identical registers and occupancy, was the largest rung of the session.

> **Rule.** On gfx906, when a kernel is limited by its dependency/issue graph,
> static instruction count is not merely a weak proxy - it can move in the
> opposite direction to performance. The optimization target is the production
> dependency graph: memory fences, exec-mask regions, and what can be issued
> before a wait. Count `s_waitcnt vmcnt(0)` and `saveexec`, not instructions.

This completes the set of proxies now measured to fail on this hardware:

    occupancy            section 6   improved, kernel got slower
    register count       section 7c  reduced, crossed no threshold, slower
    ISA op count         section 2   driving ds_bpermute to zero bought nothing
    instruction count    HERE        wrong sign entirely

The only proxy that has never misled: measured wall time on the composed engine,
with the probe proven to compile the same control-flow path as production.

---

## 9. A geometry change resets performance authority

Stated 2026-09-09, after the Qwen lane's 4-way TP changed a matrix dimension
from 12288 to 3072 and immediately exposed dispatch heuristics that had been
correct and well-tuned at the old shape.

> **Any change in M, shard width, row width, head count, selected-entry count,
> or threshold regime resets performance authority for every mechanism whose
> cost depends on that dimension. Correctness may transfer; tuning does not.**

This generalises section 6, which was one kernel: at LANES 32 -> 64 the
registers fell and the occupancy rose and `mla_g16` got SLOWER, because the
lane width silently priced the cross-lane reduction. Section 9 is the same
statement at engine scale.

### What this demotes to a hypothesis

Every number below was MEASURED, and every one is a function of a shape that
prefill or TP changes. None carries authority at a new shape:

    G16_LANES = 32          measured optimal at KVLORA=512, 2 groups per wave.
                            Shard attention to 12 heads and the reduction
                            economics change exactly as they did at 64 lanes.
    32 lanes x uint2        measured 741 GB/s against uint4's 643, at a
                            1280-byte row. Shard the row and it may invert.
    P92_SPLITS = 16         tuned for decode's entry count, 128 sinks + nsel.
    the 20-iteration        contract, not tuning - this one DOES transfer.
      Sinkhorn
    LDS staging removed     measured at 9 dispatch slots and M=1. At M=512 the
                            activation reuse per weight byte changes by 512x
                            and staging may become correct again.
    "an extra reduction     measured on a latency-bound M=1 kernel. At M=512
      is cheap/expensive"   the same reduction amortises over 512 rows.
    branchless loads        the 6->2 vmcnt win came from exec-masked loads on a
                            per-entry loop. A batched kernel has different
                            control flow and may not have the defect at all.

Do not pre-emptively re-optimise these. Do not grant them authority either.
Measure at the new shape, in the composed engine.

### Corollary: prefill is a new workload, not decode with more rows

M = 1 -> M = 512 is a larger geometry change than the TP shard that exposed the
Qwen bug. Preserving the decode ARCHITECTURE is useful for correctness and
weight ownership. Preserving its kernel TUNING should carry essentially zero
presumption of optimality.

## 10. Gate both sides AND the boundary of every dimension-controlled dispatch

The `win_cap` incident (11_PROFILING_LAW, third contamination) was one branch
missed. Enumerating openPangu's dispatch points afterwards found something
worse.

### The map, tests/p92_generate.hip:426-463

    branch  condition                    nsel            win_cap   sel
    A       dsa && npos > 2048           2048 top-k      0         USED
    B       dsa && npos <= 2048          npos            w.cap     null
    C       SWA layer                    min(npos,512)   w.cap     null

    nested in A:  npos <= 16384  ->  single-block select
                  npos >  16384  ->  chunked histogram select
    also:         slotpos = dsa ? position : (position % w.cap)   ring wrap
                  w.cap   = dsa ? MAXPOS : 512

**At the 220-token production benchmark npos never exceeds 220, so branch A
cannot execute.** This is structural, not statistical: `npos` is bounded by the
generated length, and every optimisation run on 2026-09-08 used
`p92_gen ... 220 512 148899`.

> **SECOND CORRECTION, same day.** The phrase "or `src/p92_dsa_index.hip`" is
> WRONG. Six of its kernels - `idx_hist`, `idx_pick`, `idx_count`,
> `idx_scan`, `idx_write`, `idx_pad` - run on EVERY token via the head
> top-256 refine at `tests/p92_generate.hip:608-616`. Only `k_p92_idx_scores`
> and `k_p92_idx_select` are DSA-exclusive.
>
> The correct statement is sharper and more useful: those six are exercised at
> exactly ONE shape, n=151552 / want=256 / nchunk=37, and never at the DSA
> shape, n=npos / want=2048. Tested, but granting no authority at the shape
> that matters - which is section 9 applying to this very file. See
> receipts/17_DSA_PATH_DEFECTS.md, which also records two real correctness
> defects found on the unreachable side.

State the coverage boundary exactly, and no wider: **any run whose `npos` stays
at or below 2048 cannot exercise branch A**, the sole caller of
`k_p92_idx_scores` and `k_p92_idx_select`, in a file of 351
lines, the largest kernel file in the engine, holding the indexer scoring, the
three-pass 11-bit radix select and the histogram scan. The 220-token benchmark
that produced every rung from 34.60 to 69.92 is such a run. Its nested
`npos > 16384` transition needs a run longer still.

That is a BENCHMARK-COVERAGE DEFECT, not evidence that the DSA implementation is
wrong. Nothing here says the subsystem is broken. What it says is that the
decode work grants it **zero performance or correctness authority**, and no
conclusion about it may be inherited from the 69.9 line.

(Do not extend this to "never exercised in project history" - that would require
knowing every historical workload, and the finding does not need the stronger
claim.)

It is the `win_cap` failure one level up: not a missed branch, a missed
subsystem. Dormant only because decode benchmarks are short. A prefill prompt of
any real length crosses 2048 and executes all of it - so prefill does not merely
run the same program at larger M, it runs a DIFFERENT program.

> **Rule.** Before benchmarking or closing any kernel, enumerate every
> dimension-controlled dispatch on its path and exercise N-1, N and N+1 for
> each. A benchmark that drives only one side of a threshold measures one
> program and licenses conclusions about another.

Concrete boundaries to gate for openPangu:

    2047 / 2048 / 2049     P92_IDX_TOPK, DSA dense-causal vs top-k
    16383 / 16384 / 16385  single-block vs chunked select
    511 / 512 / 513        SWA window sizing
    cap-1 / cap / cap+1    the ring wrap, position % w.cap
    win_cap 0 vs nonzero   already gated as of 5f74ed9
    plus every chunk/M threshold prefill introduces

None of these is expensive to test. All of them are cheaper than closing a
subsystem that was never run.

---

## 11. Transform data into the consumer's layout upstream

Established across the prefill build, 2026-09-09, after the same defect appeared
three times in one day in three different kernels.

> **Transform immutable or produced data into the consumer's natural layout
> upstream. Never make the hot consumer reconstruct that layout repeatedly.**

Immutable data is transformed at LOAD time; produced data is written in the
consumer's order by its PRODUCER. Either way the hot path executes its native
access pattern and reconstructs nothing.

### The evidence, one kernel, three stages

`k_pf_gateup` holds an expert weight row resident and sweeps every routed token
past it. Its lane needs the scales of groups l, 32+l, 64+l, 96+l, 128+l.

    inline UE4M3 decode at the load     16 s_waitcnt vmcnt(0)   30 loads
    load and decode split                8                      30
    scales preshuffled lane-contiguous   2                      21

The `v_dot4_i32_i8` count is 40 at every stage. The arithmetic never changed.
Only the dependency graph did.

Two distinct transforms produced that: the weight scales are permuted at load
time by `k_pf_shuffle_scales_gu`, and the ACTIVATION scales are written already
permuted by `k_pf_quant`, their producer, at `(g%32)*5 + (g/32)`.

### Where to look

Inside any hot loop, treat these as defects until measured otherwise:

- strided metadata reads - scales, offsets, indices - at a stride the lane could
  have been given contiguously
- address arithmetic recomputed per iteration from values that do not change
- a branchy decode applied AT the point of load, which fences every load
  individually; split the load from the decode even when the layout is fixed
- any per-element permutation the producer could have applied once

### Two limits

**Per-consumer, never universal.** gate/up wants five scales contiguous, down
wants two. They get different transforms. A single "prefill layout" would
pessimise both.

**Only transform what the hardware is not already happy with.** The weight
nibbles were deliberately NOT reshuffled: 32 lanes x uint2 = 256 B a step is the
measured-best gfx906 streaming pattern, 741 GB/s against 643 for uint4 and 219
for two uint4 in flight. Reshuffling those would have been a regression dressed
as an optimisation.

### The trap this opens

Changing a consumed layout silently invalidates every gate that constructs its
own inputs. The MoE gate uploaded scales in spec order while the kernel had
moved to lane-contiguous, which would have produced a false failure looking
exactly like a kernel bug. The gate must present the layout the kernel CONSUMES
while its reference stays written from the SPEC - that separation is the whole
value of an independent reference, and it evaporates the moment a layout moves
unless the gate is updated with it.

---

## 12. A profiled bucket is not a recoverable bucket

Adopted 2026-09-09 from the Qwen lane, before the openPangu prefill profile
existed - which is the only reason it did not cost a wasted campaign here.

> **recoverable critical-path ms = profiled ms x wall sensitivity**

A kernel's profiled time is what it costs to execute. What it costs the TOKEN is
a different quantity, and in any pipeline with barriers, cross-card handoffs or
concurrent streams the two diverge. Work sitting in slack carries a large profile
bucket and contributes nothing to wall. Attacking the biggest bucket is therefore
a coin flip until its sensitivity is known.

This is the fourth proxy measured to fail on this hardware, joining occupancy,
register count and instruction count - and the most dangerous, because a profile
looks like direct evidence rather than a proxy.

### Measure by DUPLICATING, never by skipping

Duplicate an idempotent stage and watch the wall:

    coefficient ~ 1.0    fully exposed, on the critical path, worth attacking
    coefficient ~ 0.0    hidden in slack, the bucket is a mirage
    0 < k < 1            partially exposed; k multiplies the bucket

Skipping the work instead produces an invalid ceiling. This project has already
burned a probe that way: removing a component changed register allocation and
scheduling so completely that a variant doing strictly MORE work ran 5.5 us
FASTER than its own subset. Duplication preserves every result and every token
and adds a known quantity of exactly the work in question.

### The positive control is not optional

Duplicate a stage known to be exposed - the last heavy one before a barrier -
and require the wall to respond nearly proportionally. If it does not, the
instrument is broken and every coefficient it produced is meaningless. Same rule
as the null-result law in 11_PROFILING_LAW: an instrument that cannot
demonstrate a positive result cannot be trusted for a negative one.

### Where this bites hardest

Any bucket you are about to spend a day on. A 1 ms stage at coefficient 0.1 is
worth 0.1 ms and a 0.4 ms stage at coefficient 1.0 is worth four times as much -
and the profile ranks them the other way round.

Built into the prefill driver as PF_SENS_STAGE / PF_SENS_REPS before its first
composed run, so the first decomposition arrives with coefficients rather than
sending someone at the largest number.
