# Profiling law: initialization contaminates whole-process counts

Backend-independent. Written 2026-09-08 after the second optimization ranking in
this port was poisoned the same way.

## The failure

`rocprof --stats` reports kernel counts and durations for the WHOLE PROCESS.
Dividing those by the number of generated tokens silently amortizes model load,
weight preparation and one-time setup across decode, fabricating a per-token tax
that does not exist.

Concretely, on the 63.1 binary the census appeared to show:

    total kernel launches 116,862 over 60 tokens = 1,947.7 per token
    launch overhead at 1.245 us = 2.425 ms/token = 15.3% of a 15.8 ms token
    to_bf16 153/token, from_bf16 108/token  -> "261 conversion launches per token"

All of it wrong. The executor normalizes the 128 compressed-KV attention sinks at
LOAD time, in a loop of three kernels per sink:

    46 layers x 128 sinks x 3 kernels = 17,664 one-time launches

Those are startup cost. Divided by 60 they became a fictitious 294 launches/token.

## The giveaway, and the cross-check that works

    2760 = 46 x 60      exactly one launch per layer per token
    5520 = 2 x 46 x 60  exactly two per layer per token

Any recurring decode kernel lands on a clean multiple of layers x tokens. A count
that does NOT factor that way is carrying fixed work. `to_bf16` at 9,188 does not
factor; 9,188 - 5,888 = 3,300, which is not a multiple of 46 either, so the split
was not even guessable — it had to be measured.

## The rule

> Never amortize whole-process profiler kernel counts over generated tokens
> unless initialization has been explicitly excluded. Derive recurring counts by
> DIFFERENTIAL profiling — counts(N2) - counts(N1), divided by (N2 - N1), after
> identical initialization — and cross-check every claimed per-token count
> against expected layer/site multiplicity before ranking anything by it.

Durations inherit the same defect as counts. A load-heavy kernel that also runs
in decode will have its average skewed by the load-time invocations.

## Prior instance

The same contamination previously distorted the `rmsnorm<512>` ranking in this
port. Two occurrences is why this is a standing law rather than an anecdote.

## What survived the correction

`to_bf16` / `from_bf16` are pure representation glue:

    y[i] = p92_m2f(x[i])

No arithmetic, just launch latency and a write/read round trip. Whatever number
of these genuinely occur in steady-state decode remains a prime fusion candidate,
because folding them into a producer or consumer removes both the launch and the
memory traffic. But the 261/token figure is WITHDRAWN, not revised downward, and
no fusion campaign starts until the differential census replaces it.

## The corrected census (2026-09-08, freeze-p92-amd-63.5)

Method: rocprof at N=20 and N=100 after identical initialization; per-token
count = (calls(100) - calls(20)) / 80. Every row cross-checked against layer
multiplicity.

    claim                          whole-process/60    differential    truth
    total launches per token              1,947.7          1,651.0     -15%
    to_bf16 per token                       153.1             55.0     -64%
    from_bf16 per token                     108.1             10.0     -91%
    rmsnorm<512> per token                  144.1             46.0     -68%

`to_bf16` at 55 factors exactly: 46 layers + 9 block_post_layernorm layers
(indices 0,4,9,14,19,24,29,34,39). `rmsnorm<512>` at 46 is one per layer. The
5,888 difference in each is the load-time sink normalization, 46 x 128.

Everything in the corrected census factors cleanly — 44 (MoE layers), 46 (all
layers), 92 (2 per layer), 93 (2 per layer + head), 1 (head). A count that does
not factor is the signal to look for fixed work.

## The second correction: dispatch is mostly hidden

1,651 launches x 1.245 us = 2.055 ms/token of CPU dispatch work, which looked
like 13.4% of a 15.35 ms token. It is not addressable time.

    steady-state GPU kernel time   15.347 ms/token  (differential census)
    wall clock                     15.746 ms/token  (63.51 tok/s)
    non-kernel gap                  0.399 ms/token  = 2.5%

Launches are asynchronous: the CPU enqueues while the GPU executes, and at 2.055
ms of dispatch against a 15.75 ms token the host stays comfortably ahead. Only
the 0.40 ms that does not overlap is recoverable, and that is bounded by the
device-transition synchronizations the 12/12/11/11 ownership requires, not by
launch count.

> **Rule.** Launch COUNT is not launch COST. Before treating dispatch as a
> frontier, subtract summed kernel time from wall time; that difference is the
> entire addressable budget. A large launch count on an asynchronous queue that
> the host keeps fed costs nothing.

Conversion-kernel fusion is therefore worth at most a fraction of 0.40 ms/token,
not the 325 us/token first claimed. It is not the next campaign.

## Where the time actually is (steady state, load excluded)

    k_p92_f4_gateup            44/tok   2068.4 us   13.5%
    k_p92_mla_g16              46/tok   1581.5 us   10.3%
    k_p92_f4_downacc           44/tok   1484.8 us    9.7%
    k_p92_proj<6144,64>        46/tok   1016.1 us    6.6%
    k_p92_mhc_collapse         92/tok    783.0 us    5.1%
    k_p92_mla_absorb           46/tok    706.9 us    4.6%
    k_p92_route_top8           44/tok    691.7 us    4.5%

Two blocks dominate: the NVFP4 expert path (gateup + downacc = 23.2%, 3.55
ms/token) and MLA (g16 + absorb + value_up + combine ~ 20%). Those are the
frontiers, not dispatch.

## Census on freeze-66.0 (2026-09-08, after four rungs)

Differential, N=20 vs N=100, load excluded.

    launches/token   1,607   (was 1,651 on 63.5; the 44 removed MoE memsets)
    kernel sum       14.76 ms
    wall             15.13 ms at 66.08 tok/s
    non-kernel        0.37 ms   unchanged, dispatch still not a frontier

    k_p92_f4_gateup        44/tok   2079.5 us   14.1%
    k_p92_mla_g16          46/tok   1629.0 us   11.0%
    k_p92_f4_downacc       44/tok   1249.2 us    8.5%   (was 1484.8, hoist)
    k_p92_proj<6144,64>    46/tok   1010.5 us    6.8%
    k_p92_mhc_collapse     92/tok    800.7 us    5.4%
    k_p92_mla_absorb       46/tok    705.5 us    4.8%
    k_p92_route_top8       44/tok    622.1 us    4.2%
    k_p92_mla_value_up     46/tok    598.4 us    4.1%

Blocks: experts 22.6 percent (3.33 ms), MLA 21.7 percent (~3.2 ms). Level now.

MLA is the larger untouched surface and mla_g16 is the second kernel overall.
gate/up is where nine order-preserving mechanisms failed; its VALU floor is
25.52 us against 46 measured and occupancy is pinned at 7 waves by a VGPR 35 the
compiler will not reduce under any pressure. Do not reopen it without a new
mechanism.

## Third contamination: the probe must exercise the production control-flow path

2026-09-08, mla_g16. The most expensive of the three, because it did not merely
mis-rank a candidate - it produced a confident CLOSURE decision on a kernel that
had a 5 percent win in it.

### What happened

k_p92_mla_g16 takes `win_cap`. A DSA layer passes 0 and reads an explicit
selection list; a windowed layer passes a nonzero cap and computes
`(win_first + b) % win_cap`. Two different control-flow paths, two different
compiled instruction streams.

Every probe written that day passed **win_cap = 0**. So did the pre-existing
bench, tests/p92_mla_test.hip. But at the 220-token benchmark the executor sets
`win_cap = w.cap` on BOTH branches (tests/p92_generate.hip:456 and :458), so all
46 layers take the windowed path. **The measured path was the one that never
runs.**

Everything derived from those probes was therefore about a different program:
the 46.75 us figure, the per-variant comparisons, and worst of all the bound
"deleting the group merge is worth 0.4 us, so every remaining cross-lane
restructuring is worth at most that". That bound was used to close MLA.

### What was actually there

Static ISA on the real path showed the loop body issuing SIX serialized global
round trips per iteration - six `s_waitcnt vmcnt(0)`, one per load - because each
load sat inside its own `if (on)` exec-masked region. Making the loads
unconditional (inactive lanes cannot affect the result: s is overwritten with
-1e30f and p is 0, so only a VALID ADDRESS is required, obtained by pointing the
off case at sink_lat/sink_pe instead of nullptr) took it to two barriers, and the
composed engine from ~66.5 to ~70 tok/s. Bit-identical.

The reduction work that the closure was built around was real but small: 3.5
percent of the kernel. The load serialization was an order of magnitude bigger
and was invisible because the probe compiled a different branch.

### The rule

> **An optimization bound is valid only if the probe exercises the same
> production control-flow and ISA path.** Before trusting any standalone
> measurement, diff the arguments the probe passes against the ones the engine
> passes, and diff the compiled inner loop. A parameter that selects a branch is
> part of the shape being measured, exactly like a tensor dimension.

Corollaries:
- A bound derived from a deliberately-broken variant (section 7a of the
  playbook) inherits this defect. Bounding is only as valid as the path bounded.
- Any kernel with a mode parameter needs BOTH modes in its gate. The MLA gate
  had neither the windowed path nor a check that it existed.
- When a kernel's standalone time and its in-engine time disagree, treat that as
  evidence of a path difference, not as profiling overhead. Here standalone was
  46.75 us against an in-engine 35.4 us average, and the difference was
  attributed to context length rather than investigated.

### Cost of the three contaminations, for calibration

    initialization in whole-process counts   mis-ranked a campaign, caught before work
    roof measured at the wrong occupancy     over-stated a gap by 40 percent
    probe on the wrong control-flow path     CLOSED a kernel with 5 percent left in it

## Fourth law: a null result requires a demonstrated positive control

Banked 2026-09-09, after a two-day-old architectural constraint was falsified.

> **A null result is not evidence of absence until the instrument has
> demonstrated that it can produce and distinguish a non-null result.**

`receipts/07_P2P_DIAGNOSIS.md` concluded MI50 P2P was unavailable, from 0/12
directed pairs across every transport the API offers. The conclusion was false.
The same binary on the same cards under Linux 6.8.0-138 scores 12/12. Only the
kernel changed; ROCm userspace stayed at 5.7.1-98.

Two instrument failures produced the false null, and neither had a positive
control:

1. **Identity-offset peer tests could not distinguish self-aliasing from a
   no-op.** A peer pointer that silently resolves to the sender's own memory at
   the same virtual address returns plausible bytes. The test could not tell
   "the transfer did nothing" from "the transfer went to the wrong place",
   because both look like "destination unchanged". The tell was found by
   accident: a read-back returned the *previous test's* write pattern.
2. **Non-root PCIe inspection suppressed extended capability information.**
   `lspci` without root does not show ACS capabilities, so "no bridge advertises
   ACS" recorded *not visible* as *not present*, and an entire causal branch was
   closed on it.

The architecture built on that null - sequential 12/12/11/11 layer ownership and
a pinned-host transport ring - was sound engineering given the belief, and the
belief was never testable by the tests that produced it.

**Practice.** Whenever a gate decides an architecture by reporting absence,
non-occurrence or non-support:

- Build a POSITIVE control the instrument must pass. If the rig cannot produce a
  known-true result, its false results mean nothing.
- Build a HOSTILE negative control: a case that must fail, and fail
  *distinguishably* from the case under test. Distinct fill patterns per phase,
  never a value the previous phase could have written.
- State the layer the null covers. "P2P does not work" was really "P2P does not
  work on this kernel's KFD topology", and nobody had looked at `p2p_links`
  because every layer above it had been exonerated.
- Prefer data-integrity checks to capability bits. `hipDeviceCanAccessPeer`
  returned 1 on all twelve pairs while nothing moved.

This is the same failure as the three profiling contaminations above, in a
different register: the measurement was of something other than what was
believed to be measured.
