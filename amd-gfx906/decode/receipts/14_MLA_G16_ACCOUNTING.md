# mla_g16: an accounting, and a RETRACTED closure

> **RETRACTION, same day.** Everything below was measured with probes passing
> `win_cap = 0`, which selects the DSA branch. The executor sets `win_cap`
> nonzero on BOTH branches (tests/p92_generate.hip:456 and :458), so at the
> 220-token benchmark all 46 layers take the WINDOWED path. The probes compiled
> a different program from production.
>
> The specific claims that do NOT survive: the 46.75 us figure as a production
> number; the per-variant comparisons; and above all the bound "deleting the
> group merge is worth 0.4 us, therefore every remaining cross-lane
> restructuring is worth at most that", which was used to close the kernel.
>
> What was actually left in it: the loop issued SIX serialized global round
> trips per iteration, one `s_waitcnt vmcnt(0)` per exec-masked load. Making the
> loads unconditional took that to two and the engine from 66.53 to 69.92 tok/s,
> +5.36 percent paired median over twelve pairs, bit-identical. See commit
> 9a149ac and receipts/11_PROFILING_LAW.md, third contamination.
>
> What DOES survive: the DPP reduction rung at 66.72 is a valid measured
> composed speedup and stays. Only its CAUSAL claim - that the reduction was
> essentially all the remaining opportunity - is withdrawn. The counter
> measurements (FETCH_SIZE 513.8 KB against 19,240 KB issued; VALUBusy 26.9;
> MemUnitStalled 0.8) also stand, and in hindsight the low utilization with no
> stalls was the signature of the serialized loads, not of an irreducible
> dependency chain.
>
> The original text is kept below unedited, because how a wrong closure was
> reached is the useful part.

---

# mla_g16: physically accounted, and closed (ORIGINAL, SUPERSEDED)

2026-09-08, from freeze-p92-amd-66.0. Geometry fixed at the measured optimum
LANES=32 / SPLITS=16. 46.77 us/call at nsel=220, 1629 us/token over 46 layers,
11.0 percent of the engine and the second-largest kernel.

## What it is NOT bound by

Every candidate was measured and eliminated.

    axis            measurement                                      verdict
    HBM             FETCH_SIZE 513.8 KB/call vs 19,240 KB issued     not memory-bound
                    distinct set 401 KB in a 4 MB L2
    VALU            VALUBusy 26.9 percent                            not ALU-bound
    memory stall    MemUnitStalled 0.8 percent                       no backpressure
    cross-lane      ALL DS work worth ~2 us of 46.77                 not DS-bound
    occupancy       3 waves/SIMD; raising to 4 made it SLOWER        not a lever

The 37x gap between issued and fetched bytes is the same effect found in the
expert kernels: all 48 heads read the same KV rows, so the traffic is replicated
from a small L2-resident working set. Any GB/s computed from issued bytes for
this kernel is L2 bandwidth, not HBM, and is meaningless as a roof.

## What it IS bound by

The online-softmax recurrence. Per entry:

    load latent+rope -> 4 FMA -> 32-lane reduction -> exp -> rescale 16 acc -> next

Each iteration's m_run, l_run and acc feed the next. At 3 waves/SIMD there is
almost nothing to interleave against that chain, which is exactly why both units
sit near a third busy with no stalls.

## The three attacks on the chain, all order-preserving, all failed

    variant                              VGPR  occ    us      bits
    production                             76    3   46.77    -
    LANES=64 (DPL 16->8)                   50    4   49.64    identical
    drop v[16], re-read latent             70    3   62.60    identical
    two entries per softmax update          -    -   68.34    identical

LANES=64 is the instructive one: the intended mechanism WORKED - registers fell,
occupancy rose - and the kernel got slower, because at 32 lanes one shfl_xor
reduces both half-wave groups at once (5 shuffles serve 2 entries) while at 64
it is 6 shuffles for 1. Changing the geometry re-priced the reduction.

## What was taken

Exact DPP substitution for the per-entry reduction, 5 DS ops -> 2 DS + 4 ALU,
same source lanes so bit-identical:

    xor 16  crosses the 16-lane row     -> ds_bpermute
    xor  8  == row_mirror, row_half_mirror
    xor  4  no DPP substitute           -> ds_bpermute
    xor  2  == quad_perm 0x4E
    xor  1  == quad_perm 0xB1

46.77 -> 45.14 us standalone (3.5 percent). Composed 66.19 -> 66.72 tok/s,
+0.80 percent over eight interleaved pairs, clean ranges control 65.99-66.58 and
candidate 66.30-66.84. 14/14 gates, 220-token output bit-identical.

## The ceiling on anything further

Deleting the 18-shuffle group merge outright - wrong maths, bound only - gives
44.72 us. So every remaining cross-lane restructuring in this kernel, including
an ABI redesign to 32 partials, is worth at most 0.4 us of 45. That number is
why this branch is closed rather than merely paused.

## Also established here

- P92_SPLITS was defined twice and resolved to 16 only by include order in a
  single translation unit. Now one contract, with gate 14 and a negative
  control. See receipts/13_MLA_SPLIT_ABI.md.
- The lane-width curve is closed at both ends: 16 rejected earlier (244.3 vs
  383.0 GB/s), 32 optimal, 64 measured slower.
- The historical "~48 VGPR" in 06_LEDGER.md no longer describes this kernel. It
  is 76, and the growth is address/control state from the sinks-balancing and
  sliding-window paths, not the v[16] array that looks expensive in the source.

## Do not reopen without

a mechanism that shortens the online-softmax recurrence itself, or that gives a
wave independent work across entries without adding live state. Parameter
sweeps on lanes, splits, block size and register pressure are all spent.
