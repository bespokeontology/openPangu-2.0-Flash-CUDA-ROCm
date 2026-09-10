# CANDIDATE, NOT PROMOTED: hoisted downacc reduction

Status: **PROMOTED 2026-09-08 as the 66.08 rung.** Operator released the
bit-identity requirement after the order-preserving avenue was exhausted.
Composed gain 2.83 percent, median 64.27 -> 66.08 tok/s. History below kept as
written, including the reasoning for holding it.
Date 2026-09-08. Baseline freeze-p92-amd-63.5 (63.51 tok/s).
Implementation preserved at `tests/zz_probe6.hip`, kernel `k_dn_hoist`.

## The mechanism

`k_p92_f4_downacc` runs a 5-step `__shfl_down` reduction inside the expert loop,
once per expert: **45 `ds_bpermute` per thread against only 36 loads**. gate/up
does one reduction for the whole row: 10.

The reduction is hoistable because the per-expert scale is lane-uniform:

    sum_k c_k * ( sum_l acc_k[l] )   ==   sum_l ( sum_k acc_k[l] * c_k )

Accumulate `lane += acc_k * c_k` per lane across all 9 experts, then reduce once.

## Measured

    production: 9 reductions, 45 ds_bpermute      32.62 us   406.9 GB/s
    hoisted:    1 reduction,   5 ds_bpermute      28.45 us   466.4 GB/s
    no reduction at all (bound, wrong maths)      28.50 us   465.6 GB/s

Hoisting reaches the no-reduction bound EXACTLY, so it removes the entire
shuffle cost. Worth 4.17 us/call x 44 MoE layers = **183 us/token, ~1.2% engine**.

## Why it is not promoted

    hoisted vs production: relative_l2 1.885e-07
                           2560/2560 rows differ in at least one bit

Mathematically identical, but it REASSOCIATES: the float additions happen in a
different order. Every rung to 63.51 has been bit-identical.

Operator decision 2026-09-08: hold. Two reasons, both strategic rather than
numerical.

1. Bit identity is a debugging instrument, not just a quality bar. While it
   holds, every change classifies instantly as performance-only versus semantic,
   and a later regression is localizable because token identity is an absolute
   discriminator. Spending it costs that leverage permanently.
2. An order-preserving avenue with materially larger payoff is still open: the
   ALU/memory-overlap gap is ~22 us/call across both kernels against ~4.17 us
   here. Do not spend the invariant for 1.2% while the clean money is unclaimed.

## When to revisit

If the ALU/overlap investigation bottoms out, promote this candidate through the
full protocol: 13 gates, deterministic generation, composed A/B against the
then-current freeze. At that point the trade is made WITH evidence about what
the order-preserving route was actually worth.

Note the asymmetry with the rejected mHC variant C: there, the reassociating form
had NO speed advantage, so rejecting it was free. Here the advantage is real, so
this is a genuine trade rather than a dominated option.
