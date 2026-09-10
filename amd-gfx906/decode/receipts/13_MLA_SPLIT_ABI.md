# The P92_SPLITS accident: compiled geometry is evidence, headers are not

Found 2026-09-08 while opening MLA from freeze-p92-amd-66.0. No live defect, but
a dormant one, and it invalidated the performance reasoning it was found during.

## What was there

`P92_SPLITS` was defined TWICE, by the producer and the consumer of the same
buffer:

    src/p92_mla_g16.hip     #ifndef P92_SPLITS / #define P92_SPLITS 8  / #endif
    src/p92_mla_split.hip   #define P92_SPLITS 16                       (unguarded)

k_p92_mla_g16 (producer) writes pacc/pm/pl indexed by P92_SPLITS.
k_p92_mla_combine (consumer, in mla_split.hip) reads them indexed by P92_SPLITS.
tests/p92_generate.hip allocates them sized by P92_SPLITS and launches the
producer with grid dim3(48, P92_SPLITS/4).

Production resolved to **16**, for two reasons that both have to hold:
single-translation-unit compilation (p92_generate.hip #includes every source),
and an include order that puts mla_split (line 23) before mla_g16 (line 24), so
the unguarded 16 wins and g16's #ifndef never fires.

Change the include order, or split the build into real translation units, and
the producer would write 8 splits per head while the consumer read 16 - reading
uninitialised memory for half of every head, silently.

## Proven, not inferred

A temporary receipt compiled into the shipped binary and then reverted:

    P92_SPLITS compiled into this TU = 16
    g16 grid dim3(48,4) x 256 = 192 blocks = 768 waves
    combine grid dim3(48) x 256, inner loop s < 16
    pacc alloc bytes = 1572864            (393,216 floats)
    producer max pacc float index = 393215
    consumer max pacc float index = 393215
    pacc capacity in floats       = 393216

Producer and consumer agree; the top index is capacity minus one. Consistent.

## Why it mattered anyway

The source *reads* as 8. Reasoning from the header would have given 48 x 2 = 96
blocks and 384 waves, and concluded that mla_g16 was starved of global
parallelism on a machine holding ~1920 waves. The truth is 192 blocks and 768
waves against ~720 resident slots at its 3 waves/SIMD - the machine is FULL, and
the constraint is register depth, not grid size. An entire optimisation
direction would have been chosen on a false premise.

> **Rule.** Compiled geometry is evidence. Header defaults, `#ifndef` fallbacks
> and comments are not. Establish a constant that crosses a producer/consumer
> boundary by compiling it into the shipped binary and printing it, or by a
> static assertion that fails the build.

Same lesson as the stale comments found in the same kernel: `G16_DPL` was
commented "32 latent dims per lane" and `G16_STEPS` "8 uint2 loads", both left
over from when G16_LANES was 16. The real values are 16 and 4. Corrected.

## The fix

`include/p92amd/p92_mla_geometry.h` is now the single definition. Both kernels
include it; neither defines it. A `P92_SPLITS_LOCKED` guard makes a second
definition a build error rather than a silent race between include order and
`#ifndef`.

## Gate 14, and its negative control

`tests/p92_mla_abi_test.hip` poisons every consumer-visible slot with a sentinel,
runs the producer, and requires that not one poisoned word survives. If the
producer's splits are fewer than the consumer's, the tail keeps the sentinel.

It also carries a NEGATIVE CONTROL, because a gate that cannot fail proves
nothing: it re-poisons and relaunches the producer at half the splits, and
requires the detector to fire. It reports 196,992 gaps - exactly 8 x 48 x 512
plus 8 x 48 - so the gate is demonstrably able to catch the bug it exists for.

Compile-time half: static_asserts that SPLITS is a multiple of 4 (the grid is
dim3(48, SPLITS/4)), that the per-lane dims tile KVLORA exactly, that the vector
steps tile the per-lane dims, and that PE divides across lanes.
