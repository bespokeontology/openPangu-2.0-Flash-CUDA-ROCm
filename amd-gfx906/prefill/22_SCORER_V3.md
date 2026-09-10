# 22. SCORER v3 KEPT — -37% on the engine's only quadratic term

Composed A/B, interleaved, warmup discarded, disjoint at both contexts:

  16384   v1 679.3 679.0   v3 684.1 684.7   +0.77%
          scorer 2863 -> 1955 ms, -31.7%
  65536   v1 582.0 574.9   v3 602.3 602.2   +4.1%
          scorer 43965 -> 27667 ms, -37.1%

The gain tracks the bucket exactly: the scorer is 3.3% of card time at 16K and
10.3% at 64K, so the same -37% buys +0.77% and +4.1% respectively. It will keep
growing, because score generation is the only quadratic operation in the engine
and everything else measures exactly linear.

Correctness is the strongest available: v3 reproduces v1 BIT-IDENTICALLY over
every scored position, worst relative 0.000e+00. Same FMA order, different memory
layout and a different issue path. Not a vacuous comparison - the same gate run
scores 3.388e-07 against an independent host reference on the same values and
rejects non-finite before any reduction.

The mechanism, for the record, is two separate fixes and both are needed:

UNIFORMITY gets off the LDS path. The query address is wave-uniform, so a
uniform GLOBAL address lets the backend issue s_load into SGPRs and feed
v_fmac_f32 the scalar operand directly. ds_read 12 -> 0, lgkmcnt waits 12 -> 1.
On its own this RAISED instruction count 58 -> 61 and would have been discarded
by anyone counting instructions.

CONTIGUITY gets the issue-rate win. q[h*128+d] strides 512 B between heads and
cannot vectorise; storing the query dim-major as [128][24] makes the 24 heads of
one dim contiguous and 24 scalar dwords become two s_load_dwordx. 39 instructions
against 58, 1.62 an FMA against 2.42.

Cards released to the Qwen lane immediately after this measurement; the freeze
of this rung is owed and will be taken when a window is next available.

