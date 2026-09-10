# 19. COOPERATIVE MULTI-WAVE ATTENTION — REFUTED ON ARITHMETIC, AND WHAT WAS BUILT INSTEAD

Directive was: attention is 57.7% of card time, tile-width tuning is closed, build a
cooperative multi-wave key-tile kernel targeting lower VGPR pressure and >2 waves/SIMD.

Attention was first confirmed **materially exposed** (receipt 18, exposure ratio 1.008),
so the premise for spending a kernel campaign here holds. The mechanism does not.

## The refutation

On gfx906, waves/SIMD = floor(256 / VGPR). Measured from the ISA, not assumed:

| kernel | VGPR | spills | waves/SIMD |
|---|---|---|---|
| k_pf_attn TQ=2 | 74 | 0 | 3 |
| k_pf_attn TQ=4 | 118 | 0 | 2 |
| k_pf_attn TQ=8 | 239 | 0 | 1 |

3 waves/SIMD requires <= 84 registers. At TQ=4 the NAMED state is acc 32 + qv 32 +
m_run 4 + l_run 4 = 72, so ~46 of the 118 are addresses, staging and temporaries.

Cooperative waves split the ACCUMULATOR across W waves, taking acc 32 -> 8 at W=4.
That lands at ~94. Even a free, infinitely-split accumulator lands at ~86. **The design
cannot reach 3 waves/SIMD no matter how far it splits, because the accumulator is not
what holds occupancy.** Its phase A would also be byte-for-byte the current inner loop.

Cutting `qv` instead means splitting the SCORE contraction across waves, which breaks
the rule the load geometry rests on: 512 bf16 = 1024 B = exactly 64 lanes x 16 B. At
W=2 the per-lane width halves to 8 B, and the estate's own roofline probe measured
8 B loads at 327 GB/s against 615 GB/s at 16 B.

Two further mined mechanisms do not apply here and are closed:
- **Split-K / flash-decoding partial combination.** The prefill grid is already
  (T/TQ, 48) ~ 1e5 blocks against 480 resident waves. There is no grid to fill; every
  split-K donor found is decode machinery.
- **Un-absorbed prefill.** Absorbed MLA is ~48 flop/B; un-absorbed is ~1 flop/B against
  an MI50 machine balance near 36. Un-absorbed is 30x under balance on this hardware.

## What the measurement actually pointed at

ISA of the k_pf_attn TQ=4 body loop, 258 instructions to consume TWO loads:

| per key | count |
|---|---|
| s_waitcnt | 26 |
| vmcnt(0) full drains | 1 |
| ds_bpermute | 24 |
| v_mul_f32 | 44 |
| v_fma | 72 |
| v_exp_f32 | 8 |

24 cross-lane ds_bpermute a key: six butterfly steps times four queries, on the LDS
path, in four dependent chains, with only 2 waves to hide the latency. And ~32 of the
44 multiplies are the accumulator rescale by alpha, which is exactly 1.0 on every key
that does not advance the running max.

## k_pf_attn2

Four changes, each aimed at one of those numbers. Semantics identical; reduction ORDER
is not, which this engine is explicitly allowed to change.

1. **DPP butterfly.** Three of the six xor steps have exact DPP equivalents that run on
   the VALU path with no lgkmcnt: xor 2 = quad_perm 0x4E, xor 1 = quad_perm 0xB1,
   xor 8 = row_mirror THEN row_half_mirror (xor 15 composed with xor 7). xor 4 has no
   DPP substitute; xor 16 and 32 cross the row. Lane mappings are the ones this project
   verified and shipped in p92_mla_g16.hip.
2. **exp2 with a prescaled query.** v_exp_f32 IS base-2 on gfx906, so every __expf pays
   an extra multiply by log2(e). Folding scale*log2(e) into the query once at load makes
   the score arrive in log2 units.
3. **Gated rescale.** The score is wave-uniform after the butterfly, so "did the running
   max advance" is a scalar branch and alpha is exactly 1.0 when it did not.
4. **Two keys in flight**, so a pair costs one vmcnt drain instead of two.

Measured effect on the per-key loop, from the ISA:

| per key | k_pf_attn | k_pf_attn2 |
|---|---|---|
| ds_bpermute | 24 | **12** |
| DPP ops | 0 | 16 |
| s_waitcnt | 26 | **14** |
| v_mul_f32 | 44 | **34** |
| VGPR (TQ=4) | 118 | 123 (still 2 waves/SIMD) |

Both kernels pass the attention gate identically: worst relative 5.794e-07 against an
independent host reference, all three tiles agreeing, causal negative control changing
786421 of 786432, and non-finite output rejected before any reduction.

Selected at runtime by PF_ATTN_KERNEL (1 = original, 2 = this). Judged on composed
prefill throughput, interleaved, six pairs — never on kernel time.
