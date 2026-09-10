# AMD gfx906 kernel notes — mechanisms, gates, negative results

## Mechanisms

- **wave64 / exact production geometry.** Every hot kernel is compiled for one
  model on one architecture: fixed rows, fixed K, fixed head counts, fixed
  shard geometry, fixed tails. No runtime branch, division, modulo, generic tail
  or dispatch in a hot loop. Code duplication is accepted.
- **Consumer-designed layouts.** A fast consumer owns its immutable weight
  representation. The dense MLA decode keeps each key's 1024-byte slice as one
  16-byte-per-lane load; the sparse path's producer emits contiguous per-query
  key streams (SPG2); the expert path decodes nibbles into int8 lanes that match
  `v_dot4_i32_i8` exactly. No universal transformed format is imposed.
- **Host-staged TP, producer push.** Peer writes ~23 GB/s, peer reads ~3.85 GB/s.
  The engine pushes state forward and never structures a hot path around
  consumers pulling remote state. Three crossings per decode token, 20,480 B each.
- **Attention.** MLA decode with 32-lane groups and 16 split slots; absorbed
  queries; branchless loads with the vmcnt(0) barrier count reduced 6 -> 2
  (measured rung 69.92 tok/s). DSA sparse attention: 24-head indexer, 192-dim
  scores, radix select top-2048, fused selector 146 us -> 26 us.
- **Everything else** (mHC Sinkhorn, MoME convolutions, routers, RMSNorm+quant,
  head) is shape-specialised; the head uses a two-stage int8 path with a
  top-256 refine.

## Measured negative results (kept)

| experiment | measurement | verdict |
|---|---|---|
| occupancy as an end in itself | VGPR 76 -> 50, occupancy 3 -> 4 waves, kernel 6.2% SLOWER | rejected; reduction economics worsened |
| larger TQ in sparse attention | TQ=4 597, TQ=8 419 tok/s vs TQ=2 669 at 64K | rejected; 3 waves/SIMD dominates key-traffic savings |
| cooperative multi-wave attention | refuted on occupancy arithmetic, then empirically by TQ=8 | rejected |
| compiler-visible union-index prefetch | LLVM re-serialised the rotation (phase-shift + re-derive) | rejected; representation change (SPG2 packing) was the fix |
| uncoalesced pack copy | 26 GB/s, +19.4 s of dsauni | rejected; coalesced (key,float4) mapping reaches ~780 GB/s |
| K-span 64 -> 512 in the CUDA backend (Spark) | 15.30 -> 14.72 tok/s | rejected (recorded for cross-backend reference) |
| device-grouped expert path (Spark) | 15.33 -> slower | rejected |
| block-wise launch tails | ~294K launches at 64K = ~0.8 s of a 115 s wall (0.7%) | not worth chasing; blocked D2H syncs were the real launch cost and were removed |

## Numerical gating policy

**Numerical bit-exactness is not used as a blanket acceptance criterion.**

Allowed without further proof: reduction order, reassociation, accumulation
order, intermediate and execution precision, storage/execution representation,
quantization and scale representation, packing, preshuffling, wave/tile
ownership, split-K structure, fusion and decomposition, producer/consumer
layout.

Exactness is required for discrete contracts: indices, bounds, ownership,
packing interpretation, nibble ordering, masks, selector membership where
membership is the contract, cache/state positions, ABI, memory safety, and
speculative control flow (accepted-prefix length, commit boundary).

Gates actually used, narrowest first:

1. host-reference comparison with an operation-appropriate tolerance and a
   negative control that must move the output (selector 3.4e-07, sparse
   attention 7.3e-07, SPG2 7.1e-07 worst relative);
2. structural checks where the contract is discrete (masks, membership,
   ordering, state transitions);
3. composed generation and coherent output at the production recipe.

No blanket cosine threshold is used either; a gate must fit the operation.
