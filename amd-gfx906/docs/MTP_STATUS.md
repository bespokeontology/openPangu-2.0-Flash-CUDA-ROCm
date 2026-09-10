# MTP status — experimental, not part of the released AMD performance claim

**AMD MTP is development work. The released AMD decode authority is plain
native target decode with MTP disabled.** The frozen decode binary contains no
MTP path, and no MTP number appears in `BENCHMARKS.md`.

## What exists (development branch `mtp-draft-20260909`, commits `d51ad52`, `b1bdead`)

| item | state |
|---|---|
| tensor census | MTP layers 46/47/48 are full decoder blocks (MLA + 256-expert MoE + shared expert + router + norms) with `eh_proj`, `enorm`, `hnorm`, own `embed_tokens` and own `shared_head`; no mHC, no DSA; SWA window 2048 |
| artifact | separate NVFP4 container built by `decode/tools/mtp_quant.cpp` (2328 records, 3.665 GB weights + 0.458 GB scales); encoder gate = UE4M3 boundary sweep, sampled round-trip worst 0.75 scale units against the 1.2 E2M1 bound |
| native draft path | implemented; per-layer body reuses the decode kernels; heads on dev0/1/2; measured 3.7-5.2 ms per three-token draft round |
| KV priming | shadow pass over prompt positions |
| state machine | full round implemented: draft -> verify -> accept/reject -> accepted-prefix commit with boundary snapshot and replay of the rejected tail |
| measured draft quality | d1 == trunk greedy next: 11/30 (36.7%) no-prompt regime, 38/150 with a 3850-token prompt in a degenerate greedy loop |

## Negative architectural result (recorded, not advertised)

Sequential target verification (three ordinary trunk passes per round) was built
and measured end to end:

| component | ms/round |
|---|---:|
| draft (three heads) | 5.24 |
| verify (3 sequential trunk passes) | 60.67 |
| commit (replay of rejected targets) | 29.09 |
| mean accepted drafts | 0.46 |
| committed throughput | 0.48 tok/s vs 55.99 tok/s plain decode |

This is a negative result: sequential verification costs about three full target
steps and cannot be the production architecture. It is **not** an MTP
performance figure.

## Remaining work

1. **Fixed-T batched target verifier** (T = 4 rows: boundary + three drafts).
   The target rows must share each expert's weight traversal (routed rows group
   to ~12-16 unique experts instead of 27 sequential loads), share selected-key
   reads in attention, and batch the mHC and projection paths. Target
   <= 25-30 ms per round. This is the make-or-break build.
2. **Accepted-prefix commit without replay** where the batched verifier makes it
   possible; discard rejected suffixes rather than recomputing them.
3. **Production sampling semantics**: greedy is a debugging mode. Speculative
   acceptance must be implemented against the engine's real sampling recipe
   (proper rejection sampling with residual correction), not "draft equals
   greedy target".
4. **Acceptance measurement with sampling** on substantive prompts. Greedy
   loops are not an acceptance gate; the 36.7%/25.3% greedy figures above are
   diagnostics of a degenerate regime, not quality statements.

## Why AMD MTP is not released

Not because the architecture is unsupported: the artifact, the draft path and
the state machine all run today. It is not released because the target verifier
that makes speculation pay is still under development and the acceptance
measurements required to claim a speed-up have not been made with production
sampling. When those land, the benchmark table gains a third row -
`AMD gfx906 / MTP` - directly comparable to the Spark MTP row, separating the
contribution of the AMD target engine from the contribution of speculation.
