# Whole-token decode cost ledger

Every row is measured on one MI50 at the stated shape, on real weights where
the mechanism has them. No estimates. Updated after each mechanism lands, so
the next optimisation goes where the cost actually is rather than where the
last kernel happened to be.

Context 8,192. The four cards run sequentially, so per-layer costs sum across
all 46 layers rather than divide by four.

| component | µs/unit | units | ms/token | share |
|---|---:|---:|---:|---:|
| MLA decode, DSA layers, composed with the indexer | 472.9 | 16 | 7.57 | 36.5% |
| routed FFN, 9-wide with the shared expert, 2 of 9 cold, H=64 | 130.6 | 44 | 5.75 | 27.7% |
| MLA decode, sliding layers | 150.8 | 30 | 4.52 | 21.8% |
| mHC, 2 sites per layer | 19.4 | 92 | 1.79 | 8.6% |
| lm_head, fast pass + top-256 refine | 565.4 | 1 | 0.57 | 2.7% |
| MoME convolutions, 3 sites per layer | 6.4 | 46 | 0.29 | 1.4% |
| dense MLP, layers 0 and 1 | 102.3 | 2 | 0.20 | 1.0% |
| **measured so far** | | | **20.69** | |

That is 48.3 tok/s for the measured components alone, before the norms, the
embedding, the four-card handoff, and whatever assembly costs. It is not a
throughput claim; it is the floor those components impose.

Not yet in the ledger: the RMSNorms (12 per layer by the oracle's order), the
merge mHC module, the embedding lookup, and the four-card handoff.

## Where the next optimisation dollar belongs

Attention is 58.3 percent of measured cost, down from 68.6 after the MLA
lane-width change. It is still the largest term and the DSA half still leads.

The composition gap recorded here earlier is CLOSED and was mostly an artifact.
Comparing composed against its own parts in one binary with the same
allocations, interleaved: the gap is +5.6 percent at 8K and +1.6 percent at
32K. The apparent 70 µs came from comparing against a microbenchmark at a
different cache size - MLA alone is 492 µs at a 4096-position cache and 529 at
8192. There is no mystery term.

## Rates, for judging whether a kernel is done

| kernel | GB/s |
|---|---:|
| int8 routed expert | 731.0 |
| dense gate/up, NVFP4 | 450.0 |
| NVFP4 routed expert on dot4 | 410.7 |
| dense MLP whole layer | 392.7 |
| dense down, NVFP4 | 324.0 |
| MLA decode, 16-lane groups (replaced) | 244.3 |
| lm_head, whole pass | 386.0 |
| MLA decode, 32-lane groups | 383.0 |
| NVFP4 routed expert, old float decode | 19.0 |

## Optimisations taken, and what measurement said

Each of these was found by breaking a composed measurement down, not by
inspection. Six of the eleven first hypotheses were wrong, and two of the wrong
ones would have looked plausible in a code review. That is the reason the rule
is to measure the dominant term rather than reason about it.

| mechanism | first hypothesis | what measurement showed | result |
|---|---|---|---|
| NVFP4 expert | representation is inherently slow | the float nibble decode was, not the format | 19.0 -> 410.7 GB/s |
| DSA selection | ten kernel launches | a serial 2048-bin walk, three times | 146 -> 26 µs |
| mHC site | launch overhead | a serial 4x4 Sinkhorn, 33.3 of 54.5 µs | 54.45 -> 19.41 µs |
| mHC logits | needed its own rms kernel | the rms scalar hoists out; the kernel cost more than the GEMV | 9.38 µs removed |
| dense down | six-shuffle reduction per row | coalescing dominates; 64 lanes already optimal | no change, hypothesis wrong |
| MLA | heads should share latent rows | head-grouping made it worse, 546 µs | reverted |
| MLA | random selection pattern is the cost | barely matters: 512 random, 526 contiguous, 591 strided | reverted |
| MLA | register pressure caps occupancy | correct: 32-lane groups, ~48 VGPRs | 512 -> 314 µs, 1.63x |
| head | two int8 components are nearly free | wrong, 483 -> 1620 µs | refine top-256 only |
| head | refine re-staging is the cost | wrong, 817 -> 826 µs | no change |
| head | candidate selection is the cost | correct: fused selector 343.8 µs at n=151552 | staged, 81.5 µs |
