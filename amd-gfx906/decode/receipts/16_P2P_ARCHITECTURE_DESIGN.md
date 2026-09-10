# openPangu on 6.8 + validated P2P: architecture design

2026-09-09. Design lane. The production engine is NOT modified.
Control: `freeze-p92-amd-69.9`, commit 9a149ac, binary sha
b28f0eef71121a0199315dfc4b5d1deee1ebbe3b8dafaad4cd40657a24d52d75.

## 1. Preserved-binary 6.8 re-baseline — DONE, and the answer is zero

Executed the preserved binary, not a rebuild. ROCm userspace unchanged at
5.7.1-98, clocks pinned 1700 MHz, identical recipe
(`p92_gen <ckpt> <artifact> <arena> 220 512 148899`).

    5.15 freeze (12 interleaved pairs)   69.92 tok/s   median
    6.8  same binary, 6 runs             69.92 / 69.88 / 69.85 / 69.80 / 69.74 / 68.71

**Steady-state decode platform gain is zero.** Any future gain is therefore
attributable to architecture, which is exactly the separation we wanted.

First token DID move: 19.99-20.46 ms on 5.15 against 17.85-18.54 ms on 6.8,
about 2 ms. That is allocation/runtime, not decode, and is NOT yet decomposed.
It is not evidence about prefill, which does not exist in this engine.

Runtime-sensitive constants re-measured on 6.8:

    launch overhead            1.245 -> 1.99 us   (+60%)
    fillBufferAligned 10 KB     3.37 -> 3.19 us   unchanged
    hipSetDevice+Synchronize            0.15 us   (idle; its engine cost is the drain)

The launch result is a second confirmation of an existing law: dispatch cost rose
60 percent and steady-state throughput did not move. **Launch count is not launch
cost.**

## 2. P2P characterisation — the governing result

Gated by data integrity, never a capability bit (under 5.15 all twelve pairs
reported canAccessPeer=1 while nothing moved). Distinct per-phase fill patterns
so a stale value cannot be read as success. **12/12 directed pairs move real
data.**

Positive control for the instrument: same-device copy 653.6 GB/s, consistent
with the 741-785 GB/s streaming roof measured during the port. The rig is sound.

**16 MB, one kernel per timing, min of 5, all 12 directed pairs:**

    local copy (control)          653.6 GB/s
    peer WRITE                22.9 - 23.2 GB/s     spread 1.3%
    peer READ                   3.8 - 3.9 GB/s     spread 2.6%
    ratio                     5.86 - 6.09x

No root-complex asymmetry: every direction behaves identically. ~23 GB/s is
about 75 percent of PCIe 4.0 x16, which is what these cards should do.

**This is the single most important number in the document, and it inverts the
obvious design.** "Consume remote data directly" is a peer READ at 3.85 GB/s -
**2.6x SLOWER than the host ring it would replace** (~10 GB/s at large payloads).
The fast direction is the producer pushing into the consumer's memory.

> **DATA-PLANE RULE. The producer PUSHES. Never make the consumer PULL.**
> A remote consumer reads at 3.85 GB/s; a remote producer writes at 23.1 GB/s
> and the consumer then reads locally at 653 GB/s. Every collective, dispatch
> and handoff in this design is expressed as a write into the destination's
> memory, followed by a local read.

Two corollaries:

- **`hipMemcpyPeer` is not the primitive.** At the 20 KB decode payload it
  measured 13.30 us against the existing host path's 13.05 - no better.
  `hipMemcpyPeerAsync` 8.02 us. The usable primitive is a kernel storing into
  peer VRAM.
- **Small-payload peer latency is not yet measured.** Two attempts produced
  launch-throughput artifacts (2.1 us at both 20 KB and 160 KB, against a 1.99 us
  null-kernel launch). Any figure below ~5 us in earlier notes is withdrawn. The
  20 KB decode-transport latency needs a completion-synchronised measurement with
  a null-kernel baseline subtracted, and it is a required input to section 10.

## 3. Where the time actually is — differential census on freeze-69.9

Load-time excluded. Total kernel 14.28 ms/token against a 14.30 ms wall, so
non-kernel time is now ~0.02 ms and there is nothing left in dispatch.

    class                              ms/tok   share
    EP    expert + router                4.21   29.5%
    TP    shards cleanly, heads/cols     5.51   38.6%
    REPL  small, latency-bound           4.56   31.9%

    largest single kernels
      f4_gateup            2083.7 us  14.6%   EP
      f4_downacc           1277.3 us   8.9%   EP
      proj<6144,64> o_proj 1034.0 us   7.2%   TP
      mla_g16               978.1 us   6.9%   TP
      mhc_collapse          797.1 us   5.6%   REPL
      mla_absorb            713.1 us   5.0%   TP
      proj<1024,32>         601.7 us   4.2%   TP
      route_top8            600.3 us   4.2%   EP
      mla_value_up          596.2 us   4.2%   TP
      rmsnorm_quant<2560>   484.4 us   3.4%   REPL

**The REPL class is the whole problem.** 4.56 ms of small, latency-bound
per-layer operations - mHC (collapse/logits/reexpand), the RMSNorms, the MoME
convs, format conversions. They are already at 3-8 waves/SIMD on tiny state
vectors. Sharding a 2560-element norm four ways gives 640 elements per card plus
an all-reduce that costs more than the kernel. **These must be REPLICATED:
every card computes them redundantly on the full state, paying the same 4.56 ms
but requiring no communication.**

That sets an Amdahl floor. Even with the other 68 percent perfectly parallel,
the ceiling is 14.28 / 4.56 = **3.13x**. Realistically far less.

## 4. Memory — this is what forces the architecture

The arena is 257 slots per MoE layer (256 routed + 1 shared), expert weights
ONLY, 44 MoE layers:

    257 x 4,423,680 B x 44 = 50.02 GB      measured arena 50.03 GB (11.37/13.64/12.51/12.51)

Per expert: gate[1024,2560] + up[1024,2560] + down[2560,1024] = 7,864,320 values
at NVFP4 0.5625 B/value (0.5 nibble + 1 B per 16-value block scale) = 4,423,680 B.

**Everything else - MLA projections, dense layers 0-1, mHC, norms, the untied
151552-row head - is about 1.2 GB.**

    experts        50.0 GB    97.7%   MUST be sharded; cannot be replicated
    everything else 1.2 GB     2.3%   CAN be replicated on all four cards, 4.8 GB total

Card capacity is 15.98 GiB each. That yields the only viable placement:

    per card: 50.0/4 = 12.5 GB of expert slots
            + 1.2 GB replicated non-expert weights
            = 13.7 GB, leaving ~2.3 GB for KV cache, activations and workspace

**KV cache is the constraint that bounds this design.** At the 512-position
benchmark the cache is 46 x 512 x 576 x 2 B = 27 MB and irrelevant. At the
model's 262,144-position context it is 13.9 GB, which does not fit alongside
12.5 GB of experts even once, let alone replicated four times. Therefore:

> Attention must remain SHARDED BY HEAD (each card owns 12 of 48 heads and only
> that shard's KV), not replicated. Replicating the small ops is affordable;
> replicating attention state is not.

## 5. Current token dependency graph

    embed -> [L0 dense] -> ... -> [L45 MoE] -> merge mHC -> norm -> head -> token

Per layer: mHC pre-mix + Sinkhorn -> input norm -> q/kv projections -> 3 MoME
convs -> RoPE -> DSA-select or window -> MLA decode (8 splits) -> combine ->
value-up -> o_proj -> post-attn norm -> mHC post -> pre-MLP norm -> MoE or dense
-> post-MLP norm -> mHC post.

Device structure: layers 0-11 on dev0, 12-23 dev1, 24-34 dev2, 35-45 dev3.
**Three device transitions per token**, each a `hipDeviceSynchronize` followed by
a 20,480 B host round trip. While dev0 runs its twelve layers, dev1-3 are idle.
Concurrent GPU count is **1**. Utilisation is 25 percent by construction.

## 6-8. The three candidate architectures

All three replicate the 1.2 GB of non-expert weights and the REPL class of
kernels. They differ in what else moves.

### A. Sharded TP

Every card computes every layer on a 1/4 shard.

- MLA: 48 heads / 4 = 12 heads per card. q/kv projections column-sharded, o_proj
  row-sharded. KV cache sharded by head - **this is the placement that makes long
  context fit.**
- Dense and shared-expert FFN: column-shard gate/up, row-shard down.
- Head: vocab-shard 151552/4, then an argmax reduction over 4 partials.
- Routed experts: cannot be TP-sharded without replicating them. **A is not
  viable alone** - it has no answer for the 50 GB.

Collectives: after each row-sharded matmul, an all-reduce of a 2560-float state.
2 per layer (attention out, MLP out) x 46 = **92 all-reduces/token**, 10,240 B
each. Under the push rule: each card writes its 10 KB partial into the other
three (30 KB out per card, concurrent), then each sums 4 partials locally.

Critical path: TP 5.51/4 = 1.38, REPL 4.56, experts unsolved.

### B. Expert parallel

Experts distributed across cards; everything else replicated.

- Each card owns 64 of 256 routed experts for every layer (12.5 GB), plus the
  shared expert replicated.
- Router runs redundantly on all cards - it is only 600 us and needs no exchange.
- Dispatch: each card pushes the 2560-element activation to the cards owning its
  selected experts, they compute, then push weighted partials back.
- Combine: sum of up to 8 partial 2560-vectors.

**Single-token load balance is the problem.** 8 experts drawn over 4 owners:
E[max load] is about 3.3 experts against a mean of 2, so the busiest card sets
the critical path and efficiency is ~61 percent. That is inherent to one token;
it disappears entirely in prefill.

Collectives: 1 dispatch + 1 combine per MoE layer x 44 = **88 exchanges/token**,
10,240 B each way.

Critical path: EP 4.21/4 x 1.65 = 1.74, TP unsharded 5.51, REPL 4.56.

### C. Hybrid TP + EP  — RECOMMENDED

Attention and dense TP-sharded by head/column; experts owned; small ops
replicated. This is the only candidate that solves both memory and utilisation.

    placement per card
      12 of 48 attention heads, and only that shard's KV cache
      64 of 256 routed experts for all 44 MoE layers        12.5 GB
      shared expert, replicated                              0.2 GB
      all non-expert weights, replicated                     1.2 GB
      = 13.9 GB of 15.98 GiB

    critical path
      EP    4.21 / 4 x 1.65 imbalance   = 1.74 ms
      TP    5.51 / 4                    = 1.38 ms
      REPL  unchanged                   = 4.56 ms
      compute subtotal                    7.68 ms
      + collectives (section 10)

**7.68 ms of compute = 130 tok/s before communication.** The REPL class is 59
percent of that critical path.

## 9. Comparison table

    metric                        A  TP        B  EP        C  Hybrid
    VRAM/card                     infeasible   13.9 GB      13.9 GB
    experts placed                no           yes          yes
    long-context KV fits          yes          NO (repl.)   yes
    compute/card/token            1.38+4.56    1.74+5.51    1.74+1.38+4.56
    collectives/token             92           88           92 + 88 = 180
    bytes/token                   92x30 KB     88x2x10 KB   ~4.6 MB
    sync boundaries/token         92           88           ~180
    concurrent GPUs               4            4 (61% bal)  4
    unavoidable serial            4.56 ms      4.56 ms      4.56 ms
    est. critical path            n/a          7.99 ms      7.68 ms + comms
    risk                          memory       imbalance    highest complexity

## 10. Timing model and its sensitivity

180 exchanges/token is the hybrid's exposure. The 20 KB peer latency is NOT yet
measured (section 2), so this is a sensitivity table, not a projection:

    per-exchange   comms/token   total     tok/s
      3 us            0.54 ms    8.22 ms    122
      5 us            0.90 ms    8.58 ms    117
     10 us            1.80 ms    9.48 ms    106
     20 us            3.60 ms   11.28 ms     89
     40 us            7.20 ms   14.88 ms     67   <- no better than today

**The design only pays if an exchange costs under ~20 us.** At the current host
path's 13 us it lands near 100 tok/s; at 40 us it is worthless. That single
measurement decides whether this programme is worth running, and it is the first
thing to do.

## 11. Risk

    highest   the REPL floor. 4.56 ms, 59% of the critical path, and every
              attempt during the port to speed small latency-bound kernels
              failed. If it cannot be reduced, 130 tok/s is the ceiling.
    high      180 synchronisation boundaries per token, each a potential
              serialisation. The port's law: the composed engine decides.
    high      EP imbalance at one token, ~61% efficiency, irreducible.
    medium    engineering scale - this is a new engine, not a modification.
    medium    KV sharding must be right or long context stops fitting.
    low       P2P correctness. 12/12 with a hostile gate, uniform directions.

## 12. Recommendation

**Prototype C, but measure the exchange first, and build PREFILL before decode.**

Prefill is the stronger case and it does not exist yet:

- It is greenfield. No frozen engine to preserve, no regression risk.
- Its payloads are T x 20,480 B, squarely in the bandwidth regime where peer
  write is 23.1 GB/s against the host ring's 10 GB/s, a measured 2.3x.
- **The REPL floor disappears.** Those kernels are latency-bound because they
  process one token's 2560-element state. At T=512 they become large batched
  GEMMs that shard perfectly. The 31.9 percent Amdahl term is a decode-only
  problem.
- EP imbalance disappears: over hundreds of tokens all 256 experts are hit and
  the distribution flattens.
- Communication amortises over T rows while compute grows with T, so the
  collective count per token falls by a factor of T.

Decode's honest ceiling is ~130 tok/s, a 1.86x, gated by a 4.56 ms floor that
resisted every optimisation attempt during the port. Prefill's ceiling is near
4x and it is unbuilt. **Build prefill on the four-GPU architecture; leave the
69.92 decoder frozen until prefill proves the substrate.**

## 13. Minimum falsifying prototype

Two experiments, in order, before any engine work.

**P1 - the exchange cost.** Completion-synchronised 20,480 B peer exchange, push
semantics, four cards, with the null-kernel baseline subtracted. Include a
4-way push all-reduce (each card writes 10 KB to three peers, then sums four
partials locally) since that is the actual primitive. **Falsifies C if a 4-way
exchange costs more than 20 us.** One afternoon.

**P2 - the REPL floor is real.** Take the three largest REPL kernels
(mhc_collapse 797 us, rmsnorm_quant 484, mhc_logits 442 = 1.72 ms) and attempt a
4-way shard with a push all-reduce. **Falsifies the replicate-the-small-ops
assumption if sharding beats replication.** If it does not, the 4.56 ms floor is
confirmed and the decode ceiling is fixed at ~130 tok/s.

Only if P1 passes and P2 confirms the floor is the full hybrid worth building.

## 14. Acceptance gates and hostile negative controls

Per the null-result law, every gate needs a positive control and a hostile
negative control that must fail distinguishably.

    G1 peer integrity      12/12 directed pairs, distinct per-phase patterns.
                           NEGATIVE: sender-local alias must be detected as a
                           distinct failure, not as "unchanged". Already built.
    G2 exchange            all-reduce result equals a CPU reference.
                           NEGATIVE: drop one card's contribution; must fail.
                           NEGATIVE: feed generation N-1 from one card; must
                           fail. Give every (iteration, card) a distinguishable
                           deterministic payload so stale, missing, duplicated
                           and mis-sourced data are all distinguishable from
                           correct.
    G3 expert ownership    every routed expert is owned exactly once; the union
                           over cards is all 257 slots x 44 layers.
                           NEGATIVE: remove one owner; the gate must report the
                           exact missing slots.
    G4 KV shard            head h's cache lives on exactly one card and attention
                           for head h reads only that card.
                           NEGATIVE: perturb one head's shard; output must change.
    G5 composition         full-layer output against the frozen 69.9 engine, with
                           operation-appropriate tolerance. Not bit-exact - the
                           reduction order changes.
                           NEGATIVE: the port's standing pattern, poison every
                           consumer-visible slot and require zero survivors.
    G6 soak                sustained run with changing inputs per iteration, to
                           catch races that a constant input hides.

## Appendix - what is measured and what is estimated

    MEASURED  6.8 re-baseline, launch/fillBuffer/sync constants, 12/12 P2P
              integrity, peer write 23.1 GB/s, peer read 3.85 GB/s, local
              control 653.6 GB/s, host ring ~10 GB/s at large payloads, the
              full 69.9 differential census, the 50.02 GB arena arithmetic.
    ESTIMATED EP imbalance 1.65x (8 balls in 4 bins), the EP/TP/REPL
              classification, every critical-path figure, all tok/s projections.
    NOT YET   20 KB peer exchange latency (two attempts produced launch-
              throughput artifacts), the TTFT decomposition behind the ~2 ms
              first-token improvement, whether REPL kernels shard.
