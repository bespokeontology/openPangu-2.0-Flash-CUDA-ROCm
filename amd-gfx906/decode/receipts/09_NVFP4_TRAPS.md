# NVFP4 on non-NVFP4 hardware: traps and results

gfx906 has no FP4 instruction and no FP8. Everything below is about executing a
4-bit block-scaled format on an integer dot-product unit, and about the places
where the STORAGE representation and the EXECUTION representation must be kept
apart. Each item cites the receipt or commit that established it.

Standing rule for this port: anything NVFP4-specific goes here and into the
chronological engineering report, both.

---

## 1. Nibble ordering is part of the ABI, and a header lied about it

Byte j holds value 2j in the LOW nibble and 2j+1 in the HIGH nibble - adjacent
pairs. The Qwen header `qf_nvfp4_wave64.h` states an INTERLEAVED order; the
`.hip` body documents that as an abandoned bug. Measured signature of getting it
wrong: correlation 0.1420 instead of 0.9955.

Establish packed layout from the real weights and the writer, never from a
header. In this estate the `.h` files are systematically stale and only `.hip`
bodies are authoritative - proven twice, the other instance being the router's
renormalisation.

*receipts/04_W8A8_CONTRACT.md section 6; receipts/00_PORT_MAP.md*

## 2. NVFP4 does not need float decode. This was worth 21.6x

E2M1 magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6}. DOUBLED they are exactly the
integers {0, 1, 2, 3, 4, 6, 8, 12}, so a 16-value group decodes losslessly into
int8 lanes and feeds `v_dot4_i32_i8`. The factor of two folds into the group
scale. Weight arithmetic stays exact - same values, integer products.

    float nibble decode, 16 FMAs per 8 bytes    19.0 GB/s
    two v_perm_b32 lookups + bitselect + dot4  410.7 GB/s

The representation was never slow. The float-decode implementation was.

**Corollary:** any target with an 8-bit integer dot product can execute the
4-bit representation exactly. The values do NOT fit a signed 4-bit dot product,
since 12 and 8 exceed 7, so `v_dot8_i32_i4` cannot consume E2M1 directly.

*commit f2c4c65; receipts/04_W8A8_CONTRACT.md section 6*

## 3. Activation scaling must match the weight artifact, not the vendor contract

Huawei's W8A8 contract uses ONE symmetric int8 activation scale per token. That
is correct FOR HUAWEI'S WEIGHTS, because jointfix smooths per-channel outliers
out of the activation and into the weights. Our arena is NVFP4 packed from the
UNSMOOTHED bf16, so the outliers are still in the activation.

Measured on the real captured layer-0 activation, 49.3x outlier:

    one scale per token   rel_err 1.09e-01,  1951 of 2560 values quantise to ZERO
    per 64                          1.67e-02,  102 lost
    per 32                          1.27e-02,   57 lost
    per 16                          9.77e-03,   36 lost

Per-16 adopted: it is the granularity the NVFP4 weights already use, and it
costs 2.3 percent. Transplanting a vendor's activation rule across a different
weight artifact destroyed most of the activation.

*commit 2fc2631*

## 4. Never drop an activation scale around a nonlinear product

The production gate/up omitted `xs[g]` on BOTH gate and up. Because the
intermediate is `silu(gate) * up`, the two omissions MULTIPLY: the error is
xs squared, about 1e3.

    gate/up + SiLU rms   618.11 -> 0.0933
    requant scales       50.9 .. 8475 -> 0.0042 .. 1.089
    MoE output rms       27088 -> 3.567

The model emitted real subwords in no order. Every kernel had passed its own
gate; only generated text exposed it. A residual-stream trace localised it in
one run: both dense layers sane at rms 0.84 and 5.7, every MoE layer 1,000 to
27,000, discontinuity exactly at layer 2.

The proximate cause is worth recording too: an earlier source patch silently
failed to match the actual formatting, so it removed the epilogue's `xs[0]` but
never added `xs[g]`. **Verify a patch landed in the compiled source; do not
trust that a textual substitution matched.**

*commit 57a124e*

## 5. `weight_scale_2` is real model state, not a formality

The expert kernels took a scalar `weight_scale_2` where the arena stores a real
per-slot value near 1e-4. Passing `1.f` puts every expert output four orders of
magnitude wrong. All three projections now index `arena.s2[slot*3 + {0,1,2}]`
for gate, up and down.

*commit 57a124e*

## 6. A packed-layout permutation is a consumer-specific contract

The even-then-odd permutation within each 16-value group exists because masking
a packed dword yields the EVEN values and shifting yields the ODD. That is a
property of the NVFP4 nibble layout. It is NOT a property of requantisation.

`k_p92_i8_requant` was shared by the NVFP4 and INT8 down paths. Moving the
permutation into it silently changed the representation the INT8 path consumes -
int8 activations have no nibble split, so permuting them is simply wrong. The
correct repair is a separate NVFP4-specific producer, `k_p92_f4_requant`, with
the generic int8 contract left byte-for-byte unchanged.

**Do not mutate a shared producer without auditing every consumer.** The two
formats may share arithmetic machinery - both end at `dot4` - while their
packing, scaling, permutation and producer/consumer ABI stay format-specific.

*commit e3f30b5 onward*

## 7. Moving layout work upstream can delete enormous redundant staging

Consumer-side permutation made every one of the down kernel's 320 blocks stage
all nine experts' hidden vectors: 9216 B of LDS a block and about 2.95 MB of
redundant traffic a call. Producing the permuted order once in the requant
removed all of it.

    LDS        9300 B/block -> 0
    VGPRs      41 -> 30
    occupancy  5 -> 8 waves/SIMD

## 8. Read the resource report next to the bandwidth number

The down kernel read HALF the bytes of gate/up and ran at 259 GB/s against 481.
That anomaly pointed at the resource report, which showed the LDS, which pointed
at the layout contract. GB/s alone would not have found it. Register and LDS
pressure have been the true cause of three separate results in this port:

    MLA          16-lane groups, ~120 VGPRs, 2 waves/SIMD -> 32-lane, ~48, 512 -> 314 us
    gate/up      full unroll 63 VGPRs 4 waves -> unroll 1, 37 VGPRs, 6 waves
    expert down  9300 B LDS, 5 waves -> 0 B, 8 waves

## 9. An observed GB/s is not a roof

410.7 GB/s was recorded as the NVFP4 rate from a standalone benchmark. In the
composed engine gate/up subsequently measured 481 GB/s. Standalone microbenchmark
rates are conditions, not ceilings; the assembled engine changes occupancy,
cache residency and neighbouring pressure.

## 10. Storage representation is not execution representation

The single most transferable idea here. NVFP4 is how the weights are STORED:
E2M1 nibbles, UE4M3 per-16 block scales, one fp32 per-tensor scale. It says
nothing about how they must be EXECUTED. On gfx906 the execution representation
is int8 lanes through `v_dot4_i32_i8`, reached by a lossless integer transform
of the stored one.

That separation is what made the format viable on hardware that has no FP4 unit
at all, and it is why the two representations now cost the same time per expert:

    int8 routed expert, per-row scales    731 GB/s   7.51 MiB an expert
    NVFP4 routed expert on dot4           481 GB/s   4.22 MiB an expert

NVFP4 reads 56 percent of the bytes at about 1.5x the cost per byte, so per
expert they tie - which is what let the whole model stay resident on 4x16 GB in
one representation, with the hot/cold split becoming a quality decision rather
than a speed one.

## 11. Per-block activation staging is a skinny-projection trap, not a one-off

Second instance of the same mechanism, and now general enough to state as a
rule: **when every output block consumes the SAME activation vector, staging it
independently per block can cost almost as much traffic as the weights and
crush occupancy at the same time.**

    downacc   320 blocks x 9216 B  = 2.95 MB staged, against 13.3 MB of weights
    o_proj    640 blocks x 6144 B  = 3.93 MB staged, against  8.85 MB of weights

Both were fixed the same way: produce the consumed order once, upstream, and
let every consumer read it from global where it stays hot in L2.

    o_proj    LDS 6144 -> 0 B    occupancy 4 waves/SIMD
    gate/up   VGPRs 37 -> 35     occupancy 6 -> 7 waves/SIMD
    whole engine 56.31 -> 59.50 tok/s

The signature to look for: a projection whose weight traffic is modest, whose
GB/s is well under a sibling kernel's, and whose resource report shows LDS
proportional to the activation width.

## 11b. LDS is a cost, not a free optimisation, on skinny decode kernels

Three instances in this engine now, all of them cases where staging in LDS made
things WORSE:

    downacc                  9300 B/block   occupancy 5 -> 8 when removed
    o_proj                   6144 B/block   occupancy 4, LDS -> 0 when removed
    rmsnorm+quant fusion    10240 B/block   whole engine flat at 60.57 with it,
                                            61.6 without

The third is the clearest, because the same mathematical fusion was measured
both ways: with 10 KB of LDS to stage the normalised vector it saved 218 us of
kernel time and delivered nothing end to end; with the LDS removed and the
vector re-read from L1 it delivered 60.59 -> 61.6.

These kernels are small enough that their data is already hot in L1, and the
occupancy LDS costs matters more than the reuse it buys. On gfx906 skinny decode
kernels, staging in LDS should require evidence rather than being the default
instinct.

## 12. Representation and scaling are ABI. Gate the producer/consumer EDGE

The single most important methodological result in this port. Component
correctness does not imply composition correctness when representation is part
of the interface, and a microtest that constructs its own input silently
bypasses the producer contract it exists to validate.

**Two live defects were found this way, both invisible to twelve passing gates.**

*Case A - order.* Making `k_p92_quant` emit the even-then-odd order is correct
for `k_p92_proj` and `k_p92_f4_gateup`, which were changed with it.
`k_p92_dense_gateup` also consumes that activation and still staged it,
permuting an already-permuted vector. Layers 0 and 1 were corrupted and the
engine emitted degenerate text - repeated characters, then markup fragments -
while ALL TWELVE GATES PASSED.

*Case B - scaling.* When per-16 activation scaling was adopted, the projections
and the experts were converted and the dense path was missed:
`k_p92_dense_gateup` kept applying a single `xs[0]`. Layers 0 and 1 ran with the
wrong activation scaling for several commits. **Generation stayed coherent the
whole time** - two of 46 layers are dense and the residual stream renormalises -
so neither the gates nor reading the output caught it. The contract gate found
it on its first run:

    quant<2560> -> dense_gateup    cosine 0.76857078  ->  1.00000000

Case B is the more instructive one. "The output looks fine" is not a correctness
criterion. A layer can be badly wrong and still be washed out by normalisation.

**The gate that catches this class** (`tests/p92_contract_test.hip`) invokes the
REAL producers and feeds their actual output into the REAL consumers. Its
reference is built by reading the producer's output back, undoing the documented
order, and reconstructing the activation - so a consumer that misreads the order
or the scaling cannot agree with it. Activations carry deliberate outliers,
since that is what per-group scaling exists for.

**Two rules follow.**

- When changing what a producer emits - order, scaling granularity, precision,
  anything - enumerate EVERY consumer mechanically. `grep p92_stage_perm` found
  four sites, of which one was wrong and three were correct for reasons worth
  checking one at a time: two consume natural-order requant output, one consumes
  a host-quantised vector. A global search-and-replace would have broken all
  three.
- A green gate suite is not evidence the engine works, and coherent output is not
  evidence it is correct. Gate the edges.

## 13. Isolate a gate to the mechanism it names

The dense down gate failed at 9.3e-04 against a 1e-5 threshold. Not a defect:
gate/up accumulates in fp32 where the scalar reference uses double, so one value
of 9216 sat on an int8 quantisation boundary and landed on the other side. That
single flipped byte moved the down result.

Building the down reference from the DEVICE's own requant output rather than the
host's isolates the arithmetic actually under test: 9.3e-04 -> 4.077e-08. The
byte-difference count is now reported rather than gated.

A gate that spans two mechanisms measures the looser one. The same reasoning put
the composed-DSA gate on the device's own selection list.

## 14. Measurement variance is itself a signal

Run-to-run spread on this engine fell from about 3 percent to about 0.1 percent
over the optimisation ladder, while serialised single-block work was being taken
off the critical path. The causal link is NOT established - the variance
components were never decomposed - but it is recorded because it changes method:
at 3 percent a 150 us improvement is indistinguishable from noise, and at 0.1
percent it is obvious. Early rungs needed median-of-three; later ones do not.
