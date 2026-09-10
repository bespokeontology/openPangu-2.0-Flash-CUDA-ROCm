# Mechanism 3 - expert execution path selected: resident INT8/NVFP4 mix

Date 2026-09-08. 4x MI50 gfx906, ROCm 5.7.1. Real packed weights from the
P92FP41 artifact, layer 2 expert 0.

## Device gate on real weights

The mechanism-1 NVFP4 kernels were run against actual artifact bytes and checked
against a CPU dequantise-and-dot reference over the same bytes.

| kernel | worst relative error |
|---|---|
| gate/up 1024x2560 | 1.109e-06 |
| down 2560x1024 | 1.867e-06 |

CORRECT. The kernels read the real artifact, including its scale layout and
per-tensor weight_scale_2 (3.48772e-05 gate, 3.56038e-05 down).

## Measured rates, same shapes, same card

| path | per expert (gate+up+down) | bytes | rate | x44 MoE layers, top-8 |
|---|---|---|---|---|
| NVFP4 | 248.79 us | 4.42 MB | 17.8 GB/s | 87.6 ms/token, 11.4 tok/s |
| INT8 dot4 | 29.57 us | 7.86 MB | 265.9 GB/s | 10.4 ms/token, 96.1 tok/s |

INT8 is 8.4x faster despite reading 1.78x more bytes. NVFP4 on gfx906 is
ALU-bound, not bandwidth-bound: there is no FP4 datapath, so each nibble costs
integer bit construction into f32, roughly 12 VALU ops per weight, while
v_dot4_i32_i8 does four weights per op. The measured 17.8 GB/s matches the
estate's recorded NVFP4 rate for the Qwen shapes (24.6 MB in 1.392 ms =
17.7 GB/s), so this is the gfx906 NVFP4 ceiling and not a defect in the
openPangu specialisation.

## Residency arithmetic

Physical HBM 4 x 17,163,091,968 B = 63.94 GiB. Non-expert state at BF16 is
about 2.6 GB, leaving roughly 66 GB for experts. There are 256 x 44 = 11,264
routed experts.

| all-INT8 | all-NVFP4 |
|---|---|
| 11,264 x 7.86 MB = 88.6 GB | 11,264 x 4.42 MB = 49.8 GB |
| does not fit, short by 22.6 GB | fits, but 11.4 tok/s |

Neither pure representation is acceptable. Solving for a resident mix:

    7.86x + 4.42(11264 - x) <= 66,000 MB
    3.44x <= 16,213
    x <= 4,713 experts = 107 per layer

## Decision

Hold every expert resident. Store the 107 most frequently routed experts per
layer as INT8 and the remaining 149 as NVFP4. No host-backed expert movement:
a miss would cost 7.86 MB over PCIe at the measured 10.32 GB/s, about 786 us,
which is worse than simply executing the NVFP4 copy in place at 248.79 us.

Projected experts-only cost depends on the hit rate of the INT8 set, which is a
property of the routing distribution and is being captured from the CUDA engine
before the Spark is returned:

| INT8 hit rate | per layer | x44 | experts-only tok/s |
|---|---|---|---|
| 100% | 0.237 ms | 10.4 ms | 96.1 |
| 90% | 0.410 ms | 18.0 ms | 55.5 |
| 80% | 0.583 ms | 25.7 ms | 39.0 |
| 42% (no skew, proportional) | 1.253 ms | 55.1 ms | 18.1 |

The 42% row is the null hypothesis: it assumes routing is uniform, so the
resident INT8 fraction equals the hit rate. Any real skew moves the result
toward the top of the table. The Qwen histogram on comparable hardware showed
top-32 of 512 covering 51% of activations, roughly eight times uniform, but
openPangu's distribution is its own and is being measured rather than assumed.

## Not yet done

The NVFP4-to-INT8 converter must be built (the Qwen one caps K at 2560 via a
float vals[10] with a 256 stride and silently drops elements above that, so it
cannot be reused unmodified for the 9216 and 6144 dense shapes). Expert
placement needs the routing histogram. Both are the next mechanisms.
