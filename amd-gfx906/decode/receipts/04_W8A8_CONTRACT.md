# openPangu-2.0-Flash quantisation contract

Recovered, not invented. Sources: the shipped `openpangu/openPangu-2.0-Flash-Int8`
checkpoint, and the Windows CUDA reference `pangu_under8_patch`
(`pangu_w8a8.cu`, `pangu_numerical.c`) preserved in the Dragon Mac unified copy.
The Windows tree is a math and contract reference only; it ran on a laptop GPU
and none of its timings mean anything here.

This file is deliberately hardware-neutral. It states what the model requires.
Kernel geometry belongs in the per-target receipts.

## 1. What Huawei actually quantises

Census of the shipped Int8 checkpoint: 73,790 tensors, 50 shards, 97.18 GiB.

| category | GiB | of which int8 |
|---|---:|---:|
| routed experts | 88.23 | 88.12 |
| embed / lm_head | 3.61 | 0 |
| norms, mHC, sinks | 2.38 | 0.17 |
| attention | 2.21 | 1.15 |
| shared expert | 0.69 | 0 |
| router | 0.06 | 0 |

Routed experts are int8. The shared expert, the router and the embedding are
bf16. Attention is partly int8. Trunk layers are 0-45; layers 46-48 are the MTP
predictor and are not resident for trunk decode.

## 2. Weight representation

    weight        int8, row-major [outputs, inputs]
    weight_scale  bf16, shape [outputs, 1] - one scale per output row
    zero point    none. The representation is symmetric.

The older export carried offset and bias tensors alongside. The Windows
reference reads them and treats any non-zero offset as a hard failure, which is
how we know the export is symmetric in practice. The current shipped Int8
checkpoint drops the offset tensor entirely and carries weight and weight_scale
only.

Per-expert byte cost, gate + up + down with scales: 7.51 MiB.

## 2a. The equalisation, and the trap

Huawei's int8 weights are NOT a direct quantisation of the published bf16
weights. They carry a per-input-channel equalisation, SmoothQuant in form:

    W_int8_dequant  ~=  W_bf16 . diag(s)
    gamma_int8      ~=  gamma_bf16 / s          folded into pre_mlp_layernorm

Measured on model.layers.5.mlp.experts.3.gate_proj, against the published bf16
weights and norms:

| hypothesis | residual rel_l2 |
|---|---:|
| remove a per-input-channel vector | 9.75e-03 |
| remove a per-output-row vector | 1.49e-01 |
| no transform at all | 8.50e-01 |

and s taken from the weights times gamma_int8/gamma_bf16 comes to a median of
0.99993, which closes the loop. post_attention_layernorm is untouched, ratio
exactly 1.0. The channel factors span 0 to 12.9 with a median near 6.35, and
the gamma ratio spans 95x, so the fold is large and not a rounding effect.

**The trap:** feeding a raw activation to Huawei's int8 weights, as one would
with any ordinary symmetric W8A8 checkpoint, is silently wrong. Measured on one
expert with a random activation: rel_l2 5.65 and cosine 0.988, which is
plausible enough to pass a careless cosine gate while being off by a factor of
five in magnitude. The activation must be divided by s first, which is what
using the int8 checkpoint's own pre_mlp_layernorm does.

**Why this matters for a mixed arena.** Hot experts held as Huawei int8 want
x/s; cold experts held as NVFP4 packed from the published bf16 want x. Both are
served from one normalisation: normalise once with gamma_bf16 to get x, then
multiply by gamma_int8/gamma_bf16 to get x/s. That vector is 2560 wide and the
extra cost is one multiply and one activation quantisation per layer. No
repacking of either arena is required.

**Settled, from Huawei's own tool rather than a measurement.** The method is
`jointfix`, shipped in openPangu-2.0-Infer under tools/quant/jointfix. Its
Pangu-specific file names the absorption topology directly - q_a_layernorm,
kv_b_proj head-interleaved V rows, the unquantised indexer.wq_b,
pre_mlp_layernorm, the router, and per-expert down_proj - which is the same
pre_mlp_layernorm the measurement above found. The smoothing scale is

    s_c = x_stat_c ^ a / w_stat_c ^ b

per input channel, with (a, b) grid-searched against a BF16 output
reconstruction objective on calibration data; a = b = 0.5 would be plain
SmoothQuant, and a = b = 0 would be no smoothing. Routed experts are searched
individually and the median (a, b) is applied across them. The weights are then
quantised with GPTQ against the smoothed inputs, not round-to-nearest.

So Huawei's arena is better than a plain absmax quantisation of the published
bf16 by construction: the smoothing factors were chosen to minimise output
error on real activations, and the weight quantisation is Hessian-aware. The
earlier comparison on a random Gaussian, where absmax won, was measuring the
one case smoothing is designed not to help. It should not be used to choose.

Using it costs 21 GB for the hot experts at H=64 plus the int8 checkpoint's own
pre_mlp_layernorm gammas. The all-NVFP4 arena is the working default until then
and needs no download.

## 3. Activation representation

One symmetric scale per token, over the whole vector:

    x_scale = max|x| / 127
    x_q[i]  = clamp(round(x[i] / x_scale), -128, 127)

Not per-channel, not per-block. This is the property worth exploiting: with a
single scalar the accumulator stays integer for an entire row instead of
converting once per scale block.

## 4. Arithmetic

    acc[o] = sum_i  w_q[o][i] * x_q[i]            int32
    y[o]   = (float)acc[o] * x_scale * w_scale[o]

int32 suffices at both openPangu extents: 2560 * 127 * 127 = 41.3M and
1024 * 127 * 127 = 16.5M, against an int32 bound of 2.1G.

## 5. Requantisation boundary

The expert FFN requantises once, between the SwiGLU and down_proj:

    h[i]     = silu(gate[i]) * up[i]              fp32
    h_scale  = max|h| / 127
    h_q[i]   = clamp(round(h[i] / h_scale), -128, 127)
    y[o]     = (float)sum_i(w_q[o][i] * h_q[i]) * h_scale * w_scale[o]

The router weight multiplies the dequantised down output, outside the integer
domain.

## 6. NVFP4 as the second resident representation

Where int8 does not fit, the same expert is held as NVFP4: E2M1 nibbles with a
UE4M3 scale per 16 values and one fp32 weight_scale_2 per tensor. Byte j holds
value 2j in the LOW nibble and 2j+1 in the HIGH nibble, adjacent pairs. Reading
these as interleaved is a real defect with a measured signature: correlation
0.1420 instead of 0.9955.

Per-expert byte cost: 4.22 MiB, 56 percent of int8.

E2M1 magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6. Doubled they are exactly the
integers 0, 1, 2, 3, 4, 6, 8, 12. An NVFP4 group therefore decodes losslessly
into int8 lanes and can use the same integer dot primitive as the int8 path,
with the factor of two folded into the group scale. This is the most important
portability fact in this file: any target with an 8-bit integer dot product can
execute the 4-bit representation exactly, without float decode. It is what took
the gfx906 cold path from 19.0 to 410.7 GB/s.

The values do not fit a signed 4-bit dot product, since 12 and 8 exceed 7, so a
4-bit integer primitive cannot consume E2M1 directly.

## 7. Residency arithmetic

Trunk only, 44 MoE layers x 256 experts.

    all int8    88.69 GB
    all nvfp4   49.83 GB
    non-expert resident (attention, embed, norms, shared, router)  4.73 GB

With H experts per layer as int8 and the rest as NVFP4:

    bytes = 44 * (H * 7.51 MiB + (256 - H) * 4.22 MiB)

Choose H as the largest value the target budget allows. Because the two
representations cost the same time per expert once both use the integer dot
path, H is a quality decision rather than a speed one: int8 is Huawei's own
calibration and is preferred wherever it fits.

## 8. Routing

Measured over 167,508 rows, 1,340,064 activations, 44 layers.

| top-N int8 | coverage |
|---:|---:|
| 64 | 77.4% |
| 96 | 88.6% |
| 128 | 94.4% |
| 192 | 99.0% |

Distinct experts firing per layer across the capture: minimum 209, median 235,
maximum 255 of 256. Every expert must be resident in some representation.
Paging or streaming on demand is not viable.

## 9. Gating

Do not gate on bit-exactness against any reference implementation. The contract
fixes the values and the integer products; it does not fix summation order, and
insisting on a serial order costs the parallelism the port exists for. Gate on
relative error against a scalar reference of the contract itself.

Observed on gfx906: activation quantisation byte exact; expert FFN relative_l2
1.466e-07; NVFP4 decode exact on all 256 byte patterns; NVFP4 gate/up
relative_l2 6.359e-08.
