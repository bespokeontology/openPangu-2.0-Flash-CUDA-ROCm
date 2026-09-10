# openPangu on 4x MI50 — state

Every number here is measured on the box, one MI50, at the stated shape. No
estimates.

## Built and gated

| # | mechanism | gate | result |
|---|---|---|---|
| 1 | NVFP4 expert GEMV | vs CPU reference | accepted earlier |
| 2 | layer ownership 12/12/11/11 | all 46 owned once | pass |
| 3 | expert path selection | measurement | int8 vs nvfp4 |
| 4 | W8A8 routed experts | relative_l2 1.466e-07, quant byte exact | pass |
| 5 | NVFP4 on dot4 | decode exact 256/256, relative_l2 6.359e-08 | pass, 21.6x |
| 6 | resident arena | 12/12 slots byte identical to artifact | pass |
| 7 | router | 0/8 ids differ, weight_sum 2.500000, tie rule | pass |
| 8 | absorbed MLA decode | relative_l2 8.427e-07 | pass, 6.45x |
| 9 | DSA indexer | set diff 0 at 2K-128K, deterministic | pass |

## Measured per-token cost, one card, decode

| component | shape | us/layer | layers | ms/token |
|---|---|---:|---:|---:|
| routed FFN, H=64 hot int8 + 192 nvfp4 | top-8 | 115.4 | 44 | 5.08 |
| MLA decode, DSA layers | 2048 selected + 128 sinks | 492.6 | 16 | 7.88 |
| MLA decode, sliding layers | 512 window + 128 sinks | 145 | 30 | 4.35 |
| DSA indexer, scorer + select | 8192 positions | 89.8 | 16 | 1.44 |

Those four total 18.8 ms/token at 8K context. The four cards run sequentially, so summing
per-layer costs across all 46 layers is the right model, not dividing by four.

## Component rates

| kernel | GB/s |
|---|---:|
| int8 routed expert | 731.0 |
| nvfp4 routed expert on dot4 | 410.7 |
| nvfp4 routed expert, old fp32 decode | 19.0 |
| MLA decode, 16-lane groups | 244.3 |

## Residency

Arena packed and on disk: 47 GiB across four device files, all 44 MoE trunk
layers, 256 experts each, NVFP4. Loads in 18.75 s for device 0. Non-expert
resident weight is 4.73 GB. An all-NVFP4 arena is 49.83 GB against 68.65 GB
physical, so it fits with room; H hot int8 experts per layer cost an extra
3.29 MiB each and the budget allows H = 64 at 77.4 percent routing coverage.

Since the dot4 rewrite the two representations cost the same time per expert,
so H is a quality choice, not a speed one. See 04_W8A8_CONTRACT.md section 2a:
Huawei's int8 is jointfix-calibrated and better than absmax by construction,
but using it needs 21 GB of their shards plus their pre_mlp_layernorm gammas.
The all-NVFP4 arena is the working default and needs no download.

## Indexer cost against context

The component that was the cost risk, now measured. Scorer plus the better of
the two selectors, one MI50, and the resulting cost over the 16 DSA layers.

| positions | scorer us | select us | ms/token |
|---:|---:|---:|---:|
| 2,048 | 47.3 | 25.9 | 1.171 |
| 8,192 | 49.1 | 40.8 | 1.437 |
| 32,768 | 75.7 | 71.7 | 2.358 |
| 131,072 | 172.3 | 79.6 | 4.031 |

Selection is close to flat because the staged selector is parallel in the
positions; the growth is the scorer, which is a genuine full scan. At 128K
native context the indexer costs 4.03 ms/token, which is bounded and modest
against the 18.8 ms the built components already cost at 8K.

## Not built

- mHC modules. Near-exact GLM donor, mhc_step_gfx906.hip.
- MoME depthwise convolutions. Four-line delta on the Qwen GDN conv.
- Dense layers 0 and 1, MLP intermediate 9216.
- Shared expert. Same shapes as a routed expert, so the existing kernels serve
  it; it needs an arena slot, not a kernel.
- lm_head, 151552 x 2560.
- Multi-GPU orchestration and the pinned-host staging transport. Measured cost
  is 16.25 us for the 20,480 B mHC activation, about 53 us per token for three
  crossings; P2P is dead on this platform and that is proven, not assumed.
- N3, T-row MLA prefill. No donor on either side. Not on the correctness path.

## Open

The combine order for the routed sum. The oracle has two semantics: the
resident path accumulates fp32 in router-selection order with one bf16 round at
the end, the fallback keeps a bf16 accumulator and rounds after each add. The
down kernel here accumulates fp32 in selection order, matching the resident
path, which is what the production engine runs. Flipping it is a five-line
change in the kernel tail if the operator rules the other way.


> **CORRECTION 2026-09-09.** Any statement in this document that MI50
> P2P is unavailable/dead at the driver is FALSE. It was true of Linux
> 5.15.0-190 only. Under Linux 6.8.0-138 the same hostile gate moves from
> 0/12 to 12/12 directed pairs with no collateral, because KFD creates
> `p2p_links` under 6.8 and created none under 5.15. Same cards, same ROCm
> 5.7.1 userspace, no ACS override. See receipts/07_P2P_DIAGNOSIS.md.
> The failure MODE described - peer pointers silently aliasing local memory
> instead of faulting - was real and is why a P2P path must be gated by a
> data-integrity check, never a capability bit.
