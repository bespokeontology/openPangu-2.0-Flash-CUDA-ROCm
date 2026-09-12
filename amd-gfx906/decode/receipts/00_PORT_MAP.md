All numbers verified independently. One material correction to the inventories: GLM's `mhc_step_gfx906` is a near-exact donor for openPangu's mHC (same 24-mix 4/4/16 split, same σ/2σ/softmax, same 20-iteration Sinkhorn, same 1e-5/1e-6 epsilons) — that moves mHC off the new-kernel schedule.

# openPangu-2.0-Flash → 4× MI50 (gfx906) PORT MAP

Every number below was recomputed independently from the checkpoint manifest and the CUDA oracle. Sources are cited inline. `[est]` marks estimates.

---

## 1. BYTE TABLE

Trunk only (46 layers). MTP layers 46–48 excluded. INT8 and 4-bit columns follow the **artifact's own quantization rule** — verified by censusing `<data>/artifact/manifest.bin` (magic `P92FP41`, 36,528 records): quantized families are routed/shared/dense MLP, `q_a_proj`, `q_b_proj`, `o_proj`, `kv_b_proj`, indexer `wq_b`/`wk`, `mlp.gate` (47 records), `lm_head`. Everything else stays BF16 — notably `kv_a_proj_with_mqa`, which has **no manifest record**.

4-bit = 9/16 B/element (0.5 B packed + 1 B UE4M3 per 16). INT8 = 1 B + fp32 per-row scale.

| tensor family | /lyr | lyr | elements | BF16 MB | INT8 MB | 4bit MB | touch el/tok | tB16 | tI8 | t4b |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| routed experts gate/up/down | 768 | 44 | 88,583,700,480 | 177167.4 | 88583.7 | 49828.3 | 2,768,240,640 | 5536.48 | 2768.24 | 1557.14 |
| shared expert gate/up/down | 3 | 44 | 346,030,080 | 692.1 | 346.0 | 194.6 | 346,030,080 | 692.06 | 346.03 | 194.64 |
| dense MLP gate/up/down | 3 | 2 | 141,557,760 | 283.1 | 141.6 | 79.6 | 141,557,760 | 283.12 | 141.56 | 79.63 |
| MLA q_a_proj [1024,2560] | 1 | 46 | 120,586,240 | 241.2 | 120.6 | 67.8 | 120,586,240 | 241.17 | 120.59 | 67.83 |
| MLA q_b_proj [9216,1024] | 1 | 46 | 434,110,464 | 868.2 | 434.1 | 244.2 | 434,110,464 | 868.22 | 434.11 | 244.19 |
| MLA kv_a_proj_with_mqa [576,2560] | 1 | 46 | 67,829,760 | 135.7 | 135.7 | 135.7 | 67,829,760 | 135.66 | 135.66 | 135.66 |
| MLA kv_b_proj [12288,512] | 1 | 46 | 289,406,976 | 578.8 | 289.4 | 162.8 | 289,406,976 | 578.81 | 289.41 | 162.79 |
| MLA o_proj [2560,6144] | 1 | 46 | 723,517,440 | 1447.0 | 723.5 | 407.0 | 723,517,440 | 1447.03 | 723.52 | 406.98 |
| DSA indexer wq_b [3072,1024] | 1 | 16 | 50,331,648 | 100.7 | 50.3 | 28.3 | 50,331,648 | 100.66 | 50.33 | 28.31 |
| DSA indexer wk [128,2560] | 1 | 16 | 5,242,880 | 10.5 | 5.2 | 2.9 | 5,242,880 | 10.49 | 5.24 | 2.95 |
| DSA indexer k_norm [128] | 1 | 16 | 2,048 | 0.0 | 0.0 | 0.0 | 2,048 | 0.00 | 0.00 | 0.00 |
| DSA indexer weights_proj [24,2560] | 1 | 16 | 983,040 | 2.0 | 2.0 | 2.0 | 983,040 | 1.97 | 1.97 | 1.97 |
| mHC phi [24,10240] (attn+mlp) | 2 | 46 | 22,609,920 | 45.2 | 45.2 | 45.2 | 22,609,920 | 45.22 | 45.22 | 45.22 |
| mHC norm_gamma [10240] | 2 | 46 | 942,080 | 1.9 | 1.9 | 1.9 | 942,080 | 1.88 | 1.88 | 1.88 |
| mHC branch_alpha[3]+beta[24] | 2 | 46 | 2,484 | 0.0 | 0.0 | 0.0 | 2,484 | 0.00 | 0.00 | 0.00 |
| MoME qa/compresskv/o conv (k=3) | 1 | 46 | 1,059,840 | 2.1 | 2.1 | 2.1 | 1,059,840 | 2.12 | 2.12 | 2.12 |
| RMSNorm in/postattn/premlp/postmlp | 4 | 46 | 471,040 | 0.9 | 0.9 | 0.9 | 471,040 | 0.94 | 0.94 | 0.94 |
| RMSNorm block_post [10240] | 1 | 9 | 92,160 | 0.2 | 0.2 | 0.2 | 92,160 | 0.18 | 0.18 | 0.18 |
| RMSNorm q_a[1024]+kv_a[512] | 1 | 46 | 70,656 | 0.1 | 0.1 | 0.1 | 70,656 | 0.14 | 0.14 | 0.14 |
| router gate [256,2560] | 1 | 44 | 28,835,840 | 57.7 | 28.8 | 16.2 | 28,835,840 | 57.67 | 28.84 | 16.22 |
| router e_score_corr_bias[256] **F32** | 1 | 44 | 11,264 | 0.0 | 0.0 | 0.0 | 11,264 | 0.05 | 0.05 | 0.05 |
| attn sinks k_pe[128,64]+ckv[128,512] | 1 | 46 | 3,391,488 | 6.8 | 6.8 | 6.8 | 3,391,488 | 6.78 | 6.78 | 6.78 |
| embed_tokens [151552,2560] | 1 | 1 | 387,973,120 | 775.9 | 775.9 | 775.9 | 2,560 | 0.01 | 0.01 | 0.01 |
| lm_head [151552,2560] | 1 | 1 | 387,973,120 | 775.9 | 388.0 | 218.2 | 387,973,120 | 775.95 | 387.97 | 218.23 |
| model.norm [2560] | 1 | 1 | 2,560 | 0.0 | 0.0 | 0.0 | 2,560 | 0.01 | 0.01 | 0.01 |
| merge_mhc phi[4,10240]+gamma+α/β | 1 | 1 | 51,205 | 0.1 | 0.1 | 0.1 | 51,205 | 0.10 | 0.10 | 0.10 |
| **TRUNK TOTAL** | | | **91,596,785,593** | **183193.6** | **92082.3** | **52221.1** | | **10786.73** | **5490.90** | **3173.97** |

**Closure check (this is what makes the table trustworthy).** Manifest trunk quantized params = 91,111,292,928. My unquantized trunk families sum to 485,492,665. Sum = **91,596,785,593** — exact. Manifest header `source_payload_bytes` = 200,272,648,050 = `model.safetensors.index.json` `total_size`, identical.

**Whole-model resident totals**

| point | bytes | GiB |
|---|--:|--:|
| BF16 (trunk) | 183,193,593,714 | 170.61 |
| INT8, artifact rule (~1.4% stays BF16) | 92,082,300,786 | **85.76** |
| INT8, pure | 91,596,819,385 | 85.31 |
| 4-bit, artifact rule | 52,221,110,130 | **48.64** |
| 4-bit, pure | 51,523,230,616 | 47.99 |
| experts-4bit / rest-BF16 | 55,854,524,274 | **52.02** |
| experts-4bit / rest-INT8 | 53,326,931,826 | 49.67 |
| experts-INT8 / rest-BF16 | 94,609,893,234 | 88.11 |

**Bytes touched per generated token** (static weights, context-independent): BF16 **10,786.73 MB**, INT8 **5,490.90 MB**, 4-bit **3,173.97 MB**, experts-4bit/rest-BF16 **6,807.39 MB**.

Context-dependent traffic is separate and is dominated by the DSA key scan over the 16 non-sliding layers:

| context | key scan | latent read | sliding | total | @894 GB/s |
|--:|--:|--:|--:|--:|--:|
| 2,048 | 8.4 MB | 37.7 | 17.7 | 63.8 MB | 0.071 ms |
| 8,192 | 33.6 | 37.7 | 17.7 | 89.0 | 0.100 |
| 32,768 | 134.2 | 37.7 | 17.7 | 189.7 | 0.212 |
| 131,072 | 536.9 | 37.7 | 17.7 | 592.3 | 0.663 |
| 524,288 | 2,147.5 | 37.7 | 17.7 | 2,202.9 | 2.464 |

At native 512K the key scan alone is **67.7 % of the entire 4-bit weight stream**. Storing DSA index keys at INT8 instead of BF16 halves it and is worth more than any expert kernel work at that context. Note the regime condition: `src/model.cpp:1641` runs dense attention with **no** key scan at all when `positions <= kDsaTopK` (2048).

`[est]`/uncertain in this table: `model.norm` shape [2560] is inferred, not read (its shard is not downloaded); byte closure forces it. `merge_mhc.branch_alpha_pre`/`branch_beta_pre` total exactly 5 BF16 elements from the closure residual; the split between them is unknown. The 894 GB/s figure is from the task statement, not measured here — every millisecond scales linearly if it is wrong.

---

## 2. RESIDENCY VERDICT

Measured HBM, from `/sys/class/kfd/kfd/topology/nodes/{1..4}/mem_banks/0/properties`: **17,163,091,968 B per card**, aggregate **68,652,367,872 B = 63.94 GiB** — not a round 64. The engine's own budget guard defaults to 15,872 MiB/card (`qf_hip4_stage.hip:804`), i.e. **62.00 GiB usable**.

| point | GiB | vs 63.94 physical | verdict |
|---|--:|--:|---|
| BF16 | 170.61 | −106.67 | no (2.67×) |
| **INT8** | **85.76** | **−21.82** | **no** |
| 4-bit | 48.64 | +15.30 | fits |
| experts-4bit / rest-BF16 | 52.02 | +11.92 | fits |
| experts-4bit / rest-INT8 | 49.67 | +14.27 | fits |

**Plainly: openPangu does not fit at INT8 and does fit at 4-bit.** INT8 misses by 21.8 GiB — a factor of 1.38, not a rounding problem. There is no INT8 arrangement of this model that fits 64 GB: the routed experts alone are 88,583,700,480 params, and at 1 B/param that is 82.5 GiB before anything else.

**Recommended point: experts-4bit / rest-BF16 = 52.02 GiB, 11.92 GiB headroom.** That headroom covers 32K context (0.76 GB), 128K (2.97 GB) and — under the worked shard in §5 — 512K native (11.83 GB attention state) with card 1 at 15.73 of 15.98 GiB. Note how little residency is bought by degrading the non-expert path: 52.02 → 48.64 GiB is 3.4 GiB for a full precision step on *every projection in the model*. Do not spend quality there.

**What 4-bit costs in kernel work — this is the central finding.**

gfx906 has no FP4 datapath. The estate's own measured receipt (`qf8_moe_i8.h:4-8`): the NVFP4 read path spends ~12 VALU ops per weight and measured **777 µs (gate/up) + 615 µs (down) per layer for 24.6 MB of packed weights** — ALU-bound at ~20 GB/s against a card measured at ~894. That is 17.66 MB/ms.

openPangu touches 9 experts (8 routed + 1 shared, verified same shape from the manifest: `shared_experts.gate_proj [1024,2560]`) × 3 matrices × 1,310,720 packed bytes = **35.39 MB per MoE layer**. At 17.66 MB/ms that is **2.005 ms/layer × 44 = 88.2 ms/token from routed experts alone ≈ 11.3 tok/s**, before attention, before the head.

So the two floors are:

| | per-token bytes | time | tok/s |
|---|--:|--:|--:|
| 4-bit **byte floor** (HBM roofline) | 3,173.97 MB | 3.55 ms | 282 |
| experts-4bit/rest-BF16 byte floor | 6,807.39 MB | 7.61 ms | 131 |
| 4-bit **ALU ceiling** (measured NVFP4 rate) | — | ~94 ms | ~10.6 |
| INT8 byte floor *(does not fit)* | 5,490.90 MB | 6.14 ms | 163 |

The layer-sharded plan in §5 runs the four cards **serially** for one token, so each card reads only its own layers and the total time is the sum — the 3.55 ms figure is the whole box, not one card.

**The gap between 3.55 ms and 94 ms is the port.** The resolution the existing tree already supports is a **mixed-format resident set** — INT8 for a hot subset, 4-bit for the rest, both resident, no host streaming. `qf_expert_slots.cpp:120-149` already allocates NVFP4, FP8 and INT8 arenas side by side. Solving `7.88x + 4.42(2816−x) = available` per card (11 MoE layers = 2,816 experts, 300 MB scratch allowance):

| context | available for experts | INT8-resident | share |
|---|--:|--:|--:|
| 32K | 14.72 GiB | 968 of 2,816 | **34.4 %** |
| 128K | 14.21 GiB | 808 | 28.7 % |
| 512K | 12.14 GiB | 168 | 6.0 % |

Whether 34 % of experts absorbs most of the lookups depends entirely on openPangu's routing skew, which is **unmeasured**. The tree already has the census tool (`qf4_stage_hist_print`, `qf_hip4.h:141`).

Host streaming of cold experts is **not** the answer. A miss streams 4,423,680 B of NVFP4, so ~99 misses/token ≈ 438 MB/token H2D — comparable to the Qwen figure the code calls "what sets decode speed here." Worse, the i8-only miss path drains the device after every converted expert (`qf_expert_slots.cpp:323-327`, `hipStreamSynchronize(compute)` under the comment "init-time only"). Per-token streaming would take a full device sync per miss. That, not the byte count, is the blocker.

---

## 3. KERNEL REUSE MAP

| operation | existing gfx906 kernel | shape difference | verdict | risk |
|---|---|---|---|---|
| **router top-8** | GLM `su_k0_route`, `moe_su_gfx906.hip:353-457` — sigmoid, `+e_score_correction_bias`, top-8 on corrected score, renormalise, `×2.5` | GEMV K 4096→2560 (`k0 = lane*16 + i*1024` over 4 iters covers 4096 only; 2560 = 64 lanes × 40 B, re-tile); NEXP 288→256 (4 cand/lane not 5); **strip `ROUTER_RENORM_EPS 1e-20`** (`router_gfx906.h:20`) — openPangu has no epsilon (`nvfp4_ops.cu:429`) | **specialise** | low. `ROUTER_SCALING 2.5f` is literally the same constant. Also emits an ascending-expert-id slot table free (`:437-457`). **Do not** port Qwen's `qf_router_wave64.hip` — it does a full-512 softmax, renormalises, and masks winners with `-1.0f`, which is a real bug once the corrected score can go negative |
| **routed expert gate/up** | `k_i8_gateup_all`, `qf8_moe_i8.hip:84-100` | K=2560 **unchanged** — half-wave/row, `st*512` × 5 steps = 2560 B is exactly right. Only grid changes: `(QF5_NFF/8, nexp)` = (80,·) → (128,·) | **reuse as-is + one grid constant** | very low |
| **routed expert down** | `k_i8_downacc_all`, `qf8_moe_i8.hip:129-143` | row width 640→1024: `i8v4 v[5]` → `v[8]` (8 lanes × 8 chunks), or 16 lanes × 4 chunks which needs a 16-lane term (`0x201F`) added to `i8_g8_sum` | **specialise** | medium — `v[8]` = 32 VGPRs held across the expert loop; which variant holds occupancy is a measurement, not a reading |
| **routed expert, 4-bit path** | `k_nvfp4_gemv_640x2560` / `_2560x640`, `qf_nvfp4_wave64.hip:109-163` | gate/up 1024×2560 keeps 160 groups/row → half-wave/5-step body survives, grid 80→128. **down 2560×1024 has 64 groups/row, not 40** → the eighth-wave/5-step body is invalid; needs 8 steps or a 16-lane group × 4 | **specialise (down = new template)** | **HIGH** — this is the ALU-bound path of §2 |
| **shared expert** | same expert kernels — manifest confirms `[1024,2560]`×2 + `[2560,1024]`, identical to a routed expert | none. But `k_shexp_add` (`qf_hip4_stage.hip:174-177`) applies `sigmoidf_(g[0]) *` — openPangu is a **plain BF16 add** (`model.cpp:1728` → `add_inplace`) | **reuse + delete the gate** | low, but silent if missed |
| **dense MLP (layers 0–1)** | `k_d8_gemv<LPR,NST>`, `qf8_dense_i8.hip:79-104` | gate/up K=2560 ✓ covered. **down K=9216 is the single uncovered K in the whole model** — add `case 9216: k_d8_gemv<64,9>` (64·9·16=9216, RPB=4). GLM's `w8_family` *rejects* it: `K>=8192 && K%4096` → 9216%4096=1024 → −1 (`w8_gfx906.hip:426-427`) | **specialise (one switch line)** | low. Separately: the NVFP4→INT8 converter caps K at 2560 via `float vals[10]` with a 256-stride (`qf8_moe_i8.hip:169-171`) — at K=9216 and K=6144 it silently drops elements and never writes `wo[k]`. Must be fixed before any K>2560 tensor goes through it |
| **RMSNorm** | `k_rmsnorm<D>`, `glmflash-hip/.../dsa_decode_gfx906.hip:88-103` | D ∈ {2560, 1024, 512, 128, 10240}, all template args | **reuse as-is** | low — but Qwen's `k_d8_hc_norm_x8` hardcodes **eps 1e-6** (`qf8_dense_i8.hip:136`) where openPangu is **1e-5** (`model.cpp:57`). Same class of silent-semantics-change as the GLM MLP's `±10.0` clamps |
| **residual add** | trivial elementwise | none | **reuse as-is** | none |
| **mHC (per-layer, ×2)** | **GLM `mhc_step_gfx906`** — `MHC_MIX 24 = (2+H)·H`, split 4/4/16, `pre=σ+1e-6`, `post=2σ`, `comb=softmax_row+1e-6`, 20-iteration Sinkhorn, `MHC_RMS_EPS 1e-5`, `MHC_HC_EPS 1e-6`, `base F32[24]`, `scale F32[3]` | **Same algebra, verified line-by-line against `nvfp4_ops.cu:516-577`.** Deltas: D 4096→2560, FLAT 16384→10240; GLM uses *Unweighted*RMSNorm, openPangu multiplies by `norm_gamma[index]` (`nvfp4_ops.cu:551-553`); α/β are F32 in GLM, BF16 in openPangu | **specialise** | **low** — this is a near-exact donor. It contradicts the earlier assessment that "openPangu's mHC is a different algebra entirely"; it is not. The phi GEMV is [24]×K=10240, and **K=10240 is already a supported INT8 K** (`qf8_dense_i8.hip:341`) |
| **mHC merge (global)** | same kernel | phi [4,10240] not [24,10240]; no 4/4/16 split, no Sinkhorn | **specialise** | low |
| **MoME convs (qa/compresskv/o)** | `qf8_gdn.hip:47-62` (decode) + `qf8_gdn_chunk.hip:37-51` (rows) | width 4→3; ring depth 3→2; **tap-major → channel-major** (`state[ch*2]`); `silu` → **residual add** (`nvfp4_ops.cu:269-285`). Channels 1024 / 512 / 6144 — note compresskv is the latent half only, **not** the 64 rope dims | **specialise** | low — 4 line-level changes in a kernel that exists in both decode and rows form. Channel-major precedent: `kda_decode_gfx906.hip:310` |
| **MLA q_a [1024,2560]** | `k_d8_gemv<32,5>` | none | **reuse as-is** | low |
| **MLA q_b [9216,1024]** | GLM `k_w8_ks` (K%1024==0) or new `k_d8_gemv<32,2>`/`<64,1>` | Qwen switch lacks K=1024 | **specialise (one line)** | low |
| **MLA kv_a [576,2560]** | `k8_gemv_vec` (bf16) or `k_d8_gemv<32,5>` | **BF16 in the artifact** — no manifest record. Needs a bf16 read path from day one, or quantize it locally | **reuse as-is** | low. Worth noting: at 135.66 MB/token it is 4.3 % of the 4-bit stream purely because the artifact left it BF16; quantizing it saves 97.4 MB/token |
| **MLA kv_b [12288,512]** | GLM `k_w8_grp<32,8>` (K=512) | Load-time split into k-half (first 128 rows/head) and v-half (next 128), confirmed by `conversion/pangu.py:211-217` and `model_ops.cu:74`. GLM binds a pre-split `Wv_b`; openPangu keeps one interleaved tensor | **specialise (+ load-time split)** | medium — this is the operand MLA absorbs the query into, the most numerically sensitive site in the layer. The artifact has 49 quantized `kv_b` records but the oracle loads BF16 and **never exercises them** |
| **MLA o_proj [2560,6144]** | `k_d8_gemv<32,12>`, `qf8_dense_i8.hip:339` | **none — K=6144 is already supported** | **reuse as-is** | low |
| **DSA index scoring** | Qwen `k8_idx_scores` (`qf8_qsa.hip:196-212`) / GLM `dsa_decode_gfx906.hip:176` | Qwen: 4 pooled-block heads, no per-head weight, `1/√128` inside relu. openPangu: **24 heads, per-position (no pooling at all), per-head weight, both scales = 1**. GLM's formula is the right shape (weights outside, relu inside) at 32 heads | **NEW** (see §4) | **HIGH** |
| **DSA top-2048 selection** | GLM `dsa_select_radix`, `dsa_select_gfx906.h` — 4× 8-bit radix passes over order-preserving flipped fp32 keys | K = 512 **pools** → 2048 **positions**; P ≤ 65536 pools → 524,288 positions; delete the pool→token flatten (`dsa_decode_gfx906.hip:203-211`) and the tail-block append | **specialise** | medium. The oracle uses a *full* CUB sort (`dsa_ops.cu:174-184`) where a selection was wanted — the AMD radix-select is strictly better and already carries a deterministic tie rule |
| **attention sinks** | **none — zero hits across both AMD trees** | 128 prepended keys, every layer, both paths, own rope half, RMSNorm'd once at load | **NEW** (see §4) | **HIGH** |
| **sliding-window attention** | **none — no windowed/circular KV anywhere** | 512-slot circular cache on 30 of 46 layers (`index%3 != 0`) | **NEW** (see §4) | medium |
| **indexed attention** | GLM `k_attend_part_m` / `k_attend_comb_m`, `dsa_decode_gfx906.hip:369-479` — split-flash over an E4M3 latent cache, gather-by-index. **`DSA_KVLORA` is already 512**, and 64 lanes × 8 dims covers it exactly | heads 64→48; scale `0.0625` → `1/√192`; `DSA_SEL_MAX` 2051 → 2176 (2048 + 128 sinks); **plus the new rope term and second cache** | **specialise-heavy** | **HIGH** — but the hardest part (compressed-latent MLA with query absorption at exactly 512) already exists and compiles under ROCm 5.7.1 (`k_attend` present in `glm_decode_gfx906`, dated Sep 4) |
| **value-up** | `k_vctx` / `k_vctx_m`, `dsa_decode_gfx906.hip:481-547` | HDIM 256→128, NHEADS 64→48 (both are `#define`s); stride into interleaved `kv_b` instead of a pre-split `Wv_b` | **specialise** | low |
| **lm_head** | `qf8_lm_head` int8 path, `qf_hip4_stage.hip:1038` | vocab 248,320 → 151,552; K=2560 unchanged | **specialise (constant)** | low. `qf8_i8_quant_bf16_host` already streams a head from a host mmap through a 4096-row staging buffer — reusable unchanged |
| **sampling / argmax** | `qf_hip4_stage` argmax | none | **reuse as-is** | none |

**Two cross-cutting warnings for anyone reading this tree.**

The `.h` headers in `/tmp/fork_tree/src` are systematically stale and must not be trusted for semantics; only `.hip` bodies are authoritative. Two proven instances: `qf_nvfp4_wave64.h:18-19` states an *interleaved* nibble order that `qf_nvfp4_wave64.hip:78-88` documents as an abandoned bug (measured corr 0.9955 adjacent vs 0.1420 interleaved); and `qf_router_wave64.h:11-14` says weights are "NOT renormalized" while `.hip:149-154` renormalizes. openPangu is definitively adjacent-pair — confirmed on both the write side (`nvfp4_ops.cu:833-839`) and the oracle's own reference read (`nvfp4_ops.cu:858-860`, `shift = (element & 1) * 4`).

The NVFP4 scale plumbing is **resolved and safe**: `nvfp4_pack.cu:255-256` reads `weight_scale_2 = tensor_amax / (6·448)` and `global_scale = 1/weight_scale_2`, so `weight_scale_2` **is** the reciprocal and maps to the AMD `scale2` with **no inversion** (`qf_nvfp4_wave64.hip:131`, `acc * scale2`). Activations are quantized with `global_scale = 1.0F`, so the alpha is complete on its own. Scale bytes are guaranteed non-negative (built from `fabsf/fmaxf` off 0.0F, validated `> 0.0F` at load), so `e = b >> 3` is safe. The on-disk scale plane is **linear** — the swizzled copy is a load-time CUDA artifact — so the AMD side reads `scales.e4m3` directly and can ignore the 6.23 GB `scales.swizzled.e4m3`.

---

## 4. NEW KERNELS REQUIRED

Only items with **no** existing analogue. Everything else in §3 is a parameter change or a re-tile of a kernel that exists and compiles.

**N1 — Unified MLA decode attention: sinks + rope term + three addressing modes.** *Difficulty: large.* This is one kernel because openPangu's oracle makes it one expression (`model_ops.cu:390-392`), and all three modes share the sink prepend.

- *Sinks.* 128 prepended keys, indices 0–127, on **every** layer and **both** paths. They are full attendable keys, not a denominator trick: they contribute to the softmax denominator *and* the value accumulator (`model_ops.cu:451`). They carry their own rope half (`param_sink_k_pe`, bound raw, no rope applied) and are RMSNorm'd once at load with the layer's `kv_a_layernorm` (`model.cpp:1139-1152`). **Nothing resembling this exists on AMD** — a `sink` grep over both trees returns exactly one hit, and it is the word "sinkhorn."
- *Rope term.* The AMD MLA is **NoPE** — a rope/`k_pe`/`inv_freq`/`cosf` grep over both GLM `dsa_decode_gfx906` files returns **zero** hits. openPangu's score is two-part: `q_absorbed·latent` over 512 dims **plus** `q_rope·k_rope` over 64, from a second cache allocated separately (`model.cpp:891-894`). New: a second cache array, a second dot in the inner loop, and q/k rope kernels. The rope *arithmetic* is available (Qwen `k8_attn_prep` does half-split (i, i+D/2) pairing at `qf8_qsa.hip:102-107`), so only the plumbing is new.
- *Three modes.* dense / circular-512 / indexed, selected by `(cache_start, cache_count, cache_capacity, selected_indices)`. The circular mode (30 of 46 layers) is three integers and one modulo — small, but it must live in the same kernel as the sink prepend.
- Shapes: 48 heads, latent 512 (64 lanes × 8 dims — the GLM lane geometry transfers unchanged), rope 64, sel length 2176.

**N2 — DSA per-position index-key cache + 24-head weighted scorer.** *Difficulty: medium.* Three structural things differ from Qwen's QSA indexer, not just constants: (a) the pooled key must become a per-position key cache — delete `pool_sum`/`pool_cnt`/finalize entirely; (b) a per-head weight vector must be threaded in (`head_scores[h] = max(dot,0)` then `score = Σ w[h]·head_scores[h]`, `dsa_ops.cu:104,112`); (c) selection granularity moves from blocks to positions, so the `blk_list`/mask/tail-block machinery is replaced by a flat 2048-entry index list. The math is the same family and GLM's scorer is the closer template (weights outside, relu inside, `dsa_decode_gfx906.hip:176`) at 32 heads → 24. **Trap:** openPangu ropes only the *first 64* of the 128 indexer dims as pairs (i, i+32); Qwen ropes all 128 as (i, i+64) (`qf8_qsa.hip:162-171`). Copying Qwen's indexer rope is silently wrong. Independently confirmed by the reference model (`npu_pangu.py:471-481`, `rotary_mode="half"`). Also: openPangu norms the *key* and not the query — the opposite of Qwen — and the query comes from the normalized q_lora (1024) while the key comes from the normalized hidden (2560), so Qwen's single fused `idx_qk` projection is the wrong wiring.

**N3 — T-row MLA prefill against a shared cache.** *Difficulty: large. This is the largest new-kernel risk in the port.* The AMD MLA is decode-only: no prefill file exists in either GLM tree, and its "M-row" form is **M independent caches** for the verify lane (`AttnMPtrs` holds `const unsigned char* lat[8]` — one cache pointer per row — with the comment "the 8 rows are independent"), not T queries over one cache. The structural template is Qwen's union-tile prefill (`k8_attn_union8`, `qf8_qsa.hip:951-1068`: one wave per 8 consecutive queries × 4 heads, every K/V block read from HBM once per wave) but that is written for 256-dim standard K/V, not a 512-dim latent plus a 64-dim rope cache. **The CUDA oracle does not have a good one either**: `mla_attention_prefill_rows_tiled` fuses at most 4 rows and otherwise falls back to one kernel launch per row (`model_ops.cu:1054-1069`), and its 4-row fused kernel is marked `[[maybe_unused]]`, unreachable, and uses a non-circular `cache_index` that would be wrong on sliding layers. There is no donor on either side.

Everything else that looked new is not. MoME conv3 is a 4-line delta on Qwen's GDN conv (which exists in both decode and rows form). The DSA top-2048 selector is a re-parameterisation of GLM's radix-select. The mHC is a near-exact GLM donor. The router is a near-exact GLM donor. `k_d8_gemv<64,9>` for K=9216 is one switch line.

---

## 5. MULTI-GPU PLACEMENT

**There is no GPU-to-GPU path on this box.** Each gfx906 node has exactly one `io_link`, pointing at node 0 (host); there is no `p2p_links` directory at all, and the four cards sit on four separate root complexes (PCI 03/43/83/c7). `hipMemcpyPeerAsync` silently no-ops. Large BAR is *not* the blocker — the cards expose a 16 GB prefetchable 64-bit BAR0 — the kernel build (`CONFIG_PCI_P2PDMA` unset) and the topology are. The tree's `qf4_peer_available` is a functional data-movement probe (writes a pattern, reads it back), so it can be relied on to select the transport safely.

**Recommendation: layer-parallel, 12 / 12 / 11 / 11.** Not expert-parallel.

| card | layers | DSA / sliding | weights (4-bit) | +KV @32K | +KV @512K |
|---|---|---|--:|--:|--:|
| 0 | 0–11 (incl. both dense) | 4 / 8 | 11.66 GiB | 11.83 | 14.41 |
| 1 | 12–23 | 4 / 8 | 12.98 | **13.16** | **15.73** |
| 2 | 24–34 | 4 / 7 | 11.90 | 12.07 | 14.65 |
| 3 | 35–45 + norm + merge + head | 4 / 7 | 12.10 | 12.28 | 14.86 |
| | | | | 49.34 box | 59.65 box |

The 12/12/11/11 split is not arbitrary: each contiguous block contains **exactly 4 multiples of 3**, so the 16 DSA layers — the only ones that hold full-context KV — distribute perfectly evenly. A 13/11/11/11 split would balance weights better but gives 5/3/4/4 DSA layers, and at 512K that KV imbalance (1.48 GB) costs more than the weight imbalance (1.3 GB) it fixes. Card 1 is the constraint at 512K: 15.73 of 15.98 GiB physical, **above** the 15.50 GiB guard budget and with only 0.25 GiB before driver reserve. Driver reserve on gfx906 is typically 200–400 MB/card `[est, not measured]` — that is the one case where it could flip the answer. At 128K everything sits at ~12.4–13.7 GiB and the split is comfortable.

**Blocker to fix before any of this runs.** `qf4_layers_total()` and `qf4_layer_base()` both round down with `v -= v % QF4_NGPU` (`qf_hip4_stage.hip:747`, `:760`; repeated at `qf_hip4_pipeline.cpp:339` and `qf_m8_region_server.cpp:211`). With 46 layers this silently yields `ltot=44`, `nl_per=11` — **layers 44 and 45 vanish**. Worse, `S->owns_head = (l0 + nl_per >= NLAYER)` then evaluates `33+11 >= 46` = false, so **no stage claims the output head either** (recoverable only via the `QF_AMD_OWNS_HEAD=1` override at `:783`). This is the exact class of silent truncation the project record already logged once ("every authority incl 19.02 ran 44 of 45 layers"). The code cannot express an uneven split at all; that has to be built.

**Crossings per token: 3.** The residual crosses card boundaries through **pinned host memory** — never D2D. `QF4_RSIZE = 10240` floats is `hc_count 4 × n_embd 2560`, and openPangu's `kFlat = kHidden * kStreams = 10240` is the identical width, so the plumbing transfers unchanged: 40,960 B fp32, or 20,480 B if the wire carries BF16 as the oracle does (`model.cpp:826-829`). Plus a ~5 KB embedding row gather per token; keep the 776 MB embedding table **pinned on the host** (the Qwen tree measured 0.29 → 1.92 ms/token when it was not). Total crossing traffic ≈ 128 KB/token — bandwidth-trivial.

**The four cards run SERIALLY for one token, and this is a data dependency, not a scheduling bug.** With contiguous layers, card *g*+1 cannot start until card *g* finishes. `qf4_stage_run_M` only enqueues; the next statement is a blocking `hipMemcpy` D2H that drains the stream (`qf_hip4_stage.hip:1224-1229`). Three of four cards are idle at every instant. The measured Qwen shape of this: 16 layers across 4 cards = 10.03 ms/token wall vs 9.405 ms of hipEvent spans, i.e. ~2.4 ms per stage and only 0.63 ms/token of launch+sync residue (`AMD_M1_LAUNCH_OVERHEAD_DISCRIMINATOR_20260906.md:19-21`).

Recovering the idle cards has exactly three levers: **token pipelining** (`qf_hip4_pipeline.cpp` exists — 2-deep mailboxes, one pthread per GPU — but production never calls it, going straight to `qf4_stage_*`), **M > 1 rows**, or **prefill chunks** (the only path that is already concurrent, with a recorded ROCm 5.7 gotcha: cross-device `hipStreamWaitEvent` blocks the *enqueueing host thread*, so the working code uses `hipEventSynchronize` instead).

**Why not expert-parallel.** Exact multivariate-hypergeometric enumeration over C(256,8) into four blocks of 64 gives **E[max experts on one card] = 3.5122**, so the speedup is **8/3.5122 = 2.28×, not 4×** (P(max≥4) = 0.436). And with P2P dead, tensor/expert-parallel needs an all-reduce of the 10,240-element mHC stream twice per layer = **92 host-mediated round trips per token** to buy a floor of ~1.27 ms `[est]` versus 3.55 ms serial. The existing expert-sharded tier (`qf_moe_4card.cpp`) already D2H-copies all four partials and sums them on the host, and is not even linked into the production server.

---

## 6. BRING-UP ORDER

**Step 0 — capture the oracle before the Spark leaves. This is the highest-priority item in the whole schedule.**

The CUDA engine has a good test suite (`tests/layer0_forward_test.cpp`, `mla_decode_test.cu`, `mla_rows_test.cu`, `dsa_test.cu`, `mhc_state_test.cu`, `mome_rows_test.cu`, `moe_forward_test.cpp`, `projection_rows_test.cpp`, `full_forward_test.cpp`) — but a grep for `ofstream|fwrite|write(` across all of them returns **nothing**. Every test self-checks and prints `P92_LAYER0_FORWARD_OK`. **They emit no reference vectors.** A dump harness must be built on the Spark while it is still here, or every AMD gate below loses its oracle.

Capture, at fp32, for a fixed prompt and a fixed seed:

1. Layer-0 per-op tensors: post-`mhc_pre_parallel` (mixes[24], h_post[4], h_res[4,4]), post-RMSNorm, q_a, q_norm, q_b, q_absorbed, q_rope, kv_a, kv_down, post-MoME-conv, attention output, o_proj, post-`mhc_post`.
2. `prepare_sinks()` output — the 128 RMSNorm'd sink latents per layer, plus raw `param_sink_k_pe`. These are load-time constants; capturing them removes a whole class of bring-up ambiguity.
3. Router: raw logits, sigmoid scores, corrected scores, selected ids, final weights, for ≥64 real hidden states across several layers. This doubles as the routing-skew census (§7 R1) **and** settles the combine-order question below.
4. DSA at a context > 2048: index scores for all positions, and the selected 2048 indices, for at least one layer.
5. If precomputing the absorbed `M`: the `M[48,512,1024]` matrix for layer 0 (`+31 MB/layer`, `+1.45 GB` across 46 — the oracle absorbs at runtime instead, `model_ops.cu:63-78`).
6. Full-forward logits for ~32 generated tokens — the end-to-end gate.
7. **An operator ruling on combine order.** The oracle has *two* combine semantics and neither is the "ascending expert id + BF16 accumulator" law stated in the brief. The resident path accumulates **fp32 in router-selection order** with one BF16 round at the end (`nvfp4_ops.cu:107-122`). The fallback keeps a **BF16 accumulator**, fp32 route weight, rounds after the add, still selection order (`nvfp4_ops.cu:209-218`, reached from `model.cpp:1707-1720`). They differ observably, the AMD machinery can implement either, and the M=1 kernel's fold is a 5-line tail. This must be decided before N1 or the expert body is written.

**Step 1 — layer 0 alone, M=1, one card, dense attention, no MoE.** Layer 0 is dense (`index < 2`) and `index%3==0` so it is a DSA layer, but at position < 2048 DSA does not fire. It therefore exercises the maximum new machinery with the minimum expert machinery: **mHC ×2, MLA with sinks, MoME convs ×3, dense MLP, RMSNorm, residual**. *Gate:* cosine ≥ 0.999 against the captured layer-0 output for the same input residual, per-op first, then end-of-layer. *Oracle:* capture (1) + (2).

**Step 2 — layer 2: first MoE layer, sliding window.** `index%3 = 2 ≠ 0` so it is sliding with **no** indexer. Adds: router top-8, routed expert gate/up/down, shared expert, expert combine, 512-slot circular cache. *Gate:* (a) exact match on the **selected expert id set** vs capture (3) — this is a discrete gate and it either passes or it does not; (b) cosine ≥ 0.999 on the layer output. Run the router gate *first and separately*: quantization noise in an INT8 router GEMV shifts the sigmoid logit, and selection is decided by `sigmoid(logit) + a small fp32 bias`, so noise can flip *which* experts are chosen, not merely how much weight they get. *Oracle:* capture (3).

**Step 3 — layer 3: first DSA layer with the indexer live, at context > 2048.** Adds N2 (per-position key cache + 24-head scorer) and the top-2048 selector. *Gate:* (a) the selected 2048-index set matches the capture exactly, or the symmetric difference is bounded and every differing entry is a genuine score tie; (b) attention output cosine ≥ 0.999. *Oracle:* capture (4). Ties are a real question — the oracle takes the first 2048 of a full descending CUB sort, while the AMD selectors define ties as ascending index; whether openPangu's scores tie often enough to matter is **unknown**.

**Step 4 — all 46 layers, M=1, one card, 4-bit experts, short context.** *Gate:* greedy token path matches the capture for ≥32 tokens. Any divergence is either the combine-order ruling or 4-bit expert quality (§7 R2) — and the layer-by-layer gates above tell you which.

**Step 5 — head and sampling.** `mhc_merge` + `model.norm` + INT8 lm_head + argmax. *Gate:* top-1 and top-5 logit ids match the capture.

**Step 6 — four cards, 12/12/11/11, serial.** Fix the `% QF4_NGPU` truncation first. *Gate:* the 4-card greedy path is **bit-identical** to the 1-card path — the crossing is a plain 40,960 B copy and must not change anything. Then measure per-stage hipEvent spans against wall to get the launch/sync residual (the Qwen gate was < 1.0 ms/token).

**Step 7 — prefill (N3), then token pipelining.** Both are throughput work and neither is on the correctness path. Do not start either before Step 6 passes.

---

## 7. THE THREE LARGEST RISKS

**R1 — 4-bit routed experts are ALU-bound on gfx906, and INT8 does not fit.** This is a hard fork in the road, not a tuning problem. The measured NVFP4 rate (777 + 615 µs per layer for 24.6 MB, `qf8_moe_i8.h:4-8`) projects to **~88 ms/token from routed experts alone ≈ 11 tok/s**, against a 4-bit byte floor of 3.55 ms. INT8 would sit at the roofline but misses residency by 21.8 GiB. If neither the mixed-format set nor a faster 4-bit kernel works, the port's ceiling is ~10 tok/s.

*Measurement that settles it early, and it is cheap:* port `k_i8_gateup_all`/`k_i8_downacc_all` to 1024/2560 (a grid constant and a 5→8 chunk change) and the NVFP4 gate/up + a new down template, then time **one** MoE layer on **one** card in both formats. That single number decides whether the mixed-format resident set is worth building. Pair it with the routing-skew census from oracle capture (3): with ~34 % of experts INT8-resident per card at 32K, the payoff is entirely a function of what fraction of lookups that hot set absorbs — and the tree already has `qf4_stage_hist_print` to answer it.

**R2 — there is no quality receipt for 4-bit routed experts, and the Qwen precedent is bad.** The only 4-bit quality datum in existence is `receipts/IMPLEMENTATION.md:31-33`: cosine **0.991788**, max abs diff **0.215007** — measured on `model.layers.0.self_attn.q_a_proj.weight`, an **attention projection**, not an expert. Meanwhile the project record for Qwen on this exact hardware reads "int4 routed experts REJECTED — first boot = TOKEN SALAD at 19.27 tok/s," with the note that a 0.985 cosine proxy measured a path the kernel never ran. openPangu's experts are 1024-wide where Qwen's are 640, and its top-8-of-256 is sparser than Qwen's top-10-of-512, so it may well behave differently — but that is a hypothesis. **4-bit is the only residency point that fits; if it fails on quality, the model does not fit this box at all.**

*Measurement that settles it early — and it must happen on the Spark before it leaves:* run the CUDA engine end-to-end with the resident 4-bit expert bank versus BF16 experts, same prompt set, same seed, and compare greedy token paths and logit cosines. This is a single A/B on hardware that already exists and already has both paths (`model.cpp:1685` selects on `layer.expert_bank != nullptr`). It is the highest-value hour available before the oracle goes away, and it is currently unrun.

**R3 — T-row MLA prefill has no donor anywhere, including the oracle.** The AMD MLA is decode-only (no prefill file in either GLM tree; its M-row form is M independent caches, not T queries over one). Qwen's union-tile prefill is written for 256-dim standard K/V, not a 512-dim latent plus a 64-dim rope cache. And the CUDA oracle itself fuses at most 4 rows and otherwise launches **one kernel per row** (`model_ops.cu:1054-1069`), with its only fused 4-row kernel dead and incorrect on sliding layers. There is nothing to steal. At 46 layers × per-row launches, a naive port makes long prompts unusable.

*Measurement that settles it early:* time the CUDA engine's own prefill at 2K and 8K on the Spark before it leaves — that establishes what "acceptable" means and whether the oracle's per-row fallback is even the path being taken. Then time a single-row AMD MLA decode kernel × T on the AMD box to price the naive floor. The ratio tells you whether N3 is a scheduling problem or a real kernel project, and it can be measured before a line of N3 is written.

*Runner-up, worth stating because it changes the shape of the model at long context:* the DSA index-key scan is the only unbounded per-token term, reaching **2,147 MB/token at 512K — 67.7 % of the entire 4-bit weight stream**. Storing index keys at INT8 rather than BF16 halves it, and the selection is a top-K over relu'd dot products where a global scale is rank-invariant. Whether INT8 keys preserve the selected set is untested; a one-shot experiment on the Spark against capture (4) would settle it.

> **CORRECTION 2026-09-09.** Any statement in this document that MI50
> P2P is unavailable/dead at the driver is FALSE. It was true of Linux
> 5.15.0-190 only. Under Linux 6.8.0-138 the same hostile gate moves from
> 0/12 to 12/12 directed pairs with no collateral, because KFD creates
> `p2p_links` under 6.8 and created none under 5.15. Same cards, same ROCm
> 5.7.1 userspace, no ACS override. See receipts/07_P2P_DIAGNOSIS.md.
> The failure MODE described - peer pointers silently aliasing local memory
> instead of faulting - was real and is why a P2P path must be gated by a
> data-integrity check, never a capability bit.
