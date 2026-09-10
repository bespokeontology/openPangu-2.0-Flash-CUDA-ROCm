# openPangu Flash92 — native NVFP4 CUDA inference engine

A direct C++/CUDA inference engine for Huawei's `openPangu-2.0-Flash` (92B total
parameters, ~6B active per token) targeting NVIDIA GB10 / SM 12.1a. The engine keeps
the model fully resident, executes NVFP4 projections on Blackwell tensor cores, and
contains no Python, no framework runtime and no llama.cpp-derived execution path.

This repository contains source only. It contains no model weights. See `NOTICE`.

## Backends

This repository contains two native backends for the same model. **The decode
rows differ in decode mode: the DGX Spark row uses MTP / speculative decoding;
the AMD row is plain native target decode with MTP disabled.** MTP on AMD is
experimental and is not part of the released claim (see amd-gfx906/docs/MTP_STATUS.md).

| Backend | Hardware | Decode mode | MTP | Decode tok/s | Prefill tok/s | Status |
|---|---|---|---:|---:|---:|---|
| NVIDIA CUDA | DGX Spark GB10 | MTP / speculative | ON | 52.10 @ 123 ctx, 96 gen ("Explain probability in one clear sentence.") | 51.4 @ 3,807 tokens | preserved baseline |
| NVIDIA CUDA | DGX Spark GB10 | plain target | OFF | 21.01 @ 331 ctx | 51.4 @ 3,807 tokens | preserved baseline |
| AMD gfx906 | 4x MI50 / Pro VII | plain target | OFF | 69.92 @ 512 ctx, fixed token stream | 710 / 731 / 732 / 725 / 698 at 4K/8K/16K/32K/64K | current |

The AMD decode measurement shown above does not use MTP or speculative decoding;
the DGX Spark decode measurement shown above does.

Full tables, methodology and sources: amd-gfx906/docs/BENCHMARKS.md.

## Model

| property | value |
|---|---|
| trunk layers | 46 |
| MTP layers | 3 (layers 46-48), optional |
| hidden size | 2560 |
| attention | MLA, `q_lora_rank` 1024, `kv_lora_rank` 512, 48 heads, 1 KV head |
| sparse attention | DSA indexer, 24 heads, key length 128, top-2048 |
| attention sinks | 128 per layer |
| sliding window | 512, on layers where `index % 3 != 0` |
| MoE | 256 experts, top-8, plus 1 shared expert, expert FFN 1024 |
| dense layers | 0 and 1 |
| RoPE | dimension 64, base 6,400,000 |
| vocabulary | 151,552, untied LM head |
| native context | 524,288 |

## Weights

The engine reads two things:

1. an **NVFP4 artifact** — the quantized projection weights packed into a `P92FP41`
   container (`manifest.bin`, `weights.nvfp4`, `scales.e4m3`, `scales.swizzled.e4m3`);
2. an **auxiliary checkpoint directory** — BF16 tensors that are not quantized
   (per-layer norms, mHC parameters, the compressed-KV projections, depthwise
   convolutions, attention sinks, routers and their correction biases, the token
   embedding), plus `tokenizer.json`.

Only the small tensors come from (2). `scripts/fetch_auxiliary_tensors.py` fetches
exactly those by HTTP byte range from the public upstream repository and assembles
them into safetensors shards, so a full 200 GB checkpoint download is not required.
The result is approximately 1.6 GB.

The catalog reads shards named `model-00001.safetensors` through
`model-00050.safetensors`; shards that hold no required tensor may be empty but must
be present and well formed.

## Build

Requires CUDA 13, a C++20 compiler, CUTLASS headers, FlashInfer headers and
`libpcre2-8`.

```
cmake -S . -B build \
  -DP92_CUTLASS_INCLUDE=/path/to/cutlass/include \
  -DP92_CUTLASS_TOOLS_INCLUDE=/path/to/cutlass/tools/util/include \
  -DP92_FLASHINFER_INCLUDE=/path/to/flashinfer/include
cmake --build build -j
```

Targets of interest: `p92_chat` (trunk decode), `p92_chat_mtp` (trunk plus the
checkpoint's three-stage MTP predictor), `p92_tokenizer_test`,
`p92_nvfp4_artifact_inspect`.

## Run

```
./build/p92_chat     <auxiliary_checkpoint_dir> <nvfp4_artifact_dir> [max_context] [max_new]
./build/p92_chat_mtp <auxiliary_checkpoint_dir> <nvfp4_artifact_dir> [max_context] [max_new]
```

Both read prompts from stdin and accept `/reset` and `/quit`.

## Measured

DGX Spark GB10, NVFP4 resident, 56.9 GB. Full conditions and method in
`ENGINEERING_REPORT.md`.

| workload | result |
|---|---|
| decode, MTP, documented gate | 52.10 tok/s, acceptance 67/81 |
| decode, MTP, harder prompt | 43.37 tok/s, acceptance 197/300 |
| decode, MTP, 800-token answer | 33.84 tok/s, acceptance 507/873 |
| decode, 331 context, trunk | 21.01 tok/s |
| decode, 4,008 context, MTP | 20.05 tok/s |
| decode, 11,139 context, trunk | 14.10 tok/s |
| prefill, 3,807 tokens | 74.08 s = 51.4 tok/s |
| prefill, 10,970 tokens | 268.42 s = 40.9 tok/s |
| cold load, page cache dropped | 59-65 s |

MTP throughput tracks draft acceptance, which depends on the prompt: the two rows above
use the same binary and context and differ only in what was asked. Decode also falls with
context because DSA index selection scans every position on 16 of the 46 layers per
token, while attention itself is bounded. Prefill is the weaker component and
`docs/PREFILL_MEASUREMENTS.md` records four changes that were measured and rejected.

A llama.cpp Q4_K_M build of the same model on the same machine measures 11.30 tok/s
decode and 243-274 tok/s prefill.

## Licence

Powered by openPangu. openPangu is a trademark of Huawei Technologies Co., Ltd.

Released under the OpenPangu Model License Agreement Version 2.0, reproduced in full in
`LICENSE`. Note section 3.1: the Model may not be used within the European Union. The
model weights and tokenizer are obtained separately from Huawei under the same
agreement; see `NOTICE`. `legal/` retains Huawei's licence and Open Source Software
Notice as exact upstream bytes.
