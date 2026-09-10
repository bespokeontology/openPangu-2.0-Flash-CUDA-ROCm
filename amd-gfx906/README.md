# openPangu-2.0-Flash — native AMD gfx906 backend

Native C++/HIP inference engine for Huawei's `openPangu-2.0-Flash` (92B total,
~6B active) on 4x AMD MI50 / Radeon Pro VII (gfx906). No Python, no framework
runtime, no rocBLAS in the hot path, no generic GEMM fallback.

This is the second native backend of the same project. The DGX Spark CUDA
backend at the repository root is the earlier implementation and is preserved
unchanged as the historical baseline and comparison point.

## Status

- **Decode: released authority, plain native target decode, MTP OFF.**
  69.92 tok/s steady, gfx906, 4 cards, 12/12/11/11 layer ownership.
- **Prefill: released authority (SPG2), 710 / 731 / 732 / 725 / 698 tok/s at
  4K / 8K / 16K / 32K / 64K prompt tokens.**
- **MTP: experimental, not part of the released performance claim.**
  A native three-head draft path and speculative state machine exist on the
  development branch; the fixed-T batched target verifier is not finished.
  See `docs/MTP_STATUS.md`.

## Hardware and software

| item | value |
|---|---|
| GPU | 4x AMD MI50 / Radeon Pro VII, gfx906, 60 CU, wave64 |
| VRAM | 17,163,091,968 B per card (15.98 GiB usable), 63.94 GiB aggregate |
| clocks | 1700 MHz (DPM `profile_peak` enforced by the benchmark lock) |
| host | Ubuntu, Linux 6.8 |
| ROCm | /opt/rocm (HIP, clang 17) |
| build | `hipcc --offload-arch=gfx906 -O3 -std=c++17` |

## Layout

```
amd-gfx906/
  decode/     trunk decode engine (p92-amd tree): src/, include/, tests/, tools/, receipts/
  prefill/    prefill engine (p92-prefill tree): src/, include/, tests/, receipts
  docs/       AMD_IMPLEMENTATION.md, BENCHMARKS.md, AMD_KERNEL_NOTES.md,
              PREFILL_REPORT.md, MTP_STATUS.md
```

## Weights

Both engines read the NVFP4 container `P92FP41` (`manifest.bin`,
`weights.nvfp4`, `scales.e4m3`) plus BF16 tensors taken from the official
checkpoint (norms, mHC parameters, compressed-KV projections, depthwise
convolutions, attention sinks, routers and correction biases, token
embeddings). The container format is specified by its consumers:
`decode/tests/p92_generate.hip` (`MHdr`/`MRec`, `nv_up`),
`decode/src/p92_nvfp4_dot4.hip` (E2M1 nibble order, UE4M3 per-16 group
scales), `decode/tools/p92_pack_arena.cpp` (arena layout).

The quantizer that produced the MTP container is included
(`decode/tools/mtp_quant.cpp`) and shows the exact packing procedure. The
trunk container is produced by the same procedure over the trunk tensor
census documented in `docs/AMD_IMPLEMENTATION.md`.

## Build (decode)

```
cd amd-gfx906/decode
/opt/rocm/bin/hipcc --offload-arch=gfx906 -O3 -std=c++17 -Iinclude \
  tests/p92_generate.hip -o p92_gen -lpthread
```

Run: `./p92_gen <checkpoint> <artifact> <arena> <tokens> <maxpos> <start-token>`

## Build (prefill)

```
cd amd-gfx906/prefill
/opt/rocm/bin/hipcc --offload-arch=gfx906 -O3 -std=c++17 -Iinclude \
  -I../../amd-gfx906/decode/include tests/p92_pf_bench.hip src/p92_pf_shuffle.hip \
  src/p92_p2p.hip src/p92_pf_drive.hip -o p92_pf_bench -lpthread
```

Run under the machine lock: `~/q27bench ./p92_pf_bench <checkpoint> <artifact> <arena> <prompt> <maxpos>`

## Context

- **Supported prompt context: 62,144 tokens** (the decode engine sizes its DSA caches from the `MAXPOS` argument; 62,144/pass it as the 5th argument).
- Prefill is verified to 65,536 tokens at 698.2 tok/s (TTFT 0.65 s); 128K and 256K exceed VRAM on the 4x16 GB configuration.
- Decode throughput at 62K context has not been measured; the DSA index scan is O(context) per token, so expect the short-context rate (67.47 tok/s at context 512) to fall at long context.

## Model / license

Source only; no weights are included. See the repository `NOTICE` and
`legal/` for upstream Huawei attribution and the OpenPangu Model License
Agreement Version 2.0.

## Chat (text in, text out)

The decoder accepts text prompts directly; no token-id preparation and no Python.

```
echo "Explain Kolmogorov complexity in two sentences." | \
  P92_TEMP=0.6 P92_TOPK=50 P92_TOPP=0.95 \
  ~/q27bench ./p92_chat <checkpoint> <artifact> <arena> 160 4096 148899 -
```

argv[7] = "-" reads the prompt from stdin (chat template + BPE encode applied in-process); pass a path instead to feed int64 token ids. P92_TEMP=0 restores greedy argmax. See decode/receipts/19_CHAT_TEXT_MODE.md.
