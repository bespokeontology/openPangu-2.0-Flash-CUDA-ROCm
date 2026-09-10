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

- **Measured prompt context: 70,008 tokens** (20.29 ms/token = 49.28 tok/s over the run; 20.87 ms at position 70,007). The decode engine sizes its DSA caches from the `MAXPOS` argument and crosses 2^16 = 65,536 with no change in behaviour, so 16-bit position and slot indices are not a limit. `MAXPOS=262144` allocates ~369 MB per DSA layer (~1.5 GB a card) and runs; the block at larger contexts is ingest, not residency.
- Prefill (a chunk pipeline that launches no indexer above 2,048 positions, and therefore measures a ceiling rather than a correct context) is verified to 65,536 tokens at 698.2 tok/s. Building a real 262K context needs the indexer inside the chunked ingest; see `decode/receipts/22_70008_POSITIONS_AND_PREFILL_CEILING.md`.
- **Retrieval at depth is verified**: in a 32,852-token context, a fact inserted at 50% depth and another at 75% depth were both quoted verbatim in the answer (basalt `58-1904`, cobalt `63-8157`) at 50.11 tok/s; at 74,891 tokens the same question is read correctly at 48.96 tok/s. Evidence: `decode/receipts/22_70008_POSITIONS_AND_PREFILL_CEILING.md`.
- Long prompt files must be built with the model's own tokenizer: **`decode/tools/p92_encode.cpp`** (`encode` / `chat` / `decode` / `check` / `ids`). Earlier long-context files were built with a different tokenizer and contained ids up to 154,841 against a 151,552-row embedding table, which makes their *content* meaningless even though their timings stand.
- Measured decode through the chat CLI (greedy, MTP off): **67.5 tok/s at context 512**, **50.4 tok/s at 5,669** (19.83 ms/token), **48.9 tok/s at 43,135** (20.46 ms/token). The cost is a one-time step when the engine switches into sparse attention at 2,048 positions, not a linear scan: the 37,466 positions past 5,669 cost 0.63 ms/token, and the whole DSA index/select path is 0.5% of token time. Profile in `decode/receipts/20_DECODE_PROFILE_AND_PIPELINE_FINDING.md`.

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
