# openPangu-2.0-Flash — native AMD gfx906 backend

Native C++/HIP inference engine for Huawei's `openPangu-2.0-Flash` (92B total,
~6B active) on 4x AMD MI50 / Radeon Pro VII (gfx906). No Python, no framework
runtime, no rocBLAS in the hot path, no generic GEMM fallback.

This is the second native backend of the same project. The DGX Spark CUDA
backend at the repository root is the earlier implementation and is preserved
unchanged as the historical baseline and comparison point.

## Status

- **Prefill ingest in the server (v1.2, 2026-09-12): released authority.** `p92_serve` ingests a
  prompt with the four-card chunk pipeline and decodes from its caches: real prompts of 1,117 /
  9,405 / 38,716 tokens at the numbers in `docs/BENCHMARKS.md` §2b (ring arena, binary defaults),
  time to the first generated token 1.15 s at 1K instead of 18.4 s for the token-walk ingest of
  v1.1 (989 / 1,038 / 1,013 / 964 tok/s at 1K / 4K / 8K / 32K). Greedy continuations are deterministic run to run and identical between the two ingest paths.
- **Decode: plain native target decode, MTP OFF.** 69.92 tok/s steady at 512 context on the fixed
  token stream (v1.1 authority, unchanged kernels); 59-60 tok/s behind a real 1K prompt, 49-50 behind
  8K-32K (the sparse-attention regime above 2,048 positions).
- **Prefill bench (v1.1's 710-732 tok/s ladder):** superseded. That harness never delivered chunk
  state to cards 1-3 (fixed in v1.2; timing was representative, outputs were not).
- **MTP: experimental, not part of the released performance claim.** See `docs/MTP_STATUS.md`.

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

## Build (server with prefill ingest — the product path since 2026-09-12)

`p92_serve` contains the four-card chunked prefill in-process: a prompt is ingested by the
prefill pipeline and the decoder continues from the same caches (`P92_PREFILL=1`, default;
`P92_PREFILL_MIN=32` is the shortest prompt that goes through the pipeline). It also needs
the arena's expert scales in both layouts (flipped in place at the phase boundary), rocBLAS
for the optional projection path, and pcre2 for the tokenizer.

```
cd amd-gfx906/decode
/opt/rocm/bin/hipcc --offload-arch=gfx906 -O3 -std=c++17 -Iinclude -I../prefill/include \
  tests/p92_serve.hip tools/p92_tokenizer.cpp ../prefill/src/p92_pf_shuffle.hip \
  ../prefill/src/p92_p2p.hip ../prefill/src/p92_pf_drive.hip -o p92_serve \
  -lpthread -lpcre2-8 -L/opt/rocm/lib -lrocblas
```

Run (prompt as int64 token ids from `p92_encode`, `NTOK` = prompt tokens + tokens to generate):

```
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-5.7.1/lib/rocblas/library P92_TEMP=0 \
  ~/q27bench ./p92_serve <checkpoint> <artifact> <arena> <NTOK> <MAXPOS> 148899 prompt.i64
```

Switches (all default to the measured-best setting, see `prefill/24_PREFILL_INGEST_RING_ATTENTION.md`):
`P92_PREFILL` (1) · `PF_CHUNK` (256) · `PF_ATTN_BLK` (2 = blocked attention v2, 1 = v1, 0 = the
v1.1 per-head-pair kernels) · `PF_PROJ_RB` (1 = rocBLAS int8 projections, warmed at init) ·
`PF_DOWN_DET` (1, deterministic MoE down) · `P92_CHAIN` (1, device-side decode crossings) ·
`PF_VERBOSE=1` prints the prefill's per-stage decomposition.

`ROCBLAS_TENSILE_LIBPATH` pins rocBLAS 5.7.1's library directory: about one run in thirty it
resolves the path without the `lib/` component and aborts at init.

## Arena layouts: contiguous and ring

`tools/p92_pack_arena.cpp ARTIFACT OUT` packs the contiguous 12/12/11/11 expert arena (the
decode-optimal map: 3 card crossings a token). `tools/p92_pack_arena.cpp ARTIFACT OUT 4` packs a
RING arena: card `b % 4` owns the layer block `b` of 4 layers, so a prefill chunk hops the ring
0 -> 1 -> 2 -> 3 -> 0 ... and the pipeline fills in 3 block-times instead of 3 chunk-times — the
short-prompt lever (+36-49 % prefill at 1K, +14 % at 4K, +6 % at 8K with the v1.1 kernels; decode
crosses cards 11 times a token, at no measurable cost with the device-side chain). The arena header records the block size and `p92_serve` reads its ownership
map from the arena it loads; `p92_pf_bench` is contiguous-only.

## Build (prefill bench)

```
cd amd-gfx906/prefill
/opt/rocm/bin/hipcc --offload-arch=gfx906 -O3 -std=c++17 -Iinclude \
  -I../../amd-gfx906/decode/include tests/p92_pf_bench.hip src/p92_pf_shuffle.hip \
  src/p92_p2p.hip src/p92_pf_drive.hip -o p92_pf_bench -lpthread
```

Run under the machine lock: `~/q27bench ./p92_pf_bench <checkpoint> <artifact> <arena> <prompt> <maxpos> [148899 prompt.i64]`
(argv[7] = a real prompt as int64 ids; without it the bench uses its synthetic token stream).

## Context

- **Measured prompt context: 70,008 tokens** (20.29 ms/token = 49.28 tok/s over the run; 20.87 ms at position 70,007). The decode engine sizes its DSA caches from the `MAXPOS` argument and crosses 2^16 = 65,536 with no change in behaviour, so 16-bit position and slot indices are not a limit. `MAXPOS=262144` allocates ~369 MB per DSA layer (~1.5 GB a card) and runs; the block at larger contexts is ingest, not residency.
- Prefill (a chunk pipeline that launches no indexer above 2,048 positions, and therefore measures a ceiling rather than a correct context) is verified to 65,536 tokens at 698.2 tok/s. Building a real 262K context needs the indexer inside the chunked ingest; see `decode/receipts/22_70008_POSITIONS_AND_PREFILL_CEILING.md`.
- **Retrieval at depth works; recall at depth is not perfect.** At 32,852 tokens, a fact inserted at 50% depth and another at 75% depth were both quoted verbatim (basalt `58-1904`, cobalt `63-8157`) at 50.11 tok/s. At 74,891 tokens the same document is still being read — the 25%- and 75%-depth needles came back verbatim — but the requested 50%-depth needle was missed. Accessibility of deep context is demonstrated; perfect needle recall is not. Evidence: `decode/receipts/23_RETRIEVAL_AT_DEPTH_AND_PERSISTENT_SERVER.md`.
- **Persistent serving verified**: `decode/tests/p92_serve.hip` (`P92_SERVE=1`) keeps an ingested context resident and answers further requests from stdin, appending each and advancing positions monotonically (`[serve] request 30 tokens, at position 194` then `at position 288`), so a long ingest is paid once and then asked many questions.
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
