# Benchmarks — DGX Spark (CUDA, MTP ON) and AMD gfx906 (plain, MTP OFF)

Every number below is copied from a committed receipt. Where the two backends
were not measured with the same methodology, the methodology column says so;
no apples-to-apples claim is manufactured.

**Decode-mode distinction, stated once and repeated in every table:
the DGX Spark decode figures below use MTP / speculative decoding
(`p92_chat_mtp`); the AMD gfx906 decode figure below is plain native target
decode with MTP disabled (`p92_gen`).**

## 1. Decode

| backend | hardware | decode mode | MTP | context | prompt | generated | tok/s | source |
|---|---|---|---:|---:|---|---:|---:|---|
| NVIDIA CUDA | DGX Spark GB10 | MTP / speculative | ON | 123 | "Explain probability in one clear sentence." | 96 | 52.1041 | Spark `ENGINEERING_REPORT.md` §3.1 (runs 46.2825 / 52.1041, acceptance 67/81) |
| NVIDIA CUDA | DGX Spark GB10 | MTP / speculative | ON | 331 | "Summarise Kolmogorov complexity in two sentences." | 300 | 43.3694 | Spark `ENGINEERING_REPORT.md` §3.4 (acceptance 197/300) |
| NVIDIA CUDA | DGX Spark GB10 | plain target | OFF | 331 | "Summarise Kolmogorov complexity in two sentences." | 300 | 21.0090 | Spark `ENGINEERING_REPORT.md` §3.4 |
| NVIDIA CUDA | DGX Spark GB10 | plain target | OFF | 4,008 | same | 200 | 15.2955 | Spark `ENGINEERING_REPORT.md` §3.4 |
| **AMD gfx906** | **4x MI50 / Pro VII** | **plain target** | **OFF** | **512** | **fixed token stream** | **12** | **69.92** | `decode/ENGINEERING_REPORT.md` (rung ladder; frozen binary `freeze-69.9/p92_gen`, sha256 b28f0eef...) |

Note on the two decode regimes: the Spark rows are different prompts and
generation lengths on a different machine; the AMD row is the frozen
authority rung measured on the project's own fixed token stream at short
context. The Spark MTP rows are not comparable to the AMD row as a
backend-versus-backend comparison, because one uses speculative decoding and
the other does not.

## 2. Prefill

| backend | hardware | prompt tokens | tok/s | context limit | source |
|---|---|---:|---:|---:|---|
| NVIDIA CUDA | DGX Spark GB10 | 3,807 | 51.4 | 8,192 | Spark `ENGINEERING_REPORT.md` §4 |
| NVIDIA CUDA | DGX Spark GB10 | 10,970 | 40.9 | 16,384 | Spark `ENGINEERING_REPORT.md` §4 |
| **AMD gfx906** | **4x MI50 / Pro VII** | **4,096** | **710.0** | 4,096 | `prefill/23_SPG2_SHIPS.md` ladder |
| **AMD gfx906** | **4x MI50 / Pro VII** | **8,192** | **730.7** | 8,192 | same |
| **AMD gfx906** | **4x MI50 / Pro VII** | **16,384** | **732.4** | 16,384 | same |
| **AMD gfx906** | **4x MI50 / Pro VII** | **32,768** | **724.9** | 32,768 | same |
| **AMD gfx906** | **4x MI50 / Pro VII** | **65,536** | **698.2** | 65,536 | same; TTFT 0.65 s |

Prefill methodology: the AMD engine prefills a prompt of the stated length and
reports whole-prompt tokens/second at chunk 256, pipeline depth 4, all four
cards, chunked prefill. The Spark rows are its own prompt files at its own
context limits. The two prefill measurements are the same physical quantity
(prompt tokens processed per second, cold) but on different hardware, different
prompts and different engines; they are presented adjacently without a
derived ratio.

Absolute peak reference for the AMD cards: HBM streaming measured at
~894 GB/s per card in this project's own roofline work.

## 3. Memory

| backend | resident | notes |
|---|---|---|
| NVIDIA CUDA | 56.9 GB | NVFP4 projections 56,051,047,104 B + auxiliary BF16/F32 |
| AMD gfx906 | 47 GiB expert arena + ~1.2 GB non-expert per machine | all-NVFP4 expert arena, 257 slots a layer, 12/12/11/11 ownership |
| AMD gfx906 (prefill) | ~12.8 GiB per card, 4.35 GiB spare | P4 placement; 64K context fits, 128K does not (see `PREFILL_REPORT.md`) |

## 4. What was not measured together

- The Spark engine was not run on this AMD machine and the AMD engine was not
  run on the Spark machine. There is no same-hardware cross-backend row.
- Spark MTP acceptance is prompt-dependent (82.7% and 62.6% on two prompts at
  identical context, same binary); no single Spark MTP number is quoted here
  without its prompt and generation length.
- The AMD engine's MTP path is experimental and is not benchmarked in this
  release. `docs/MTP_STATUS.md` records what exists and what is a negative
  architectural result rather than a performance figure.
