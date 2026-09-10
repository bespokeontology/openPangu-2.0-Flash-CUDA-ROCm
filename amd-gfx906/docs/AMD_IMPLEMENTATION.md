# AMD gfx906 implementation

## 1. Hardware / platform

| item | value |
|---|---|
| GPU | 4x AMD MI50 / Radeon Pro VII, gfx906 (Vega 20), 60 CU, wave64 |
| VRAM | 17,163,091,968 B per card from `/sys/class/kfd` topology; 63.94 GiB aggregate |
| clocks | 1700 MHz; DPM `profile_peak` enforced by the machine lock before any run |
| host | Ubuntu, Linux 6.8 |
| toolchain | ROCm HIP, clang 17 (`/opt/rocm`), `--offload-arch=gfx906` |
| transport | host-staged peer copies (producer push), measured 23 GB/s push vs 3.85 GB/s peer read |

## 2. Model

openPangu-2.0-Flash: 46 trunk layers (0-1 dense, 2-45 MoE), MLA attention
(q_lora 1024, kv_lora 512, 48 heads, 1 KV head, 128 nope + 64 rope), 128
attention sinks, DSA sparse attention on 16 layers (24-head indexer, top-2048),
512 sliding window elsewhere, 256 routed experts top-8 + 1 shared per MoE
layer, 4-stream mHC hyper-connections with 20-iteration Sinkhorn per sublayer,
three MoME depthwise causal convolutions per attention block, 3 MTP layers
(46-48) that the release does not execute.

## 3. Storage vs execution representation

Storage is NVFP4 (`P92FP41` container): E2M1 nibbles, 2 per byte, even index in
the low nibble; one UE4M3 scale per 16-element group, row-major. Execution is
NOT the storage format: the hot MoE path decodes nibbles to int8 lanes that are
exactly 2x the E2M1 magnitudes and feeds `v_dot4_i32_i8`, with the factor of
two folded into the group scale (the weight arithmetic is exact integer
products). Storage representation is not execution representation anywhere in
this engine.

## 4. Runtime architecture (decode)

Per token, layers execute in owner order 12/12/11/11 across the four cards, the
four-stream mHC state crossing cards as a 20,480-byte host-staged push. Every
kernel is openPangu-shape specialised: no rocBLAS, no generic GEMM, no tail
handling, no runtime dispatch in the hot path. Kernel families:

| family | file | notes |
|---|---|---|
| MLA decode attention | `src/p92_mla_g16.hip` | 32-lane groups, 16 split slots, branchless loads, 2 vmcnt(0) barriers |
| MLA prep / absorb / value-up | `src/p92_mla_prep.hip` | absorbed queries, latent cache |
| DSA indexer + selector | `src/p92_dsa_index.hip` | 24-head indexer, radix select top-2048, fused selector 146 us -> 26 us |
| NVFP4 experts | `src/p92_nvfp4_dot4.hip` | `p92_dq4` + `v_dot4_i32_i8`, 257-slot arena |
| INT8 expert path | `src/p92_i8_expert.hip` | alternate representation |
| MoE router + group | `src/p92_router.hip` | DPP cross-lane argmax |
| mHC (4-stream, Sinkhorn) | `src/p92_mhc.hip` | 24-mix 4/4/16 split, sigma/2sigma softmax, 20 iterations |
| MoME convolutions | `src/p92_mome.hip` | depthwise causal k=3, three sites |
| projections / norms / rope | `src/p92_proj.hip` | fused RMSNorm+quant, GEMV per shape |
| head + top-256 refine | `src/p92_head.hip` | two-stage int8 head |
| transport | `src/p92_transport.cpp` | host-staged ring, producer push |

## 5. Prefill architecture

Chunked prefill (chunk 256) with a four-card layer pipeline
(`prefill/src/p92_pf_drive.hip`): each card owns 11-12 layers of the 46 and
runs them for every chunk, pushing the chunk's four-stream state to the next
card (producer push). Pipeline depth 4 overlaps chunks; measured overlap 3.7-3.8x
of 4. Stage budget at 64K: attention 51.2%, proj 10.7%, gateup 9.9%, down 7.8%,
DSA score 6.7%, everything else 5.4%. See `PREFILL_REPORT.md`.

## 6. Collectives / transport

Three host-staged crossings per decode token (20,480 B each). Peer writes reach
~23 GB/s; peer READS measured only 3.85 GB/s, so the engine never structures a
hot path around consumers pulling remote state. The prefill pipeline's landing
buffer is 5.24 MB per card with an integrity gate that verifies an ordered-pair
pattern before enabling peer paths (a capability bit is never trusted).

## 7. Numerics / gating policy

**Numerical bit-exactness is not used as a blanket acceptance criterion.**
Kernels may change reduction order, reassociation, intermediate precision,
execution representation, packing and layout. Gates used instead:

- structural/exact contracts: indices, bounds, ownership, packing, nibble
  order, selector membership, mask interpretation, state transitions;
- kernel-level: comparison against an independent host reference with a
  tolerance appropriate to the operation (e.g. DSA selector 3.4e-07, sparse
  attention 7.3e-07, G2 head-shared attention 7.1e-07 worst relative), plus
  negative controls that must move the output;
- engine-level: composed generation, coherent output, accepted quality gates
  by the operator.

## 8. Frozen authority (this release)

| artifact | commit / tag | sha256 |
|---|---|---|
| decode binary `p92_gen` | frozen tree `p92-amd` @ `7f5ccc7` (report 69.92 rung) | `b28f0eef71121a0199315dfc4b5d1deee1ebbe3b8dafaad4cd40657a24d52d75` |
| prefill binary `p92_pf_bench` | tag `freeze-p92-prefill-spg2-20260909` -> `17d0fa9` | `d5ad29f368d33d67bdcbae7dd7d12f2a53379bd55df0ddcc084c69fca503ae72` |

**The released AMD decode authority has MTP disabled.** The binary contains no
MTP execution path. MTP work lives on a development branch and is described in
`MTP_STATUS.md`.

## 9. Build

Decode: `hipcc --offload-arch=gfx906 -O3 -std=c++17 -Iinclude tests/p92_generate.hip -o p92_gen -lpthread`
Prefill: see `README.md`. Both compile with no dependencies beyond ROCm.

Machine lock: prefill and decode runs are serialized by a machine-wide lock
that fails closed if a foreign VRAM holder is present, if any card is off
`profile_peak`, or if the driver cannot be queried. Benchmarks are never run
without it.
