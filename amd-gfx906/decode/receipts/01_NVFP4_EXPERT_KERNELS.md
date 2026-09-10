# Mechanism 1 - NVFP4 routed/shared expert GEMV on gfx906

Date 2026-09-08. Target 4x MI50 (gfx906, wave64), ROCm 5.7.1.

## What was done

Specialised the proven Qwen3.8 gfx906 NVFP4 kernels (qf_nvfp4_wave64.hip) to the
openPangu expert shapes. The NVFP4 decode (E2M1 nibble, UE4M3 block scale), the
group dot, the wave64 sub-reduction geometry and the packing contract are copied
unchanged. Only dimensions differ.

| property | Qwen3.8 | openPangu | effect on the kernel |
|---|---|---|---|
| hidden (K for gate/up) | 2560 | 2560 | none, inner loop identical |
| expert FFN | 640 | 1024 | gate/up rows 640 to 1024, grid 80 to 128 blocks |
| down projection K | 640 | 1024 | groups 40 to 64, iterations 5 to 8 |
| experts per layer | 512 | 256 | arena slot count only |
| top-k | 10 | 8 | router only, not these kernels |

Per-expert strides: gate or up W 1,310,720 B, S 163,840 B; down W 1,310,720 B,
S 163,840 B.

Packing note: the header of the Qwen kernel states an interleaved nibble order.
That comment is stale. The .hip file documents the correction and the code uses
adjacent pairs (value 2j in the low nibble of byte j, value 2j+1 in the high
nibble), measured against the BF16 original at corr 0.9955 / maxerr 0.0066
versus corr 0.1420 / maxerr 0.0809 for the interleaved reading. The openPangu
kernels follow the code, not the stale header.

## Correctness gate

tests/p92_nvfp4_ref_test.cpp compares the kernel lane decomposition against an
independent scalar reference that walks every value of a row in natural order.
64 random trials per shape, random packed bytes and scales (UE4M3 saturation
pattern excluded), activations uniform in [-1, 1].

| kernel | K | lanes | iterations | worst relative error |
|---|---|---|---|---|
| gate/up 1024x2560 | 2560 | 32 | 5 | 8.291e-06 |
| down 2560x1024 | 1024 | 8 | 8 | 1.373e-06 |

P92_NVFP4_GATE_OK. Threshold 2e-5. Reproduce:
  g++ -O2 -DP92_HOST_CHECK -Iinclude -o t tests/p92_nvfp4_ref_test.cpp && ./t

## Compilation

hipcc --offload-arch=gfx906 -O3 -Iinclude -c src/p92_nvfp4_wave64.hip

| kernel | SGPR | VGPR | occupancy | VGPR spill | LDS |
|---|---|---|---|---|---|
| gemv_1024x2560 | 24 | 64 | 4 waves/SIMD | 177 | 10,240 B |
| gemv_2560x1024 | 24 | 64 | 4 waves/SIMD | 335 | 4,096 B |
| gateup_1024x2560 | 32 | 64 | 4 waves/SIMD | 374 | 10,240 B |

The spill counts are not a regression introduced here. The proven Qwen kernels
compiled with the same toolchain report 177, 177 and 376 with identical VGPR
count and occupancy, so the spilling is a property of the shared decode
structure under this compiler and is present in a kernel already in production
use.

## Not yet done

No GPU execution. This gate covers kernel arithmetic and compilation only.
A device run against real packed expert weights, checked against the CUDA
engine, is the next gate.
