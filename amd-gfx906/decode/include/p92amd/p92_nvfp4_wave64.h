// p92_nvfp4_wave64.h - gfx906 (MI50) wave64 NVFP4 expert kernels for
// openPangu-2.0-Flash. Derived from the proven Qwen3.8 gfx906 kernels
// (qf_nvfp4_wave64.h/.hip) by shape specialisation only; the NVFP4 decode,
// wave64 reduction geometry and packing contract are unchanged.
//
// Shape differences from the Qwen kernels:
//   hidden (K for gate/up)   2560 -> 2560   unchanged
//   expert FFN (rows/K down) 640  -> 1024
//   experts per layer        512  -> 256
//   experts selected         10   -> 8
//
// Packing contract (follows the CORRECTED comment in qf_nvfp4_wave64.hip,
// not the stale header text there): a row of K values is K/2 packed bytes
// plus K/16 UE4M3 scale bytes. A scale group is 16 values = 8 packed bytes
// + 1 scale byte. Byte j of a group holds value 2j in the LOW nibble and
// value 2j+1 in the HIGH nibble (adjacent pairs). The interleaved
// {v[j] lo, v[j+8] hi} reading is wrong and was measured wrong: adjacent
// pairs give corr 0.9955 / maxerr 0.0066 against the BF16 original, the
// interleaved reading gives corr 0.1420 / maxerr 0.0809.
//   val = sign * E2M1[nib & 7] * UE4M3(scale_byte) * scale2
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifndef P92_HOST_CHECK
#include <hip/hip_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define P92_NEMBD    2560   // hidden
#define P92_NFF      1024   // routed/shared expert intermediate
#define P92_NEXP     256    // routed experts per MoE layer
#define P92_NEXPUSED 8      // top-k
#define P92_DISPATCH 9      // 8 routed plus the shared expert

// Per-expert byte strides in a packed slot arena.
//   gate or up : [P92_NFF, P92_NEMBD]  -> 1024*1280 W bytes, 1024*160 S bytes
//   down       : [P92_NEMBD, P92_NFF]  -> 2560*512  W bytes, 2560*64  S bytes
#define P92_GU_W_BYTES ((size_t)P92_NFF   * (P92_NEMBD / 2))
#define P92_GU_S_BYTES ((size_t)P92_NFF   * (P92_NEMBD / 16))
#define P92_DN_W_BYTES ((size_t)P92_NEMBD * (P92_NFF / 2))
#define P92_DN_S_BYTES ((size_t)P92_NEMBD * (P92_NFF / 16))

#ifndef P92_GATEUP_THREADS
#define P92_GATEUP_THREADS 256
#endif

// Rows per block: one row per half-wave (32 lanes).
#define P92_GU_ROWS_PER_BLOCK (P92_GATEUP_THREADS / 32)
#define P92_GU_BLOCKS         (P92_NFF / P92_GU_ROWS_PER_BLOCK)
// down: one row per eighth-wave (8 lanes), 256 threads -> 32 rows per block.
#define P92_DN_ROWS_PER_BLOCK 32
#define P92_DN_BLOCKS         (P92_NEMBD / P92_DN_ROWS_PER_BLOCK)

#ifdef __cplusplus
}
#endif
