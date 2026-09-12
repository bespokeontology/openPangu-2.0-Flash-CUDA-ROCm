# Transfer package: gfx906 / MI50 lessons from the openPangu port, for AMD Qwen3.8-27B

Written 2026-09-08 by the Pangu window. You own AMD-only Qwen3.8-27B. This is
everything the Pangu port learned that should transfer.

## 0. What you are and are not being told to do

**Your product target is gfx906 / MI50 (Radeon Pro VII / Instinct MI50).** Build
a native HIP engine for Qwen3.8-27B on those cards.

**Do NOT optimize the existing CUDA Qwen engine.** It is not the product. Use it
as two things and nothing else:
1. an **architecture and correctness oracle** - the authoritative statement of
   what the model computes, shapes, layouts, and the order of operations;
2. a **source of proven algorithms** - things already measured to work.

**Pangu is your implementation playbook**, not your architecture. The two models
differ; the hardware does not.

The single most important inherited rule: **a decision made on CUDA is a
hypothesis on gfx906, not a constraint.** Pangu proved this in both directions.
Huawei overlaps mHC Sinkhorn work on a side stream; the CUDA port reproduced it
exactly and REJECTED it (frozen 1751.9/1762.3/1834.7 ms vs overlapped
1791.9/1828.5/1886.5) because event cost exceeded hidden work - and the AMD port
then WON with the same idea done intra-kernel (wave 0 running Sinkhorn while
waves 1-3 collapse, no events). Conversely the CUDA side rejected a reassociated
mHC reduction, and on AMD the reassociating variant turned out to have no speed
advantage at all, so it was rejected for a *different, locally measured* reason.
Bank the other backend's finding as a test case. Confirm or falsify it here.

## 1. Read these first

On the AMD host these live under `<data>/p92-amd/receipts/`, and they are
published (so link them from your README instead of copying them):
`amd-gfx906/decode/receipts/` in the openPangu AMD tree on GitHub and Gitee.
The old `<data>/p92-amd/receipts/` path in the first version of this file no
longer exists.

The six that matter most:

    10_GFX906_PLAYBOOK.md    the hardware lessons. READ IN FULL. 7 sections.
    11_PROFILING_LAW.md      how measurement lies. Three named contaminations.
    08_REJECTED.md           ~20 mechanisms that failed, with numbers and causes.
    09_NVFP4_TRAPS.md        14 NVFP4-specific traps.
    13_MLA_SPLIT_ABI.md      a dormant producer/consumer ABI landmine + its gate.
    14_MLA_G16_ACCOUNTING.md a kernel accounted for - including a WRONG closure.

The rejected ledger is the highest-value document per minute of reading. Most of
what looks obviously worth doing on this hardware has already been measured and
lost.

## 2. Things to actively SEARCH FOR in your kernels

Not "remember these" - go looking for them, with these exact tools.

### 2.1 Conditional/exec-masked VMEM serialization  ** the biggest single find **

This was worth ~6 percent of the whole Pangu engine in one change.

**Symptom.** A load inside `if (cond) { ... }` where `cond` varies by lane. The
compiler puts each load in its own exec-masked region and fences EVERY ONE with
`s_waitcnt vmcnt(0)`. N guarded loads become N serialized global round trips.

**How to find it:**

    hipcc --offload-arch=gfx906 -O3 -S -o k.s -c kernel.hip
    awk '/BB0_4/,/BB0_28/' k.s | grep -c 'vmcnt(0)'     # per-loop barriers
    awk '/BB0_4/,/BB0_28/' k.s | grep -c 'saveexec'      # exec-mask churn

If barriers ~= number of loads, you have it.

**The fix.** Make the loads unconditional and give inactive lanes a VALID
address. In Pangu's `k_p92_mla_g16` the masked results were *already* discarded -
`s` is overwritten with `-1e30f` and the softmax weight `p` is 0 when off - so
only the address mattered. Pointing the off case at an always-resident buffer
(`sink_lat`/`sink_pe`) instead of `nullptr` made every load unconditional:

    6 -> 2 s_waitcnt vmcnt(0), 12 -> 7 saveexec, bit-identical output.

**Verify before you do this:** prove the inactive lane's loaded value cannot
reach the result, and prove the substitute address is in-bounds for every
inactive lane. Do not trade a branch for a speculative OOB read.

**The counter-intuitive part, which is the real lesson:** the winning variant had
MORE instructions - 283 -> 422 in the loop - at identical VGPR (77) and occupancy
(3). Instruction count went up 49 percent and it got faster. Dependency structure
is the objective; instruction count is not.

### 2.2 Production-path vs probe-path control-flow mismatch  ** most expensive mistake **

Pangu CLOSED a kernel that had 5 percent left in it because every probe passed a
mode parameter (`win_cap = 0`) that selects a different branch from the one
production takes. The pre-existing bench had the same defect. All measurements,
comparisons and a "remaining prize is bounded at 0.4 us" conclusion were about a
program that never runs.

**Before trusting any standalone measurement:** diff the arguments your probe
passes against what the engine passes, and diff the compiled inner loop. A
parameter that selects a branch is part of the shape being measured, exactly like
a tensor dimension. If a kernel has modes, gate BOTH.

**Tell-tale:** standalone time and in-engine time disagreeing. Pangu saw 46.75 us
standalone vs 35.4 us in-engine and attributed it to context length instead of
investigating. That was the clue.

### 2.3 s_waitcnt / vmcnt dependency chains generally

Look at where waits sit relative to loads. The pattern you want is: issue several
independent loads, then one wait. The pattern to hunt is: load, wait, use, load,
wait, use. `grep -n 'global_load\|s_waitcnt' k.s` and read the interleaving.

### 2.4 Runtime division and modulo - but expect ZERO

`x % n` with runtime `n` compiles to ~22 instructions (`v_rcp_iflag_f32`, two
`v_mul_lo_u32`, two `v_mul_hi_u32`, ...). Find them: `grep -c 'v_rcp_iflag_f32'`.

Pangu found one executing once per entry on every layer, replaced it with an
exact compare-subtract (reduce the offset modulo the cap on the HOST, then
`t >= cap ? t - cap : t` is exact when both operands are < cap), removed all
three integer-division ops - and measured **exactly zero** speedup.

Keep this as a calibration: on a latency-bound kernel, eliminating 22
instructions per entry can be worth nothing. Fix them for clarity; do not expect
a rung, and never let an instruction count justify a change on its own.

### 2.5 DPP vs ds_bpermute, and the exact lane semantics

`__shfl*` lowers to `ds_bpermute_b32` on gfx906 - the DS pipe, not the ALU. In a
kernel with occupancy to spare it is hidden. In a single-wave or low-occupancy
kernel it dominates. Pangu's router went 8.96 -> 6.04 us (33 percent) by
replacing it with DPP; the SAME substitution LOST in the expert kernels at 7-8
waves/SIMD. **Replace DS ops only where measurement shows they are on the
critical path.**

**VERIFIED lane mappings** (proved on hardware, do not take these from anywhere
else, including this document, without re-running the check):

    row_shl:N  == __shfl_down(v,N)      lane n gets lane n+N
    row_shr:N  == __shfl_up(v,N)        lane n gets lane n-N
    quad_perm 0xB1 == xor 1             exact
    quad_perm 0x4E == xor 2             exact
    row_half_mirror 0x141 == xor 7      exact
    row_mirror 0x140      == xor 15     exact
    xor 8  == row_mirror THEN row_half_mirror,  (l^15)^7 == l^8   exact, 2 ops
    xor 4  == NO exact substitute
    xor 16, xor 32 == cross the 16-lane row, NO substitute

    #define DPPF(x,c) __int_as_float(__builtin_amdgcn_update_dpp( \
        __float_as_int(x), __float_as_int(x), (c), 0xF, 0xF, false))

**An asserted lane semantic is a hypothesis.** An analysis agent told the Pangu
window that `row_shr` was the `__shfl_down` substitute. It is `row_shl`. That was
integrated without checking and produced a kernel that looked fine, ran at a
plausible speed, and was silently NOT bit-exact. It was caught only because a
bit-comparison happened to be running. Write the 12-line probe every time:

    __global__ void k(float*o){ int l=threadIdx.x; float v=(float)(l+1);
      o[l*3+0]=__shfl_down(v,4,64);
      o[l*3+1]=DPPF(v,0x100u|4u);   // row_shl:4
      o[l*3+2]=DPPF(v,0x110u|4u); } // row_shr:4

**Reassociation trap.** `row_ror` sums the same SET of lanes as `xor` but in a
different PAIRING. For lane 4: xor gives `(v4+v0)+(v12+v8)`, ror gives
`(v4+v8)+(v12+v0)`. Float addition is not associative. Lanes 0-3 coincide, so a
spot check of the first quad passes and the kernel still diverges. **Check the
pairing at a lane >= 4, never only lane 0.**

### 2.6 Dead-code elimination masquerading as a speedup

A DPP router rewrite measured FASTER than the measured no-communication floor and
emitted `0xBEBEBEBE`. A patch had consumed the one assignment that gave the loop
an output dependency, so the compiler deleted it.

> A kernel that beats its own measured lower bound has stopped doing the work.
> Every optimization measurement is invalid until its correctness gate passes on
> the same binary that produced the timing.

Three instances of this class occurred on Pangu. Also: a sweep that left the
launch grid fixed while shrinking rows/block (the kernel wrote a fraction of its
rows and looked faster), and an unscoped `re.sub` that rewrote a neighbouring
kernel's lane math.

### 2.7 Occupancy and VGPR are proxies, not objectives

Measured on `mla_g16`: LANES 32 -> 64 took VGPR 76 -> 50 and occupancy 3 -> 4 -
both target metrics improved - and the kernel got **6.2 percent slower**,
bit-identical. Changing the lane width re-priced the reduction: at 32 lanes one
`__shfl_xor` reduces BOTH half-wave groups at once (5 shuffles serve 2 entries);
at 64 it is 6 shuffles for 1 entry, 2.4x the DS traffic per unit work.

> Never promote on a resource metric. When a geometry parameter changes,
> re-derive the cost of EVERY mechanism that depends on it.

Also: before spending effort on registers, establish WHICH live ranges hold them
and WHICH occupancy threshold the reduction would cross. gfx906: waves =
floor(256/VGPR) with granule 4, so 8 waves needs <=32, 5 needs <=51, 4 needs <=64.
A register count with no threshold behind it is not a target. On `mla_g16` the
obvious 16-register array turned out to hold only 6 of the 28 extra registers,
and removing it crossed no threshold and cost 34 percent.

### 2.8 LDS staging and replication

Pangu's expert down-projection staged all nine experts' activations into LDS:
9216 B/block and 2.95 MB of redundant traffic per call, capping occupancy at 5
waves/SIMD. **Removing it entirely was a win** - 30 VGPRs, 8 waves. The
permutation the consumer needed was pushed into the PRODUCER instead, so the
consumer needs no staging at all.

> Do not assume LDS helps. Require measurement. Prefer moving a layout
> requirement into the producer over staging in the consumer.

### 2.9 Skinny GEMV memory behaviour

- 32 lanes x `uint2` (8 B/lane) = 256 B per step is the best-measured streaming
  pattern on these cards. **Wider per-lane loads are WORSE**: uint4 (512 B/step)
  measured 643 GB/s against 741 for uint2; two uint4 in flight collapsed to 219.
  Do not re-propose uint4 without new evidence.
- A pure streaming read at that pattern reaches **741-785 GB/s** per card. The
  int8 expert kernel achieves 731 GB/s in production. Those are your real roofs.
- **Measure a roof at the TARGET KERNEL'S occupancy.** A streaming probe at 10
  waves/SIMD is not a roof for a kernel that runs at 3. Pin it:
  `__attribute__((amdgpu_waves_per_eu(N,N)))`. Pangu's "34.85 us memory roof" was
  really 38.07 at the kernel's actual occupancy - a 40 percent overstatement of
  the gap.
- **Issued bytes are not fetched bytes.** In both the expert and MLA kernels all
  consumers read the same weights/KV rows, so traffic is replicated from a small
  L2-resident working set. `mla_g16` issues 19,240 KB and FETCHES 513.8 KB. Any
  GB/s computed from issued bytes is L2 bandwidth and is meaningless as a roof.
  Measure `FETCH_SIZE` with rocprof, do not compute it.

## 3. NVFP4 and quantization ABI

Qwen may or may not use NVFP4. If it does, all of this transfers verbatim.

### 3.1 Storage representation is not execution representation

**The central trick.** NVFP4 is E2M1 nibbles with UE4M3 per-16 block scales and a
per-tensor fp32 `weight_scale_2`. gfx906 has NO FP4 datapath. But the DOUBLED
E2M1 magnitudes are exactly {0,1,2,3,4,6,8,12} - all representable in int8 - so a
nibble decodes losslessly into an int8 lane for `v_dot4_i32_i8` via two
`v_perm_b32` byte-table lookups plus a bitselect, and a `0.5f` in the epilogue
undoes the doubling.

    float nibble decode, 16 FMAs per 8 bytes       19.0 GB/s
    two v_perm_b32 + bitselect + dot4            410.7 GB/s     21.6x

See `src/p92_nvfp4_dot4.hip:26-52` (`p92_dq4`, `p92_d4`, `p92_group_dot4`).
`v_dot8_i32_i4` is NOT usable for E2M1 because 12 and 8 exceed the 4-bit signed
range.

### 3.2 The nibble ABI

Byte j holds value 2j in the LOW nibble and 2j+1 in the HIGH nibble - adjacent
pairs, not interleaved halves. Masking a packed dword yields the EVEN values and
shifting yields the ODD ones. This forces the activation layout: the consumer
wants activations permuted even-then-odd within each 16-value group.

**Put that permutation in the PRODUCER.** Pangu emits it from `k_p92_quant` and
`k_p92_f4_requant` so the consumers need no LDS staging.

**Then guard the contract.** Moving the permutation into a SHARED requant kernel
silently changed what the INT8 path consumed. Keep separate requant kernels per
consumer ABI (`k_p92_f4_requant` vs `k_p92_i8_requant`) and gate the composition.

### 3.3 Activation scales and secondary scale state

Two traps that both produced ~1e3 errors on Pangu:

- The per-16 **activation** scale must be folded into the group scale INSIDE the
  dot loop, not applied in the epilogue. Dropping it cost a factor of `xs` on
  gate and again on up, and since the intermediate is `silu(gate)*up` that is
  `xs` SQUARED.
- `weight_scale_2` is a per-tensor fp32 around 1e-4, stored per slot as
  `s2[slot*3 + {gate,up,down}]`. Passing `1.f` instead is a silent 1e4 error.

> Nonlinear scale propagation: whenever a scale crosses a nonlinearity (SiLU,
> softmax, a norm), work out on paper where it must be applied. An error that
> would be a harmless constant in a linear path becomes a squared or exponential
> error through a nonlinearity.

### 3.4 Double-application

After the permutation moved into the producer, one consumer still staged it -
producing a double permutation that corrupted two layers while **all twelve gates
passed** and the engine still emitted plausible text. Composition gates exist
because of this.

## 4. Measurement laws - these cost the most to learn

### 4.1 Initialization contaminates whole-process profiles

`rocprof --stats` counts the WHOLE PROCESS. Dividing by generated tokens
amortizes model load and setup into decode and fabricates a per-token tax. Pangu
computed "1,947.7 launches/token, 15.3 percent dispatch overhead" this way. The
truth was 1,651 - the difference being 46 layers x 128 sinks x 3 kernels = 17,664
ONE-TIME launches.

**Use differential profiling:** `counts(N2) - counts(N1)` over `(N2 - N1)` after
identical initialization. Then cross-check every recurring count against
structure - a real per-token kernel lands on a clean multiple of
layers x tokens. A count that does not factor that way is carrying fixed work.

### 4.2 Launch count is not launch cost

Even at a correct 1,651 launches/token x 1.245 us = 2.055 ms of dispatch, the
work is asynchronous and the host stays ahead. Summed kernel time was 14.76 ms
against a 15.13 ms wall - only **0.37 ms** is non-kernel and recoverable.

> Before treating dispatch as a frontier, subtract summed kernel time from wall
> time. That difference is the entire addressable budget.

### 4.3 Source ablation does not decompose cost additively

A variant with a component removed is a DIFFERENT PROGRAM with different register
allocation and scheduling. In one Pangu attempt a variant doing strictly MORE
work ran 5.5 us FASTER than its own subset. That invalidates the method, not the
kernel. Use hardware counters and ISA, never differences of programs.

Useful counters: `FETCH_SIZE VALUBusy MemUnitBusy MemUnitStalled`.

### 4.4 Low utilization with low stalls is a dependency signature

`mla_g16`: VALUBusy 26.9 percent, MemUnitBusy 35.6, MemUnitStalled 0.8. The naive
reading is 4x ALU headroom. Wrong - the machine has capacity, the WAVE cannot
expose independent work because each online-softmax step depends on the previous.

Distinguish by trying to ADD independent work. If that makes it worse, the chain
is the answer.

### 4.5 Bound the prize before restructuring

Before designing an ABI change to remove an 18-shuffle merge, Pangu simply
DELETED it - mathematically wrong, loudly labelled - to see what it was worth.
0.4 us of a 45 us kernel. One measurement made the redesign obviously irrational.
(Caveat: a bound inherits 2.2 - it is only valid on the production path.)

### 4.6 Measurement protocol that survived

- Interleaved A/B, control and candidate alternating, never batched.
- Median of >= 6 pairs; report the full distribution and the ranges.
- Require the clean distributions to be DISJOINT before banking.
- The box produces occasional 3-5 percent outliers that hit BOTH arms; exclude
  them as a regime, never one-sided.
- Freeze every accepted rung: binary + sha256 + tag + commit + the distribution.

## 5. Composition gates and negative controls

Unit gates pass while composed paths are broken. Pangu shipped a corrupted layer
with twelve unit gates green.

**Every producer/consumer boundary needs a gate that runs BOTH sides.** And every
such gate needs a NEGATIVE CONTROL proving it can fail:

`tests/p92_mla_abi_test.hip` poisons every consumer-visible slot with a sentinel,
runs the producer, and requires zero survivors. Then it deliberately relaunches
the producer at HALF the splits and requires the detector to fire - it reports
196,992 gaps, exactly the missing half. A gate that cannot fail proves nothing.

**Also gate the geometry contract itself.** Pangu had `P92_SPLITS` defined twice -
unguarded as 16 in the consumer's file, `#ifndef`-guarded as 8 in the producer's.
Production resolved to 16 only because every source is `#include`d into ONE
translation unit AND the include order put the unguarded one first. Change either
and the producer writes 8 splits per head while the consumer reads 16, silently.

> **Compiled geometry is evidence. Header defaults and comments are not.**
> Establish a constant that crosses a producer/consumer boundary by compiling it
> into the shipped binary and printing it, or by a static assertion.

Prove the full address range, not just the value: producer max index, consumer
max index, and allocation capacity. Pangu's receipt records 393215 / 393215 /
393216.

Stale comments are the same class of hazard: `G16_DPL` was commented "32 latent
dims per lane" when it is 16, and `G16_STEPS` "8 uint2 loads" when it is 4, both
left over from an earlier geometry. Fix them when you find them.

## 6. Pangu code worth copying or adapting

All under `<data>/p92-amd/src/` on the AMD host.

| file | what to take | why |
|---|---|---|
| `p92_nvfp4_dot4.hip` | `p92_dq4`, `p92_d4`, `p92_group_dot4`, `p92_ue4m3_d` | The NVFP4 -> int8 dot4 decode, 21.6x over float decode. Directly reusable if Qwen is NVFP4. `p92_ue4m3_d`'s two branches are an OPTIMIZATION (execz-skips a rare path); do not "fix" them branchless, it measured 22 percent slower. |
| `p92_i8_expert.hip` | the whole int8 routed-expert kernel | Achieves 731 GB/s, the best rate in the engine. The reference for what a good streaming expert kernel looks like on this hardware. |
| `p92_router.hip` | the single-wave top-k with the DPP argmax reduction | 8.96 -> 6.04 us. The pattern for any small top-k. Note `readlane` needs a wave-uniform index. |
| `p92_mla_g16.hip` | the branchless-load structure, and the online-softmax split/merge | The 6->2 vmcnt fix is here. Also the sinks/body joint-run balancing, which removed a ~110 us floor caused by giving all sinks to split 0. |
| `p92_proj.hip` | `k_p92_rmsnorm_quant<N>` fused norm+quantize | LDS-free, 16 B LDS, 33 VGPRs, 7 waves. The pattern for fusing a norm into its consumer's quantization. |
| `p92_arena.cpp` | the resident-arena layout | No allocation after READY. Weights laid out per-device for the layer split. |
| `p92_transport.cpp` | the pinned-host ring | Cross-device layer handoff. NOTE: P2P is dead at the driver on this box; see `07_P2P_DIAGNOSIS.md` - remote pointers silently ALIAS LOCAL MEMORY, so a P2P path can appear to work while moving no data. Test with a data-integrity check, not a capability bit. |
| `tests/p92_mla_abi_test.hip` | the poison + negative-control gate pattern | Copy this structure for every producer/consumer boundary you build. |
| `tests/p92_contract_test.hip` | the composition gate | Caught a live defect twelve unit gates missed. |

## 7. Hardware facts

    4x MI50 / Radeon Pro VII, gfx906, Vega20, wave64
    60 CUs, 4 SIMDs/CU, 256 VGPRs/lane, granule 4
    sclk pinned 1700 MHz (rocm-smi --setperflevel high)
    16 GiB VRAM per card as reported by rocm-smi
    L2 4 MB per card
    HBM2: theoretical 1024 GB/s; measured streaming 741-785 GB/s
    v_dot4_i32_i8 available; v_dot8_i32_i4 present but unusable for E2M1
    NO MFMA usable for this workload; no FP4 datapath
    launch overhead 1.245 us; HIP graphs buy only ~15 percent of that
    P2P dead at the driver - three missing kernel config options, see 07_

`ssh <amd-host>`. Do not touch `<data>/p92-amd` except read-only.

## 8. The two rules that matter most

1. **Measure the composed engine.** The ISA explains, the microbenchmark
   diagnoses, the composed engine decides. Six of Pangu's first eleven
   hypotheses were wrong, several of which would pass a code review.
2. **Believe the number over the reasoning.** Every single one of the mechanisms
   in section 2 was found by measuring something that contradicted a confident
   expectation.

## 9. Prior-art inventory, with what each is good for

Everything below is on the AMD host unless marked. **Read-only.**

### 9.1 gfx906 HIP that already exists — start here

`~/glmflash-hip/src/hip/` is GLM running on gfx906 in HIP. It
is the closest precedent to what you are building: real AMD kernels for these
exact cards, not CUDA to be translated.

    glm_expert_v2_gfx906.hip    routed-expert kernel
    mlp_gfx906.hip              dense MLP
    dsa_decode_gfx906.hip       sparse-attention decode
    glm_bodies_gfx906.hip       layer bodies
    mtp_decode_gfx906.hip       MTP decode — relevant if Qwen gets speculation
    prefix_cache_gfx906.hip     prefix cache
    router_gfx906.h             routing
    exl3_dequant_gfx906.hip     EXL3 dequantization on gfx906
    bench_attn_roofline_gfx906.hip   a roofline harness already written
    bench_native_gemv_gfx906.hip     a GEMV bench already written
    bench_exl3_bw_gfx906.hip         EXL3 bandwidth bench

Sibling trees (`glmflash-amd40`, `glmflash-ep4`, `glmflash-dsa`,
`glmflash-m3-*`, `glmflash-arenas`, `glmflash-appliance-moe`, and others under
`~/`) are variants from that programme. `/srv/glmflash-preserve`
is the preserved copy. Pangu's MLA decode was itself ported from the GLM gfx906
tiled online-softmax kernel — that lineage already worked once.

**Caveat:** GLM's engine predates most of the lessons in sections 2-5 of this
document. Treat its kernels as structural starting points, then re-measure. In
particular check it for exec-masked VMEM serialization (2.1) — nothing in that
programme was looking for it.

### 9.2 Qwen material

    /data/qwen38-27b/model            the checkpoint
    /data/qwen38-27b/oracle_capture   captured oracle activations + reference
    /data/qwen38-27b/dflash-aligned-v5

Plus the frozen Qwen states in `~/freeze/`:
`FREEZE_QWEN_OVERNIGHT_20260907`, `FREEZE_QWEN_PREFILL_1543_20260906`,
`FREEZE_QWEN_FORK_PROOF_20260906`.

**The oracle capture is your correctness spine.** Use it the way Pangu used the
CUDA engine: per-op reference activations to gate each kernel as you write it,
with operation-appropriate tolerances rather than bit-exactness.

**One warning from the Pangu programme:** a captured oracle can be INVALID. One
Spark capture was proved wrong by an invariant its own source forced. Before
trusting a capture, verify it satisfies at least one property derivable from the
model definition independently of the capture.

### 9.3 The CUDA Qwen engine

Architecture and correctness oracle, and a source of proven algorithms. **Not
the product, and not a bit-exact target.** See section 0.

### 9.4 Pangu

`<data>/p92-amd` — this port. Section 1 lists the receipts, section 6
the kernels worth copying.

`<data>/engine/openpangu-flash92-native` — the CUDA oracle for Pangu.
Its receipts are worth reading even for Qwen, because they record what Huawei
and the CUDA port measured about MoE/expert execution: grouped-GEMM shapes,
expert batching, shared-expert fusion, reordering by (projection, expert), and
cross-row reuse factors, each with numbers and a verdict.

### 9.5 Not on this box — algorithm references to fetch

AITER, Composable Kernel (CK), FLA, SGLang, FlashInfer, EXL3/exllamav3 and
KTransformers are not present. Fetch them as **algorithm references**, never as
binaries or as a runtime dependency:

- **AITER / CK** — AMD's own kernel patterns for this ISA family. The single
  most relevant external source. Read for reduction structure, buffer-load usage,
  and MFMA-free paths. Note the standing project position: AITER is the algorithm
  reference, not a source of binaries.
- **FLA (flash-linear-attention)** — linear/GDN attention algorithms; relevant if
  Qwen3.8 uses any gated-delta or linear-attention component.
- **FlashInfer / SGLang** — attention kernel and scheduling structure. CUDA, so
  translate the ALGORITHM and re-derive the geometry for wave64/gfx906.
- **EXL3 / exllamav3** — quantized-weight kernel design. There is already a
  gfx906 EXL3 dequantizer in `glmflash-hip` to compare against.
- **KTransformers** — CPU/GPU-split MoE scheduling, relevant only if Qwen does
  not fit resident on four cards.

**Retrieval-first is a standing law on this programme.** Before deriving any
mechanism, search these sources and the Pangu receipts for whether it has already
been measured. Two Pangu optimization campaigns were saved by finding a proposed
mechanism already tried, with numbers.

## 10. Standing constraints for the Qwen build

1. **AMD/gfx906 is the target.** The MI50s are the product, not a porting
   exercise.
2. **The Spark CUDA engine is an oracle, NOT a bit-exact target.** Cross-backend
   token identity is explicitly not a contract. Pangu's own sequence diverges
   from its CUDA oracle after token 2 and that is fine.
3. **Custom AMD kernels are expected.** No generic GEMM/GEMV library in the
   production hot path. Specialize to the model's exact shapes.
4. **NVFP4 storage is not NVFP4 execution.** Section 3.1. Choose the best gfx906
   representation and arithmetic even where it looks nothing like the CUDA path.
5. **Re-prove Qwen's packing and scaling ABI from its own checkpoint.** Do not
   inherit Pangu's nibble order, scale layout, or `weight_scale_2` convention.
   Verify each against the Qwen artifact before writing a kernel against it.
6. **Optimize useful throughput and measured production wall time**, never a
   proxy. Every proxy tried on Pangu has now failed at least once: occupancy,
   register count, ISA op count, and static instruction count — the last with
   the wrong sign.

## 11. If you read only one thing

The Pangu engine went 34.60 -> 69.92 tok/s. The single largest rung, +5.36
percent, came from noticing that five loads guarded by `if (on)` compiled to five
serialized `s_waitcnt vmcnt(0)` barriers, and that inactive lanes only needed a
VALID ADDRESS because their values were already discarded.

It was found after the kernel had been declared closed, because every probe had
passed a mode parameter that selected the other branch.

Look at the ISA of the path production actually takes.


> **CORRECTION 2026-09-09.** Any statement in this document that MI50
> P2P is unavailable/dead at the driver is FALSE. It was true of Linux
> 5.15.0-190 only. Under Linux 6.8.0-138 the same hostile gate moves from
> 0/12 to 12/12 directed pairs with no collateral, because KFD creates
> `p2p_links` under 6.8 and created none under 5.15. Same cards, same ROCm
> 5.7.1 userspace, no ACS override. See receipts/07_P2P_DIAGNOSIS.md.
> The failure MODE described - peer pointers silently aliasing local memory
> instead of faulting - was real and is why a P2P path must be gated by a
> data-integrity check, never a capability bit.
