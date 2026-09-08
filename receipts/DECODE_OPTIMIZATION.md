# Decode optimization receipt — 2026-08-22

This pass profiled one warmed selective-mode target token with Nsight Systems,
then changed only measured CUDA bottlenecks. The frozen resident baseline under
`frozen/native-target-prefill-dsa-20260822` was not modified.

## Starting profile

Command target: `p92_decode_profile`, context 64, input token 2772, expected
next token 5015. The target projection working set was about 4.06 GB.

- Host wall: 93.7866 ms, 10.6625 tok/s.
- `mhc_pre_kernel`: 24.757 ms across 92 calls (33.9% of summed GPU time).
- NVFP4 CUTLASS projections: 22.134 ms across 1,365 calls.
- latent attention: 9.831 ms across 46 calls.
- router top-8: 4.597 ms across 44 calls.
- 44 routed-expert ID reads are pageable device-to-host copies followed by a
  stream synchronization; together with 16 DSA key copies, embedding upload,
  and argmax read, Nsight attributes 58.35 ms of CUDA API time to 62 async
  copies. This host route fence is the largest remaining execution-structure
  problem.

## Exact-order parallel mHC

The original mHC kernel projected 24 coefficient rows with only eight warps in
one block, serializing three waves inside every call. The new path launches 24
independent one-warp blocks per activation row. Each warp retains the original
lane-stride accumulation and shuffle order. Norm, coefficient/Sinkhorn, and
mixed-output work are separate kernels with persistent model-owned scratch.

An initial 256-thread reduction was rejected: its local cosine was 1.0 but the
full target's serial BOS branch changed. The shipped one-warp reduction has
zero coefficient delta against the old kernel and restores the prior target
sequence.

- mHC state gate: cosine 1, coefficient delta 0.
- `mhc_phi_parallel_kernel`: 6.098 ms across 92 calls, down from 24.757 ms.
- Full target gate after this cut: `148899 -> 2772 -> 5015`.

## Parallel router selection

The old router stored two 256-float arrays in thread-local memory and selected
eight experts serially in thread 0. The replacement computes sigmoid and bias
in 256 threads, stores scores in shared memory, and performs deterministic
block-wide max reductions for each of eight slots.

- Exact route gate: `123,204,232,142,203,11,178,3`, weights sum to 2.5.
- Full target sequence remains deterministic.
- Warm target wall after mHC plus router: 64.147 ms, 15.5892 tok/s.

## Tiled online MLA attention

The reference decode kernel applied two block synchronizations for every key.
Even at position 1 it always attends 128 learned sinks plus the live cache key,
costing about 214 us per layer. The new kernel scores keys with eight warps and
updates online softmax once per 64-key tile. It supports linear, circular SWA,
and indexed DSA caches and is also used by causal row-bank prefill.

- Direct A/B gate at 129 keys: cosine 1.0, maximum BF16 latent delta
  4.76837e-7.
- Serial and block-prefill execution agree after the change.
- The artificial BOS-only greedy branch changes to `148899 -> 14627 -> 287`;
  exact-token identity was not made a requirement for this optimized kernel.
- A resident normal-prompt gate was therefore run. Prompt:
  `Explain probability in one clear sentence.` The engine produced coherent
  reasoning beginning `First, the user asked...`; 26 prompt tokens prefilling
  took 0.381005 s and 32 generated tokens decoded at 15.7561 tok/s.
- Warm selective target wall: 56.1571 ms, 17.8072 tok/s.

The resident validation process exited and GPU process accounting was empty
afterward.

## Net result and next bottleneck

Measured warm target decode improved from 93.7866 ms / 10.6625 tok/s to
56.1571 ms / 17.8072 tok/s: 40.0% lower wall time and 67.0% higher throughput.
Interactive resident decode measured 15.7561 tok/s over 32 tokens.

## Device-scheduled grouped routed experts

The resident model now binds a compact device table containing each routed
expert's packed weight, swizzled block scales, and F32 dequantization scalar.
The router's device-resident top-8 IDs index that table directly. No selected ID
is copied to the host in the resident path.

Gate and up execute as one 16-problem SM121 CUTLASS pointer-array grouped
NVFP4 GEMM. SwiGLU consumes the resulting eight row pairs on the device. Down
executes as one eight-problem grouped GEMM, and a CUDA combine kernel applies
the device-resident route weights. The implementation uses the Blackwell
SM120/SM121 block-scaled tensor-op path with per-problem alpha pointers; the
selective loader retains its original serial admission path.

The low-memory A/B gate loaded only the eight layer-2 experts selected by the
real router:

- Route: `123,204,232,142,203,11,178,3`, total weight `2.5`.
- Grouped versus serial routed output: cosine `0.999992`, maximum BF16
  difference `0.000732422`.
- Serial routed chain: `0.580936 ms`; grouped chain: `0.194131 ms`; speedup
  `2.99249x`.
- Selective serial and two-row prefill paths still agree after reset:
  `148899 -> 14627 -> 287`.

The full 56.05 GB resident gate then executed the new branch and exited cleanly:

- Resident bytes: `56,051,047,104`; auxiliary bytes: `831,547,250`.
- Second target token: `44.126 ms`, or `22.6624 tok/s`.
- Normal prompt: `Explain probability in one clear sentence.`
- Prefill: 26 tokens in `0.395138 s`; generation: 48 tokens at
  `21.2046 tok/s`; output was coherent and began by restating the requested
  one-sentence task before answering it.
- GPU process accounting was empty after both resident tests.

The target engine has therefore moved from 10.66 tok/s before decode work to
roughly 21–23 tok/s on the working resident path. The three shipped predictor
layers are the next major throughput lever: execute their proposal state on the
device and verify accepted prefixes with the existing M<=8 target block path.
That work must report target steps/s, accepted tokens/step, and effective
emitted tok/s separately.
