# Producer-push peer transport: gated, 2026-09-09

Foundation for the four-GPU prefill path. Frozen decode engine untouched.
Tree: <data>/p92-prefill. Kernel 6.8.0-138, ROCm 5.7.1-98,
four MI50s at profile_peak 1700 MHz. Run under the shared machine lock
`~/q27bench`, which serialises benchmarks between the two lanes.

## Result

    p92_p2p: integrity gate 12/12 directed pairs, enabled
    positive  correct push 0->1                     verified
    negative  missing push detected                 yes
    negative  stale gen 3 rejected as gen 7         yes
    negative  wrong source (dev2 not dev0) detected yes
    negative  partial push detected                 yes
    all pairs source-tagged and verified            12/12
    PASS

## What it establishes

The transport implements the data-plane rule from receipts/16: a kernel on the
producer STORES into the consumer's landing buffer (measured 23.1 GB/s), and the
consumer reads its own buffer locally (653.6 GB/s). A consumer never
dereferences a peer pointer, because peer READ measured 3.85 GB/s - 2.6x slower
than the host staging it replaces. `hipMemcpyPeer` is not used anywhere: it
measured 13.30 us at the 20 KB payload against host staging's 13.05.

`p92_p2p_init` refuses to enable unless all N*(N-1) directed pairs demonstrably
move data. A capability bit is never trusted - under 5.15 every pair reported
canAccessPeer=1 while nothing moved and peer pointers silently aliased
sender-local memory.

## Why the negative controls are the point

A passing output proves nothing on its own; that is this project's most
expensive lesson. Each fault the transport can suffer has a control that must
fail, and the payload encodes (generation, source, index) so the failures are
distinguishable from each other rather than all reading as "unchanged":

    missing push       flag never advances; wait times out
    stale generation   gen 3 payload must not verify as gen 7
    wrong source       dev2's push must not verify as dev0's
    partial push       half a payload must not verify as whole

All four fire.

## A defect found by the gate, in the gate

The first run hung for five minutes holding the machine lock. `p92_p2p_wait`
bounded itself with a SPIN COUNT of 200,000,000 hipMemcpy polls - about 16
minutes - and the "missing push" control waits by construction for a generation
that never arrives. A spin count is not a timeout.

Now wall-clock bounded at P92_P2P_WAIT_MS = 2000. **A negative control must fail
fast**, because the failure path is the one it always takes.

Killed by exact PID, never by pattern. The peer lane lost its own SSH session
the same day to a `pkill` whose pattern matched the command line carrying it.

## Known issue, carried forward

`p92_p2p_wait` polls host-side with a hipMemcpy per iteration. Adequate for a
correctness gate, far too expensive for a per-layer boundary. The intended
primitive is `hipStreamWaitValue32`, so the consumer's stream waits on the flag
with no host involvement. Correctness first; that is a measurement-phase change.

## Scope discipline

This window was granted seconds of GPU time to establish the transport
foundation and nothing else. The threshold suite (511/512/513, 2047/2048/2049,
16383/16384/16385, cap+-1) was NOT run and remains queued. Cards handed back
idle immediately after this gate.
