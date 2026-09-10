# CORRECTION 2026-09-09: the conclusion was FALSE. P2P works on these MI50s.

> **This document correctly describes the behaviour of Linux 5.15.0-190 on this
> machine. Its CONCLUSION - that P2P is unavailable on these MI50s - is false,
> and was a statement about a software stack mistaken for a statement about
> hardware.**
>
> On the same four cards, with the same hostile gate binary:
>
>     Linux 5.15.0-190    0/12 directed pairs. Peer operations produced
>                         destructive sender-local aliasing / shadow behaviour.
>     Linux 6.8.0-138    12/12 directed pairs are true P2P, zero unintended
>                         collateral.
>
> The mechanism: **KFD creates `p2p_links` under 6.8 and created none under
> 5.15.** No ACS override, no custom kernel, and no change to ROCm userspace
> (5.7.1-98) was required.
>
> Two specific claims below are therefore wrong or unsupported:
>
> 1. **"Not the platform"** - it was precisely the platform, in the one place
>    not examined: the kernel's KFD P2P topology. Every layer above it was
>    checked and exonerated; the layer that was actually broken was never
>    inspected because `p2p_links` was not known to be the thing to look at.
> 2. **"Not ACS routing. No bridge above any of the four GPUs advertises an
>    Access Control Services capability."** That inspection was run without
>    root. Non-root `lspci` suppresses extended capability information, so
>    "not visible" was recorded as "not present". The claim is unsupported as
>    written, independent of whether ACS mattered.
>
> What survives unchanged is the *diagnostic work*: the discriminator that
> bypassed the copy API with a peer-dereferencing kernel, and the finding that
> peer pointers silently aliased local memory rather than faulting. That failure
> mode is real, was correctly characterised, and is the reason a P2P path must
> always be gated by a data-integrity check and never by a capability bit. Under
> 5.15 `hipDeviceCanAccessPeer` returned 1 on all twelve pairs while nothing
> moved.
>
> **Method law this cost us**, now banked in `11_PROFILING_LAW.md`:
>
> > A null result is not evidence of absence until the instrument has
> > demonstrated it can produce and distinguish a non-null result.
>
> Both failures here were instrument failures. The identity-offset peer tests
> could not distinguish self-aliasing from a no-op, and the non-root PCIe
> inspection could not distinguish an absent capability from a hidden one.
> Neither had a positive control.
>
> **Consequences for the engine.** openPangu's 12/12/11/11 sequential layer
> ownership and its pinned-host transport ring (`src/p92_transport.cpp`) were
> both chosen under the constraint this document asserts. That constraint is
> gone. See the 6.8 + P2P architecture design lane.
>
> Documents that repeat the old conclusion and are annotated rather than
> rewritten: `00_PORT_MAP.md`, `05_STATE.md`, `15_QWEN_TRANSFER_PACKAGE.md`,
> `ENGINEERING_REPORT.md`, and `/data/QWEN_GFX906_TRANSFER_PACKAGE.md`.

---

# MI50 P2P: diagnosed, not written off

The symptom was internally inconsistent with the documented API: all 12 ordered
pairs report `hipDeviceCanAccessPeer` 1, accept `hipDeviceEnablePeerAccess`, and
return success from `hipMemcpyPeer`, while the destination keeps its pre-fill
byte. That is a broken path, not absent hardware, so it was diagnosed rather
than assumed.

## What was ruled out

**Not a HIP copy-API bug.** The discriminator was to bypass the copy API: a
kernel on device A dereferencing an allocation owned by device B.

    remote STORE, A-side kernel writes B's buffer   0/12 pairs
    remote LOAD,  A-side kernel reads  B's buffer   0/12 pairs
    hipMemcpyPeer / PeerAsync / D2D / Default       none move data

The read test is the informative one. It read back `0xAB000000`, which was the
*write* pattern from the preceding store test, not the `0xCD` seed it had just
put in B's buffer. B's allocation had been freed and reissued at the same
address, so the A-side kernel was hitting **A's own local memory at that virtual
address**. Peer pointers are not mapped into the other device's address space,
and dereferencing them silently aliases local memory. No fault, wrong data.

**Not ACS routing.** No bridge above any of the four GPUs advertises an Access
Control Services capability, so there is nothing to redirect peer transactions
upstream.

**Not BAR addressability.** Large BAR is enabled: each card exposes
`Region 0: 64-bit prefetchable [size=16G]` at a 64-bit address, so the whole
framebuffer is mappable.

**Not fine-grain PCIe mode.** AMD documents `HSA_FORCE_FINE_GRAIN_PCIE=1` as
required for PCIe P2P transport. It was unset; setting it changes nothing,
0/12 on every test.

**Not the platform.** EPYC Zen 2, four root ports, no IOMMU arguments on the
command line. rocm-smi reports PCIE link type, weight 40 and 2 hops between all
pairs, so ROCm itself believes the pairs are connected.

## The actual cause

The running kernel is built without every option PCIe P2P DMA needs.

    kernel   5.15.0-190-generic, stock Ubuntu, inbox amdgpu
    ROCm     5.7.1-98

    CONFIG_PCI_P2PDMA          not set
    CONFIG_DMABUF_MOVE_NOTIFY  not set
    CONFIG_HSA_AMD_P2P         not set
    CONFIG_ZONE_DEVICE         y

That accounts for every symptom at once. HIP's capability flag and
`enablePeerAccess` are runtime bookkeeping and succeed regardless; the KFD
cannot create the peer mapping, so the copy silently does nothing and remote
pointers alias.

This is consistent with ROCm issue 4793, "MI50 32GB p2p not working", which
reports the same failure on gfx906 across ROCm 5.7.1 and 6.4 and is still
open.

## The smallest change that would test it

Boot a kernel built with `CONFIG_PCI_P2PDMA=y` and `CONFIG_DMABUF_MOVE_NOTIFY=y`
- Ubuntu HWE 6.x carries these - with the DKMS amdgpu stack rather than the
inbox module, then re-run `tests/p92_p2p_diag.hip` unchanged. It reports all 12
directed pairs for remote store, remote load and all three copy APIs, so it
answers the question in one run.

Nothing was changed. The box runs production Qwen and boots to Windows on a
crash, so a kernel change is an operator decision, not a side effect of a
diagnosis.

## What it is worth

Almost nothing for this model, which is why it did not block assembly.

    3 crossings a token, host staged, serial     48.19 us   0.048 ms/token
    hidden behind concurrent compute             39.02 us
    effective                                    ~0.009 ms/token

Against a 20.69 ms token, perfect zero-cost P2P could recover at most 0.23
percent, and 0.04 percent of what is not already hidden. openPangu makes three
crossings of 20,480 B; the payload is far below the point where bandwidth
matters, and the cost is latency the staging path already absorbs.

The reason to fix it is not openPangu. It is that a tensor-parallel model with
per-layer collectives would cross far more often and far wider, and this box
would then be crippled by a missing kernel option.
