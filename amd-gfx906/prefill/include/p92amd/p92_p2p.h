// Producer-push peer transport for the four-GPU prefill path.
//
// SUPERSEDES the host-staging assumption in p92_transport.h. That header states
// "P2P is not available on this box ... peer copy must never be used here".
// That was true of Linux 5.15.0-190 and is FALSE on 6.8.0-138: the same hostile
// gate moves 0/12 -> 12/12 because KFD creates p2p_links under 6.8 and created
// none under 5.15. Same cards, same ROCm 5.7.1-98, no ACS override.
// See receipts/07_P2P_DIAGNOSIS.md and 16_P2P_ARCHITECTURE_DESIGN.md.
//
// THE RULE THIS HEADER EXISTS TO ENFORCE. Measured on 6.8, 16 MB, all twelve
// directed pairs, against a 653.6 GB/s same-device positive control:
//
//     peer WRITE   22.9 - 23.2 GB/s
//     peer READ     3.8 -  3.9 GB/s        5.86 - 6.09x slower
//     host ring    ~10    GB/s at large payloads
//
// A remote READ is 2.6x SLOWER than the host staging it would replace. So the
// producer PUSHES into the consumer's memory and the consumer reads locally at
// 653 GB/s. Never expose a peer pointer to a consumer kernel as an input.
//
// hipMemcpyPeer is NOT the primitive: 13.30 us for the 20 KB decode payload
// against the host path's 13.05. The primitive is a kernel storing into peer
// VRAM.
#ifndef P92AMD_P2P_H
#define P92AMD_P2P_H

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>

#define P92_P2P_NGPU 4

// Wall-clock bound for p92_p2p_wait. Must be short: the gate deliberately
// waits for a generation that never arrives, and a slow failure holds the
// machine lock. 2 s is far beyond any real boundary and fails a negative
// control promptly.
#define P92_P2P_WAIT_MS 2000

struct P92P2P {
    // dst[from][to] is a pointer, valid on device `from`, into device `to`'s
    // landing buffer. Producer-side only: a consumer never dereferences these.
    void   *dst[P92_P2P_NGPU][P92_P2P_NGPU];
    void   *land[P92_P2P_NGPU];         // this device's own landing buffer
    uint32_t *flag[P92_P2P_NGPU];       // arrival flags, device-local
    hipStream_t push[P92_P2P_NGPU];
    size_t  bytes;
    int     devices;
    int     enabled;                    // 0 until the integrity gate passes
};

// Enables peer access on every ordered pair, allocates one landing buffer and
// flag block per device, and RUNS THE INTEGRITY GATE. Returns 0 only if all
// N*(N-1) directed pairs demonstrably move real data. A capability bit is never
// trusted: under 5.15 every pair reported canAccessPeer=1 while nothing moved
// and peer pointers silently aliased sender-local memory.
int  p92_p2p_init(P92P2P *p, size_t bytes);
void p92_p2p_free(P92P2P *p);

// Producer push: device `from` writes `n` bytes from its local `src` into
// device `to`'s landing buffer at `off`, then raises to's arrival flag `slot`.
// Async on from's push stream. The store is a kernel, not hipMemcpyPeer.
int  p92_p2p_push(P92P2P *p, int from, int to, const void *src,
                  size_t off, size_t n, int slot, uint32_t generation);

// Consumer wait: spin on the local flag until `generation` arrives. Local read.
int  p92_p2p_wait(P92P2P *p, int dev, int slot, uint32_t generation);

// Hostile self-test: verifies the gate can DETECT a missing push, a stale
// generation and a wrong source. Returns 0 if every fault is caught.
int  p92_p2p_selftest(P92P2P *p);

#endif
