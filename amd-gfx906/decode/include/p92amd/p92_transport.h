// Cross-card transport for the 12/12/11/11 layer split.
//
// P2P is not available on this box, and that is measured rather than inferred:
// all 12 ordered pairs report canAccessPeer 1 and accept hipDeviceEnablePeerAccess,
// and hipMemcpyPeer returns success, but 0 of 12 actually move bytes - a
// destination pre-filled with 0x11 still reads 0x11 after copying a buffer of
// 0xAB. KFD reports io_links 1 and p2p_links 0 on every node, and lspci shows
// each card behind its own bridge chain on a separate root port. Silent data
// loss, so peer copy must never be used here.
//
// Host staging is the path, and it is much cheaper than the figure previously
// recorded. Measured device 0 to device 1, pinned, for the 20 KB payload:
//   blocking 15.88 us   async 15.62 us   overlapped 10.03 us
// against a recorded 70-90 us that bundled blocking copies and hipSetDevice.
// Cost is latency-bound, not bandwidth-bound: 5 KB and 20 KB cost the same, so
// there is no reason to send anything separately that can travel together.
//
// WHAT CROSSES. By the oracle's layer order the carried state between layers is
// the four-stream mHC block, current_streams / other_streams, 4 x 2560 bf16 =
// 20,480 B. The residual IS those streams; there is no second payload. KV stays
// on the device that owns its layers. So a token makes 3 crossings of 20,480 B,
// about 30 us overlapped, which is 0.15 percent of a 20.69 ms token.
#ifndef P92AMD_TRANSPORT_H
#define P92AMD_TRANSPORT_H

#include <hip/hip_runtime.h>
#include <cstddef>

#define P92_TX_NGPU     4
#define P92_TX_BOUNDS   (P92_TX_NGPU - 1)      // 3 crossings a token
#define P92_TX_STREAMS  4
#define P92_TX_HIDDEN   2560
#define P92_TX_BYTES    ((size_t)P92_TX_STREAMS * P92_TX_HIDDEN * 2)   // 20,480

struct P92Transport {
    void        *stage[P92_TX_BOUNDS];         // pinned host, one per boundary
    hipStream_t  copy[P92_TX_NGPU];            // one copy stream per device
    size_t       bytes;
    int          devices;
};

// Allocates one pinned staging buffer per boundary and one copy stream per
// device. bytes defaults to the four-stream block when passed 0.
int  p92_transport_init(P92Transport *t, size_t bytes);
void p92_transport_free(P92Transport *t);

// Push device `from`'s carried state into the boundary buffer. Async on that
// device's copy stream.
int  p92_transport_send(P92Transport *t, int from, const void *src);

// Pull the boundary buffer into device `from + 1`. Async on that device's copy
// stream. Call p92_transport_wait on the receiving device before using dst.
int  p92_transport_recv(P92Transport *t, int from, void *dst);

int  p92_transport_wait(P92Transport *t, int device);

#endif
