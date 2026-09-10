// p92_placement.h - layer ownership and cross-card transport for 4x MI50.
//
// Ownership replaces the Qwen scheme, which computed layers per GPU as
//   v = total; v -= v % QF4_NGPU;
// That truncation is correct only when the layer count divides the GPU count.
// openPangu has 46 trunk layers on 4 cards: 46 - (46 % 4) = 44, silently
// dropping layers 44 and 45. A model missing two trunk layers is not the
// model. Here every layer is owned exactly once and the remainder is
// distributed, giving 12/12/11/11.
//
// The split is not arbitrary. DSA layers are those with index % 3 == 0 and
// are the only ones holding full-context KV. 12/12/11/11 puts exactly 4 of
// them on each card, so KV grows evenly; 13/11/11/11 balances weights better
// but gives 5/3/4/4 and costs more at long context than it saves.
//
// Transport: this platform has no working GPU-to-GPU path. Measured on the
// box: hipDeviceCanAccessPeer reports 1 and hipDeviceEnablePeerAccess
// succeeds for all 12 ordered pairs, but hipMemcpyPeer returns success while
// leaving the destination unmodified, and KFD reports io_links 0 with no
// p2p_links on all four nodes. Crossings therefore stage through pinned host
// memory. Measured cost for the 20,480 B mHC activation: 16.25 us pinned
// against 21.08 us pageable, so roughly 53 us per token for three crossings.
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define P92_TRUNK_LAYERS 46
#define P92_NGPU 4

// Layers owned by device d: [p92_layer_begin(d), p92_layer_end(d)).
// Remainder layers go to the lowest-numbered devices, giving 12/12/11/11.
static inline int p92_layer_begin(int d) {
    const int base = P92_TRUNK_LAYERS / P92_NGPU;   // 11
    const int rem  = P92_TRUNK_LAYERS % P92_NGPU;   // 2
    return d * base + (d < rem ? d : rem);
}
static inline int p92_layer_end(int d) {
    const int base = P92_TRUNK_LAYERS / P92_NGPU;
    const int rem  = P92_TRUNK_LAYERS % P92_NGPU;
    return (d + 1) * base + ((d + 1) < rem ? (d + 1) : rem);
}
static inline int p92_layer_count(int d) { return p92_layer_end(d) - p92_layer_begin(d); }
static inline int p92_layer_owner(int layer) {
    for (int d = 0; d < P92_NGPU; d++)
        if (layer >= p92_layer_begin(d) && layer < p92_layer_end(d)) return d;
    return -1;
}
// A DSA layer holds full-context KV; sliding layers hold a 512 window.
static inline int p92_layer_is_dsa(int layer) { return (layer % 3) == 0; }

#ifdef __cplusplus
}
#endif
