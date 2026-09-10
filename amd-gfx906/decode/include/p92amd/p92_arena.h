// Resident routed-expert arena: one file per GPU, six contiguous planes plus a
// per-expert weight_scale_2 table. Built offline by tools/p92_pack_arena.cpp.
// Slot addressing is slot = local_moe_index * 256 + expert, which is what the
// gate/up and down kernels index with.
#ifndef P92AMD_ARENA_H
#define P92AMD_ARENA_H

#include <hip/hip_runtime.h>
#include <cstdint>

#pragma pack(push,1)
struct P92ArenaHdr {
    char     magic[8];              // "P92AREN"
    uint32_t version, device, layer_begin, layer_end;
    uint32_t moe_layers, experts, pad;
    uint64_t off_gw, off_uw, off_dw, off_gs, off_us, off_ds, off_s2, total;
};
#pragma pack(pop)

struct P92Arena {
    P92ArenaHdr h;
    uint8_t *gw, *uw, *dw;          // device: packed E2M1 nibbles
    uint8_t *gs, *us, *ds;          // device: UE4M3 block scales
    float   *s2;                    // device: [slots][3] gate, up, down
    int      slots;
};

// Loads arena_dev<device>.bin onto the current HIP device. Returns 0 on
// success. Reads each plane in one call; no per-tensor walk at boot.
int  p92_arena_load(const char *path, P92Arena *out);
void p92_arena_free(P92Arena *a);

// slot for a trunk layer index and expert id, or -1 if the layer is not owned
// by this arena or is one of the two dense layers.
static inline int p92_arena_slot(const P92Arena *a, int layer, int expert) {
    if (layer < (int)a->h.layer_begin || layer >= (int)a->h.layer_end || layer < 2) return -1;
    const int li = layer - (a->h.layer_begin < 2 ? 2 : (int)a->h.layer_begin);
    return li * (int)a->h.experts + expert;
}

// The shared expert has exactly one routed expert's shapes, so it lives in the
// same arena as slot 256 of its layer and is dispatched as a ninth expert with
// route weight 1.0. The reference adds it unweighted:
//   final_hidden_states = final_hidden_states + shared_output
// and its intermediate size is moe_intermediate_size * n_shared_experts,
// which is 1024 * 1.
#define P92_SHARED_INDEX 256
static inline int p92_arena_shared_slot(const P92Arena *a, int layer) {
    return p92_arena_slot(a, layer, P92_SHARED_INDEX);
}

#endif
