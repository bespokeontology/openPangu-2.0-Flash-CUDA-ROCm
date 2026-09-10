// Grouped MoE for openPangu prefill on gfx906.
//
// NOT the decode path widened. Decode dispatches 9 fixed slots for one token
// and is weight-bandwidth bound: 39.8 MB of expert weights per layer to serve a
// single row. Prefill inverts that. At T=512 a layer routes 512*8 = 4096
// (row, expert) pairs over at most 256 experts, so an expert averages 16 rows
// and its 4.42 MB of weights are amortised 16x. The kernel must therefore read
// each weight ONCE and apply it to every row routed to that expert.
//
// SCHEDULE, taken from glmflash-hip/src/hip/glm_expert_v2m_gfx906.hip:
// collapse M rows x top-k into unique-expert groups in ascending expert id,
// each carrying its row list, and load the group's weight pointers once. That
// grouping mechanism transfers directly. Its BODY does not: v2m_k1_row
// processes one row at a time and re-stages the activation into LDS per row,
// which is correct for decode and throws away the entire prefill win.
//
// WHAT REPLACES THE BODY. Each 32-lane group owns one output row of the expert
// and holds that row's 2560 packed nibbles in REGISTERS - 1280 B over 32 lanes
// is 40 B/lane, ten dwords - then sweeps every token in the group against it.
// Weight traffic per expert becomes independent of M_g. Activations stream from
// the chunk, which at 512 x 2560 int8 is 1.3 MB and L2-resident.
//
// Representation stays NVFP4 -> int8 dot4 (p92_dq4): 410.7 GB/s against 19.0
// for a float nibble decode, 21.6x, already proven on this hardware. Storage
// representation is not execution representation.
#ifndef P92AMD_PF_MOE_H
#define P92AMD_PF_MOE_H

#include <hip/hip_runtime.h>
#include <cstdint>

#define PF_T_MAX      1024   // largest prefill chunk
#define PF_NEXP        256   // routed experts per MoE layer
#define PF_TOPK          8
#define PF_SHARED_SLOT 256   // the always-on shared expert
#define PF_NEMBD      2560
#define PF_NFF        1024
#define PF_GROUP_MAX  (PF_NEXP + 1)

// One expert's work for a chunk: which rows routed to it, and their weights.
// Row lists live in a separate flat array so the struct stays small enough to
// keep the group table in cache; nrows==0 groups are skipped by the grid.
struct PfGroup {
    int32_t eid;        // arena slot for this expert, -1 if unused
    int32_t nrows;      // rows in this chunk routed here
    int32_t first;      // offset into the shared row/weight arrays
};

struct PfRouting {
    PfGroup *groups;    // [PF_GROUP_MAX]
    int32_t *rows;      // [T * (PF_TOPK+1)] token index per (group, slot)
    float   *wts;       // [T * (PF_TOPK+1)] routing weight, parallel to rows
    int32_t *ngroups;   // device scalar: groups with nrows > 0
    int32_t *order;     // [PF_GROUP_MAX] compacted list of live group indices
};

// Device-side grouping. One block. Counts rows per expert, prefix-sums to get
// each group's offset, then scatters (row, weight) pairs into the flat arrays.
// Ascending expert id, so the arena is walked in address order.
__global__ void k_pf_group(const int32_t *__restrict__ ids,
                           const float *__restrict__ wts,
                           int T, int slot_base, int experts,
                           PfGroup *__restrict__ groups,
                           int32_t *__restrict__ rows,
                           float *__restrict__ rwt,
                           int32_t *__restrict__ order,
                           int32_t *__restrict__ ngroups);

// gate/up + SwiGLU for every row of every live group.
// grid (PF_NFF/8, ngroups), block 256. A 32-lane group owns one output row and
// holds its packed weights in registers across all rows of the expert.
__global__ void k_pf_gateup(const uint8_t *__restrict__ Wg, const uint8_t *__restrict__ Sg,
                            const uint8_t *__restrict__ Wu, const uint8_t *__restrict__ Su,
                            const int8_t *__restrict__ xq, const float *__restrict__ xs,
                            const float *__restrict__ s2,
                            const PfGroup *__restrict__ groups,
                            const int32_t *__restrict__ order,
                            const int32_t *__restrict__ rows,
                            float *__restrict__ hidden);

#endif
