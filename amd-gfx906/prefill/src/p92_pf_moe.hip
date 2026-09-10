#include "p92amd/p92_pf_moe.h"
#include "../../decode/src/p92_nvfp4_dot4.hip"   // p92_dq4, p92_group_dot4, p92_ue4m3_d

// ---- device-side grouping --------------------------------------------------
// One block. Count rows per expert, exclusive-scan, scatter (row, weight).
// Ascending expert id, so the arena is walked in address order.
__global__ void __launch_bounds__(256)
k_pf_group(const int32_t *__restrict__ ids, const float *__restrict__ wts,
           int T, int slot_base, int experts,
           PfGroup *__restrict__ groups, int32_t *__restrict__ rows,
           float *__restrict__ rwt, int32_t *__restrict__ order,
           int32_t *__restrict__ ngroups)
{
    __shared__ int32_t cnt[PF_GROUP_MAX];
    __shared__ int32_t off[PF_GROUP_MAX];
    const int t = threadIdx.x;
    for (int e = t; e < PF_GROUP_MAX; e += 256) cnt[e] = 0;
    __syncthreads();
    // count: every token contributes topk routed experts plus the shared one
    for (int i = t; i < T * PF_TOPK; i += 256) {
        const int e = ids[i];
        if (e >= 0 && e < PF_NEXP) atomicAdd(&cnt[e], 1);   // never fault on a bad id
    }
    for (int r = t; r < T; r += 256)          atomicAdd(&cnt[PF_SHARED_SLOT], 1);
    __syncthreads();
    // exclusive scan, one thread: PF_GROUP_MAX is 257, the scan is trivial
    // against the GEMM that follows and a parallel scan would cost more in
    // barriers than it saves.
    // DEAD-SLOT CONTRACT (09-09): consumers launch a FIXED PF_GROUP_MAX grid,
    // so every order slot past the live list must deterministically name a
    // group with nrows == 0. Slot PF_GROUP_MAX is pinned dead, order[] is
    // initialised to it, and the live scan overwrites the prefix. This removes
    // the blocking D2H read of the live-group count (a drained-queue host sync
    // per MoE layer) without touching the kernels' early-outs.
    for (int e = t; e < PF_GROUP_MAX; e += 256) order[e] = PF_GROUP_MAX;
    if (t == 0) {
        int run = 0, live = 0;
        for (int e = 0; e < PF_GROUP_MAX; e++) {
            off[e] = run; run += cnt[e];
            groups[e].eid   = cnt[e] ? (slot_base + e) : -1;
            groups[e].nrows = cnt[e];
            groups[e].first = off[e];
            if (cnt[e]) order[live++] = e;
        }
        groups[PF_GROUP_MAX].eid   = -1;
        groups[PF_GROUP_MAX].nrows = 0;
        groups[PF_GROUP_MAX].first = 0;
        *ngroups = live;
    }
    __syncthreads();
    for (int e = t; e < PF_GROUP_MAX; e += 256) cnt[e] = 0;
    __syncthreads();
    // scatter
    for (int i = t; i < T * PF_TOPK; i += 256) {
        const int e = ids[i];
        if (e < 0 || e >= PF_NEXP) continue;
        const int k = off[e] + atomicAdd(&cnt[e], 1);
        rows[k] = i / PF_TOPK;
        rwt[k]  = wts[i];
    }
    for (int r = t; r < T; r += 256) {
        const int k = off[PF_SHARED_SLOT] + atomicAdd(&cnt[PF_SHARED_SLOT], 1);
        rows[k] = r;
        rwt[k]  = 1.0f;                      // shared expert is unweighted
    }
}

// ---- gate/up + SwiGLU, weights resident across the group's rows ------------
// grid (PF_NFF/8, ngroups), block 256. Eight 32-lane groups per block, one
// output row each. THE POINT: the row's 2560 packed nibbles are loaded ONCE
// into registers - 40 B/lane, five uint2 for gate and five for up - and swept
// against every token routed to this expert. Weight traffic is independent of
// the group size, which is the entire prefill win over the decode path.
//
// LOAD DISCIPLINE. Every load in the token loop is issued before any is
// consumed, and the four activation words of a 16-group are ONE uint4 rather
// than four dwords pretending to be separate. A hoisted weight load sitting
// next to un-hoisted activation and scale loads reads as done and is not.
__global__ void __launch_bounds__(256)
k_pf_gateup(const uint8_t *__restrict__ Wg, const uint8_t *__restrict__ Sg,
            const uint8_t *__restrict__ Wu, const uint8_t *__restrict__ Su,
            const int8_t *__restrict__ xq, const float *__restrict__ xs,
            const float *__restrict__ s2,
            const PfGroup *__restrict__ groups, const int32_t *__restrict__ order,
            const int32_t *__restrict__ rows, float *__restrict__ hidden)
{
    const int g   = order[blockIdx.y];
    const PfGroup grp = groups[g];
    if (grp.nrows <= 0) return;
    const int s   = grp.eid;
    const int l   = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);

    const uint8_t *wg = Wg + (size_t)s * P92_GU_W_BYTES + (size_t)row * (PF_NEMBD / 2);
    const uint8_t *wu = Wu + (size_t)s * P92_GU_W_BYTES + (size_t)row * (PF_NEMBD / 2);
    const uint8_t *sg = Sg + (size_t)s * P92_GU_S_BYTES + (size_t)row * (PF_NEMBD / 16);
    const uint8_t *su = Su + (size_t)s * P92_GU_S_BYTES + (size_t)row * (PF_NEMBD / 16);

    // ---- hoist the ENTIRE weight row into registers, once ------------------
    uint2 pg[5], pu[5];
    float dg[5], du[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) { const int q = i * 32 + l;
        pg[i] = *(const uint2 *)(wg + (size_t)q * 8);
        pu[i] = *(const uint2 *)(wu + (size_t)q * 8); }
    // Load the scale BYTES first, decode them second. p92_ue4m3_d branches, so
    // decoding inline put every one-byte load in its own exec-masked region and
    // the compiler fenced each with s_waitcnt vmcnt(0) - ten serialized round
    // trips inside a loop this file claims is hoisted. Splitting load from
    // decode leaves the branches operating on registers.
    // Scales arrive PRESHUFFLED at load time so this lane's five are contiguous
    // (k_pf_shuffle_scales_gu). Five strided one-byte loads become one 32-bit
    // plus one 8-bit, which is what actually removes the fences rather than
    // halving them - the branchy UE4M3 decode then operates purely on registers.
    uint8_t sgb[5], sub[5];
    {   const uint32_t g0 = *(const uint32_t *)(sg + l * 5);
        const uint32_t u0 = *(const uint32_t *)(su + l * 5);
        sgb[0]=(uint8_t)g0; sgb[1]=(uint8_t)(g0>>8); sgb[2]=(uint8_t)(g0>>16); sgb[3]=(uint8_t)(g0>>24);
        sub[0]=(uint8_t)u0; sub[1]=(uint8_t)(u0>>8); sub[2]=(uint8_t)(u0>>16); sub[3]=(uint8_t)(u0>>24);
        sgb[4]=sg[l*5+4]; sub[4]=su[l*5+4]; }
    #pragma unroll
    for (int i = 0; i < 5; i++) { dg[i] = p92_ue4m3_d(sgb[i]); du[i] = p92_ue4m3_d(sub[i]); }

    const float f2g = 0.5f * s2[(size_t)s * 3 + 0];
    const float f2u = 0.5f * s2[(size_t)s * 3 + 1];

    // ---- sweep every token routed to this expert ---------------------------
    for (int r = 0; r < grp.nrows; r++) {
        const int tok = rows[grp.first + r];
        const int8_t *x = xq + (size_t)tok * PF_NEMBD;
        const float  *a = xs + (size_t)tok * (PF_NEMBD / 16);
        // every load first, then every consumption
        uint4 xv[5]; float av[5];
        #pragma unroll
        for (int i = 0; i < 5; i++) xv[i] = *(const uint4 *)(x + (size_t)(i * 32 + l) * 16);
        {   // scales arrive lane-contiguous from k_pf_quant: two loads, not five
            const float4 s4 = *(const float4 *)(a + l * 5);
            av[0]=s4.x; av[1]=s4.y; av[2]=s4.z; av[3]=s4.w; av[4]=a[l*5+4]; }
        float ag = 0.f, au = 0.f;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            const uint2 xe = make_uint2(xv[i].x, xv[i].y);
            const uint2 xo = make_uint2(xv[i].z, xv[i].w);
            ag = fmaf(dg[i] * av[i], (float)p92_group_dot4(pg[i], xe, xo), ag);
            au = fmaf(du[i] * av[i], (float)p92_group_dot4(pu[i], xe, xo), au);
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) {
            ag += __shfl_down(ag, off, 64);
            au += __shfl_down(au, off, 64);
        }
        if (l == 0) {
            const float gv = ag * f2g, uv = au * f2u;
            hidden[(size_t)(grp.first + r) * PF_NFF + row] = gv / (1.f + __expf(-gv)) * uv;
        }
    }
}
