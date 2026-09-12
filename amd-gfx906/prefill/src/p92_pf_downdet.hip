// Deterministic MoE down projection for openPangu prefill.
//
// k_pf_down scatters each (expert, token) contribution into the token's hidden row with
// fp32 atomicAdd, so the summation order across a token's nine experts varies run to run
// and a near-tied argmax flips between runs (two distinct greedy continuations of the same
// 1117-token prompt were observed on 2026-09-12). Here every grouped row writes its own
// partial (no atomics) and a combine adds a token's partials in a FIXED order: the eight
// routed experts in routing order, then the shared expert. Traffic: nrows x 2560 floats
// written and read once (23 MB a chunk-layer at T=256), against the atomics it retires.
#include <hip/hip_runtime.h>
#include <cstdint>
#include "p92amd/p92_pf_moe.h"

#define PF_TOKSLOTS (PF_TOPK + 1)

// k_pf_group's scatter, extended: tokslot[t * 9 + j] = the grouped row index of token t's
// j-th expert (j < 8 routed in routing order, j = 8 the shared expert).
__global__ void __launch_bounds__(256)
k_pf_group_det(const int32_t *__restrict__ ids, const float *__restrict__ wts,
               int T, int slot_base, int experts,
               PfGroup *__restrict__ groups, int32_t *__restrict__ rows,
               float *__restrict__ rwt, int32_t *__restrict__ order,
               int32_t *__restrict__ ngroups, int32_t *__restrict__ tokslot)
{
    __shared__ int32_t cnt[PF_GROUP_MAX];
    __shared__ int32_t off[PF_GROUP_MAX];
    const int t = threadIdx.x;
    for (int e = t; e < PF_GROUP_MAX; e += 256) cnt[e] = 0;
    __syncthreads();
    for (int i = t; i < T * PF_TOPK; i += 256) {
        const int e = ids[i];
        if (e >= 0 && e < PF_NEXP) atomicAdd(&cnt[e], 1);
    }
    for (int r = t; r < T; r += 256) atomicAdd(&cnt[PF_SHARED_SLOT], 1);
    __syncthreads();
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
        groups[PF_GROUP_MAX].eid = -1; groups[PF_GROUP_MAX].nrows = 0; groups[PF_GROUP_MAX].first = 0;
        *ngroups = live;
    }
    __syncthreads();
    for (int e = t; e < PF_GROUP_MAX; e += 256) cnt[e] = 0;
    __syncthreads();
    for (int i = t; i < T * PF_TOPK; i += 256) {
        const int e = ids[i];
        const int tok = i / PF_TOPK, j = i - tok * PF_TOPK;
        if (e < 0 || e >= PF_NEXP) { tokslot[tok * PF_TOKSLOTS + j] = -1; continue; }
        const int k = off[e] + atomicAdd(&cnt[e], 1);
        rows[k] = tok; rwt[k] = wts[i];
        tokslot[tok * PF_TOKSLOTS + j] = k;
    }
    for (int r = t; r < T; r += 256) {
        const int k = off[PF_SHARED_SLOT] + atomicAdd(&cnt[PF_SHARED_SLOT], 1);
        rows[k] = r; rwt[k] = 1.0f;
        tokslot[r * PF_TOKSLOTS + PF_TOPK] = k;
    }
}

// k_pf_down with the scatter replaced by a per-grouped-row partial (routing weight folded in).
__global__ void __launch_bounds__(256)
k_pf_down_part(const uint8_t *__restrict__ Wd, const uint8_t *__restrict__ Sd,
               const int8_t *__restrict__ hq, const float *__restrict__ hs,
               const float *__restrict__ s2,
               const PfGroup *__restrict__ groups, const int32_t *__restrict__ order,
               const float *__restrict__ rwt, float *__restrict__ part)
{
    const int g = order[blockIdx.y];
    const PfGroup grp = groups[g];
    if (grp.nrows <= 0) return;
    const int s   = grp.eid;
    const int l   = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const uint8_t *wr = Wd + (size_t)s * P92_DN_W_BYTES + (size_t)row * (PF_NFF / 2);
    const uint8_t *sr = Sd + (size_t)s * P92_DN_S_BYTES + (size_t)row * (PF_NFF / 16);
    uint2 pk[2]; uint8_t sb[2]; float ds[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) { const int q = i * 32 + l; pk[i] = *(const uint2 *)(wr + (size_t)q * 8); }
    #pragma unroll
    for (int i = 0; i < 2; i++) sb[i] = sr[i * 32 + l];
    #pragma unroll
    for (int i = 0; i < 2; i++) ds[i] = p92_ue4m3_d(sb[i]);
    const float f2 = 0.5f * s2[(size_t)s * 3 + 2];
    for (int r = 0; r < grp.nrows; r++) {
        const int gr = grp.first + r;
        const int8_t *h = hq + (size_t)gr * PF_NFF;
        uint4 xv[2];
        #pragma unroll
        for (int i = 0; i < 2; i++) xv[i] = *(const uint4 *)(h + (size_t)(i * 32 + l) * 16);
        const float hsr = hs[gr], w = rwt[gr];
        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const uint2 xe = make_uint2(xv[i].x, xv[i].y);
            const uint2 xo = make_uint2(xv[i].z, xv[i].w);
            acc = fmaf(ds[i], (float)p92_group_dot4(pk[i], xe, xo), acc);
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) acc += __shfl_down(acc, off, 64);
        if (l == 0) part[(size_t)gr * PF_NEMBD + row] = acc * hsr * f2 * w;
    }
}

// y[t][:] = sum over the token's nine partials, fixed order. grid (T), block 256.
__global__ void __launch_bounds__(256)
k_pf_down_combine(const float *__restrict__ part, const int32_t *__restrict__ tokslot,
                  int T, float *__restrict__ y)
{
    const int t = blockIdx.x;
    if (t >= T) return;
    int ks[PF_TOKSLOTS];
    #pragma unroll
    for (int j = 0; j < PF_TOKSLOTS; j++) ks[j] = tokslot[t * PF_TOKSLOTS + j];
    for (int d = threadIdx.x; d < PF_NEMBD; d += 256) {
        float a = 0.f;
        #pragma unroll
        for (int j = 0; j < PF_TOKSLOTS; j++) if (ks[j] >= 0) a += part[(size_t)ks[j] * PF_NEMBD + d];
        y[(size_t)t * PF_NEMBD + d] = a;
    }
}
