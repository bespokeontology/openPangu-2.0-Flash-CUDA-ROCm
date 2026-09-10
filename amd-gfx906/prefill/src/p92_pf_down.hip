#include "p92amd/p92_pf_moe.h"
#include "../../decode/src/p92_nvfp4_dot4.hip"

// Requantise the grouped hidden state for the down projection.
// One block per grouped row. Emits int8 in the even-then-odd order the packed
// nibble ABI requires, so the consumer needs no staging - the same
// producer-owns-the-layout contract as the decode path.
__global__ void __launch_bounds__(256)
k_pf_requant(const float *__restrict__ hidden, int nrows,
             int8_t *__restrict__ hq, float *__restrict__ hs)
{
    const int r = blockIdx.x;
    if (r >= nrows) return;
    const size_t base = (size_t)r * PF_NFF;
    float v[4]; float m = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; i++) { v[i] = hidden[base + i * 256 + threadIdx.x];
                                  m = fmaxf(m, fabsf(v[i])); }
    __shared__ float red[4];
    #pragma unroll
    for (int o = 32; o; o >>= 1) m = fmaxf(m, __shfl_down(m, o, 64));
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = m;
    __syncthreads();
    m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    const float sc = m > 0.f ? m * (1.f / 127.f) : 1.f;
    if (threadIdx.x == 0) hs[r] = sc;
    const float inv = 1.f / sc;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int q = (int)rintf(v[i] * inv);
        q = q < -128 ? -128 : (q > 127 ? 127 : q);
        const int idx = i * 256 + threadIdx.x, g = idx >> 4, j = idx & 15;
        hq[base + (g << 4) + ((j & 1) ? 8 + (j >> 1) : (j >> 1))] = (int8_t)q;
    }
}

// Down projection, weights resident across the group's rows.
// grid (PF_NEMBD/8, ngroups), block 256. A 32-lane group owns one of the 2560
// output rows and holds its 1024 packed nibbles in registers - 512 B over 32
// lanes is 16 B/lane, two uint2 - then sweeps every row of the group. Cheaper
// to hold resident than gate/up, whose row is 40 B/lane.
//
// SCATTER. A token receives contributions from up to 9 experts, so the write is
// an accumulate. atomicAdd is used, which makes the summation order across
// experts vary run to run. That is a deliberate reassociation, consistent with
// the promoted decode rung that hoisted downacc's reduction, and it is the
// reason PF_DETERMINISTIC exists: defining it writes per-group partials for a
// separate ordered combine instead, at the cost of nrows*2560 floats of traffic.
__global__ void __launch_bounds__(256)
k_pf_down(const uint8_t *__restrict__ Wd, const uint8_t *__restrict__ Sd,
          const int8_t *__restrict__ hq, const float *__restrict__ hs,
          const float *__restrict__ s2,
          const PfGroup *__restrict__ groups, const int32_t *__restrict__ order,
          const int32_t *__restrict__ rows, const float *__restrict__ rwt,
          float *__restrict__ y)
{
    const int g = order[blockIdx.y];
    const PfGroup grp = groups[g];
    if (grp.nrows <= 0) return;
    const int s   = grp.eid;
    const int l   = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);

    const uint8_t *wr = Wd + (size_t)s * P92_DN_W_BYTES + (size_t)row * (PF_NFF / 2);
    const uint8_t *sr = Sd + (size_t)s * P92_DN_S_BYTES + (size_t)row * (PF_NFF / 16);

    // hoist the whole weight row, once. Loads first, branchy UE4M3 decode
    // second - decoding inline puts each one-byte scale load in its own
    // exec-masked region and fences it, which cost this file 8 extra
    // s_waitcnt vmcnt(0) in k_pf_gateup before it was split.
    uint2 pk[2]; uint8_t sb[2]; float ds[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) { const int q = i * 32 + l;
        pk[i] = *(const uint2 *)(wr + (size_t)q * 8); }
    #pragma unroll
    for (int i = 0; i < 2; i++) sb[i] = sr[i * 32 + l];
    #pragma unroll
    for (int i = 0; i < 2; i++) ds[i] = p92_ue4m3_d(sb[i]);
    const float f2 = 0.5f * s2[(size_t)s * 3 + 2];

    for (int r = 0; r < grp.nrows; r++) {
        const int gr  = grp.first + r;
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
        if (l == 0) atomicAdd(&y[(size_t)rows[gr] * PF_NEMBD + row], acc * hsr * f2 * w);
    }
}
