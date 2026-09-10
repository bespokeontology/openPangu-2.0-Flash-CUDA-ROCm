#include "p92amd/p92_pf_moe.h"

// Load-time weight preshuffle. Immutable, done once at arena load, never in the
// hot path.
//
// CONSUMER-DESIGNED, not one universal prefill layout. Each consumer states what
// it wants and gets its own transform, because the operations have different
// access shapes and forcing a shared layout would pessimise all of them.
//
// WHAT GATE/UP WANTS. Its 32-lane group holds a whole weight row resident, and
// lane l needs the scales of groups l, 32+l, 64+l, 96+l, 128+l - stride 32, so
// five separate one-byte loads. Those five are what the branchy UE4M3 decode
// fenced: 16 s_waitcnt vmcnt(0) before load and decode were split, 8 after.
// Reordering so lane l's five scales are CONTIGUOUS at [l*5 .. l*5+4] makes it
// one 8-byte load, five loads becoming one and the fence problem disappearing
// rather than being halved.
//
// The weight nibbles are NOT reshuffled. 32 lanes x uint2 = 256 B a step is the
// measured-best streaming pattern on gfx906 (741 GB/s, against 643 for uint4 and
// 219 for two uint4 in flight), so the existing layout is already what the
// hardware wants and a wider per-lane load would be a regression.
//
// Layout, per expert row of PF_NEMBD/16 = 160 groups:
//   packed[l*5 + i] = original[i*32 + l]      l in [0,32), i in [0,5)
// 160 bytes in, 160 out, pure permutation.
__global__ void __launch_bounds__(256)
k_pf_shuffle_scales_gu(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst,
                       int rows_total)
{
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows_total) return;
    const int l = threadIdx.x & 31;
    const uint8_t *s = src + (size_t)row * (PF_NEMBD / 16);
    uint8_t *d = dst + (size_t)row * (PF_NEMBD / 16);
    #pragma unroll
    for (int i = 0; i < 5; i++) d[l * 5 + i] = s[i * 32 + l];
}

// DOWN wants the same thing at its own width: 64 groups a row, lane l needs
// groups l and 32+l, so two bytes become one 16-bit load.
//   packed[l*2 + i] = original[i*32 + l]      l in [0,32), i in [0,2)
__global__ void __launch_bounds__(256)
k_pf_shuffle_scales_dn(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst,
                       int rows_total)
{
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows_total) return;
    const int l = threadIdx.x & 31;
    const uint8_t *s = src + (size_t)row * (PF_NFF / 16);
    uint8_t *d = dst + (size_t)row * (PF_NFF / 16);
    #pragma unroll
    for (int i = 0; i < 2; i++) d[l * 2 + i] = s[i * 32 + l];
}

// Verifies a shuffle is a permutation: every source byte lands exactly once.
// Cheap, and it runs at load time where cheap does not matter - a silently
// wrong immutable layout would corrupt every subsequent token.
__global__ void __launch_bounds__(256)
k_pf_shuffle_check(const uint8_t *__restrict__ src, const uint8_t *__restrict__ dst,
                   int rows_total, int per_row, int fan, uint32_t *__restrict__ bad)
{
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows_total) return;
    const int l = threadIdx.x & 31;
    const uint8_t *s = src + (size_t)row * per_row;
    const uint8_t *d = dst + (size_t)row * per_row;
    for (int i = 0; i < fan; i++)
        if (d[l * fan + i] != s[i * 32 + l]) atomicAdd(bad, 1u);
}
