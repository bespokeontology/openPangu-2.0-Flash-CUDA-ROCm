#pragma once
// Absorbed MLA decode, 16-lane groups. Third form.
//
// The block-per-head form ran at 3183 us/layer, barrier bound. Split-K with a
// wave per (head, split) removed every barrier and reached 567 us, but raising
// the split count past 8 did nothing, which said the limit was no longer
// occupancy. It is the reduction: a 64-lane score reduction is six
// ds_bpermute steps plus a broadcast, seven LDS-path instructions for every
// position, against eight useful fmaf.
//
// Here a wave is split into lane groups and each group owns its own position,
// so several positions are scored concurrently and the reduction is a few
// shuffles shared across them. Each lane covers 512/G16_LANES latent dims as
// uint2 loads of four bf16, which also turns 2-byte per-lane loads into 8-byte
// ones.
//
// The group width was swept, because the first two hypotheses about what
// limits this kernel were both wrong: grouping four heads per block so they
// share latent rows made it WORSE (546 us), and the access pattern barely
// matters (random 512, contiguous 526, strided 591 us). What actually caps it
// is register pressure. At 16 lanes a group each lane holds qv[32], acc[32]
// and v[32], about 120 VGPRs, which allows roughly two waves a SIMD and leaves
// memory latency exposed.
//
//   lanes per group    8        16       32
//   us/layer         456.65   512.46   314.26
//   GB/s             263.5    234.8    382.9
//   gate             FAIL     pass     pass
//
// 32 lanes is 2 groups a wave, 16 dims a lane, 4 steps - about 48 VGPRs, so
// several times the occupancy - and is 1.63x the 16-lane form. 8 lanes fails
// its gate at relative_l2 3.081e-03 and is not usable.
//
// The four groups hold four independent online-softmax states over the same
// head, so they merge inside the wave with two shuffle steps before the
// partial is written.
#include <hip/hip_runtime.h>
#include <cstdint>
#ifndef P92_NH
#define P92_NH        48
#define P92_KVLORA   512
#define P92_PE        64
#define P92_SINKS    128
#define P92_QK_DIM   192
typedef uint16_t p92_bf;
#endif
#include "p92amd/p92_mla_geometry.h"

// 32 is a measured optimum, not a default. 16 lanes: G16_DPL 32, ~96 live
// floats, rejected earlier (244.3 against 383.0 GB/s). 64 lanes: G16_DPL 8, VGPR
// 76 -> 50 and occupancy 3 -> 4, and STILL slower, 49.64 against 46.75 us,
// because at 32 lanes one __shfl_xor reduces BOTH half-wave groups at once (5
// shuffles serve 2 entries) while at 64 it is 6 shuffles for 1 - 2.4x the DS
// traffic per entry, which outweighs the occupancy gain. Do not re-sweep this
// axis without a new mechanism. Verified bit-identical at both widths.
#define G16_LANES_CFG 32
#define G16_GROUPS   (64 / G16_LANES_CFG)   // groups per wave
#define G16_LANES   G16_LANES_CFG        // lanes per group
#define G16_VEC      4                   // bf16 per load
#define G16_DPL     (P92_KVLORA / G16_LANES)          // 16 latent dims per lane at LANES=32
#define G16_STEPS   (G16_DPL / G16_VEC)               // 4 uint2 loads at LANES=32
#define G16_PEPL    (P92_PE / G16_LANES)              // 4 rope dims per lane

static __device__ __forceinline__ void g16_unpack(uint32_t a, uint32_t b, float *o) {
    o[0] = __uint_as_float( a         << 16);
    o[1] = __uint_as_float((a & 0xFFFF0000u));
    o[2] = __uint_as_float( b         << 16);
    o[3] = __uint_as_float((b & 0xFFFF0000u));
}

__global__ void __launch_bounds__(256)
k_p92_mla_g16(const float *__restrict__ qlat, const float *__restrict__ qpe_a,
              const p92_bf *__restrict__ lat, const p92_bf *__restrict__ kpe,
              const p92_bf *__restrict__ sink_lat, const p92_bf *__restrict__ sink_pe,
              const int *__restrict__ sel, int nsel, int nsink,
              int win_first, int win_cap,
              float *__restrict__ pacc, float *__restrict__ pm, float *__restrict__ pl) {
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int grp  = lane / G16_LANES;         // which position
    const int li   = lane % G16_LANES;         // which dim slice
    const int h    = blockIdx.x;
    const int sp   = blockIdx.y * 4 + wave;
    if (sp >= P92_SPLITS) return;

    const float *q   = qlat  + (size_t)h * P92_KVLORA;
    const float *qpe = qpe_a + (size_t)h * P92_PE;
    const float scale = rsqrtf((float)P92_QK_DIM);

    float qv[G16_DPL], acc[G16_DPL];
    #pragma unroll
    for (int s = 0; s < G16_STEPS; s++)
        #pragma unroll
        for (int j = 0; j < G16_VEC; j++) {
            qv [s * G16_VEC + j] = q[s * (G16_LANES * G16_VEC) + li * G16_VEC + j];
            acc[s * G16_VEC + j] = 0.f;
        }
    float qp[G16_PEPL];
    #pragma unroll
    for (int j = 0; j < G16_PEPL; j++) qp[j] = qpe[li * G16_PEPL + j];

    float m_run = -1e30f, l_run = 0.f;

    // Sinks and body form ONE run of nsink + nsel entries, split evenly.
    // Giving every sink to split 0 left it with 132 entries at short context
    // while the other seven took 4 each, so the cost was set by a single wave
    // and the kernel showed a ~110 us floor independent of nsel: 119 us at
    // nsel=32 against 292 at nsel=2048, where the marginal rate makes the 128
    // sinks worth only 11 us. Balancing removes the floor.
    const int total = nsink + nsel;
    const int per   = (total + P92_SPLITS - 1) / P92_SPLITS;
    const int lo    = sp * per;
    int hi = lo + per; if (hi > total) hi = total;
    const int nrun  = hi - lo;

    for (int j0 = 0; j0 < nrun; j0 += G16_GROUPS) {
        const bool live = (j0 + grp) < nrun;
        const int  ent  = lo + j0 + grp;             // index into the joint run
        const bool is_sink = live && ent < nsink;
        // A sliding layer's position list is (first + i) mod cap, identical for
        // all 30 sliding layers of a token and derivable here, so sel is null
        // for them: no window kernel, no list write, and no 640 int reads a
        // layer. A DSA layer passes the indexer's list and win_cap is 0.
        int tok = -1;
        if (live && !is_sink) {
            const int b = ent - nsink;               // body index
            tok = win_cap ? ((win_first + b) % win_cap) : sel[b];
        }
        const bool on = live && (is_sink || tok >= 0);

        // Branchless loads. Each of the five loads sat in its own `if (on)`
        // exec-masked region and the compiler fenced every one with
        // s_waitcnt vmcnt(0): SIX serialized global round trips per iteration.
        // Inactive lanes cannot affect the result - s is overwritten with
        // -1e30f and p is 0 when !on - so the only requirement is a VALID
        // address. Point the off case at sink_lat/sink_pe, always resident and
        // at least 128 rows, instead of nullptr, and the loads go unconditional.
        const p92_bf *c = on ? (is_sink ? sink_lat + (size_t)ent * P92_KVLORA
                                        : lat + (size_t)tok * P92_KVLORA)
                             : sink_lat;
        const p92_bf *e = on ? (is_sink ? sink_pe + (size_t)ent * P92_PE
                                        : kpe + (size_t)tok * P92_PE)
                             : sink_pe;
        float v[G16_DPL]; float s = 0.f;
        #pragma unroll
        for (int st = 0; st < G16_STEPS; st++) {
            float t4[G16_VEC] = {0.f, 0.f, 0.f, 0.f};
            {
                const uint2 w = *(const uint2 *)(c + st * (G16_LANES * G16_VEC) + li * G16_VEC);
                g16_unpack(w.x, w.y, t4);
            }
            #pragma unroll
            for (int j2 = 0; j2 < G16_VEC; j2++) {
                v[st * G16_VEC + j2] = t4[j2];
                s = fmaf(qv[st * G16_VEC + j2], t4[j2], s);
            }
        }
        {
            // NOTE: this reads 4 bf16 and uses G16_PEPL of them. At LANES=32,
            // G16_PEPL is 2 and lane 31 reads elements 62..65 of a 64-element
            // row - already 2 past the row, landing inside the next row of the
            // same allocation, and the extra values are discarded. It is only
            // safe by construction at this geometry; any change to G16_LANES
            // must replace it with an exactly-sized per-lane load.
            const uint2 w = *(const uint2 *)(e + li * G16_PEPL);
            float p4[4]; g16_unpack(w.x, w.y, p4);
            #pragma unroll
            for (int j2 = 0; j2 < G16_PEPL; j2++) s = fmaf(qp[j2], p4[j2], s);
        }
        // Exact DPP substitution for the 32-lane xor reduction. Verified lane
        // mappings (receipts/10_GFX906_PLAYBOOK.md section 5):
        //   xor 16 crosses the 16-lane row  -> no DPP, stays ds_bpermute
        //   xor  8 == row_mirror then row_half_mirror, since (l^15)^7 == l^8
        //   xor  4 has NO DPP substitute    -> stays ds_bpermute
        //   xor  2 == quad_perm [2,3,0,1] = 0x4E
        //   xor  1 == quad_perm [1,0,3,2] = 0xB1
        // Same source lanes throughout, so this is bit-identical, not a
        // reassociation. 5 DS ops -> 2 DS + 4 ALU. This kernel runs at 3
        // waves/SIMD, the lowest in the engine, so DS latency is the least
        // hidden here of anywhere.
        #define G16_DPPF(x, c) __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(x), __float_as_int(x), (c), 0xF, 0xF, false))
        s += __shfl_xor(s, 16, 64);
        { const float t = G16_DPPF(s, 0x140u); s += G16_DPPF(t, 0x141u); }
        s += __shfl_xor(s, 4, 64);
        s += G16_DPPF(s, 0x4Eu);
        s += G16_DPPF(s, 0xB1u);
        #undef G16_DPPF
        s = on ? s * scale : -1e30f;

        const float mn = fmaxf(m_run, s);
        const float al = __expf(m_run - mn);
        const float p  = on ? __expf(s - mn) : 0.f;
        l_run = l_run * al + p;
        #pragma unroll
        for (int i = 0; i < G16_DPL; i++) acc[i] = acc[i] * al + p * v[i];
        m_run = mn;
    }

    // merge the four groups: same dims live in lanes li, li+16, li+32, li+48
    #pragma unroll
    for (int off = G16_LANES; off < 64; off <<= 1) {
        const float om = __shfl_xor(m_run, off, 64);
        const float ol = __shfl_xor(l_run, off, 64);
        const float mn = fmaxf(m_run, om);
        const float a0 = __expf(m_run - mn), a1 = __expf(om - mn);
        #pragma unroll
        for (int i = 0; i < G16_DPL; i++)
            acc[i] = acc[i] * a0 + __shfl_xor(acc[i], off, 64) * a1;
        l_run = l_run * a0 + ol * a1;
        m_run = mn;
    }

    const size_t base = ((size_t)h * P92_SPLITS + sp) * P92_KVLORA;
    if (grp == 0) {
        #pragma unroll
        for (int st = 0; st < G16_STEPS; st++)
            #pragma unroll
            for (int j2 = 0; j2 < G16_VEC; j2++)
                pacc[base + st * (G16_LANES * G16_VEC) + li * G16_VEC + j2] = acc[st * G16_VEC + j2];
        if (li == 0) { pm[h * P92_SPLITS + sp] = m_run; pl[h * P92_SPLITS + sp] = l_run; }
    }
}
