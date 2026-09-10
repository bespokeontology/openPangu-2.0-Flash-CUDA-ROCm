// k_p92_idx_scores_v2 - long-context indexer score scan.
//
// v1 (k_p92_idx_scores) is one thread per position running a SERIAL 128-iteration
// loop with 24 accumulators, grid (npos+255)/256. At 5,669 context that is 23
// blocks against 240 SIMD slots and the scan measured 0.97 us/position/token -
// about 0.85% of the card's MAC rate. The cost is exposed latency, not FLOPs.
//
// v2 splits BOTH axes inside the block: P92V2_TILE positions x P92V2_DCHUNK
// dimension chunks. The serial chain per thread falls 128 -> 32, the LDS query
// broadcast is amortised over 64 positions instead of 256, and the block count
// rises 4x. Partial accumulators are reduced through LDS in two steps.
//
// Arithmetic is identical to v1: score[p] = sum_h w[h] * relu(dot(q[h],k[p])*scale).
#pragma once
// Included AFTER p92_dsa_index.hip by the harness: relies on its typedefs/constants.


#define P92V2_TILE   64
#define P92V2_DCHUNK 4
#define P92V2_DLEN   (P92_IDX_DIM / P92V2_DCHUNK)

__global__ void __launch_bounds__(256)
k_p92_idx_scores_v2(const float *__restrict__ q,        // [24][128]
                    const p92_bf *__restrict__ cache,   // [128][cap] transposed
                    const float *__restrict__ wh,       // [24]
                    int cap, int n, float scale,
                    float *__restrict__ scores) {
    __shared__ float sq[P92_IDX_HEADS * P92_IDX_DIM];              // 12 KB
    __shared__ float sw[P92_IDX_HEADS];
    __shared__ float part[P92V2_DCHUNK][P92V2_TILE][P92_IDX_HEADS]; // 24 KB

    for (int i = threadIdx.x; i < P92_IDX_HEADS * P92_IDX_DIM; i += 256)
        sq[i] = q[i];
    if (threadIdx.x < P92_IDX_HEADS) sw[threadIdx.x] = wh[threadIdx.x];
    __syncthreads();

    const int t  = threadIdx.x;
    const int pi = t & (P92V2_TILE - 1);        // position within tile
    const int dc = t >> 6;                      // dimension chunk 0..3
    const int p  = blockIdx.x * P92V2_TILE + pi;
    const int d0 = dc * P92V2_DLEN;

    float acc[P92_IDX_HEADS];
    #pragma unroll
    for (int h = 0; h < P92_IDX_HEADS; h++) acc[h] = 0.f;

    if (p < n) {
        const p92_bf *col = cache + p;
        #pragma unroll 8
        for (int d = d0; d < d0 + P92V2_DLEN; d++) {
            const float kv = p92_ibf(col[(size_t)d * cap]);
            #pragma unroll
            for (int h = 0; h < P92_IDX_HEADS; h++)
                acc[h] = fmaf(sq[h * P92_IDX_DIM + d], kv, acc[h]);
        }
    }
    #pragma unroll
    for (int h = 0; h < P92_IDX_HEADS; h++) part[dc][pi][h] = acc[h];
    __syncthreads();

    // one thread per position: sum the 4 chunks, relu*scale, weighted head sum
    if (t < P92V2_TILE) {
        const int pp = blockIdx.x * P92V2_TILE + t;
        if (pp < n) {
            float s = 0.f;
            #pragma unroll
            for (int h = 0; h < P92_IDX_HEADS; h++) {
                float v = 0.f;
                #pragma unroll
                for (int c = 0; c < P92V2_DCHUNK; c++) v += part[c][t][h];
                s = fmaf(sw[h], fmaxf(v * scale, 0.f), s);
            }
            scores[pp] = s;
        }
    }
}
