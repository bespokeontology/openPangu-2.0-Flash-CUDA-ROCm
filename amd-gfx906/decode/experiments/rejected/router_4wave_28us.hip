// REJECTED. Router top-8 by wave-local top-8 plus a 32-candidate merge.
// Correct - identical ids, identical tie rule, weight sum exactly 2.500000,
// coherent generation - but 28.19 us a call against the 25.00 us baseline and
// 52.88 tok/s whole-engine median against 53.30. It trades 23 barriers for 56
// shuffles a wave, and on gfx906 a shuffle is ds_bpermute, the same LDS path a
// barrier uses; three of the four waves then idle through the merge.
// Superseded by the single-wave form at 19.93 us, which is in production.
// See receipts/08_REJECTED.md entry R1. Kept, not deleted: fusion, batch width
// or MTP could change the conditions that made it lose.
//
// __global__ void __launch_bounds__(256)
// k_p92_route_top8(const float *logits, const float *bias, int *ids, float *wts) {
//     const int t = threadIdx.x, lane = t & 63, wv = t >> 6;
//     __shared__ float sc[256]; __shared__ float cv[32]; __shared__ int ci[32];
//     const float s = 1.f / (1.f + __expf(-logits[t]));
//     sc[t] = s; float v = s + bias[t]; int i = t;
//     __syncthreads();
//     for (int k = 0; k < 8; k++) {                 // per-wave top-8, no barrier
//         float bv = v; int bi = i;
//         for (int off = 32; off; off >>= 1) {
//             const float ov = __shfl_down(bv, off, 64);
//             const int   oi = __shfl_down(bi, off, 64);
//             if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
//         bv = __shfl(bv, 0, 64); bi = __shfl(bi, 0, 64);
//         if (lane == 0) { cv[wv * 8 + k] = bv; ci[wv * 8 + k] = bi; }
//         if (i == bi) v = -3.0e38f; }
//     __syncthreads();
//     if (wv == 0) {                                 // merge 32 candidates
//         float mv = (lane < 32) ? cv[lane] : -3.0e38f;
//         int   mi = (lane < 32) ? ci[lane] : 256;
//         float wsum = 0.f;
//         for (int k = 0; k < 8; k++) {
//             float bv = mv; int bi = mi;
//             for (int off = 16; off; off >>= 1) {
//                 const float ov = __shfl_down(bv, off, 64);
//                 const int   oi = __shfl_down(bi, off, 64);
//                 if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
//             bi = __shfl(bi, 0, 64);
//             const float w = sc[bi]; wsum += w;
//             if (lane == 0) { ids[k] = bi; wts[k] = w; }
//             if (mi == bi) mv = -3.0e38f; }
//         if (lane == 0) { const float inv = 2.5f / wsum;
//             for (int k = 0; k < 8; k++) wts[k] *= inv; } } }
