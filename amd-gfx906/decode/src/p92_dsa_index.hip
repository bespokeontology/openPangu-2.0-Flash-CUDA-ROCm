#pragma once
// openPangu DSA indexer (N2): per-position index-key cache, 24-head weighted
// scorer, top-2048 selection. gfx906 wave64.
//
// This is the one component whose cost is unbounded in context: it scores
// EVERY position on the 16 DSA layers of every token, while attention itself
// is bounded to 2048 selected plus 128 sinks. It is why decode throughput
// falls with context on the CUDA engine. So the layout is chosen to make the
// scan as close to free as a scan can be.
//
// The cache is stored TRANSPOSED, [128][cap] rather than [cap][128]. That one
// choice removes every cross-lane reduction from the scorer: a thread owns a
// whole position and walks the 128 dims, so consecutive threads read
// consecutive addresses at each dim and the 24 head dots accumulate in
// registers. Appending a position writes 128 strided values, which is nothing
// once per token per layer.
//
// Traps this kernel is written against, both from the port map and confirmed
// in the reference:
//   - openPangu ropes only the FIRST 64 of the 128 indexer dims, as pairs
//     (i, i+32). Qwen ropes all 128 as (i, i+64). Copying Qwen is silently
//     wrong.
//   - openPangu norms the KEY, not the query. Qwen norms the query.
//   - the query comes from the normalised q_lora (1024) and the key from the
//     normalised hidden (2560), so they are two different projections and
//     Qwen's single fused idx_qk wiring does not apply.
#include <hip/hip_runtime.h>
#include <cstdint>

#define P92_IDX_HEADS 24
#define P92_IDX_DIM  128
#define P92_IDX_ROPE  64          // only the first 64 dims are roped
#define P92_IDX_TOPK 2048
#define P92_IDX_THREADS 256

typedef uint16_t p92_bf;
static __device__ __forceinline__ float p92_ibf(p92_bf b){ return __uint_as_float((uint32_t)b<<16); }
static __device__ __forceinline__ p92_bf p92_fbi(float f){
    uint32_t u = __float_as_uint(f); return (p92_bf)((u + 0x8000u) >> 16); }

// ---- append one position's index key, transposed --------------------------
// k[128] fp32 -> cache[d * cap + slot]. 128 threads, one dim each.
__global__ void __launch_bounds__(128)
k_p92_idx_append(const float *__restrict__ k, p92_bf *__restrict__ cache,
                 int cap, int slot) {
    const int d = threadIdx.x;
    if (d < P92_IDX_DIM) cache[(size_t)d * cap + slot] = p92_fbi(k[d]);
}

// ---- rope the first 64 dims as pairs (i, i+32) ---------------------------
// rope_interleave is false, so a pair is (i, i + rope_dim/2) inside the roped
// prefix, i.e. (i, i+32) over 64 dims - NOT (i, i+64) over 128.
__global__ void __launch_bounds__(64)
k_p92_idx_rope(float *__restrict__ v, const float *__restrict__ cosv,
               const float *__restrict__ sinv) {
    const int i = threadIdx.x;
    if (i >= P92_IDX_ROPE / 2) return;
    const float c = cosv[i], s = sinv[i];
    const float a = v[i], b = v[i + P92_IDX_ROPE / 2];
    v[i]                      = a * c - b * s;
    v[i + P92_IDX_ROPE / 2]   = b * c + a * s;
}

// ---- weighted 24-head score for every position ---------------------------
// score[p] = sum_h w[h] * relu(dot(q[h], k[p]) * scale)
// One thread per position. The 24x128 query block lives in LDS and every
// thread reads the same element at the same step, which is an LDS broadcast.
__global__ void __launch_bounds__(P92_IDX_THREADS)
k_p92_idx_scores(const float *__restrict__ q,        // [24][128]
                 const p92_bf *__restrict__ cache,   // [128][cap] transposed
                 const float *__restrict__ wh,       // [24]
                 int cap, int n, float scale,
                 float *__restrict__ scores) {
    __shared__ float sq[P92_IDX_HEADS * P92_IDX_DIM];
    __shared__ float sw[P92_IDX_HEADS];
    for (int i = threadIdx.x; i < P92_IDX_HEADS * P92_IDX_DIM; i += P92_IDX_THREADS)
        sq[i] = q[i];
    if (threadIdx.x < P92_IDX_HEADS) sw[threadIdx.x] = wh[threadIdx.x];
    __syncthreads();

    const int p = blockIdx.x * P92_IDX_THREADS + threadIdx.x;
    if (p >= n) return;

    float acc[P92_IDX_HEADS];
    #pragma unroll
    for (int h = 0; h < P92_IDX_HEADS; h++) acc[h] = 0.f;

    for (int d = 0; d < P92_IDX_DIM; d++) {
        const float kv = p92_ibf(cache[(size_t)d * cap + p]);
        #pragma unroll
        for (int h = 0; h < P92_IDX_HEADS; h++)
            acc[h] = fmaf(sq[h * P92_IDX_DIM + d], kv, acc[h]);
    }
    float s = 0.f;
    #pragma unroll
    for (int h = 0; h < P92_IDX_HEADS; h++) s = fmaf(sw[h], fmaxf(acc[h] * scale, 0.f), s);
    scores[p] = s;
}

// ---- top-2048 by radix threshold on order-preserving flipped fp32 --------
// Two 11-bit passes over the top 22 bits find a threshold that admits at least
// TOPK; the gather then takes entries above it and fills the remainder from
// the tie band in ascending position order, which is the deterministic rule
// the AMD selectors use.
static __device__ __forceinline__ uint32_t p92_flip(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
#define P92_RBITS 11
#define P92_RBINS (1 << P92_RBITS)

// Three 11-bit passes narrow the flipped key to an EXACT threshold. One pass
// only narrows to a bin, and members of a bin are not ties - that is worth
// stating because a single-pass version selected 2048 entries with a symmetric
// difference of 636 against the true top-2048 while looking plausible.
//
// state[0] = prefix accumulated so far, state[1] = mask of decided bits,
// state[2] = count already known to be strictly above, state[3] = chosen bin.
__global__ void __launch_bounds__(256)
k_p92_idx_hist(const float *__restrict__ scores, int n, int shift,
               const uint32_t *__restrict__ state, uint32_t *__restrict__ hist) {
    __shared__ uint32_t h[P92_RBINS];
    const uint32_t prefix = state[0], mask = state[1];
    for (int i = threadIdx.x; i < P92_RBINS; i += 256) h[i] = 0;
    __syncthreads();
    for (int p = blockIdx.x * 256 + threadIdx.x; p < n; p += gridDim.x * 256) {
        const uint32_t f = p92_flip(scores[p]);
        if ((f & mask) == prefix) atomicAdd(&h[(f >> shift) & (P92_RBINS - 1)], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < P92_RBINS; i += 256)
        if (h[i]) atomicAdd(&hist[i], h[i]);
}

// Pick the threshold bin with a parallel suffix scan over the histogram. The
// serial form - one thread stepping 2048 bins, three times per selection -
// was the fixed cost of the whole selection at short context: replacing it
// took the fused selector from 146 us to 26 us at n=2048.
__global__ void __launch_bounds__(1024)
k_p92_idx_pick(const uint32_t *__restrict__ hist, int want, int shift,
               uint32_t *__restrict__ state) {
    __shared__ uint32_t suf[P92_RBINS], tmp[P92_RBINS];
    __shared__ uint32_t sbin, snext;
    const int t = threadIdx.x;
    const uint32_t already = state[2];
    for (int i = t; i < P92_RBINS; i += 1024) suf[i] = hist[i];
    __syncthreads();
    for (int d = 1; d < P92_RBINS; d <<= 1) {
        for (int i = t; i < P92_RBINS; i += 1024)
            tmp[i] = (i + d < P92_RBINS) ? suf[i] + suf[i + d] : suf[i];
        __syncthreads();
        for (int i = t; i < P92_RBINS; i += 1024) suf[i] = tmp[i];
        __syncthreads();
    }
    for (int i = t; i < P92_RBINS; i += 1024) {
        const uint32_t hi_excl = (i + 1 < P92_RBINS) ? suf[i + 1] : 0u;
        if (already + suf[i] >= (uint32_t)want && already + hi_excl < (uint32_t)want) {
            sbin = (uint32_t)i; snext = already + hi_excl;
        }
    }
    __syncthreads();
    if (t == 0) {
        state[0] |= sbin << shift;
        state[1] |= ((uint32_t)(P92_RBINS - 1)) << shift;
        state[2]  = snext;
        state[3]  = sbin;
    }
}

// Ordered gather, three phases. A single-block ordered walk is deterministic
// but serial, and measured 210 us at 8K against 48 us for the scorer it
// serves - the selection cost more than the scan. This keeps the exact same
// output while running in parallel: fixed contiguous chunks are counted
// independently, the chunk counts are scanned once, and each chunk writes at
// its own offset in ascending order. Order is a function of chunk boundaries
// only, so it is identical every run.
#define P92_GCHUNK 4096
#define P92_GMAXC  256

__global__ void __launch_bounds__(256)
k_p92_idx_count(const float *__restrict__ scores, int n,
                const uint32_t *__restrict__ state, uint32_t *__restrict__ cnt) {
    const uint32_t thr = state[0];
    const int c = blockIdx.x;
    const int lo = c * P92_GCHUNK;
    int hi = lo + P92_GCHUNK; if (hi > n) hi = n;
    uint32_t a = 0, e = 0;
    for (int p = lo + threadIdx.x; p < hi; p += 256) {
        const uint32_t f = p92_flip(scores[p]);
        if (f > thr) a++; else if (f == thr) e++;
    }
    __shared__ uint32_t sa[256], se[256];
    sa[threadIdx.x] = a; se[threadIdx.x] = e;
    __syncthreads();
    for (int st = 128; st; st >>= 1) {
        if (threadIdx.x < st) { sa[threadIdx.x] += sa[threadIdx.x + st];
                                se[threadIdx.x] += se[threadIdx.x + st]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { cnt[c] = sa[0]; cnt[P92_GMAXC + c] = se[0]; }
}

// Exclusive scan of the chunk counts, above-band first then the tie band, so
// the two bands stay contiguous and ascending.
__global__ void __launch_bounds__(256)
k_p92_idx_scan(uint32_t *__restrict__ cnt, int nchunk, int want,
               uint32_t *__restrict__ off) {
    if (threadIdx.x) return;
    uint32_t r = 0;
    for (int c = 0; c < nchunk; c++) { const uint32_t v = cnt[c]; off[c] = r; r += v; }
    const uint32_t above = r;
    for (int c = 0; c < nchunk; c++) {
        const uint32_t v = cnt[P92_GMAXC + c]; off[P92_GMAXC + c] = r; r += v; }
    off[2 * P92_GMAXC]     = above;
    off[2 * P92_GMAXC + 1] = r < (uint32_t)want ? r : (uint32_t)want;
}

__global__ void __launch_bounds__(256)
k_p92_idx_write(const float *__restrict__ scores, int n,
                const uint32_t *__restrict__ state, const uint32_t *__restrict__ off,
                int want, int *__restrict__ sel) {
    const uint32_t thr = state[0];
    const int c = blockIdx.x;
    const int lo = c * P92_GCHUNK;
    int hi = lo + P92_GCHUNK; if (hi > n) hi = n;
    __shared__ uint32_t va[4], ve[4];
    __shared__ uint32_t ca, ce;
    const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
    if (threadIdx.x == 0) { ca = off[c]; ce = off[P92_GMAXC + c]; }
    __syncthreads();
    for (int base = lo; base < hi; base += 256) {
        const int p = base + threadIdx.x;
        bool ta = false, te = false;
        if (p < hi) { const uint32_t f = p92_flip(scores[p]);
            ta = f > thr; te = f == thr; }
        const uint64_t ba = __ballot(ta), be = __ballot(te);
        const uint32_t pa = (uint32_t)__popcll(ba & ((1ull << lane) - 1ull));
        const uint32_t pe = (uint32_t)__popcll(be & ((1ull << lane) - 1ull));
        if (lane == 0) { va[wv] = (uint32_t)__popcll(ba); ve[wv] = (uint32_t)__popcll(be); }
        __syncthreads();
        uint32_t wa = 0, we = 0, ta_t = 0, te_t = 0;
        for (int w = 0; w < 4; w++) { if (w < wv) { wa += va[w]; we += ve[w]; }
                                      ta_t += va[w]; te_t += ve[w]; }
        if (ta) { const uint32_t s = ca + wa + pa; if (s < (uint32_t)want) sel[s] = p; }
        if (te) { const uint32_t s = ce + we + pe; if (s < (uint32_t)want) sel[s] = p; }
        __syncthreads();
        if (threadIdx.x == 0) { ca += ta_t; ce += te_t; }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(256)
k_p92_idx_pad(const uint32_t *__restrict__ off, int want, int *__restrict__ sel) {
    for (int i = off[2 * P92_GMAXC + 1] + blockIdx.x * 256 + threadIdx.x; i < want;
         i += gridDim.x * 256) sel[i] = -1;
}

// ---- fused single-launch selection ---------------------------------------
// The staged version was correct but cost 193 us at n=2048 and 290 us at
// n=32768 - almost flat in n, which is the signature of launch overhead, not
// work: ten launches at roughly 19 us each. The whole selection reads n floats
// four times, which is 128 KB at 8K context; one block can do that in tens of
// microseconds. So the three radix passes and the ordered gather all run in a
// single block, one launch, with the histogram in LDS.
#define P92_SEL_THREADS 1024
#define P92_SEL_WAVES   (P92_SEL_THREADS / 64)

__global__ void __launch_bounds__(P92_SEL_THREADS)
k_p92_idx_select(const float *__restrict__ scores, int n, int want,
                 int *__restrict__ sel) {
    __shared__ uint32_t hist[P92_RBINS], suf[P92_RBINS], tmp[P92_RBINS];
    __shared__ uint32_t sprefix, smask, sabove, sbin, sabove_next;
    __shared__ uint32_t wv_a[P92_SEL_WAVES], wv_e[P92_SEL_WAVES];
    __shared__ int cur_a, cur_e;
    const int t = threadIdx.x, lane = t & 63, wv = t >> 6;

    if (t == 0) { sprefix = 0; smask = 0; sabove = 0; sabove_next = 0; }
    __syncthreads();

    const int shifts[3] = {21, 10, 0};
    #pragma unroll
    for (int pass = 0; pass < 3; pass++) {
        const int sh = shifts[pass];
        for (int i = t; i < P92_RBINS; i += P92_SEL_THREADS) hist[i] = 0;
        __syncthreads();
        const uint32_t pf = sprefix, mk = smask;
        for (int p = t; p < n; p += P92_SEL_THREADS) {
            const uint32_t f = p92_flip(scores[p]);
            if ((f & mk) == pf) atomicAdd(&hist[(f >> sh) & (P92_RBINS - 1)], 1u);
        }
        __syncthreads();
        // Parallel bin walk. A single thread stepping 2048 bins three times is
        // 6144 serial steps and was the fixed cost of the whole selection at
        // short context. This is an inclusive suffix scan over the bins in 11
        // Hillis-Steele steps; the chosen bin is the highest one whose suffix
        // total reaches what is still wanted.
        for (int i = t; i < P92_RBINS; i += P92_SEL_THREADS) suf[i] = hist[i];
        __syncthreads();
        for (int d = 1; d < P92_RBINS; d <<= 1) {
            for (int i = t; i < P92_RBINS; i += P92_SEL_THREADS)
                tmp[i] = (i + d < P92_RBINS) ? suf[i] + suf[i + d] : suf[i];
            __syncthreads();
            for (int i = t; i < P92_RBINS; i += P92_SEL_THREADS) suf[i] = tmp[i];
            __syncthreads();
        }
        // suf[b] = count of entries in bins >= b. Pick the lowest b whose
        // suffix still reaches want; that is the bin the threshold lands in.
        for (int i = t; i < P92_RBINS; i += P92_SEL_THREADS) {
            const uint32_t hi_excl = (i + 1 < P92_RBINS) ? suf[i + 1] : 0u;
            if (sabove + suf[i] >= (uint32_t)want && sabove + hi_excl < (uint32_t)want) {
                sbin = (uint32_t)i; sabove_next = sabove + hi_excl;
            }
        }
        __syncthreads();
        if (t == 0) {
            sprefix |= sbin << sh;
            smask   |= ((uint32_t)(P92_RBINS - 1)) << sh;
            sabove   = sabove_next;
        }
        __syncthreads();
    }

    // ordered gather against the exact threshold key
    const uint32_t thr = sprefix;
    if (t == 0) { cur_a = 0; cur_e = 0; }
    __syncthreads();
    const int above_total = (int)sabove;
    for (int base = 0; base < n; base += P92_SEL_THREADS) {
        const int p = base + t;
        bool ta = false, te = false;
        if (p < n) { const uint32_t f = p92_flip(scores[p]);
            ta = f > thr; te = f == thr; }
        const uint64_t ba = __ballot(ta), be = __ballot(te);
        const uint32_t pa = (uint32_t)__popcll(ba & ((1ull << lane) - 1ull));
        const uint32_t pe = (uint32_t)__popcll(be & ((1ull << lane) - 1ull));
        if (lane == 0) { wv_a[wv] = (uint32_t)__popcll(ba); wv_e[wv] = (uint32_t)__popcll(be); }
        __syncthreads();
        uint32_t wa = 0, we = 0, tota = 0, tote = 0;
        for (int w = 0; w < P92_SEL_WAVES; w++) {
            if (w < wv) { wa += wv_a[w]; we += wv_e[w]; }
            tota += wv_a[w]; tote += wv_e[w];
        }
        if (ta) { const int s = cur_a + (int)(wa + pa); if (s < want) sel[s] = p; }
        if (te) { const int s = above_total + cur_e + (int)(we + pe); if (s < want) sel[s] = p; }
        __syncthreads();
        if (t == 0) { cur_a += (int)tota; cur_e += (int)tote; }
        __syncthreads();
    }
    const int filled = above_total + cur_e < want ? above_total + cur_e : want;
    for (int i = filled + t; i < want; i += P92_SEL_THREADS) sel[i] = -1;
}
