// Prefill-native DSA selection for openPangu on gfx906.
//
// SEMANTICS ARE THE MODEL'S, transcribed from the decode engine and not
// reinterpreted. On the 16 layers where index%3==0:
//   key   : proj(ix_wk) -> 128 dims, RMS-normed with ix_knorm, RoPE on dims
//           0..63 as pairs (i, i+32), stored bf16 into a DIM-MAJOR [128][cap]
//           cache. Written at EVERY position, not only above the threshold -
//           decode carries a scar comment about exactly that bug.
//   query : proj(ix_wqb) -> 24 heads x 128, same RoPE rule per head, NOT normed.
//   score : score[p] = sum_h wh[h] * relu(scale * dot(q[h], cache[:,p])),
//           scale = 1/sqrt(128), wh = bf16_gemv(ix_wproj, norm).
//   select: top-2048 by score. Below 2048 eligible positions the model takes a
//           dense path and no selection runs.
//
// WHAT IS PREFILL-NATIVE HERE, rather than decode replayed T times:
//
// Decode launches one block per POSITION and re-reads the whole 24x128 query
// from global memory in every one of them. That is the right shape when there
// is a single query and the parallelism has to come from positions. Prefill has
// T queries and npos positions, so the parallelism is already there twice over,
// and the query can be staged ONCE per block and reused across a tile of 256
// positions.
//
// Query batching was considered and is NOT needed, on arithmetic. Per position
// the scorer reads 128 bf16 = 256 B of cache and does 24*128 = 3072 FMA, so one
// query per block already runs at 12 FMA/byte against an MI50 balance near 8.8.
// The scorer is compute-bound before any query reuse, and at these context
// lengths the whole cache (128 x npos x 2 B = 1 MB at 4096) sits inside the
// 4 MB L2 anyway. Batching queries would cost LDS and buy nothing.
#include <hip/hip_runtime.h>
#include <cstdint>

#ifndef PF_DSA_H
#define PF_DSA_H
#define PF_IDX_HEADS   24
#define PF_IDX_DIM    128
#define PF_IDX_TOPK  2048
#define PF_SEL_THREADS 1024
#define PF_SEL_WAVES  (PF_SEL_THREADS / 64)
#define PF_RBITS       11
#define PF_RBINS      (1 << PF_RBITS)
#endif

static __device__ __forceinline__ float pfd_b2f(uint16_t b) {
    return __uint_as_float((uint32_t)b << 16);
}
static __device__ __forceinline__ uint16_t pfd_f2b(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
// order-preserving float -> uint key, so a radix pass on the bits is a radix
// pass on the values
static __device__ __forceinline__ uint32_t pfd_flip(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ---- indexer key: norm, rope, transpose into the dim-major cache ----------
// grid(T), block 128 (== PF_IDX_DIM, so every lane reaches the barrier).
__global__ void __launch_bounds__(PF_IDX_DIM)
k_pf_ix_key(const float *__restrict__ kidx, const uint16_t *__restrict__ knorm,
            const float *__restrict__ cosv, const float *__restrict__ sinv,
            int T, int base, int cap, float eps, uint16_t *__restrict__ cache)
{
    const int t = blockIdx.x;
    if (t >= T) return;
    const int d = threadIdx.x;
    const float *k = kidx + (size_t)t * PF_IDX_DIM;
    const float kd = k[d];
    float ss = kd * kd;
    #pragma unroll
    for (int o = 32; o; o >>= 1) ss += __shfl_down(ss, o, 64);
    __shared__ float red[2];
    if ((d & 63) == 0) red[d >> 6] = ss;
    __syncthreads();
    const float inv = rsqrtf((red[0] + red[1]) / (float)PF_IDX_DIM + eps);
    const int pos = base + t;
    const float *cs = cosv + (size_t)pos * 32, *sn = sinv + (size_t)pos * 32;
    // the RoPE mixes d with d+-32, so a lane forms its partner's NORMALISED
    // value directly rather than exchanging it: inv is wave-uniform and the two
    // extra loads are L1 hits
#define PF_NK(i) (k[(i)] * inv * pfd_b2f(knorm[(i)]))
    float v = PF_NK(d);
    if (d < 32)      { const float a=PF_NK(d),    b=PF_NK(d+32); v = a*cs[d] - b*sn[d]; }
    else if (d < 64) { const float a=PF_NK(d-32), b=PF_NK(d);    v = b*cs[d-32] + a*sn[d-32]; }
#undef PF_NK
    cache[(size_t)d * cap + pos] = pfd_f2b(v);
}

// ---- indexer query: same RoPE rule per head, no norm ---------------------
// grid(24, T), block 32.
__global__ void __launch_bounds__(32)
k_pf_ix_query(float *__restrict__ q, const float *__restrict__ cosv,
              const float *__restrict__ sinv, int T, int base)
{
    const int h = blockIdx.x, t = blockIdx.y;
    if (t >= T) return;
    const int i = threadIdx.x;
    float *v = q + ((size_t)t * PF_IDX_HEADS + h) * PF_IDX_DIM;
    const int pos = base + t;
    const float c = cosv[(size_t)pos * 32 + i], s = sinv[(size_t)pos * 32 + i];
    const float a = v[i], b = v[i + 32];
    v[i]      = a * c - b * s;
    v[i + 32] = b * c + a * s;
}

// ---- scoring: one query a block-column, 256 positions a block ------------
// grid((maxn+255)/256, T), block 256. The 24x128 query is staged once and every
// position in the tile reuses it; decode re-read it per position.
__global__ void __launch_bounds__(256)
k_pf_idx_scores(const float *__restrict__ q, const uint16_t *__restrict__ cache,
                const float *__restrict__ wh, int cap, int T, int base,
                float scale, float *__restrict__ scores, int sstride)
{
    const int t = blockIdx.y;
    if (t >= T) return;
    const int npos = base + t + 1;              // causal: this query's prefix
    __shared__ float sq[PF_IDX_HEADS * PF_IDX_DIM];
    __shared__ float sw[PF_IDX_HEADS];
    const float *qr = q + (size_t)t * PF_IDX_HEADS * PF_IDX_DIM;
    for (int i = threadIdx.x; i < PF_IDX_HEADS * PF_IDX_DIM; i += 256) sq[i] = qr[i];
    if (threadIdx.x < PF_IDX_HEADS) sw[threadIdx.x] = wh[(size_t)t * PF_IDX_HEADS + threadIdx.x];
    __syncthreads();

    const int p = blockIdx.x * 256 + threadIdx.x;
    if (p >= npos) return;

    float acc[PF_IDX_HEADS];
    #pragma unroll
    for (int h = 0; h < PF_IDX_HEADS; h++) acc[h] = 0.f;
    for (int d = 0; d < PF_IDX_DIM; d++) {
        const float kv = pfd_b2f(cache[(size_t)d * cap + p]);   // coalesced in p
        #pragma unroll
        for (int h = 0; h < PF_IDX_HEADS; h++)
            acc[h] = fmaf(sq[h * PF_IDX_DIM + d], kv, acc[h]);
    }
    float s = 0.f;
    #pragma unroll
    for (int h = 0; h < PF_IDX_HEADS; h++) s = fmaf(sw[h], fmaxf(acc[h] * scale, 0.f), s);
    scores[(size_t)t * sstride + p] = s;
}


// ---- scorer v2: the query belongs in SCALAR registers, not LDS -----------
//
// v1 stages the 24x128 query in LDS and reads it back inside the hot loop. The
// ISA says what that costs: one d iteration is 24 v_fma, 12 ds_read and TWELVE
// lgkmcnt waits in 58 instructions. Twenty-four useful multiply-adds paying for
// twelve serialised LDS round trips, which is why the scorer measures ~37% of
// this machine's FMA peak while owning 89% of the selector and being its only
// quadratic term.
//
// The query address is WAVE-UNIFORM: t is blockIdx.y and h, d are loop indices,
// so every lane reads the same word. That is precisely what the scalar unit is
// for. Reading it from global with a uniform address lets the backend issue
// s_load into SGPRs and feed v_fmac_f32 v, s, v directly - the load stops being
// an instruction in the loop at all, rather than becoming a cheaper one. The
// scalar cache holds the 12 KB query comfortably and it is re-read per block,
// amortised across the block's 256 positions exactly as the LDS copy was.
//
// __builtin_amdgcn_readfirstlane makes the uniformity explicit rather than
// hoping divergence analysis proves it through the pointer.
__global__ void __launch_bounds__(256)
k_pf_idx_scores2(const float *__restrict__ q, const uint16_t *__restrict__ cache,
                 const float *__restrict__ wh, int cap, int T, int base,
                 float scale, float *__restrict__ scores, int sstride)
{
    const int t = blockIdx.y;
    if (t >= T) return;
    const int npos = base + t + 1;
    const int p = blockIdx.x * 256 + threadIdx.x;
    if (p >= npos) return;

    const float *qr = q  + (size_t)__builtin_amdgcn_readfirstlane(t) * PF_IDX_HEADS * PF_IDX_DIM;
    const float *wr = wh + (size_t)__builtin_amdgcn_readfirstlane(t) * PF_IDX_HEADS;

    float acc[PF_IDX_HEADS];
    #pragma unroll
    for (int h = 0; h < PF_IDX_HEADS; h++) acc[h] = 0.f;
    for (int d = 0; d < PF_IDX_DIM; d++) {
        const float kv = pfd_b2f(cache[(size_t)d * cap + p]);   // coalesced in p
        #pragma unroll
        for (int h = 0; h < PF_IDX_HEADS; h++)
            acc[h] = fmaf(qr[h * PF_IDX_DIM + d], kv, acc[h]);
    }
    float s = 0.f;
    #pragma unroll
    for (int h = 0; h < PF_IDX_HEADS; h++) s = fmaf(wr[h], fmaxf(acc[h] * scale, 0.f), s);
    scores[(size_t)t * sstride + p] = s;
}


// ---- v3: DIM-MAJOR query, so the 24 heads become one scalar load ---------
//
// v2 removed the LDS traffic but left 24 separate s_load_dword per d, because
// q[h*128+d] strides 512 bytes between heads and cannot vectorise. Storing the
// query dim-major as [128][24] makes the 24 heads of one dim CONTIGUOUS, and 24
// scalar dwords become three s_load_dwordx8. Same arithmetic, a third of the
// scalar issue.
//
// The transpose is free: it folds into the RoPE pass that already rewrites every
// element of the query, and the query is small (T x 3072 floats).
//
// grid(24, T), block 64: lanes 0..31 rotate the pair (i, i+32) - openPangu ropes
// only the first half - and all 64 lanes copy one of the unroped dims 64..127.
__global__ void __launch_bounds__(64)
k_pf_ix_query_t(const float *__restrict__ q, float *__restrict__ qT,
                const float *__restrict__ cosv, const float *__restrict__ sinv,
                int T, int base)
{
    const int h = blockIdx.x, t = blockIdx.y;
    if (t >= T) return;
    const int i = threadIdx.x;
    const float *src = q  + ((size_t)t * PF_IDX_HEADS + h) * PF_IDX_DIM;
    float       *dst = qT + (size_t)t * PF_IDX_DIM * PF_IDX_HEADS;
    const int pos = base + t;
    if (i < 32) {
        const float c = cosv[(size_t)pos * 32 + i], sn = sinv[(size_t)pos * 32 + i];
        const float a = src[i], b = src[i + 32];
        dst[(size_t)(i)      * PF_IDX_HEADS + h] = a * c - b * sn;
        dst[(size_t)(i + 32) * PF_IDX_HEADS + h] = b * c + a * sn;
    }
    dst[(size_t)(64 + i) * PF_IDX_HEADS + h] = src[64 + i];   // unroped half
}

__global__ void __launch_bounds__(256)
k_pf_idx_scores3(const float *__restrict__ qT, const uint16_t *__restrict__ cache,
                 const float *__restrict__ wh, int cap, int T, int base,
                 float scale, float *__restrict__ scores, int sstride)
{
    const int t = blockIdx.y;
    if (t >= T) return;
    const int npos = base + t + 1;
    const int p = blockIdx.x * 256 + threadIdx.x;
    if (p >= npos) return;

    const int tu = __builtin_amdgcn_readfirstlane(t);
    const float *qr = qT + (size_t)tu * PF_IDX_DIM * PF_IDX_HEADS;
    const float *wr = wh + (size_t)tu * PF_IDX_HEADS;

    float acc[PF_IDX_HEADS];
    #pragma unroll
    for (int h = 0; h < PF_IDX_HEADS; h++) acc[h] = 0.f;
    for (int d = 0; d < PF_IDX_DIM; d++) {
        const float kv = pfd_b2f(cache[(size_t)d * cap + p]);
        const float *qd = qr + (size_t)d * PF_IDX_HEADS;   // 24 CONTIGUOUS, uniform
        #pragma unroll
        for (int h = 0; h < PF_IDX_HEADS; h++)
            acc[h] = fmaf(qd[h], kv, acc[h]);
    }
    float s = 0.f;
    #pragma unroll
    for (int h = 0; h < PF_IDX_HEADS; h++) s = fmaf(wr[h], fmaxf(acc[h] * scale, 0.f), s);
    scores[(size_t)t * sstride + p] = s;
}

// ---- top-2048 per query, one block a query -------------------------------
// Three 11-bit radix passes over the order-preserving flipped key, then an
// ordered gather against the exact threshold. Ported from the decode selector
// and batched over queries; below the threshold the model takes a dense path,
// so this writes the identity list and the true count instead.
__global__ void __launch_bounds__(PF_SEL_THREADS)
k_pf_idx_select(const float *__restrict__ scores, int T, int base, int want,
                int sstride, int *__restrict__ sel, int *__restrict__ nsel)
{
    const int q = blockIdx.x;
    if (q >= T) return;
    const int n = base + q + 1;
    const int t = threadIdx.x;
    int *selq = sel + (size_t)q * want;

    if (n <= want) {                       // dense path: every position, in order
        for (int i = t; i < n; i += PF_SEL_THREADS) selq[i] = i;
        for (int i = n + t; i < want; i += PF_SEL_THREADS) selq[i] = -1;
        if (t == 0) nsel[q] = n;
        return;
    }
    if (t == 0) nsel[q] = want;

    const float *sc = scores + (size_t)q * sstride;
    __shared__ uint32_t hist[PF_RBINS], suf[PF_RBINS], tmp[PF_RBINS];
    __shared__ uint32_t sprefix, smask, sabove, sbin, sabove_next;
    __shared__ uint32_t wv_a[PF_SEL_WAVES], wv_e[PF_SEL_WAVES];
    __shared__ int cur_a, cur_e;
    const int lane = t & 63, wv = t >> 6;

    if (t == 0) { sprefix = 0; smask = 0; sabove = 0; sabove_next = 0; }
    __syncthreads();

    const int shifts[3] = {21, 10, 0};
    #pragma unroll
    for (int pass = 0; pass < 3; pass++) {
        const int sh = shifts[pass];
        for (int i = t; i < PF_RBINS; i += PF_SEL_THREADS) hist[i] = 0;
        __syncthreads();
        const uint32_t pf = sprefix, mk = smask;
        for (int p = t; p < n; p += PF_SEL_THREADS) {
            const uint32_t f = pfd_flip(sc[p]);
            if ((f & mk) == pf) atomicAdd(&hist[(f >> sh) & (PF_RBINS - 1)], 1u);
        }
        __syncthreads();
        for (int i = t; i < PF_RBINS; i += PF_SEL_THREADS) suf[i] = hist[i];
        __syncthreads();
        for (int d = 1; d < PF_RBINS; d <<= 1) {          // Hillis-Steele suffix sum
            for (int i = t; i < PF_RBINS; i += PF_SEL_THREADS)
                tmp[i] = (i + d < PF_RBINS) ? suf[i] + suf[i + d] : suf[i];
            __syncthreads();
            for (int i = t; i < PF_RBINS; i += PF_SEL_THREADS) suf[i] = tmp[i];
            __syncthreads();
        }
        for (int i = t; i < PF_RBINS; i += PF_SEL_THREADS) {
            const uint32_t hi_excl = (i + 1 < PF_RBINS) ? suf[i + 1] : 0u;
            if (sabove + suf[i] >= (uint32_t)want && sabove + hi_excl < (uint32_t)want) {
                sbin = (uint32_t)i; sabove_next = sabove + hi_excl;
            }
        }
        __syncthreads();
        if (t == 0) {
            sprefix |= sbin << sh;
            smask   |= ((uint32_t)(PF_RBINS - 1)) << sh;
            sabove   = sabove_next;
        }
        __syncthreads();
    }

    const uint32_t thr = sprefix;
    if (t == 0) { cur_a = 0; cur_e = 0; }
    __syncthreads();
    const int above_total = (int)sabove;
    for (int b = 0; b < n; b += PF_SEL_THREADS) {
        const int p = b + t;
        bool ta = false, te = false;
        if (p < n) { const uint32_t f = pfd_flip(sc[p]); ta = f > thr; te = f == thr; }
        const uint64_t ba = __ballot(ta), be = __ballot(te);
        const uint32_t pa = (uint32_t)__popcll(ba & ((1ull << lane) - 1ull));
        const uint32_t pe = (uint32_t)__popcll(be & ((1ull << lane) - 1ull));
        if (lane == 0) { wv_a[wv] = (uint32_t)__popcll(ba); wv_e[wv] = (uint32_t)__popcll(be); }
        __syncthreads();
        uint32_t wa = 0, we = 0, tota = 0, tote = 0;
        for (int w = 0; w < PF_SEL_WAVES; w++) {
            if (w < wv) { wa += wv_a[w]; we += wv_e[w]; }
            tota += wv_a[w]; tote += wv_e[w];
        }
        if (ta) { const int s = cur_a + (int)(wa + pa); if (s < want) selq[s] = p; }
        if (te) { const int s = above_total + cur_e + (int)(we + pe); if (s < want) selq[s] = p; }
        __syncthreads();
        if (t == 0) { cur_a += (int)tota; cur_e += (int)tote; }
        __syncthreads();
    }
    const int filled = above_total + cur_e < want ? above_total + cur_e : want;
    for (int i = filled + t; i < want; i += PF_SEL_THREADS) selq[i] = -1;
}
