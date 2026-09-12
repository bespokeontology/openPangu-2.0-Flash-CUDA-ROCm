// Prefill projections on rocBLAS (Tensile int8x4 / v_dot4 kernels, gfx906) for openPangu.
//
// WHY. k_pf_proj holds one NVFP4 weight row per 32-lane group and sweeps the chunk's
// tokens past it: correct reuse, but every (token, row) pair ends in a 5-step cross-lane
// reduction, so the kernel issues ~1 instruction per 2 MACs and measures ~2 TMAC/s.
// The Qwen lane measured Tensile's int8 GEMMs at 16-20 TMAC/s at M=256 on this card.
// The three NVFP4 projections (q_a, q_b, o_proj) and the two DSA indexer projections get
// a load-time int8 PER-ROW mirror (row scale = absmax/127 of the dequantised row); the
// bf16 ones (kv_a, router gate, indexer weights_proj) get an fp32 copy and run on sgemm,
// which also retires the per-token GEMV loops (k_pf_bf16_gemv_rows / k_pf_route_gemv
// re-read the whole weight once PER TOKEN: 755 MB a chunk-layer for kv_a alone).
//
// REPRESENTATION. Activations are quantised per TOKEN (one scale a row, natural order);
// the MoE keeps its own per-16 permuted quantisation (k_pf_quant_k), untouched.
// C[M][N] = X8[M][K] . W8[N][K]^T in int32 (rocblas_gemm_ex, op T / op N, the packed
// int8x4 layout coincides with these natural row-major layouts), then an epilogue
// y = C * rs[n] * xs[t]. Precision: per-row weights and per-token activations are
// coarser than the per-16 groups of the hand-written path; the whole-engine gate is
// the token stream and the text, not bit identity (operator's law).
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdint>
#include <cstdio>

// ---- NVFP4 row -> fp32 -> int8 per-row mirror. One block a row, K <= 6144. ----
// value = 0.5 * s2 * ue4m3(scale of the 16-group) * dq4(nibble); nibble order: byte j
// holds element 2j (low) and 2j+1 (high). Scales are in SPEC order (never shuffled
// for the projections).
template <int K>
__global__ void __launch_bounds__(256)
k_pf_rb_mirror(const uint8_t *__restrict__ W, const uint8_t *__restrict__ S, float s2,
               int N, int8_t *__restrict__ W8, float *__restrict__ rs)
{
    constexpr int PER = K / 256;                 // elements a thread (10 / 4 / 24 / 12)
    const int row = blockIdx.x;
    if (row >= N) return;
    const uint8_t *wr = W + (size_t)row * (K / 2);
    const uint8_t *sr = S + (size_t)row * (K / 16);
    float v[PER]; float m = 0.f;
    #pragma unroll
    for (int i = 0; i < PER; i++) {
        const int e = i * 256 + threadIdx.x;
        const uint32_t byte = wr[e >> 1];
        const uint32_t nib = (e & 1) ? (byte >> 4) : (byte & 15u);
        const int mag = nib & 7u;
        const int dq = (mag < 4) ? mag : (mag == 4 ? 4 : (mag == 5 ? 6 : (mag == 6 ? 8 : 12)));   // doubled E2M1
        const float w = 0.5f * s2 * p92_ue4m3_d(sr[e >> 4]) * (float)((nib & 8u) ? -dq : dq);
        v[i] = w; m = fmaxf(m, fabsf(w));
    }
    __shared__ float red[4];
    #pragma unroll
    for (int o = 32; o; o >>= 1) m = fmaxf(m, __shfl_down(m, o, 64));
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = m;
    __syncthreads();
    m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    const float sc = m > 0.f ? m * (1.f / 127.f) : 1.f;
    const float inv = m > 0.f ? 127.f / m : 0.f;
    if (threadIdx.x == 0) rs[row] = sc;
    #pragma unroll
    for (int i = 0; i < PER; i++) {
        const int e = i * 256 + threadIdx.x;
        int q = (int)rintf(v[i] * inv);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        W8[(size_t)row * K + e] = (int8_t)q;
    }
}

// bf16 [N][K] -> fp32 [N][K]
__global__ void __launch_bounds__(256)
k_pf_rb_bf16_f32(const uint16_t *__restrict__ src, float *__restrict__ dst, size_t n)
{
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = __uint_as_float((uint32_t)src[i] << 16);
}

// ---- per-token activation quantiser: one block a token, natural order, one scale ----
template <int K>
__global__ void __launch_bounds__(256)
k_pf_rb_quant_tok(const float *__restrict__ x, int T, int8_t *__restrict__ xq, float *__restrict__ xs)
{
    constexpr int PER = K / 256;
    const int t = blockIdx.x;
    if (t >= T) return;
    const float *xr = x + (size_t)t * K;
    float v[PER]; float m = 0.f;
    #pragma unroll
    for (int i = 0; i < PER; i++) { v[i] = xr[i * 256 + threadIdx.x]; m = fmaxf(m, fabsf(v[i])); }
    __shared__ float red[4];
    #pragma unroll
    for (int o = 32; o; o >>= 1) m = fmaxf(m, __shfl_down(m, o, 64));
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = m;
    __syncthreads();
    m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    const float sc = m > 0.f ? m * (1.f / 127.f) : 1.f;
    const float inv = m > 0.f ? 127.f / m : 0.f;
    if (threadIdx.x == 0) xs[t] = sc;
    #pragma unroll
    for (int i = 0; i < PER; i++) {
        int q = (int)rintf(v[i] * inv);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        xq[(size_t)t * K + i * 256 + threadIdx.x] = (int8_t)q;
    }
}

// ---- epilogue: y[t][n] = C[t][n] * rs[n] * xs[t] ----
__global__ void __launch_bounds__(256)
k_pf_rb_epi(const int32_t *__restrict__ C, const float *__restrict__ rs, const float *__restrict__ xs,
            int T, int N, float *__restrict__ y)
{
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= (size_t)T * N) return;
    const int t = (int)(i / N), n = (int)(i - (size_t)t * N);
    y[i] = (float)C[i] * rs[n] * xs[t];
}

// ---- host side ----
struct PfRb {
    rocblas_handle h[4];
    int32_t *C[4];          // [T][Nmax] int32 scratch
    int8_t  *xq2560[4], *xq1024[4], *xq6144[4];   // per-token int8 activations by input width
    float   *xs2560[4], *xs1024[4], *xs6144[4];   // per-token scales
    int      ok;
};

static int pf_rb_init(PfRb *rb, int T, int nmax) {
    rb->ok = 0;
    for (int d = 0; d < 4; d++) {
        if (hipSetDevice(d) != hipSuccess) return -1;
        if (rocblas_create_handle(&rb->h[d]) != rocblas_status_success) { fprintf(stderr, "rocblas_create_handle(%d) failed\n", d); return -1; }
        if (rocblas_set_stream(rb->h[d], 0) != rocblas_status_success) return -1;
        if (hipMalloc(&rb->C[d], (size_t)T * nmax * 4) != hipSuccess) return -1;
        if (hipMalloc(&rb->xq2560[d], (size_t)T * 2560) != hipSuccess) return -1;
        if (hipMalloc(&rb->xq1024[d], (size_t)T * 1024) != hipSuccess) return -1;
        if (hipMalloc(&rb->xq6144[d], (size_t)T * 6144) != hipSuccess) return -1;
        if (hipMalloc(&rb->xs2560[d], (size_t)T * 4) != hipSuccess) return -1;
        if (hipMalloc(&rb->xs1024[d], (size_t)T * 4) != hipSuccess) return -1;
        if (hipMalloc(&rb->xs6144[d], (size_t)T * 4) != hipSuccess) return -1;
    }
    rb->ok = 1;
    return 0;
}

// int8 GEMM: C[M][N] = X8[M][K] . W8[N][K]^T (int32)
static inline int pf_rb_gemm_i8(rocblas_handle h, const int8_t *W8, int N, int K, const int8_t *X8, int M, int32_t *C) {
    const int32_t alpha = 1, beta = 0;
    const rocblas_status st = rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, N, M, K, &alpha,
            W8, rocblas_datatype_i8_r, K, X8, rocblas_datatype_i8_r, K, &beta,
            C, rocblas_datatype_i32_r, N, C, rocblas_datatype_i32_r, N,
            rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, rocblas_gemm_flags_pack_int8x4);
    if (st != rocblas_status_success) { fprintf(stderr, "rocblas_gemm_ex int8 status %d (N=%d K=%d M=%d)\n", (int)st, N, K, M); return -1; }
    return 0;
}
// fp32 GEMM: Y[M][N] = X[M][K] . Wf[N][K]^T
static inline int pf_rb_gemm_f32(rocblas_handle h, const float *Wf, int N, int K, const float *X, int M, float *Y) {
    const float alpha = 1.f, beta = 0.f;
    const rocblas_status st = rocblas_sgemm(h, rocblas_operation_transpose, rocblas_operation_none, N, M, K, &alpha,
            Wf, K, X, K, &beta, Y, N);
    if (st != rocblas_status_success) { fprintf(stderr, "rocblas_sgemm status %d (N=%d K=%d M=%d)\n", (int)st, N, K, M); return -1; }
    return 0;
}
