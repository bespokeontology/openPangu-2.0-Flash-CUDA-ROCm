// Correctness gate for the openPangu NVFP4 expert kernels.
// Scalar CPU reference decodes the packed bytes independently and compares
// against a host emulation of the kernel arithmetic. Runs without a GPU.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include "p92amd/p92_nvfp4_wave64.h"

static float e2m1(uint32_t nib) {
    uint32_t mag = nib & 7u, e = mag >> 1, m = mag & 1u; float v;
    if (e) { uint32_t bits = ((e + 126u) << 23) | (m << 22); memcpy(&v, &bits, 4); }
    else v = m ? 0.5f : 0.f;
    return (nib & 8u) ? -v : v;
}
static float ue4m3(uint32_t b) {
    uint32_t e = b >> 3, m = b & 7u; float v;
    if (e == 0) return (float)m * 0x1p-9f;
    if (e == 15 && m == 7) return INFINITY;
    uint32_t bits = ((e + 120u) << 23) | (m << 20); memcpy(&v, &bits, 4); return v;
}
// Independent reference: walk every value of a row in natural order.
// Byte j of a 16-value group holds value 2j low nibble, value 2j+1 high nibble.
static float ref_row_dot(const uint8_t *w, const uint8_t *s, const float *x, int K) {
    double acc = 0.0;
    for (int g = 0; g < K / 16; g++) {
        double gsum = 0.0;
        for (int j = 0; j < 8; j++) {
            uint32_t b = w[(size_t)g * 8 + j];
            gsum += (double)e2m1(b & 15u) * x[g * 16 + 2 * j];
            gsum += (double)e2m1(b >> 4)  * x[g * 16 + 2 * j + 1];
        }
        acc += gsum * (double)ue4m3(s[g]);
    }
    return (float)acc;
}
// Host emulation of the kernel: same lane decomposition and group_dot order.
static float kern_row_dot(const uint8_t *w, const uint8_t *s, const float *x,
                          int K, int lanes, int iters) {
    std::vector<float> lane_acc(lanes, 0.f);
    for (int l = 0; l < lanes; l++) {
        float acc = 0.f;
        for (int i = 0; i < iters; i++) {
            int g = i * lanes + l;
            const uint8_t *pk = w + (size_t)g * 8;
            const float *xs = x + g * 16;
            float a = 0.f;
            for (int j = 0; j < 4; j++) {
                uint32_t b0 = pk[j], b1 = pk[j + 4];
                a = fmaf(e2m1(b0 & 15u), xs[2 * j],     a);
                a = fmaf(e2m1(b0 >> 4),  xs[2 * j + 1], a);
                a = fmaf(e2m1(b1 & 15u), xs[2 * j + 8], a);
                a = fmaf(e2m1(b1 >> 4),  xs[2 * j + 9], a);
            }
            acc = fmaf(ue4m3(s[g]), a, acc);
        }
        lane_acc[l] = acc;
    }
    for (int off = lanes / 2; off; off >>= 1)
        for (int l = 0; l < off; l++) lane_acc[l] += lane_acc[l + off];
    return lane_acc[0];
}
static int check(const char *name, int K, int lanes, int iters) {
    std::mt19937 rng(12345 + K + lanes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<uint8_t> w(K / 2), s(K / 16);
    std::vector<float> x(K);
    int bad = 0; float worst = 0.f;
    for (int trial = 0; trial < 64; trial++) {
        for (auto &b : w) b = (uint8_t)byte(rng);
        for (auto &b : s) { uint8_t v; do { v = (uint8_t)byte(rng); } while ((v >> 3) == 15 && (v & 7) == 7); b = v; }
        for (auto &v : x) v = uni(rng);
        float r = ref_row_dot(w.data(), s.data(), x.data(), K);
        float k = kern_row_dot(w.data(), s.data(), x.data(), K, lanes, iters);
        float den = fabsf(r) > 1e-6f ? fabsf(r) : 1.f;
        float rel = fabsf(r - k) / den;
        if (rel > worst) worst = rel;
        if (rel > 2e-5f) bad++;
    }
    printf("  %-28s K=%4d lanes=%2d iters=%d  worst_rel=%.3e  %s\n",
           name, K, lanes, iters, worst, bad ? "FAIL" : "ok");
    return bad;
}
int main() {
    printf("P92 NVFP4 kernel arithmetic gate (lane decomposition vs independent reference)\n");
    int bad = 0;
    bad += check("gate/up 1024x2560", P92_NEMBD, 32, 5);
    bad += check("down    2560x1024", P92_NFF,    8, 8);
    printf("  strides: GU_W=%zu GU_S=%zu DN_W=%zu DN_S=%zu\n",
           P92_GU_W_BYTES, P92_GU_S_BYTES, P92_DN_W_BYTES, P92_DN_S_BYTES);
    printf("  grid:    gate/up %d blocks x %d threads, down %d blocks x 256\n",
           P92_GU_BLOCKS, P92_GATEUP_THREADS, P92_DN_BLOCKS);
    printf("%s\n", bad ? "P92_NVFP4_GATE_FAIL" : "P92_NVFP4_GATE_OK");
    return bad ? 1 : 0;
}
