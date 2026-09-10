// Gate: (1) the v_perm_b32 nibble decode is exactly twice the E2M1 value for
// all 256 byte patterns; (2) the dot4 gate/up kernel matches a CPU reference
// built from exact E2M1 values and the same int8 activation.
#include "../src/p92_nvfp4_dot4.hip"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)

__global__ void k_dq_probe(uint32_t *out) {          // byte b -> two decoded int8
    const uint32_t b = threadIdx.x;                  // 0..255
    out[b] = p92_dq4((b & 0x0Fu) | ((b >> 4) << 8)); // low nibble in byte0, high in byte1
}
static float e2m1(uint32_t nib) {
    uint32_t mag = nib & 7u, e = mag >> 1, m = mag & 1u;
    float v = e ? ldexpf(1.f + 0.5f * m, (int)e - 1) : (m ? 0.5f : 0.f);
    return (nib & 8u) ? -v : v;
}
static float ue4m3(uint32_t b) {
    uint32_t e = b >> 3, m = b & 7u;
    if (e == 0) return (float)m * 0x1p-9f;
    if (e == 15 && m == 7) return INFINITY;
    return ldexpf(1.f + m / 8.f, (int)e - 7);
}
int main() {
    // ---- (1) exhaustive decode ------------------------------------------
    uint32_t *d; CHK(hipMalloc(&d, 256 * 4));
    hipLaunchKernelGGL(k_dq_probe, dim3(1), dim3(256), 0, 0, d);
    CHK(hipDeviceSynchronize());
    std::vector<uint32_t> h(256);
    CHK(hipMemcpy(h.data(), d, 256 * 4, hipMemcpyDeviceToHost));
    int bad = 0;
    for (uint32_t b = 0; b < 256; b++) {
        int lo = (int8_t)(h[b] & 0xFF), hi = (int8_t)((h[b] >> 8) & 0xFF);
        if (lo != (int)lroundf(2.f * e2m1(b & 15u)) ||
            hi != (int)lroundf(2.f * e2m1(b >> 4))) bad++;
    }
    printf("decode: %d/256 byte patterns wrong\n", bad);

    // ---- (2) gate/up against a CPU reference ----------------------------
    const int E = 4, N = P92_NFF, K = P92_NEMBD;
    std::mt19937 rng(20260908);
    std::vector<uint8_t> Wg((size_t)E*N*K/2), Wu((size_t)E*N*K/2),
                         Sg((size_t)E*N*K/16), Su((size_t)E*N*K/16);
    for (auto &v : Wg) v = (uint8_t)(rng() & 0xFF);
    for (auto &v : Wu) v = (uint8_t)(rng() & 0xFF);
    for (auto &v : Sg) v = (uint8_t)(56 + (rng() % 16));      // sane ue4m3 exponents
    for (auto &v : Su) v = (uint8_t)(56 + (rng() % 16));
    std::vector<int8_t> xq(K), xq_nat(K);
    for (auto &v : xq_nat) v = (int8_t)((int)(rng() % 255) - 127);
    for (int g=0; g<K/16; g++) for (int i=0;i<16;i++)      // permuted for the kernel
        xq[g*16 + ((i&1)? 8+(i>>1) : (i>>1))] = xq_nat[g*16+i];
    const float xsc = 0.0187f, sc2g = 1.31f, sc2u = 0.77f;
    std::vector<int> slot(P92_DISPATCH); std::vector<float> wts(P92_DISPATCH, .125f);
    for (int k = 0; k < P92_DISPATCH; k++) slot[k] = k % E;
    slot[3] = -1;

    std::vector<float> ref((size_t)P92_DISPATCH * N, 0.f);
    for (int k = 0; k < P92_DISPATCH; k++) {
        if (slot[k] < 0) continue;
        const int s = slot[k];
        for (int r = 0; r < N; r++) {
            double ag = 0, au = 0;
            for (int g = 0; g < K / 16; g++) {
                const uint8_t *pg = &Wg[(size_t)s*N*K/2 + (size_t)r*(K/2) + g*8];
                const uint8_t *pu = &Wu[(size_t)s*N*K/2 + (size_t)r*(K/2) + g*8];
                double sg = 0, su = 0;
                for (int j = 0; j < 8; j++) {
                    sg += (double)e2m1(pg[j] & 15u) * xq_nat[g*16 + 2*j]
                        + (double)e2m1(pg[j] >> 4)  * xq_nat[g*16 + 2*j + 1];
                    su += (double)e2m1(pu[j] & 15u) * xq_nat[g*16 + 2*j]
                        + (double)e2m1(pu[j] >> 4)  * xq_nat[g*16 + 2*j + 1];
                }
                ag += sg * ue4m3(Sg[(size_t)s*N*K/16 + (size_t)r*(K/16) + g]);
                au += su * ue4m3(Su[(size_t)s*N*K/16 + (size_t)r*(K/16) + g]);
            }
            const float gg = (float)(ag * xsc * sc2g), uu = (float)(au * xsc * sc2u);
            ref[(size_t)k*N + r] = gg / (1.f + expf(-gg)) * uu;
        }
    }
    uint8_t *dWg,*dWu,*dSg,*dSu; int8_t *dxq; float *dxs,*dh; int *dslot;
    CHK(hipMalloc(&dWg,Wg.size())); CHK(hipMalloc(&dWu,Wu.size()));
    CHK(hipMalloc(&dSg,Sg.size())); CHK(hipMalloc(&dSu,Su.size()));
    CHK(hipMalloc(&dxq,K)); CHK(hipMalloc(&dxs,(size_t)(K/16)*4));
    CHK(hipMalloc(&dh,(size_t)P92_DISPATCH*N*4)); CHK(hipMalloc(&dslot,P92_DISPATCH*4));
    CHK(hipMemcpy(dWg,Wg.data(),Wg.size(),hipMemcpyHostToDevice));
    CHK(hipMemcpy(dWu,Wu.data(),Wu.size(),hipMemcpyHostToDevice));
    CHK(hipMemcpy(dSg,Sg.data(),Sg.size(),hipMemcpyHostToDevice));
    CHK(hipMemcpy(dSu,Su.data(),Su.size(),hipMemcpyHostToDevice));
    CHK(hipMemcpy(dxq,xq.data(),K,hipMemcpyHostToDevice));
    { std::vector<float> xsg(K/16, xsc);   // one scale per 16 values
      CHK(hipMemcpy(dxs,xsg.data(),xsg.size()*4,hipMemcpyHostToDevice)); }
    CHK(hipMemcpy(dslot,slot.data(),P92_DISPATCH*4,hipMemcpyHostToDevice));
    CHK(hipMemset(dh,0,(size_t)P92_DISPATCH*N*4));
    // gate/up now takes the per-slot weight_scale_2 table the arena stores,
    // [slot][3] as gate, up, down, rather than two scalars
    float *ds2; CHK(hipMalloc(&ds2,(size_t)E*3*4));
    { std::vector<float> h2((size_t)E*3);
      for(int e2=0;e2<E;e2++){ h2[e2*3+0]=sc2g; h2[e2*3+1]=sc2u; h2[e2*3+2]=1.f; }
      CHK(hipMemcpy(ds2,h2.data(),h2.size()*4,hipMemcpyHostToDevice)); }
    hipLaunchKernelGGL(k_p92_f4_gateup, dim3(N/8,P92_DISPATCH), dim3(256), 0, 0,
                       dWg,dSg,dWu,dSu,dxq,dxs,dslot,ds2,dh);
    CHK(hipDeviceSynchronize());
    std::vector<float> got((size_t)P92_DISPATCH*N);
    CHK(hipMemcpy(got.data(),dh,got.size()*4,hipMemcpyDeviceToHost));
    double es=0, rs=0, mx=0;
    for (int k = 0; k < P92_DISPATCH; k++) {
        if (slot[k] < 0) continue;
        for (int r = 0; r < N; r++) {
            double a=got[(size_t)k*N+r], b=ref[(size_t)k*N+r], e=a-b;
            es+=e*e; rs+=b*b; mx=fmax(mx,fabs(e));
        }
    }
    const double rel = rs>0 ? sqrt(es/rs) : 0;
    printf("gate/up dot4 vs exact E2M1 reference: relative_l2=%.3e max_abs=%.3e\n", rel, mx);
    const bool pass = (bad==0) && (rel < 1e-5);
    printf("%s (gate: decode exact, relative_l2 < 1e-5)\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
