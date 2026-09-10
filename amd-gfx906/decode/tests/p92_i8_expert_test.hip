// Gate: the specialised gfx906 W8A8 expert path against a scalar CPU reference
// of Huawei's contract. Relative error, not bit-exactness - the GPU reduction
// order and __expf differ from the serial reference by design.
#include "../src/p92_i8_expert.hip"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)

static const int E = 12;      // resident experts in the test arena
static const int N = P92_I8_NFF, K = P92_I8_NEMBD;

int main() {
    std::mt19937 rng(20260908);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    std::uniform_real_distribution<float> xd(-3.f, 3.f);

    std::vector<int8_t> Wg((size_t)E*N*K), Wu((size_t)E*N*K), Wd((size_t)E*K*N);
    std::vector<float>  Sg((size_t)E*N), Su((size_t)E*N), Sd((size_t)E*K);
    for (auto &v : Wg) v = (int8_t)qd(rng);
    for (auto &v : Wu) v = (int8_t)qd(rng);
    for (auto &v : Wd) v = (int8_t)qd(rng);
    for (auto &v : Sg) v = sd(rng);
    for (auto &v : Su) v = sd(rng);
    for (auto &v : Sd) v = sd(rng);

    std::vector<float> x(K);
    for (auto &v : x) v = xd(rng);
    std::vector<int>   slot(P92_I8_DISPATCH);
    std::vector<float> wts(P92_I8_DISPATCH);
    for (int k = 0; k < P92_I8_DISPATCH; k++) { slot[k] = k % E; wts[k] = 0.05f + 0.02f * k; }
    slot[5] = -1;
    if (P92_I8_DISPATCH > 8) slot[8] = -1;   // shared slot unused in this microtest

    // ---- CPU reference, exactly the contract -----------------------------
    float xm = 0.f; for (float v : x) xm = fmaxf(xm, fabsf(v));
    const float xsc = xm > 0.f ? xm / 127.f : 1.f;
    std::vector<int8_t> xq(K);
    for (int i = 0; i < K; i++) {
        int q = (int)rintf(x[i] / xsc);
        xq[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
    std::vector<float> ref(K, 0.f);
    std::vector<std::vector<int8_t>> rhq(P92_I8_DISPATCH, std::vector<int8_t>(N));
    std::vector<float> rhs(P92_I8_DISPATCH, 1.f);
    for (int k = 0; k < P92_I8_DISPATCH; k++) {
        if (slot[k] < 0) continue;
        const int s = slot[k];
        std::vector<float> h(N);
        float hm = 0.f;
        for (int r = 0; r < N; r++) {
            int ag = 0, au = 0;
            const int8_t *wg = &Wg[(size_t)s*N*K + (size_t)r*K];
            const int8_t *wu = &Wu[(size_t)s*N*K + (size_t)r*K];
            for (int i = 0; i < K; i++) { ag += (int)wg[i]*xq[i]; au += (int)wu[i]*xq[i]; }
            const float g = (float)ag * xsc * Sg[(size_t)s*N + r];
            const float u = (float)au * xsc * Su[(size_t)s*N + r];
            h[r] = g / (1.f + expf(-g)) * u;
            hm = fmaxf(hm, fabsf(h[r]));
        }
        const float hsc = hm > 0.f ? hm / 127.f : 1.f;
        rhs[k] = hsc;
        for (int r = 0; r < N; r++) {
            int q = (int)rintf(h[r] / hsc);
            rhq[k][r] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
        }
    }
    for (int r = 0; r < K; r++) {
        float total = 0.f;
        for (int k = 0; k < P92_I8_DISPATCH; k++) {
            if (slot[k] < 0) continue;
            const int s = slot[k];
            const int8_t *wr = &Wd[(size_t)s*K*N + (size_t)r*N];
            int acc = 0;
            for (int i = 0; i < N; i++) acc += (int)wr[i]*rhq[k][i];
            total = fmaf((float)acc * rhs[k] * wts[k], Sd[(size_t)s*K + r], total);
        }
        ref[r] = total;
    }

    // ---- GPU -------------------------------------------------------------
    int8_t *dWg,*dWu,*dWd,*dxq,*dhq; float *dSg,*dSu,*dSd,*dx,*dxs,*dh,*dhs,*dy,*dw; int *dslot;
    CHK(hipMalloc(&dWg,Wg.size())); CHK(hipMalloc(&dWu,Wu.size())); CHK(hipMalloc(&dWd,Wd.size()));
    CHK(hipMalloc(&dSg,Sg.size()*4)); CHK(hipMalloc(&dSu,Su.size()*4)); CHK(hipMalloc(&dSd,Sd.size()*4));
    CHK(hipMalloc(&dx,K*4)); CHK(hipMalloc(&dxq,K)); CHK(hipMalloc(&dxs,4));
    CHK(hipMalloc(&dh,(size_t)P92_I8_DISPATCH*N*4)); CHK(hipMalloc(&dhq,(size_t)P92_I8_DISPATCH*N));
    CHK(hipMalloc(&dhs,P92_I8_DISPATCH*4)); CHK(hipMalloc(&dy,K*4));
    CHK(hipMalloc(&dslot,P92_I8_DISPATCH*4)); CHK(hipMalloc(&dw,P92_I8_DISPATCH*4));
    CHK(hipMemcpy(dWg,Wg.data(),Wg.size(),hipMemcpyHostToDevice));
    CHK(hipMemcpy(dWu,Wu.data(),Wu.size(),hipMemcpyHostToDevice));
    CHK(hipMemcpy(dWd,Wd.data(),Wd.size(),hipMemcpyHostToDevice));
    CHK(hipMemcpy(dSg,Sg.data(),Sg.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dSu,Su.data(),Su.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dSd,Sd.data(),Sd.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dx,x.data(),K*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dslot,slot.data(),P92_I8_DISPATCH*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dw,wts.data(),P92_I8_DISPATCH*4,hipMemcpyHostToDevice));
    CHK(hipMemset(dy,0,K*4));

    P92I8ExpertArena a{dWg,dWu,dWd,dSg,dSu,dSd,E};
    p92_i8_quantise(dx,dxq,dxs,0);
    p92_i8_expert_ffn(a,dxq,dxs,dslot,dw,dh,dhq,dhs,dy,0);
    CHK(hipDeviceSynchronize());

    std::vector<int8_t> gxq(K); std::vector<float> gy(K), gxs(1);
    CHK(hipMemcpy(gxq.data(),dxq,K,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(gxs.data(),dxs,4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(gy.data(),dy,K*4,hipMemcpyDeviceToHost));

    int qmis = 0; for (int i=0;i<K;i++) if (gxq[i]!=xq[i]) qmis++;
    double es=0, rs=0, mx=0;
    for (int i=0;i<K;i++){ double d=gy[i]-ref[i]; es+=d*d; rs+=(double)ref[i]*ref[i]; mx=fmax(mx,fabs(d)); }
    const double rel = rs>0 ? sqrt(es/rs) : 0.0;
    printf("activation quant: scale gpu=%.9g cpu=%.9g  byte mismatches=%d/%d\n", gxs[0], xsc, qmis, K);
    printf("expert ffn:       relative_l2=%.3e  max_abs_err=%.3e  ref_l2=%.6g\n", rel, mx, sqrt(rs));
    const bool pass = (qmis==0) && (rel < 1e-4);
    printf("%s (gate: quant exact, relative_l2 < 1e-4)\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
