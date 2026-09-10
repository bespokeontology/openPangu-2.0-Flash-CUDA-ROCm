// Where do 18.18 us go in an 8-of-256 selection? Time the kernel with the round
// count varied: if cost scales with rounds it is the dependent shuffle chain,
// if it does not it is something else entirely.
#include "../src/p92_router.hip"
#include <cstdio>
#include <vector>
#include <random>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)

template <int ROUNDS>
__global__ void __launch_bounds__(64)
k_probe(const float *__restrict__ logits, const float *__restrict__ bias,
        int *__restrict__ ids, float *__restrict__ wts) {
    const int lane = threadIdx.x;
    float sc[4], sel[4]; int idx[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int e = j * 64 + lane;
        const float s = 1.f / (1.f + __expf(-logits[e]));
        sc[j] = s; sel[j] = s + bias[e]; idx[j] = e;
    }
    float wsum = 0.f;
    #pragma unroll
    for (int k = 0; k < ROUNDS; k++) {
        float bv = sel[0]; int bi = idx[0];
        #pragma unroll
        for (int j = 1; j < 4; j++)
            if (sel[j] > bv || (sel[j] == bv && idx[j] < bi)) { bv = sel[j]; bi = idx[j]; }
        #pragma unroll
        for (int off = 32; off; off >>= 1) {
            const float ov = __shfl_down(bv, off, 64);
            const int   oi = __shfl_down(bi, off, 64);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        bi = __shfl(bi, 0, 64);
        float w = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; j++) if (idx[j] == bi) { w = sc[j]; sel[j] = -3.0e38f; }
        w = __shfl(w, bi & 63, 64);
        wsum += w;
        if (lane == 0) { ids[k] = bi; wts[k] = w; }
    }
    if (lane == 0 && wsum > 0.f) wts[0] *= 2.5f / wsum;
}


// Is the cross-lane op the cost, or the dependent chain in general? Same round
// count, same arithmetic, but the reduction replaced by ALU ops of equal
// dependency depth. Semantics are wrong on purpose; this measures only cost.
template <int ROUNDS>
__global__ void __launch_bounds__(64)
k_noshuf(const float *__restrict__ logits, const float *__restrict__ bias,
         int *__restrict__ ids, float *__restrict__ wts) {
    const int lane = threadIdx.x;
    float sc[4], sel[4]; int idx[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int e = j * 64 + lane;
        const float s = 1.f / (1.f + __expf(-logits[e]));
        sc[j] = s; sel[j] = s + bias[e]; idx[j] = e;
    }
    float wsum = 0.f;
    #pragma unroll
    for (int k = 0; k < ROUNDS; k++) {
        float bv = sel[0]; int bi = idx[0];
        #pragma unroll
        for (int j = 1; j < 4; j++)
            if (sel[j] > bv || (sel[j] == bv && idx[j] < bi)) { bv = sel[j]; bi = idx[j]; }
        #pragma unroll
        for (int off = 32; off; off >>= 1) {          // same depth, ALU only
            const float ov = bv * 1.0000001f + 1e-9f;
            const int   oi = bi ^ off;
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) if (idx[j] == bi) { wsum += sc[j]; sel[j] = -3.0e38f; }
        if (lane == 0) { ids[k] = bi; wts[k] = wsum; }
    }
    if (lane == 0 && wsum > 0.f) wts[0] *= 2.5f / wsum;
}

// DPP row_shr/row_bcast: the same reduction on the ALU path instead of the LDS
// path. This is what AMD aiter mHC kernel uses for its cross-lane work.
#define P92_DPP(x, ctrl) __builtin_amdgcn_update_dpp(0, (x), (ctrl), 0xF, 0xF, false)
template <int ROUNDS>
__global__ void __launch_bounds__(64)
k_dpp(const float *__restrict__ logits, const float *__restrict__ bias,
      int *__restrict__ ids, float *__restrict__ wts) {
    const int lane = threadIdx.x;
    float sc[4], sel[4]; int idx[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int e = j * 64 + lane;
        const float s = 1.f / (1.f + __expf(-logits[e]));
        sc[j] = s; sel[j] = s + bias[e]; idx[j] = e;
    }
    float wsum = 0.f;
    #pragma unroll
    for (int k = 0; k < ROUNDS; k++) {
        float bv = sel[0]; int bi = idx[0];
        #pragma unroll
        for (int j = 1; j < 4; j++)
            if (sel[j] > bv || (sel[j] == bv && idx[j] < bi)) { bv = sel[j]; bi = idx[j]; }
        // row_shr 1,2,4,8 then row_bcast15, row_bcast31; the control must be a
        // literal so the steps are written out
        #define P92_STEP(C) { const float ov = __int_as_float(P92_DPP(__float_as_int(bv), C)); \
                              const int oi = P92_DPP(bi, C); \
                              if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
        P92_STEP(0x111) P92_STEP(0x112) P92_STEP(0x114) P92_STEP(0x118)
        P92_STEP(0x142) P92_STEP(0x143)
        #undef P92_STEP
        bi = __builtin_amdgcn_readlane(bi, 63);
        #pragma unroll
        for (int j = 0; j < 4; j++) if (idx[j] == bi) { wsum += sc[j]; sel[j] = -3.0e38f; }
        if (lane == 0) { ids[k] = bi; wts[k] = wsum; }
    }
    if (lane == 0 && wsum > 0.f) wts[0] *= 2.5f / wsum;
}

int main(){
    float *dl,*db,*dw; int *di;
    CHK(hipMalloc(&dl,256*4)); CHK(hipMalloc(&db,256*4));
    CHK(hipMalloc(&dw,8*4)); CHK(hipMalloc(&di,8*4));
    std::vector<float> h(256); std::mt19937 r(1);
    std::normal_distribution<float> nd(0,1);
    for(auto&v:h) v=nd(r); CHK(hipMemcpy(dl,h.data(),256*4,hipMemcpyHostToDevice));
    for(auto&v:h) v=nd(r)*0.1f; CHK(hipMemcpy(db,h.data(),256*4,hipMemcpyHostToDevice));
    hipEvent_t a,b; CHK(hipEventCreate(&a)); CHK(hipEventCreate(&b));
    const int IT=3000;
    auto run=[&](const char*name, auto fn){
        for(int i=0;i<200;i++) fn(); (void)hipDeviceSynchronize();
        (void)hipEventRecord(a,0); for(int i=0;i<IT;i++) fn();
        (void)hipEventRecord(b,0); (void)hipEventSynchronize(b);
        float ms; (void)hipEventElapsedTime(&ms,a,b);
        printf("  %-28s %7.2f us\n", name, ms*1000.0/IT); };
    run("production, 8 rounds",[&]{ hipLaunchKernelGGL(k_p92_route_top8,dim3(1),dim3(64),0,0,dl,db,di,dw); });
    run("probe, 8 rounds",[&]{ hipLaunchKernelGGL((k_probe<8>),dim3(1),dim3(64),0,0,dl,db,di,dw); });
    run("probe, 4 rounds",[&]{ hipLaunchKernelGGL((k_probe<4>),dim3(1),dim3(64),0,0,dl,db,di,dw); });
    run("probe, 2 rounds",[&]{ hipLaunchKernelGGL((k_probe<2>),dim3(1),dim3(64),0,0,dl,db,di,dw); });
    run("probe, 1 round",[&]{ hipLaunchKernelGGL((k_probe<1>),dim3(1),dim3(64),0,0,dl,db,di,dw); });
    run("probe, 0 rounds (load+sigmoid)",[&]{ hipLaunchKernelGGL((k_probe<0>),dim3(1),dim3(64),0,0,dl,db,di,dw); });
    run("no cross-lane, 8 rounds",[&]{ hipLaunchKernelGGL((k_noshuf<8>),dim3(1),dim3(64),0,0,dl,db,di,dw); });
    run("DPP reduction, 8 rounds",[&]{ hipLaunchKernelGGL((k_dpp<8>),dim3(1),dim3(64),0,0,dl,db,di,dw); });
    return 0;
}
