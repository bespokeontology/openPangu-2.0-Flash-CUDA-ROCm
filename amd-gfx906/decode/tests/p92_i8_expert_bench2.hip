// Production-shape benchmark: 106 resident int8 experts (the H determined from
// the shipped checkpoint bytes), top-8 selection, one layer per iteration.
#include "../src/p92_i8_expert.hip"
#include <cstdio>
#include <vector>
#include <random>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
int main(int argc,char**argv){
    const int E = argc>1?atoi(argv[1]):106;
    const int N=P92_I8_NFF,K=P92_I8_NEMBD;
    int8_t *dWg,*dWu,*dWd,*dxq,*dhq; float *dSg,*dSu,*dSd,*dxs,*dh,*dhs,*dy,*dw; int *dslot;
    size_t gw=(size_t)E*N*K, dwb=(size_t)E*K*N;
    CHK(hipMalloc(&dWg,gw)); CHK(hipMalloc(&dWu,gw)); CHK(hipMalloc(&dWd,dwb));
    CHK(hipMalloc(&dSg,(size_t)E*N*4)); CHK(hipMalloc(&dSu,(size_t)E*N*4)); CHK(hipMalloc(&dSd,(size_t)E*K*4));
    CHK(hipMalloc(&dxq,K)); CHK(hipMalloc(&dxs,4));
    CHK(hipMalloc(&dh,(size_t)P92_I8_DISPATCH*N*4)); CHK(hipMalloc(&dhq,(size_t)P92_I8_DISPATCH*N));
    CHK(hipMalloc(&dhs,P92_I8_DISPATCH*4)); CHK(hipMalloc(&dy,K*4));
    CHK(hipMalloc(&dslot,P92_I8_DISPATCH*4)); CHK(hipMalloc(&dw,P92_I8_DISPATCH*4));
    CHK(hipMemset(dWg,3,gw)); CHK(hipMemset(dWu,2,gw)); CHK(hipMemset(dWd,1,dwb));
    CHK(hipMemset(dxq,5,K));
    std::vector<float> ones((size_t)E*K,0.01f);
    CHK(hipMemcpy(dSg,ones.data(),(size_t)E*N*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dSu,ones.data(),(size_t)E*N*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dSd,ones.data(),(size_t)E*K*4,hipMemcpyHostToDevice));
    float xs1=0.02f; CHK(hipMemcpy(dxs,&xs1,4,hipMemcpyHostToDevice));
    std::vector<int> slot(P92_I8_DISPATCH); std::vector<float> wt(P92_I8_DISPATCH,0.125f);
    std::mt19937 rng(7); for(int k=0;k<P92_I8_DISPATCH;k++) slot[k]=rng()%E;
    CHK(hipMemcpy(dslot,slot.data(),P92_I8_DISPATCH*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dw,wt.data(),P92_I8_DISPATCH*4,hipMemcpyHostToDevice));
    P92I8ExpertArena a{dWg,dWu,dWd,dSg,dSu,dSd,E};
    for(int i=0;i<40;i++) p92_i8_expert_ffn(a,dxq,dxs,dslot,dw,dh,dhq,dhs,dy,0);
    CHK(hipDeviceSynchronize());
    hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
    const int IT=400; hipEventRecord(t0,0);
    for(int i=0;i<IT;i++) p92_i8_expert_ffn(a,dxq,dxs,dslot,dw,dh,dhq,dhs,dy,0);
    hipEventRecord(t1,0); CHK(hipEventSynchronize(t1));
    float ms=0; hipEventElapsedTime(&ms,t0,t1);
    const double us=ms*1000.0/IT;
    const double bytes=(double)P92_I8_DISPATCH*(2.0*N*K+(double)K*N+ (2.0*N+K)*4);
    printf("arena %d experts (%.0f MiB)\n", E, (double)(2*gw+dwb+(size_t)E*(2*N+K)*4)/1048576.0);
    printf("layer top-8 int8 ffn: %.2f us   %.2f MB   %.1f GB/s\n", us, bytes/1e6, bytes/(us/1e6)/1e9);
    printf("44 MoE layers:        %.2f ms/token -> %.1f tok/s (experts only)\n", us*44/1000.0, 1000.0/(us*44/1000.0));
    printf("nvfp4 reference:      248.79 us/expert = %.2f ms/token -> %.1f tok/s\n", 248.79*8*44/1000.0, 1000.0/(248.79*8*44/1000.0));
    return 0;
}
