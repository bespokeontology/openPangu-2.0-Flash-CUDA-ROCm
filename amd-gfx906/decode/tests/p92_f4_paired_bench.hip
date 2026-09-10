// Paired A/B: mechanism-1 fp32-decode NVFP4 gate/up against the dot4 rewrite,
// same arena, same shapes, interleaved arms.
#include "../src/p92_nvfp4_wave64.hip"
#include "../src/p92_nvfp4_dot4.hip"
#include <cstdio>
#include <vector>
#include <random>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
int main(int argc,char**argv){
    const int E = argc>1?atoi(argv[1]):150;
    const int N=P92_NFF,K=P92_NEMBD;
    uint8_t *Wg,*Wu,*Sg,*Su; int8_t*xq; float*xs,*xf,*h; int*slot;
    size_t wb=(size_t)E*N*K/2, sb=(size_t)E*N*K/16;
    CHK(hipMalloc(&Wg,wb)); CHK(hipMalloc(&Wu,wb));
    CHK(hipMalloc(&Sg,sb)); CHK(hipMalloc(&Su,sb));
    CHK(hipMalloc(&xq,K)); CHK(hipMalloc(&xs,4)); CHK(hipMalloc(&xf,K*4));
    CHK(hipMalloc(&h,(size_t)P92_DISPATCH*N*4)); CHK(hipMalloc(&slot,P92_DISPATCH*4));
    CHK(hipMemset(Wg,0x53,wb)); CHK(hipMemset(Wu,0x35,wb));
    CHK(hipMemset(Sg,60,sb));   CHK(hipMemset(Su,61,sb));
    CHK(hipMemset(xq,7,K));
    float one=0.02f; CHK(hipMemcpy(xs,&one,4,hipMemcpyHostToDevice));
    std::vector<int> sl(P92_DISPATCH); std::mt19937 r(3);
    for(int k=0;k<P92_DISPATCH;k++) sl[k]=r()%E;
    CHK(hipMemcpy(slot,sl.data(),P92_DISPATCH*4,hipMemcpyHostToDevice));

    auto A=[&]{ for(int k=0;k<P92_DISPATCH;k++)
        hipLaunchKernelGGL(k_p92_nvfp4_gateup_1024x2560,dim3(N/8),dim3(256),0,0,
                           Wg,Sg,Wu,Su,xf,h+(size_t)k*N,1.f,1.f,sl[k]); };
    auto B=[&]{ hipLaunchKernelGGL(k_p92_f4_gateup,dim3(N/8,P92_DISPATCH),dim3(256),0,0,
                           Wg,Sg,Wu,Su,xq,xs,slot,1.f,1.f,h); };
    for(int i=0;i<20;i++){A();B();} CHK(hipDeviceSynchronize());
    hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
    const int IT=200; double ta=0,tb=0; float ms;
    for(int rep=0;rep<3;rep++){
      hipEventRecord(t0,0); for(int i=0;i<IT;i++) A(); hipEventRecord(t1,0);
      CHK(hipEventSynchronize(t1)); hipEventElapsedTime(&ms,t0,t1); ta+=ms;
      hipEventRecord(t0,0); for(int i=0;i<IT;i++) B(); hipEventRecord(t1,0);
      CHK(hipEventSynchronize(t1)); hipEventElapsedTime(&ms,t0,t1); tb+=ms;
    }
    const double ua=ta*1000.0/(IT*3), ub=tb*1000.0/(IT*3);
    const double by=(double)P92_DISPATCH*2*(N*K/2+N*K/16);
    printf("arena %d cold experts, top-8 gate+up\n",E);
    printf("  A mechanism-1 fp32 decode : %8.2f us  %6.1f GB/s\n", ua, by/(ua/1e6)/1e9);
    printf("  B dot4 rewrite            : %8.2f us  %6.1f GB/s\n", ub, by/(ub/1e6)/1e9);
    printf("  speedup                   : %8.2fx\n", ua/ub);
    return 0;
}
