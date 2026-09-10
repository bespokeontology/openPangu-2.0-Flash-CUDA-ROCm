// Full routed-FFN layer: hot int8 experts + cold nvfp4 experts, one dispatch,
// at the production H. Measures what a MoE layer actually costs per token.
#include "../src/p92_i8_expert.hip"
#include "../src/p92_nvfp4_dot4.hip"
#include <cstdio>
#include <vector>
#include <random>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
int main(int argc,char**argv){
    const int H  = argc>1?atoi(argv[1]):64;      // resident int8 experts
    const int C  = 256-H;                        // resident nvfp4 experts
    const int nc = argc>2?atoi(argv[2]):2;       // selections landing cold
    const int N=P92_NFF,K=P92_NEMBD;
    int8_t *iWg,*iWu,*iWd,*xq,*hq; float *iSg,*iSu,*iSd,*xs,*hid,*hs,*y,*wt; int *si,*sc;
    uint8_t *fWg,*fWu,*fWd,*fSg,*fSu,*fSd;
    size_t iw=(size_t)H*N*K, fw=(size_t)C*N*K/2, fs=(size_t)C*N*K/16;
    CHK(hipMalloc(&iWg,iw)); CHK(hipMalloc(&iWu,iw)); CHK(hipMalloc(&iWd,iw));
    CHK(hipMalloc(&iSg,(size_t)H*N*4)); CHK(hipMalloc(&iSu,(size_t)H*N*4)); CHK(hipMalloc(&iSd,(size_t)H*K*4));
    CHK(hipMalloc(&fWg,fw)); CHK(hipMalloc(&fWu,fw)); CHK(hipMalloc(&fWd,fw));
    CHK(hipMalloc(&fSg,fs)); CHK(hipMalloc(&fSu,fs)); CHK(hipMalloc(&fSd,fs));
    CHK(hipMalloc(&xq,K)); CHK(hipMalloc(&xs,4)); CHK(hipMalloc(&hid,(size_t)P92_DISPATCH*N*4));
    CHK(hipMalloc(&hq,(size_t)P92_DISPATCH*N)); CHK(hipMalloc(&hs,P92_DISPATCH*4));
    CHK(hipMalloc(&y,K*4)); CHK(hipMalloc(&wt,P92_DISPATCH*4));
    CHK(hipMalloc(&si,P92_DISPATCH*4)); CHK(hipMalloc(&sc,P92_DISPATCH*4));
    CHK(hipMemset(iWg,3,iw)); CHK(hipMemset(iWu,2,iw)); CHK(hipMemset(iWd,1,iw));
    CHK(hipMemset(fWg,0x53,fw)); CHK(hipMemset(fWu,0x35,fw)); CHK(hipMemset(fWd,0x44,fw));
    CHK(hipMemset(fSg,60,fs)); CHK(hipMemset(fSu,61,fs)); CHK(hipMemset(fSd,60,fs));
    CHK(hipMemset(xq,5,K));
    std::vector<float> o((size_t)H*K,0.01f);
    CHK(hipMemcpy(iSg,o.data(),(size_t)H*N*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(iSu,o.data(),(size_t)H*N*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(iSd,o.data(),(size_t)H*K*4,hipMemcpyHostToDevice));
    float one=0.02f; CHK(hipMemcpy(xs,&one,4,hipMemcpyHostToDevice));
    std::vector<float> w(P92_DISPATCH,0.125f);
    CHK(hipMemcpy(wt,w.data(),P92_DISPATCH*4,hipMemcpyHostToDevice));
    // first (8-nc) selections hot, last nc cold
    std::vector<int> vi(P92_DISPATCH,-1), vc(P92_DISPATCH,-1); std::mt19937 r(11);
    for(int k=0;k<P92_DISPATCH-nc;k++) vi[k]=r()%H;
    for(int k=P92_DISPATCH-nc;k<P92_DISPATCH;k++) vc[k]=r()%C;
    CHK(hipMemcpy(si,vi.data(),P92_DISPATCH*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(sc,vc.data(),P92_DISPATCH*4,hipMemcpyHostToDevice));
    P92I8ExpertArena a{iWg,iWu,iWd,iSg,iSu,iSd,H};

    auto layer=[&]() -> void {
        (void)hipMemsetAsync(y,0,K*4,0);
        hipLaunchKernelGGL(k_p92_i8_gateup,dim3(128,P92_DISPATCH),dim3(256),0,0,
                           iWg,iSg,iWu,iSu,xq,xs,si,hid);
        hipLaunchKernelGGL(k_p92_f4_gateup,dim3(N/8,P92_DISPATCH),dim3(256),0,0,
                           fWg,fSg,fWu,fSu,xq,xs,sc,1.f,1.f,hid);
        hipLaunchKernelGGL(k_p92_i8_requant,dim3(P92_DISPATCH),dim3(256),0,0,hid,si,hq,hs);
        hipLaunchKernelGGL(k_p92_i8_requant,dim3(P92_DISPATCH),dim3(256),0,0,hid,sc,hq,hs);
        hipLaunchKernelGGL(k_p92_i8_downacc,dim3(80),dim3(256),0,0,iWd,iSd,hq,hs,wt,si,y);
        hipLaunchKernelGGL(k_p92_f4_downacc,dim3(80),dim3(256),0,0,fWd,fSd,hq,hs,wt,sc,1.f,y);
    };
    for(int i=0;i<30;i++) layer(); CHK(hipDeviceSynchronize());
    hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
    const int IT=300; hipEventRecord(t0,0);
    for(int i=0;i<IT;i++) layer();
    hipEventRecord(t1,0); CHK(hipEventSynchronize(t1));
    float ms; hipEventElapsedTime(&ms,t0,t1);
    const double us=ms*1000.0/IT;
    const double gb=(double)(3*iw+3*fw+3*fs+(size_t)H*(2*N+K)*4)/1e9;
    printf("H=%3d int8 + %3d nvfp4 resident (%.2f GB/layer arena), %d of 8 selections cold\n",H,C,gb,nc);
    printf("  layer routed ffn : %7.2f us\n", us);
    printf("  44 MoE layers    : %7.2f ms/token -> %6.1f tok/s (experts only)\n", us*44/1000, 1000/(us*44/1000));
    printf("  whole-model arena: %6.2f GB\n", gb*44);
    return 0;
}
