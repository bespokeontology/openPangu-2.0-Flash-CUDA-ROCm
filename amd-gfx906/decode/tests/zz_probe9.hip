// Round 9. Counters say both kernels are dependency-latency bound: gate/up
// VALUBusy 56.7 / MemBusy 55.1 / MemStall 1.9. Neither unit is saturated.
// p92_group_dot4 chains FOUR dot4 through one accumulator. That accumulator is
// int32, and integer addition is exactly associative (and wrapping addition is
// associative mod 2^32), so splitting the chain is BIT-IDENTICAL, not a
// reassociation. Break the 4-chain into independent chains to expose ILP.
#include "../src/p92_nvfp4_dot4.hip"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)

// two independent 2-chains
__device__ __forceinline__ int gd4_2(uint2 pk,const uint2 xe,const uint2 xo){
    int a0=p92_d4(p92_dq4( pk.x     &0x0F0F0F0Fu),xe.x,0);
        a0=p92_d4(p92_dq4((pk.x>>4) &0x0F0F0F0Fu),xo.x,a0);
    int a1=p92_d4(p92_dq4( pk.y     &0x0F0F0F0Fu),xe.y,0);
        a1=p92_d4(p92_dq4((pk.y>>4) &0x0F0F0F0Fu),xo.y,a1);
    return a0+a1;
}
// four fully independent dot4, then a balanced tree
__device__ __forceinline__ int gd4_4(uint2 pk,const uint2 xe,const uint2 xo){
    const int a0=p92_d4(p92_dq4( pk.x     &0x0F0F0F0Fu),xe.x,0);
    const int a1=p92_d4(p92_dq4((pk.x>>4) &0x0F0F0F0Fu),xo.x,0);
    const int a2=p92_d4(p92_dq4( pk.y     &0x0F0F0F0Fu),xe.y,0);
    const int a3=p92_d4(p92_dq4((pk.y>>4) &0x0F0F0F0Fu),xo.y,0);
    return (a0+a1)+(a2+a3);
}
#define GU_ARGS const uint8_t*__restrict__ Wg,const uint8_t*__restrict__ Sg, \
    const uint8_t*__restrict__ Wu,const uint8_t*__restrict__ Su, \
    const int8_t*__restrict__ xq,const float*__restrict__ xs, \
    const int*__restrict__ slot,const float*__restrict__ s2,float*__restrict__ hidden
#define GU_PRE const int k=blockIdx.y; const int s=slot[k]; if(s<0) return; \
    const int8_t*sx=xq; const int l=threadIdx.x&31; \
    const int row=blockIdx.x*8+(threadIdx.x>>5); \
    const uint8_t*wg=Wg+(size_t)s*P92_GU_W_BYTES+(size_t)row*(P92_NEMBD/2); \
    const uint8_t*wu=Wu+(size_t)s*P92_GU_W_BYTES+(size_t)row*(P92_NEMBD/2); \
    const uint8_t*sg=Sg+(size_t)s*P92_GU_S_BYTES+(size_t)row*(P92_NEMBD/16); \
    const uint8_t*su=Su+(size_t)s*P92_GU_S_BYTES+(size_t)row*(P92_NEMBD/16); \
    float ag=0.f,au=0.f;
#define GU_POST _Pragma("unroll") for(int off=16;off;off>>=1){ ag+=__shfl_down(ag,off,64); au+=__shfl_down(au,off,64);} \
    if(l==0){ const float g=ag*0.5f*s2[(size_t)s*3+0],u=au*0.5f*s2[(size_t)s*3+1]; \
        hidden[(size_t)k*P92_NFF+row]=g/(1.f+__expf(-g))*u; }
#define GU_KERNEL(NAME,FN) \
__global__ void __launch_bounds__(256) NAME(GU_ARGS){ GU_PRE \
    _Pragma("unroll 1") for(int i=0;i<5;i++){ const int g=i*32+l; \
        const uint2 xe=*(const uint2*)(sx+g*16), xo=*(const uint2*)(sx+g*16+8); \
        const uint2 pg=*(const uint2*)(wg+(size_t)g*8), pu=*(const uint2*)(wu+(size_t)g*8); \
        ag=fmaf(p92_ue4m3_d(sg[g])*xs[g],(float)FN(pg,xe,xo),ag); \
        au=fmaf(p92_ue4m3_d(su[g])*xs[g],(float)FN(pu,xe,xo),au); } GU_POST }
GU_KERNEL(k_gu_c2,gd4_2)
GU_KERNEL(k_gu_c4,gd4_4)

#define DN_ARGS const uint8_t*__restrict__ Wd,const uint8_t*__restrict__ Sd, \
    const int8_t*__restrict__ hq,const float*__restrict__ hs,const float*__restrict__ wts, \
    const int*__restrict__ slot,const float*__restrict__ s2,float*__restrict__ y
#define DN_KERNEL(NAME,FN) \
__global__ void __launch_bounds__(256) NAME(DN_ARGS){ \
    __shared__ int ssl[P92_DISPATCH]; __shared__ float swt[P92_DISPATCH]; \
    if(threadIdx.x<P92_DISPATCH){ ssl[threadIdx.x]=slot[threadIdx.x]; swt[threadIdx.x]=wts[threadIdx.x]; } \
    __syncthreads(); \
    const int l=threadIdx.x%32; const int row=blockIdx.x*8+(threadIdx.x/32); float total=0.f; \
    _Pragma("unroll 2") for(int k=0;k<P92_DISPATCH;k++){ const int s=ssl[k]; if(s<0) continue; \
        const uint8_t*wr=Wd+(size_t)s*P92_DN_W_BYTES+(size_t)row*(P92_NFF/2); \
        const uint8_t*sr=Sd+(size_t)s*P92_DN_S_BYTES+(size_t)row*(P92_NFF/16); \
        const int8_t*hx=hq+(size_t)k*P92_NFF; float acc=0.f; \
        _Pragma("unroll") for(int i=0;i<2;i++){ const int g=i*32+l; \
            const uint2 xe=*(const uint2*)(hx+g*16), xo=*(const uint2*)(hx+g*16+8); \
            const uint2 pk=*(const uint2*)(wr+(size_t)g*8); \
            acc=fmaf(p92_ue4m3_d(sr[g]),(float)FN(pk,xe,xo),acc); } \
        _Pragma("unroll") for(int off=16;off;off>>=1) acc+=__shfl_down(acc,off,64); \
        total=fmaf(acc*hs[k]*0.5f*s2[(size_t)s*3+2],swt[k],total); } \
    if(l==0) y[row]+=total; }
DN_KERNEL(k_dn_c2,gd4_2)
DN_KERNEL(k_dn_c4,gd4_4)

int main(int argc,char**argv){
    const int E=argc>1?atoi(argv[1]):256; const int N=P92_NFF,K=P92_NEMBD;
    uint8_t *Wg,*Wu,*Sg,*Su,*Wd,*Sd; int8_t*xq,*hq; float*xs,*hs,*h0,*h1,*y0,*y1,*wts,*s2; int*slot;
    size_t gw=(size_t)E*N*K/2, gs=(size_t)E*N*K/16, dw=(size_t)E*K*N/2, ds=(size_t)E*K*N/16;
    CHK(hipMalloc(&Wg,gw)); CHK(hipMalloc(&Wu,gw)); CHK(hipMalloc(&Sg,gs)); CHK(hipMalloc(&Su,gs));
    CHK(hipMalloc(&Wd,dw)); CHK(hipMalloc(&Sd,ds));
    CHK(hipMalloc(&xq,K)); CHK(hipMalloc(&xs,(K/16)*4)); CHK(hipMalloc(&s2,(size_t)E*3*4));
    CHK(hipMalloc(&hq,(size_t)P92_DISPATCH*N)); CHK(hipMalloc(&hs,P92_DISPATCH*4));
    CHK(hipMalloc(&h0,(size_t)P92_DISPATCH*N*4)); CHK(hipMalloc(&h1,(size_t)P92_DISPATCH*N*4));
    CHK(hipMalloc(&y0,K*4)); CHK(hipMalloc(&y1,K*4));
    CHK(hipMalloc(&wts,P92_DISPATCH*4)); CHK(hipMalloc(&slot,P92_DISPATCH*4));
    std::mt19937 r(11);
    { std::vector<uint8_t> t(gw); for(auto&v:t) v=(uint8_t)r(); CHK(hipMemcpy(Wg,t.data(),gw,hipMemcpyHostToDevice));
      for(auto&v:t) v=(uint8_t)r(); CHK(hipMemcpy(Wu,t.data(),gw,hipMemcpyHostToDevice)); }
    { std::vector<uint8_t> t(dw); for(auto&v:t) v=(uint8_t)r(); CHK(hipMemcpy(Wd,t.data(),dw,hipMemcpyHostToDevice)); }
    { std::vector<uint8_t> t(gs); for(auto&v:t) v=(uint8_t)(32+r()%80); CHK(hipMemcpy(Sg,t.data(),gs,hipMemcpyHostToDevice));
      for(auto&v:t) v=(uint8_t)(32+r()%80); CHK(hipMemcpy(Su,t.data(),gs,hipMemcpyHostToDevice)); }
    { std::vector<uint8_t> t(ds); for(auto&v:t) v=(uint8_t)(32+r()%80); CHK(hipMemcpy(Sd,t.data(),ds,hipMemcpyHostToDevice)); }
    { std::vector<int8_t> t(K); for(auto&v:t) v=(int8_t)(r()%255-127); CHK(hipMemcpy(xq,t.data(),K,hipMemcpyHostToDevice)); }
    { std::vector<int8_t> t((size_t)P92_DISPATCH*N); for(auto&v:t) v=(int8_t)(r()%255-127); CHK(hipMemcpy(hq,t.data(),t.size(),hipMemcpyHostToDevice)); }
    { std::vector<float> t(K/16); for(auto&v:t) v=0.01f+0.001f*(r()%17); CHK(hipMemcpy(xs,t.data(),(K/16)*4,hipMemcpyHostToDevice)); }
    { std::vector<float> t(P92_DISPATCH,0.02f); CHK(hipMemcpy(hs,t.data(),P92_DISPATCH*4,hipMemcpyHostToDevice)); }
    { std::vector<float> t(P92_DISPATCH); for(int i=0;i<P92_DISPATCH;i++) t[i]=0.1f+0.05f*i; CHK(hipMemcpy(wts,t.data(),P92_DISPATCH*4,hipMemcpyHostToDevice)); }
    { std::vector<float> t((size_t)E*3,1e-4f); CHK(hipMemcpy(s2,t.data(),(size_t)E*3*4,hipMemcpyHostToDevice)); }
    std::vector<int> sl(P92_DISPATCH); for(int k=0;k<P92_DISPATCH;k++) sl[k]=r()%E;
    CHK(hipMemcpy(slot,sl.data(),P92_DISPATCH*4,hipMemcpyHostToDevice));
    const double gby=(double)P92_DISPATCH*2.0*((double)N*K/2+(double)N*K/16);
    const double dby=(double)P92_DISPATCH*((double)K*N/2+(double)K*N/16);
    hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
    auto run=[&](const char*n,double by,auto fn){
        for(int i=0;i<30;i++) fn(); CHK(hipDeviceSynchronize());
        const int IT=300; double best=1e30;
        for(int q=0;q<3;q++){ hipEventRecord(t0,0); for(int i=0;i<IT;i++) fn(); hipEventRecord(t1,0);
            CHK(hipEventSynchronize(t1)); float ms; hipEventElapsedTime(&ms,t0,t1); best=fmin(best,ms*1000.0/IT); }
        printf("  %-40s %8.2f us  %6.1f GB/s\n",n,best,by/(best/1e6)/1e9); return 0; };
    printf("gate/up (%.1f MB)\n",gby/1e6);
    run("production, 4-deep dot4 chain",gby,[&]{ hipLaunchKernelGGL(k_p92_f4_gateup,dim3(N/8,P92_DISPATCH),dim3(256),0,0,Wg,Sg,Wu,Su,xq,xs,slot,s2,h0); });
    run("two independent 2-chains",gby,[&]{ hipLaunchKernelGGL(k_gu_c2,dim3(N/8,P92_DISPATCH),dim3(256),0,0,Wg,Sg,Wu,Su,xq,xs,slot,s2,h1); });
    run("four independent, balanced tree",gby,[&]{ hipLaunchKernelGGL(k_gu_c4,dim3(N/8,P92_DISPATCH),dim3(256),0,0,Wg,Sg,Wu,Su,xq,xs,slot,s2,h1); });
    printf("down (%.1f MB)\n",dby/1e6);
    run("production, 4-deep dot4 chain",dby,[&]{ hipLaunchKernelGGL(k_p92_f4_downacc,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,y0); });
    run("two independent 2-chains",dby,[&]{ hipLaunchKernelGGL(k_dn_c2,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,y1); });
    run("four independent, balanced tree",dby,[&]{ hipLaunchKernelGGL(k_dn_c4,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,y1); });
    // bit-exactness, compared on raw bits
    auto chk=[&](const char*n,auto pf,auto vf,float*b0,float*b1,size_t cnt){
        CHK(hipMemset(b0,0,cnt*4)); CHK(hipMemset(b1,0,cnt*4)); pf(b0); vf(b1); CHK(hipDeviceSynchronize());
        std::vector<float> a(cnt),b(cnt); CHK(hipMemcpy(a.data(),b0,cnt*4,hipMemcpyDeviceToHost));
        CHK(hipMemcpy(b.data(),b1,cnt*4,hipMemcpyDeviceToHost));
        size_t d=0; for(size_t i=0;i<cnt;i++) if(__builtin_memcmp(&a[i],&b[i],4)) d++;
        printf("  %-40s %s (%zu/%zu)\n",n,d?"NOT BIT-EXACT":"bit-exact",d,cnt); return 0; };
    printf("\n");
    chk("gate/up 2-chain",[&](float*o){ hipLaunchKernelGGL(k_p92_f4_gateup,dim3(N/8,P92_DISPATCH),dim3(256),0,0,Wg,Sg,Wu,Su,xq,xs,slot,s2,o); },
                         [&](float*o){ hipLaunchKernelGGL(k_gu_c2,dim3(N/8,P92_DISPATCH),dim3(256),0,0,Wg,Sg,Wu,Su,xq,xs,slot,s2,o); },h0,h1,(size_t)P92_DISPATCH*N);
    chk("gate/up 4-chain",[&](float*o){ hipLaunchKernelGGL(k_p92_f4_gateup,dim3(N/8,P92_DISPATCH),dim3(256),0,0,Wg,Sg,Wu,Su,xq,xs,slot,s2,o); },
                         [&](float*o){ hipLaunchKernelGGL(k_gu_c4,dim3(N/8,P92_DISPATCH),dim3(256),0,0,Wg,Sg,Wu,Su,xq,xs,slot,s2,o); },h0,h1,(size_t)P92_DISPATCH*N);
    chk("down 2-chain",[&](float*o){ hipLaunchKernelGGL(k_p92_f4_downacc,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,o); },
                       [&](float*o){ hipLaunchKernelGGL(k_dn_c2,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,o); },y0,y1,K);
    chk("down 4-chain",[&](float*o){ hipLaunchKernelGGL(k_p92_f4_downacc,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,o); },
                       [&](float*o){ hipLaunchKernelGGL(k_dn_c4,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,o); },y0,y1,K);
    return 0; }
