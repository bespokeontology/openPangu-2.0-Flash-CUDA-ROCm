// Round 11. The 32-lane reduction is off=16,8,4,2,1. row_shr:N gives lane n the
// value of lane n+N within a 16-lane row -- exactly __shfl_down(v,N) for
// N=8,4,2,1, SAME SOURCE LANE, so the pairing and therefore the float result is
// identical. off=16 crosses the row boundary and stays ds_bpermute.
// Lanes near the row top read out of range and are garbage; only lane 0 (and 32)
// is consumed, and its chain is 0<-8, 0<-4, 0<-2, 0<-1, all in-row.
// 5 DS ops -> 1 DS + 4 ALU. gate/up 10 -> 2 DS. downacc 45 -> 9 DS.
#include "../src/p92_nvfp4_dot4.hip"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
#define DPPF(x,c) __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(x),__float_as_int(x),(c),0xF,0xF,false))
#define RSHR(n) (0x100u|(n))   // row_shl, verified == shfl_down
#define RED32(v) { v+=__shfl_down(v,16,64); v+=DPPF(v,RSHR(8)); v+=DPPF(v,RSHR(4)); v+=DPPF(v,RSHR(2)); v+=DPPF(v,RSHR(1)); }
#define GU_ARGS const uint8_t*__restrict__ Wg,const uint8_t*__restrict__ Sg, \
    const uint8_t*__restrict__ Wu,const uint8_t*__restrict__ Su, \
    const int8_t*__restrict__ xq,const float*__restrict__ xs, \
    const int*__restrict__ slot,const float*__restrict__ s2,float*__restrict__ hidden
__global__ void __launch_bounds__(256) k_gu_dpp(GU_ARGS){
    const int k=blockIdx.y; const int s=slot[k]; if(s<0) return;
    const int8_t*sx=xq; const int l=threadIdx.x&31;
    const int row=blockIdx.x*8+(threadIdx.x>>5);
    const uint8_t*wg=Wg+(size_t)s*P92_GU_W_BYTES+(size_t)row*(P92_NEMBD/2);
    const uint8_t*wu=Wu+(size_t)s*P92_GU_W_BYTES+(size_t)row*(P92_NEMBD/2);
    const uint8_t*sg=Sg+(size_t)s*P92_GU_S_BYTES+(size_t)row*(P92_NEMBD/16);
    const uint8_t*su=Su+(size_t)s*P92_GU_S_BYTES+(size_t)row*(P92_NEMBD/16);
    float ag=0.f,au=0.f;
    #pragma unroll 1
    for(int i=0;i<5;i++){ const int g=i*32+l;
        const uint2 xe=*(const uint2*)(sx+g*16), xo=*(const uint2*)(sx+g*16+8);
        const uint2 pg=*(const uint2*)(wg+(size_t)g*8), pu=*(const uint2*)(wu+(size_t)g*8);
        ag=fmaf(p92_ue4m3_d(sg[g])*xs[g],(float)p92_group_dot4(pg,xe,xo),ag);
        au=fmaf(p92_ue4m3_d(su[g])*xs[g],(float)p92_group_dot4(pu,xe,xo),au); }
    RED32(ag) RED32(au)
    if(l==0){ const float g=ag*0.5f*s2[(size_t)s*3+0],u=au*0.5f*s2[(size_t)s*3+1];
        hidden[(size_t)k*P92_NFF+row]=g/(1.f+__expf(-g))*u; }
}
__global__ void __launch_bounds__(256)
k_dn_dpp(const uint8_t*__restrict__ Wd,const uint8_t*__restrict__ Sd,
         const int8_t*__restrict__ hq,const float*__restrict__ hs,
         const float*__restrict__ wts,const int*__restrict__ slot,
         const float*__restrict__ s2,float*__restrict__ y){
    __shared__ int ssl[P92_DISPATCH]; __shared__ float swt[P92_DISPATCH];
    if(threadIdx.x<P92_DISPATCH){ ssl[threadIdx.x]=slot[threadIdx.x]; swt[threadIdx.x]=wts[threadIdx.x]; }
    __syncthreads();
    const int l=threadIdx.x%32; const int row=blockIdx.x*8+(threadIdx.x/32);
    float total=0.f;
    #pragma unroll 2
    for(int k=0;k<P92_DISPATCH;k++){
        const int s=ssl[k]; if(s<0) continue;
        const uint8_t*wr=Wd+(size_t)s*P92_DN_W_BYTES+(size_t)row*(P92_NFF/2);
        const uint8_t*sr=Sd+(size_t)s*P92_DN_S_BYTES+(size_t)row*(P92_NFF/16);
        const int8_t*hx=hq+(size_t)k*P92_NFF;
        float acc=0.f;
        #pragma unroll
        for(int i=0;i<2;i++){ const int g=i*32+l;
            const uint2 xe=*(const uint2*)(hx+g*16), xo=*(const uint2*)(hx+g*16+8);
            const uint2 pk=*(const uint2*)(wr+(size_t)g*8);
            acc=fmaf(p92_ue4m3_d(sr[g]),(float)p92_group_dot4(pk,xe,xo),acc); }
        RED32(acc)
        total=fmaf(acc*hs[k]*0.5f*s2[(size_t)s*3+2],swt[k],total); }
    if(l==0) y[row]=total;
}
int main(int argc,char**argv){
    const int E=argc>1?atoi(argv[1]):256; const int N=P92_NFF,K=P92_NEMBD;
    uint8_t *Wg,*Wu,*Sg,*Su,*Wd,*Sd; int8_t*xq,*hq; float*xs,*hs,*h0,*h1,*y0,*y1,*wts,*s2; int*slot;
    size_t gw=(size_t)E*N*K/2, gs=(size_t)E*N*K/16, dw=(size_t)E*K*N/2, ds=(size_t)E*K*N/16;
    CHK(hipMalloc(&Wg,gw)); CHK(hipMalloc(&Wu,gw)); CHK(hipMalloc(&Sg,gs)); CHK(hipMalloc(&Su,gs));
    CHK(hipMalloc(&Wd,dw)); CHK(hipMalloc(&Sd,ds)); CHK(hipMalloc(&xq,K)); CHK(hipMalloc(&xs,(K/16)*4));
    CHK(hipMalloc(&hq,(size_t)P92_DISPATCH*N)); CHK(hipMalloc(&hs,P92_DISPATCH*4));
    CHK(hipMalloc(&h0,(size_t)P92_DISPATCH*N*4)); CHK(hipMalloc(&h1,(size_t)P92_DISPATCH*N*4));
    CHK(hipMalloc(&y0,K*4)); CHK(hipMalloc(&y1,K*4)); CHK(hipMalloc(&wts,P92_DISPATCH*4));
    CHK(hipMalloc(&s2,(size_t)E*3*4)); CHK(hipMalloc(&slot,P92_DISPATCH*4));
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
    printf("gate/up (%.1f MB): 10 ds_bpermute -> 2\n",gby/1e6);
    run("production",gby,[&]{ hipLaunchKernelGGL(k_p92_f4_gateup,dim3(N/8,P92_DISPATCH),dim3(256),0,0,Wg,Sg,Wu,Su,xq,xs,slot,s2,h0); });
    run("DPP row_shr reduction",gby,[&]{ hipLaunchKernelGGL(k_gu_dpp,dim3(N/8,P92_DISPATCH),dim3(256),0,0,Wg,Sg,Wu,Su,xq,xs,slot,s2,h1); });
    printf("down (%.1f MB): 45 ds_bpermute -> 9\n",dby/1e6);
    run("production",dby,[&]{ hipLaunchKernelGGL(k_p92_f4_downacc,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,y0); });
    run("DPP row_shr reduction",dby,[&]{ hipLaunchKernelGGL(k_dn_dpp,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,y1); });
    printf("\n");
    { CHK(hipMemset(h0,0,(size_t)P92_DISPATCH*N*4)); CHK(hipMemset(h1,0,(size_t)P92_DISPATCH*N*4));
      hipLaunchKernelGGL(k_p92_f4_gateup,dim3(N/8,P92_DISPATCH),dim3(256),0,0,Wg,Sg,Wu,Su,xq,xs,slot,s2,h0);
      hipLaunchKernelGGL(k_gu_dpp,dim3(N/8,P92_DISPATCH),dim3(256),0,0,Wg,Sg,Wu,Su,xq,xs,slot,s2,h1);
      CHK(hipDeviceSynchronize());
      std::vector<float> a((size_t)P92_DISPATCH*N),b(a.size());
      CHK(hipMemcpy(a.data(),h0,a.size()*4,hipMemcpyDeviceToHost)); CHK(hipMemcpy(b.data(),h1,b.size()*4,hipMemcpyDeviceToHost));
      size_t d=0; for(size_t i=0;i<a.size();i++) if(__builtin_memcmp(&a[i],&b[i],4)) d++;
      printf("  gate/up DPP                              %s (%zu/%zu)\n",d?"NOT BIT-EXACT":"bit-exact",d,a.size()); }
    { CHK(hipMemset(y0,0,K*4)); CHK(hipMemset(y1,0,K*4));
      hipLaunchKernelGGL(k_p92_f4_downacc,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,y0);
      hipLaunchKernelGGL(k_dn_dpp,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,y1);
      CHK(hipDeviceSynchronize());
      std::vector<float> a(K),b(K);
      CHK(hipMemcpy(a.data(),y0,K*4,hipMemcpyDeviceToHost)); CHK(hipMemcpy(b.data(),y1,K*4,hipMemcpyDeviceToHost));
      size_t d=0; for(size_t i=0;i<(size_t)K;i++) if(__builtin_memcmp(&a[i],&b[i],4)) d++;
      printf("  down DPP                                 %s (%zu/%d)\n",d?"NOT BIT-EXACT":"bit-exact",d,K); }
    return 0; }
