// Round 6. downacc does a 5-step shfl_down reduction PER EXPERT: 45 ds_bpermute
// per thread against only 36 loads. gate/up does 10. Hoist the reduction out of
// the expert loop - legal because the per-expert scale is lane-uniform:
//   sum_k c_k * (sum_l acc_k[l])  ==  sum_l (sum_k acc_k[l] * c_k)
// mathematically identical, but it REASSOCIATES, so this is a diagnostic first.
#include "../src/p92_nvfp4_dot4.hip"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)

__global__ void __launch_bounds__(256)
k_dn_hoist(const uint8_t*__restrict__ Wd,const uint8_t*__restrict__ Sd,
           const int8_t*__restrict__ hq,const float*__restrict__ hs,
           const float*__restrict__ wts,const int*__restrict__ slot,
           const float*__restrict__ s2,float*__restrict__ y){
    __shared__ int ssl[P92_DISPATCH]; __shared__ float swt[P92_DISPATCH];
    if(threadIdx.x<P92_DISPATCH){ ssl[threadIdx.x]=slot[threadIdx.x]; swt[threadIdx.x]=wts[threadIdx.x]; }
    __syncthreads();
    const int l=threadIdx.x%32; const int row=blockIdx.x*8+(threadIdx.x/32);
    float lane=0.f;                       // per-lane running total across experts
    #pragma unroll 2
    for(int k=0;k<P92_DISPATCH;k++){
        const int s=ssl[k]; if(s<0) continue;
        const uint8_t*wr=Wd+(size_t)s*P92_DN_W_BYTES+(size_t)row*(P92_NFF/2);
        const uint8_t*sr=Sd+(size_t)s*P92_DN_S_BYTES+(size_t)row*(P92_NFF/16);
        const int8_t*hx=hq+(size_t)k*P92_NFF;
        float acc=0.f;
        #pragma unroll
        for(int i=0;i<2;i++){
            const int g=i*32+l;
            const uint2 xe=*(const uint2*)(hx+g*16);
            const uint2 xo=*(const uint2*)(hx+g*16+8);
            const uint2 pk=*(const uint2*)(wr+(size_t)g*8);
            acc=fmaf(p92_ue4m3_d(sr[g]),(float)p92_group_dot4(pk,xe,xo),acc);
        }
        lane=fmaf(acc*hs[k]*0.5f*s2[(size_t)s*3+2],swt[k],lane);   // scale is lane-uniform
    }
    #pragma unroll
    for(int off=16;off;off>>=1) lane+=__shfl_down(lane,off,64);    // ONE reduction
    if(l==0) y[row]+=lane;
}
// and a no-reduction-at-all control, to bound what the shuffles cost
__global__ void __launch_bounds__(256)
k_dn_noreduce(const uint8_t*__restrict__ Wd,const uint8_t*__restrict__ Sd,
              const int8_t*__restrict__ hq,const float*__restrict__ hs,
              const float*__restrict__ wts,const int*__restrict__ slot,
              const float*__restrict__ s2,float*__restrict__ y){
    __shared__ int ssl[P92_DISPATCH]; __shared__ float swt[P92_DISPATCH];
    if(threadIdx.x<P92_DISPATCH){ ssl[threadIdx.x]=slot[threadIdx.x]; swt[threadIdx.x]=wts[threadIdx.x]; }
    __syncthreads();
    const int l=threadIdx.x%32; const int row=blockIdx.x*8+(threadIdx.x/32);
    float lane=0.f;
    #pragma unroll 2
    for(int k=0;k<P92_DISPATCH;k++){
        const int s=ssl[k]; if(s<0) continue;
        const uint8_t*wr=Wd+(size_t)s*P92_DN_W_BYTES+(size_t)row*(P92_NFF/2);
        const uint8_t*sr=Sd+(size_t)s*P92_DN_S_BYTES+(size_t)row*(P92_NFF/16);
        const int8_t*hx=hq+(size_t)k*P92_NFF;
        float acc=0.f;
        #pragma unroll
        for(int i=0;i<2;i++){
            const int g=i*32+l;
            const uint2 xe=*(const uint2*)(hx+g*16);
            const uint2 xo=*(const uint2*)(hx+g*16+8);
            const uint2 pk=*(const uint2*)(wr+(size_t)g*8);
            acc=fmaf(p92_ue4m3_d(sr[g]),(float)p92_group_dot4(pk,xe,xo),acc);
        }
        lane=fmaf(acc*hs[k]*0.5f*s2[(size_t)s*3+2],swt[k],lane);
    }
    if(l==0) y[row]+=lane;      // WRONG maths on purpose, timing bound only
}
int main(int argc,char**argv){
    const int E=argc>1?atoi(argv[1]):256; const int N=P92_NFF,K=P92_NEMBD;
    uint8_t *Wd,*Sd; int8_t*hq; float*hs,*y,*y2,*wts,*s2; int*slot;
    size_t dw=(size_t)E*K*N/2, ds=(size_t)E*K*N/16;
    CHK(hipMalloc(&Wd,dw)); CHK(hipMalloc(&Sd,ds)); CHK(hipMalloc(&hq,(size_t)P92_DISPATCH*N));
    CHK(hipMalloc(&hs,P92_DISPATCH*4)); CHK(hipMalloc(&y,K*4)); CHK(hipMalloc(&y2,K*4));
    CHK(hipMalloc(&wts,P92_DISPATCH*4)); CHK(hipMalloc(&s2,(size_t)E*3*4)); CHK(hipMalloc(&slot,P92_DISPATCH*4));
    CHK(hipMemset(Wd,0x47,dw)); CHK(hipMemset(Sd,60,ds));
    std::vector<int8_t> hv2((size_t)P92_DISPATCH*N); std::mt19937 rr(9);
    for(auto&v:hv2) v=(int8_t)(rr()%255-127);
    CHK(hipMemcpy(hq,hv2.data(),hv2.size(),hipMemcpyHostToDevice));
    CHK(hipMemset(y,0,K*4)); CHK(hipMemset(y2,0,K*4));
    std::vector<float> hv(P92_DISPATCH,0.02f); CHK(hipMemcpy(hs,hv.data(),P92_DISPATCH*4,hipMemcpyHostToDevice));
    std::vector<float> wv(P92_DISPATCH); for(int i=0;i<P92_DISPATCH;i++) wv[i]=0.1f+0.05f*i;
    CHK(hipMemcpy(wts,wv.data(),P92_DISPATCH*4,hipMemcpyHostToDevice));
    std::vector<float> sv((size_t)E*3,1e-4f); CHK(hipMemcpy(s2,sv.data(),(size_t)E*3*4,hipMemcpyHostToDevice));
    std::vector<int> sl(P92_DISPATCH); std::mt19937 r(3);
    for(int k=0;k<P92_DISPATCH;k++) sl[k]=r()%E;
    CHK(hipMemcpy(slot,sl.data(),P92_DISPATCH*4,hipMemcpyHostToDevice));
    const double dn_by=(double)P92_DISPATCH*((double)K*N/2+(double)K*N/16);
    hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
    auto run=[&](const char*n,auto fn){
        for(int i=0;i<30;i++) fn(); CHK(hipDeviceSynchronize());
        const int IT=300; double best=1e30;
        for(int r2=0;r2<3;r2++){ hipEventRecord(t0,0); for(int i=0;i<IT;i++) fn(); hipEventRecord(t1,0);
            CHK(hipEventSynchronize(t1)); float ms; hipEventElapsedTime(&ms,t0,t1); best=fmin(best,ms*1000.0/IT); }
        printf("  %-46s %8.2f us  %6.1f GB/s\n",n,best,dn_by/(best/1e6)/1e9); return 0; };
    printf("down, %.3f MB per call, roof from pure-read probe 784.6 GB/s\n\n",dn_by/1e6);
    run("production: 9 reductions, 45 ds_bpermute",[&]{ hipLaunchKernelGGL(k_p92_f4_downacc,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,y); });
    run("hoisted: 1 reduction, 5 ds_bpermute",[&]{ hipLaunchKernelGGL(k_dn_hoist,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,y2); });
    run("no reduction at all (bound, wrong maths)",[&]{ hipLaunchKernelGGL(k_dn_noreduce,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,y2); });
    // how far apart are production and hoisted numerically?
    CHK(hipMemset(y,0,K*4)); CHK(hipMemset(y2,0,K*4));
    hipLaunchKernelGGL(k_p92_f4_downacc,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,y);
    hipLaunchKernelGGL(k_dn_hoist,dim3(320),dim3(256),0,0,Wd,Sd,hq,hs,wts,slot,s2,y2);
    CHK(hipDeviceSynchronize());
    std::vector<float> a(K),b(K); CHK(hipMemcpy(a.data(),y,K*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(b.data(),y2,K*4,hipMemcpyDeviceToHost));
    double e=0,s=0; int nbit=0;
    for(int i=0;i<K;i++){ double d=a[i]-b[i]; e+=d*d; s+=(double)a[i]*a[i]; if(a[i]!=b[i]) nbit++; }
    printf("\n  hoisted vs production: relative_l2 %.3e, %d/%d rows differ in any bit\n",
           s>0?sqrt(e/s):0.0,nbit,K);
    return 0; }
