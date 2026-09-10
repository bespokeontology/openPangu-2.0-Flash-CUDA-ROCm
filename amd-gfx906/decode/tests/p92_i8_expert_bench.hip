// INT8 dot4 expert path at openPangu shapes vs the measured NVFP4 rate.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#define NEMBD 2560
#define NFF   1024
__device__ __forceinline__ int dot4x4(int4 w,int4 x){
  int d=__builtin_amdgcn_sdot4(w.x,x.x,0,false);
  d=__builtin_amdgcn_sdot4(w.y,x.y,d,false);
  d=__builtin_amdgcn_sdot4(w.z,x.z,d,false);
  d=__builtin_amdgcn_sdot4(w.w,x.w,d,false); return d; }
// gate/up: [NFF, NEMBD] int8, one row per half-wave, 8 rows per 256-thread block
__global__ void __launch_bounds__(256) k_i8_gu(const int8_t* __restrict__ W,
    const float* __restrict__ rs, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, float* __restrict__ y){
  __shared__ int8_t sx[NEMBD]; __shared__ float ss[NEMBD/32];
  for(int i=threadIdx.x;i<NEMBD;i+=blockDim.x) sx[i]=xq[i];
  for(int i=threadIdx.x;i<NEMBD/32;i+=blockDim.x) ss[i]=xs[i];
  __syncthreads();
  int l=threadIdx.x&31, row=blockIdx.x*8+(threadIdx.x>>5);
  const int8_t* wr=W+(size_t)row*NEMBD;
  float acc=0.f;
  for(int blk=l; blk<NEMBD/32; blk+=32){
    const int4* w4=(const int4*)(wr+blk*32); const int4* x4=(const int4*)(sx+blk*32);
    int s=dot4x4(w4[0],x4[0])+dot4x4(w4[1],x4[1]);
    acc=fmaf((float)s, ss[blk], acc);
  }
  for(int off=16;off;off>>=1) acc+=__shfl_down(acc,off,64);
  if(l==0) y[row]=acc*rs[row];
}
// down: [NEMBD, NFF] int8, one row per eighth-wave, 32 rows per block
__global__ void __launch_bounds__(256) k_i8_dn(const int8_t* __restrict__ W,
    const float* __restrict__ rs, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, float* __restrict__ y){
  __shared__ int8_t sx[NFF]; __shared__ float ss[NFF/32];
  for(int i=threadIdx.x;i<NFF;i+=blockDim.x) sx[i]=xq[i];
  for(int i=threadIdx.x;i<NFF/32;i+=blockDim.x) ss[i]=xs[i];
  __syncthreads();
  int l=threadIdx.x&7, row=blockIdx.x*32+(threadIdx.x>>3);
  const int8_t* wr=W+(size_t)row*NFF;
  float acc=0.f;
  for(int blk=l; blk<NFF/32; blk+=8){
    const int4* w4=(const int4*)(wr+blk*32); const int4* x4=(const int4*)(sx+blk*32);
    int s=dot4x4(w4[0],x4[0])+dot4x4(w4[1],x4[1]);
    acc=fmaf((float)s, ss[blk], acc);
  }
  for(int off=4;off;off>>=1) acc+=__shfl_down(acc,off,64);
  if(l==0) y[row]=acc*rs[row];
}
int main(){
  int8_t *Wg,*Wd,*xq,*xqd; float *rsg,*rsd,*xs,*xsd,*yg,*yd;
  size_t gw=(size_t)NFF*NEMBD, dw=(size_t)NEMBD*NFF;
  hipMalloc(&Wg,gw); hipMalloc(&Wd,dw);
  hipMalloc(&rsg,NFF*4); hipMalloc(&rsd,NEMBD*4);
  hipMalloc(&xq,NEMBD); hipMalloc(&xs,NEMBD/32*4);
  hipMalloc(&xqd,NFF); hipMalloc(&xsd,NFF/32*4);
  hipMalloc(&yg,NFF*4); hipMalloc(&yd,NEMBD*4);
  hipMemset(Wg,3,gw); hipMemset(Wd,2,dw); hipMemset(xq,5,NEMBD); hipMemset(xqd,4,NFF);
  hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b);
  const int IT=500;
  for(int i=0;i<50;i++){
    hipLaunchKernelGGL(k_i8_gu,dim3(NFF/8),dim3(256),0,0,Wg,rsg,xq,xs,yg);
    hipLaunchKernelGGL(k_i8_gu,dim3(NFF/8),dim3(256),0,0,Wg,rsg,xq,xs,yg);
    hipLaunchKernelGGL(k_i8_dn,dim3(NEMBD/32),dim3(256),0,0,Wd,rsd,xqd,xsd,yd);
  }
  hipDeviceSynchronize(); hipEventRecord(a,0);
  for(int i=0;i<IT;i++){
    hipLaunchKernelGGL(k_i8_gu,dim3(NFF/8),dim3(256),0,0,Wg,rsg,xq,xs,yg);
    hipLaunchKernelGGL(k_i8_gu,dim3(NFF/8),dim3(256),0,0,Wg,rsg,xq,xs,yg);
    hipLaunchKernelGGL(k_i8_dn,dim3(NEMBD/32),dim3(256),0,0,Wd,rsd,xqd,xsd,yd);
  }
  hipEventRecord(b,0); hipEventSynchronize(b);
  float ms=0; hipEventElapsedTime(&ms,a,b);
  double us=ms*1000.0/IT, bytes=(double)gw*2+dw;
  printf("INT8 one expert (gate+up+down): %.2f us, %.2f MB, %.1f GB/s\n", us, bytes/1e6, bytes/(us/1e6)/1e9);
  printf("top-8 layer: %.3f ms  x44 layers: %.1f ms/token -> %.1f tok/s (experts only)\n",
         us*8/1000.0, us*8*44/1000.0, 1000.0/(us*8*44/1000.0));
  printf("NVFP4 measured earlier: 248.79 us/expert, 17.8 GB/s, 87.6 ms/token\n");
  printf("INT8 speedup: %.1fx\n", 248.79/us);
  return 0;
}
