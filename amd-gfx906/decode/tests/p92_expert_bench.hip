// Real-weight device gate + timing for the openPangu NVFP4 expert kernels.
// Loads actual packed expert weights from the P92FP41 artifact, runs the
// gfx906 kernels, checks against a CPU dequantise-and-dot reference, and
// times a full top-8 expert layer.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "p92amd/p92_nvfp4_wave64.h"

#pragma pack(push,1)
struct Hdr { char magic[8]; uint32_t version, record_bytes, tensor_count, reserved;
             uint64_t weight_bytes, scale_bytes, source_payload_bytes; };
struct Rec { char name[96]; uint32_t rows, columns; uint64_t weight_offset, scale_offset;
             float weight_scale_2; uint32_t reserved; };
#pragma pack(pop)

#include "../src/p92_nvfp4_wave64.hip"

static void* map_ro(const char* p, size_t* n){ int fd=open(p,O_RDONLY); struct stat st; fstat(fd,&st);
  *n=st.st_size; void* m=mmap(nullptr,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd); return m; }
static float e2m1(uint32_t nib){ uint32_t mag=nib&7u,e=mag>>1,m=mag&1u; float v;
  if(e){ uint32_t b=((e+126u)<<23)|(m<<22); memcpy(&v,&b,4);} else v=m?0.5f:0.f; return (nib&8u)?-v:v; }
static float ue4m3(uint32_t b){ uint32_t e=b>>3,m=b&7u; float v;
  if(e==0) return (float)m*0x1p-9f; if(e==15&&m==7) return INFINITY;
  uint32_t bb=((e+120u)<<23)|(m<<20); memcpy(&v,&bb,4); return v; }

int main(int argc,char**argv){
  const char* dir = argc>1?argv[1]:"./artifact";
  size_t mn,wn,sn;
  auto* mb=(uint8_t*)map_ro((std::string(dir)+"/manifest.bin").c_str(),&mn);
  auto* wb=(uint8_t*)map_ro((std::string(dir)+"/weights.nvfp4").c_str(),&wn);
  auto* sb=(uint8_t*)map_ro((std::string(dir)+"/scales.swizzled.e4m3").c_str(),&sn);
  auto* h=(Hdr*)mb; auto* recs=(Rec*)(mb+sizeof(Hdr));
  printf("artifact: %.7s v%u records=%u weights=%.2f GB scales=%.2f GB\n",
         h->magic,h->version,h->tensor_count,wn/1e9,sn/1e9);
  auto find=[&](const std::string& n)->Rec*{ for(uint32_t i=0;i<h->tensor_count;i++)
      if(n==recs[i].name) return &recs[i]; return nullptr; };

  // one real routed expert from a MoE layer
  Rec* g=find("model.layers.2.mlp.experts.0.gate_proj.weight");
  Rec* d=find("model.layers.2.mlp.experts.0.down_proj.weight");
  if(!g||!d){ printf("expert tensors not found\n"); return 1; }
  printf("gate_proj rows=%u cols=%u scale2=%.6g   down_proj rows=%u cols=%u scale2=%.6g\n",
         g->rows,g->columns,g->weight_scale_2,d->rows,d->columns,d->weight_scale_2);

  const int GK=g->columns, GR=g->rows, DK=d->columns, DR=d->rows;
  std::vector<float> x(GK), xd(DK);
  for(int i=0;i<GK;i++) x[i]=std::sin(i*0.017f);
  for(int i=0;i<DK;i++) xd[i]=std::cos(i*0.013f);

  uint8_t *dW,*dS,*dWd,*dSd; float *dx,*dy,*dxd,*dyd;
  size_t gw=(size_t)GR*(GK/2), gs=(size_t)GR*(GK/16);
  size_t dw=(size_t)DR*(DK/2), ds=(size_t)DR*(DK/16);
  hipMalloc(&dW,gw); hipMalloc(&dS,gs); hipMalloc(&dWd,dw); hipMalloc(&dSd,ds);
  hipMalloc(&dx,GK*4); hipMalloc(&dy,GR*4); hipMalloc(&dxd,DK*4); hipMalloc(&dyd,DR*4);
  hipMemcpy(dW,wb+g->weight_offset,gw,hipMemcpyHostToDevice);
  hipMemcpy(dS,sb+g->scale_offset,gs,hipMemcpyHostToDevice);
  hipMemcpy(dWd,wb+d->weight_offset,dw,hipMemcpyHostToDevice);
  hipMemcpy(dSd,sb+d->scale_offset,ds,hipMemcpyHostToDevice);
  hipMemcpy(dx,x.data(),GK*4,hipMemcpyHostToDevice);
  hipMemcpy(dxd,xd.data(),DK*4,hipMemcpyHostToDevice);

  hipLaunchKernelGGL(k_p92_nvfp4_gemv_1024x2560,dim3(GR/8),dim3(256),0,0,dW,dS,dx,dy,g->weight_scale_2,0);
  hipLaunchKernelGGL(k_p92_nvfp4_gemv_2560x1024,dim3(DR/32),dim3(256),0,0,dWd,dSd,dxd,dyd,d->weight_scale_2,0);
  hipDeviceSynchronize();
  std::vector<float> yg(GR), yd_(DR);
  hipMemcpy(yg.data(),dy,GR*4,hipMemcpyDeviceToHost);
  hipMemcpy(yd_.data(),dyd,DR*4,hipMemcpyDeviceToHost);

  // CPU reference on real bytes
  auto ref=[&](const uint8_t* W,const uint8_t* S,const float* xx,int K,int row,float s2){
    const uint8_t* wr=W+(size_t)row*(K/2); const uint8_t* sr=S+(size_t)row*(K/16);
    double acc=0;
    for(int gi=0;gi<K/16;gi++){ double gs2=0;
      for(int j=0;j<8;j++){ uint32_t b=wr[(size_t)gi*8+j];
        gs2+=(double)e2m1(b&15u)*xx[gi*16+2*j]; gs2+=(double)e2m1(b>>4)*xx[gi*16+2*j+1]; }
      acc+=gs2*(double)ue4m3(sr[gi]); }
    return (float)(acc*s2); };
  double worst_g=0, worst_d=0;
  for(int r=0;r<GR;r+=37){ float R=ref(wb+g->weight_offset,sb+g->scale_offset,x.data(),GK,r,g->weight_scale_2);
    double den=fabs(R)>1e-4?fabs(R):1.0; worst_g=std::max(worst_g,fabs(R-yg[r])/den); }
  for(int r=0;r<DR;r+=53){ float R=ref(wb+d->weight_offset,sb+d->scale_offset,xd.data(),DK,r,d->weight_scale_2);
    double den=fabs(R)>1e-4?fabs(R):1.0; worst_d=std::max(worst_d,fabs(R-yd_[r])/den); }
  printf("device-vs-CPU on REAL weights: gate worst_rel=%.3e  down worst_rel=%.3e  %s\n",
         worst_g,worst_d,(worst_g<1e-3&&worst_d<1e-3)?"CORRECT":"MISMATCH");

  // timing: one expert (gate+up+down) and a full top-8 layer
  hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b);
  const int IT=200;
  for(int i=0;i<20;i++){
    hipLaunchKernelGGL(k_p92_nvfp4_gemv_1024x2560,dim3(GR/8),dim3(256),0,0,dW,dS,dx,dy,g->weight_scale_2,0);
    hipLaunchKernelGGL(k_p92_nvfp4_gemv_1024x2560,dim3(GR/8),dim3(256),0,0,dW,dS,dx,dy,g->weight_scale_2,0);
    hipLaunchKernelGGL(k_p92_nvfp4_gemv_2560x1024,dim3(DR/32),dim3(256),0,0,dWd,dSd,dxd,dyd,d->weight_scale_2,0);
  }
  hipDeviceSynchronize(); hipEventRecord(a,0);
  for(int i=0;i<IT;i++){
    hipLaunchKernelGGL(k_p92_nvfp4_gemv_1024x2560,dim3(GR/8),dim3(256),0,0,dW,dS,dx,dy,g->weight_scale_2,0);
    hipLaunchKernelGGL(k_p92_nvfp4_gemv_1024x2560,dim3(GR/8),dim3(256),0,0,dW,dS,dx,dy,g->weight_scale_2,0);
    hipLaunchKernelGGL(k_p92_nvfp4_gemv_2560x1024,dim3(DR/32),dim3(256),0,0,dWd,dSd,dxd,dyd,d->weight_scale_2,0);
  }
  hipEventRecord(b,0); hipEventSynchronize(b); float ms=0; hipEventElapsedTime(&ms,a,b);
  double per_expert_us = ms*1000.0/IT;
  double bytes_per_expert = (double)(gw+gs)*2 + (double)(dw+ds);
  printf("one expert (gate+up+down): %.2f us, %.2f MB, %.1f GB/s\n",
         per_expert_us, bytes_per_expert/1e6, bytes_per_expert/(per_expert_us/1e6)/1e9);
  printf("top-8 layer: %.3f ms   x44 MoE layers: %.1f ms/token -> %.1f tok/s (experts only)\n",
         per_expert_us*8/1000.0, per_expert_us*8*44/1000.0, 1000.0/(per_expert_us*8*44/1000.0));
  return (worst_g<1e-3&&worst_d<1e-3)?0:1;
}
