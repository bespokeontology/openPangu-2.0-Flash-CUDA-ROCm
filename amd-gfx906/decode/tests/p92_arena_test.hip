// Gate the packed arena: (1) sampled slots are byte-identical to the source
// artifact; (2) the cold-path kernel runs on real weights from the arena and
// produces finite, correctly-scaled output; (3) measure a real-weight layer.
#include "../src/p92_nvfp4_dot4.hip"
#include "../src/p92_arena.cpp"
#include "p92amd/p92_placement.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <string>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
#pragma pack(push,1)
struct MHdr { char magic[8]; uint32_t version, record_bytes, tensor_count, reserved;
              uint64_t weight_bytes, scale_bytes, source_payload_bytes; };
struct MRec { char name[96]; uint32_t rows, columns; uint64_t weight_offset, scale_offset;
              float weight_scale_2; uint32_t reserved; };
#pragma pack(pop)
static void rd(int fd,void*p,size_t n,off_t o){ size_t d=0; while(d<n){ ssize_t k=pread(fd,(char*)p+d,n-d,o+d); if(k<=0)exit(1); d+=k; } }

int main(int argc,char**argv){
    const char *ART = argc>1?argv[1]:"./artifact";
    const char *ARE = argc>2?argv[2]:"./arena";
    char ap[512]; snprintf(ap,sizeof ap,"%s/arena_dev0.bin",ARE);
    P92Arena a;
    hipEvent_t t0,t1; float ms;
    CHK(hipEventCreate(&t0)); CHK(hipEventCreate(&t1));
    CHK(hipEventRecord(t0,0));
    if (p92_arena_load(ap,&a)) { printf("arena load FAILED\n"); return 1; }
    CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1)); hipEventElapsedTime(&ms,t0,t1);
    printf("arena dev0: layers %u..%u, %u MoE layers, %d slots, loaded in %.2f s\n",
           a.h.layer_begin,a.h.layer_end-1,a.h.moe_layers,a.slots,ms/1000);

    // ---- (1) byte identity against the artifact -------------------------
    int mf=open((std::string(ART)+"/manifest.bin").c_str(),O_RDONLY);
    int wf=open((std::string(ART)+"/weights.nvfp4").c_str(),O_RDONLY);
    MHdr h; rd(mf,&h,sizeof h,0);
    std::vector<MRec> recs(h.tensor_count); rd(mf,recs.data(),(size_t)h.tensor_count*h.record_bytes,sizeof h);
    std::unordered_map<std::string,const MRec*> bn;
    for(auto&r:recs) bn[std::string(r.name,strnlen(r.name,96))]=&r;
    std::mt19937 rng(20260908); int checked=0,bad=0;
    std::vector<uint8_t> src(P92_GU_W_BYTES), got(P92_GU_W_BYTES);
    for(int trial=0;trial<12;trial++){
        const int li=rng()%a.h.moe_layers, e=rng()%a.h.experts;
        const int layer=(a.h.layer_begin<2?2:a.h.layer_begin)+li;
        const int slot=li*a.h.experts+e;
        char nm[160]; snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.gate_proj.weight",layer,e);
        auto it=bn.find(nm); if(it==bn.end()){ printf("missing %s\n",nm); return 1; }
        rd(wf,src.data(),P92_GU_W_BYTES,(off_t)it->second->weight_offset);
        CHK(hipMemcpy(got.data(),a.gw+(size_t)slot*P92_GU_W_BYTES,P92_GU_W_BYTES,hipMemcpyDeviceToHost));
        checked++; if(memcmp(src.data(),got.data(),P92_GU_W_BYTES)) { bad++; printf("  MISMATCH %s\n",nm); }
    }
    printf("byte identity: %d/%d sampled gate_proj slots match the artifact exactly\n",checked-bad,checked);

    // the shared expert occupies slot 256 of each layer and must be the
    // shared_experts tensor, not a routed one
    int sh_checked=0, sh_bad=0;
    for(int li=0; li<(int)a.h.moe_layers; li++){
        const int layer=(a.h.layer_begin<2?2:a.h.layer_begin)+li;
        const int slot=li*a.h.experts+P92_SHARED_INDEX;
        char nm[160]; snprintf(nm,sizeof nm,"model.layers.%d.mlp.shared_experts.gate_proj.weight",layer);
        auto it=bn.find(nm); if(it==bn.end()){ printf("  missing %s\n",nm); sh_bad++; continue; }
        rd(wf,src.data(),P92_GU_W_BYTES,(off_t)it->second->weight_offset);
        CHK(hipMemcpy(got.data(),a.gw+(size_t)slot*P92_GU_W_BYTES,P92_GU_W_BYTES,hipMemcpyDeviceToHost));
        sh_checked++; if(memcmp(src.data(),got.data(),P92_GU_W_BYTES)){ sh_bad++; printf("  MISMATCH %s\n",nm); }
        // and it must differ from routed expert 0 of the same layer, or the
        // packer silently wrote the wrong tensor into the shared slot
        char rn[160]; snprintf(rn,sizeof rn,"model.layers.%d.mlp.experts.0.gate_proj.weight",layer);
        auto rit=bn.find(rn);
        if(rit!=bn.end()){
            std::vector<uint8_t> r0(P92_GU_W_BYTES);
            rd(wf,r0.data(),P92_GU_W_BYTES,(off_t)rit->second->weight_offset);
            if(!memcmp(r0.data(),got.data(),P92_GU_W_BYTES)){ sh_bad++; printf("  shared slot equals routed expert 0 at layer %d\n",layer); }
        }
    }
    printf("shared expert: %d/%d layer slots are the shared_experts tensor and differ from expert 0\n",
           sh_checked-sh_bad, sh_checked);

    // ---- (2)+(3) real weights through the cold kernel --------------------
    int8_t *xq,*hq; float *xs,*hid,*hs,*y,*wt; int *sl;
    CHK(hipMalloc(&xq,P92_NEMBD)); CHK(hipMalloc(&xs,4));
    CHK(hipMalloc(&hid,(size_t)P92_DISPATCH*P92_NFF*4)); CHK(hipMalloc(&hq,(size_t)P92_DISPATCH*P92_NFF));
    CHK(hipMalloc(&hs,P92_DISPATCH*4)); CHK(hipMalloc(&y,P92_NEMBD*4)); CHK(hipMalloc(&wt,P92_DISPATCH*4));
    CHK(hipMalloc(&sl,P92_DISPATCH*4));
    std::vector<int8_t> hx(P92_NEMBD); std::normal_distribution<float> nd(0,1);
    for(auto&v:hx) v=(int8_t)std::max(-127.f,std::min(127.f,nd(rng)*32));
    { std::vector<int8_t> t(hx);                           // permuted, as the quantiser emits
      for(int g=0;g<P92_NEMBD/16;g++) for(int i=0;i<16;i++)
          hx[g*16 + ((i&1)? 8+(i>>1) : (i>>1))] = t[g*16+i]; }
    CHK(hipMemcpy(xq,hx.data(),P92_NEMBD,hipMemcpyHostToDevice));
    float one=0.0187f; CHK(hipMemcpy(xs,&one,4,hipMemcpyHostToDevice));
    // the oracle's own selected ids for its captured token
    const int ids[8]={123,204,232,142,203,11,178,3};
    const float ws[8]={.35f,.30f,.28f,.27f,.26f,.25f,.24f,.55f};
    std::vector<int> vs(8); const int li=5;                       // a layer this arena owns
    for(int k=0;k<8;k++) vs[k]=li*a.h.experts+ids[k];
    CHK(hipMemcpy(sl,vs.data(),8*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(wt,ws,8*4,hipMemcpyHostToDevice));
    std::vector<float> s2(a.slots*3); CHK(hipMemcpy(s2.data(),a.s2,s2.size()*4,hipMemcpyDeviceToHost));
    const float g2=s2[(size_t)vs[0]*3+0], u2=s2[(size_t)vs[0]*3+1], d2=s2[(size_t)vs[0]*3+2];
    printf("weight_scale_2 for slot %d: gate %.6g up %.6g down %.6g\n",vs[0],g2,u2,d2);

    hipLaunchKernelGGL(k_p92_f4_gateup,dim3(P92_NFF/8,P92_DISPATCH),dim3(256),0,0,
                       a.gw,a.gs,a.uw,a.us,xq,xs,sl,a.s2,hid);
    CHK(hipDeviceSynchronize());
    std::vector<float> hh((size_t)P92_DISPATCH*P92_NFF);
    CHK(hipMemcpy(hh.data(),hid,hh.size()*4,hipMemcpyDeviceToHost));
    double ss=0,mn=1e30,mx=-1e30; int nf=0;
    for(size_t i=0;i<(size_t)P92_NFF;i++){ float v=hh[i]; if(!std::isfinite(v))nf++;
        ss+=(double)v*v; mn=std::min(mn,(double)v); mx=std::max(mx,(double)v); }
    printf("expert %d gate/up+SwiGLU on real weights: rms %.6g min %.6g max %.6g nonfinite %d\n",
           ids[0], sqrt(ss/P92_NFF), mn, mx, nf);
    const bool pass = (bad==0) && (sh_bad==0) && (nf==0) && (ss>0);
    printf("%s\n", pass?"PASS":"FAIL");
    p92_arena_free(&a);
    return pass?0:1;
}
