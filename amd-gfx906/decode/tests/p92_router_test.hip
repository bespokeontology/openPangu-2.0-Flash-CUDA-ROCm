// Gate the router against a scalar CPU reference of the shipped semantics,
// on the real bf16 router weights and correction bias, plus a forced-tie case
// to check the deterministic lower-index rule.
#include "../src/p92_router.hip"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)

static std::vector<uint16_t> load_raw(const char *dir, const char *name, size_t want_bytes) {
    // minimal safetensors reader: index -> shard -> header -> slice
    std::string ip = std::string(dir) + "/model.safetensors.index.json";
    int f = open(ip.c_str(), O_RDONLY);
    if (f < 0) return {};
    std::string js; { char b[1<<16]; ssize_t k; while ((k=read(f,b,sizeof b))>0) js.append(b,k); } close(f);
    std::string key = std::string("\"") + name + "\"";
    size_t p = js.find(key); if (p == std::string::npos) return {};
    size_t q = js.find('"', js.find(':', p) + 1); size_t r = js.find('"', q + 1);
    std::string shard = js.substr(q + 1, r - q - 1);
    std::string sp = std::string(dir) + "/" + shard;
    f = open(sp.c_str(), O_RDONLY); if (f < 0) return {};
    uint64_t hl; if (pread(f,&hl,8,0)!=8) { close(f); return {}; }
    std::string hdr(hl, 0); if (pread(f,&hdr[0],hl,8)!=(ssize_t)hl) { close(f); return {}; }
    p = hdr.find(key); if (p == std::string::npos) { close(f); return {}; }
    size_t off = hdr.find("data_offsets", p); off = hdr.find('[', off);
    long a = atol(hdr.c_str() + off + 1);
    long b = atol(hdr.c_str() + hdr.find(',', off) + 1);
    std::vector<uint16_t> out((b - a) / 2);
    if ((size_t)(b-a) != want_bytes) { printf("  %s size %ld want %zu\n", name, b-a, want_bytes); close(f); return {}; }
    if (pread(f,out.data(),b-a,8+hl+a)!=(b-a)) { close(f); return {}; }
    close(f); return out;
}
static float bf(uint16_t v){ uint32_t u=(uint32_t)v<<16; float f; memcpy(&f,&u,4); return f; }

int main(int argc,char**argv){
    const char *CK = argc>1?argv[1]:"./checkpoint";
    const int L = argc>2?atoi(argv[2]):5;
    char wn[128], bn[128];
    snprintf(wn,sizeof wn,"model.layers.%d.mlp.gate.weight",L);
    snprintf(bn,sizeof bn,"model.layers.%d.mlp.e_score_correction_bias",L);
    auto W  = load_raw(CK,wn,(size_t)P92_RTR_NEXP*P92_RTR_NEMBD*2);   // bf16
    auto Bs = load_raw(CK,bn,(size_t)P92_RTR_NEXP*4);                 // fp32
    bool real = !W.empty() && !Bs.empty();
    std::mt19937 rng(20260908); std::normal_distribution<float> nd(0,1);
    if (!real) { printf("real router weights unavailable, using synthetic\n");
        W.resize((size_t)P92_RTR_NEXP*P92_RTR_NEMBD); Bs.resize(P92_RTR_NEXP*2);
        for(auto&v:W){ float t=nd(rng)*0.02f; uint32_t u; memcpy(&u,&t,4); v=(uint16_t)(u>>16);} 
        for(int i=0;i<P92_RTR_NEXP;i++){ float t=nd(rng)*0.05f; memcpy(&Bs[i*2],&t,4);} }
    else printf("router weights: real, layer %d\n", L);
    std::vector<float> B(P92_RTR_NEXP); memcpy(B.data(),Bs.data(),P92_RTR_NEXP*4);   // fp32 bias
    std::vector<float> x(P92_RTR_NEMBD); for(auto&v:x) v=nd(rng);

    // ---- CPU reference ---------------------------------------------------
    std::vector<double> lg(P92_RTR_NEXP);
    for(int e=0;e<P92_RTR_NEXP;e++){ double a=0;
        for(int i=0;i<P92_RTR_NEMBD;i++) a += (double)bf(W[(size_t)e*P92_RTR_NEMBD+i])*x[i];
        lg[e]=a; }
    std::vector<float> sc(P92_RTR_NEXP), sel(P92_RTR_NEXP);
    for(int e=0;e<P92_RTR_NEXP;e++){ sc[e]=1.f/(1.f+expf(-(float)lg[e])); sel[e]=sc[e]+B[e]; }
    std::vector<int> rid(P92_RTR_TOPK); std::vector<float> rwt(P92_RTR_TOPK);
    { std::vector<float> t=sel; double s=0;
      for(int k=0;k<P92_RTR_TOPK;k++){ int bi=0; for(int e=1;e<P92_RTR_NEXP;e++)
            if(t[e]>t[bi]||(t[e]==t[bi]&&e<bi)) bi=e;
          rid[k]=bi; rwt[k]=sc[bi]; s+=sc[bi]; t[bi]=-3e38f; }
      for(int k=0;k<P92_RTR_TOPK;k++) rwt[k]=(float)(rwt[k]*(P92_RTR_SCALE/s)); }

    // ---- GPU -------------------------------------------------------------
    uint16_t *dW; float *dx,*dl,*db,*dw; int *di;
    CHK(hipMalloc(&dW,W.size()*2)); CHK(hipMalloc(&dx,P92_RTR_NEMBD*4));
    CHK(hipMalloc(&dl,P92_RTR_NEXP*4)); CHK(hipMalloc(&db,P92_RTR_NEXP*4));
    CHK(hipMalloc(&dw,P92_RTR_TOPK*4)); CHK(hipMalloc(&di,P92_RTR_TOPK*4));
    CHK(hipMemcpy(dW,W.data(),W.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dx,x.data(),P92_RTR_NEMBD*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(db,B.data(),P92_RTR_NEXP*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_p92_route_gemv,dim3(P92_RTR_NEXP/4),dim3(256),0,0,dW,dx,dl);
    hipLaunchKernelGGL(k_p92_route_top8,dim3(1),dim3(64),0,0,dl,db,di,dw);
    CHK(hipDeviceSynchronize());
    std::vector<int> gid(P92_RTR_TOPK); std::vector<float> gwt(P92_RTR_TOPK), gl(P92_RTR_NEXP);
    CHK(hipMemcpy(gid.data(),di,P92_RTR_TOPK*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(gwt.data(),dw,P92_RTR_TOPK*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(gl.data(),dl,P92_RTR_NEXP*4,hipMemcpyDeviceToHost));

    double le=0,lr=0; for(int e=0;e<P92_RTR_NEXP;e++){ double d=gl[e]-lg[e]; le+=d*d; lr+=lg[e]*lg[e]; }
    int idbad=0; double wmax=0, wsum=0;
    for(int k=0;k<P92_RTR_TOPK;k++){ if(gid[k]!=rid[k]) idbad++;
        wmax=std::max(wmax,(double)fabsf(gwt[k]-rwt[k])); wsum+=gwt[k]; }
    printf("logits    : relative_l2 %.3e\n", sqrt(le/lr));
    printf("selection : %d/%d ids differ\n", idbad, P92_RTR_TOPK);
    printf("ids gpu   : "); for(int k=0;k<P92_RTR_TOPK;k++) printf("%d%s",gid[k],k<7?",":"\n");
    printf("ids cpu   : "); for(int k=0;k<P92_RTR_TOPK;k++) printf("%d%s",rid[k],k<7?",":"\n");
    printf("weights   : max_abs_err %.3e   weight_sum %.6f (config routed_scaling_factor 2.5)\n", wmax, wsum);

    // ---- forced tie: lower index must win --------------------------------
    std::vector<float> tl(P92_RTR_NEXP,-20.f); tl[200]=3.f; tl[7]=3.f; tl[64]=3.f;
    std::vector<float> zb(P92_RTR_NEXP,0.f);
    CHK(hipMemcpy(dl,tl.data(),P92_RTR_NEXP*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(db,zb.data(),P92_RTR_NEXP*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_p92_route_top8,dim3(1),dim3(64),0,0,dl,db,di,dw);
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(gid.data(),di,P92_RTR_TOPK*4,hipMemcpyDeviceToHost));
    const bool tie_ok = gid[0]==7 && gid[1]==64 && gid[2]==200;
    printf("tie rule  : three equal scores at 7,64,200 -> %d,%d,%d %s\n",
           gid[0],gid[1],gid[2], tie_ok?"(ascending index, correct)":"(WRONG)");

    const bool pass = idbad==0 && wmax<1e-5 && fabs(wsum-2.5)<1e-4 && sqrt(le/lr)<1e-5 && tie_ok;
    printf("%s\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
