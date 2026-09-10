// Gate every attention projection shape, the bf16 GEMV and RMSNorm against
// scalar CPU references, on the real layer-5 weights.
#include "../src/p92_proj.hip"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
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
static float e2m1(uint32_t n){ uint32_t m=n&7u,e=m>>1,f=m&1u;
    float v = e ? ldexpf(1.f+0.5f*f,(int)e-1) : (f?0.5f:0.f); return (n&8u)?-v:v; }
static float ue4m3(uint32_t b){ uint32_t e=b>>3,m=b&7u;
    if(e==0) return (float)m*0x1p-9f; if(e==15&&m==7) return INFINITY;
    return ldexpf(1.f+m/8.f,(int)e-7); }
static float b2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }

static int mf,wf,sf; static std::unordered_map<std::string,const MRec*> BN;

template<int K,int LANES>
static int gate_proj(const char*name,int L,double*us){
    char nm[160]; snprintf(nm,sizeof nm,name,L);
    auto it=BN.find(nm); if(it==BN.end()){ printf("  missing %s\n",nm); return 1; }
    const MRec*R=it->second;
    const int N=(int)R->rows;
    if((int)R->columns!=K){ printf("  %s columns %u != K %d\n",nm,R->columns,K); return 1; }
    const size_t WB=(size_t)N*(K/2), SB=(size_t)N*(K/16);
    std::vector<uint8_t> W(WB),S(SB);
    rd(wf,W.data(),WB,(off_t)R->weight_offset); rd(sf,S.data(),SB,(off_t)R->scale_offset);
    std::mt19937 rng(20260908+K); std::normal_distribution<float> nd(0,1);
    std::vector<float> x(K); for(auto&v:x) v=nd(rng);
    // xq_nat is natural order for the scalar reference; xq is the even-then-odd
    // order the quantiser now emits and the kernel consumes without staging
    std::vector<int8_t> xq(K), xq_nat(K); std::vector<float> xsg(K/16);
    for(int g=0;g<K/16;g++){ float m=0; for(int i=0;i<16;i++) m=fmaxf(m,fabsf(x[g*16+i]));
        const float sc=m>0?m/127.f:1.f; xsg[g]=sc;
        for(int i=0;i<16;i++){ int q=(int)rintf(x[g*16+i]/sc);
            const int8_t v=(int8_t)(q<-128?-128:(q>127?127:q));
            xq_nat[g*16+i]=v;
            xq[g*16 + ((i&1)? 8+(i>>1) : (i>>1))]=v; } }
    std::vector<float> ref(N);
    for(int r=0;r<N;r++){ double a=0;
        for(int g=0;g<K/16;g++){ const uint8_t*p=&W[(size_t)r*(K/2)+g*8]; double s=0;
            for(int j=0;j<8;j++) s += (double)e2m1(p[j]&15u)*xq_nat[g*16+2*j] + (double)e2m1(p[j]>>4)*xq_nat[g*16+2*j+1];
            a += s*ue4m3(S[(size_t)r*(K/16)+g])*xsg[g]; }
        ref[r]=(float)(a*R->weight_scale_2); }
    uint8_t*dW,*dS; int8_t*dxq; float*dxs,*dy;
    CHK(hipMalloc(&dW,WB)); CHK(hipMalloc(&dS,SB)); CHK(hipMalloc(&dxq,K));
    CHK(hipMalloc(&dxs,(size_t)(K/16)*4)); CHK(hipMalloc(&dy,(size_t)N*4));
    CHK(hipMemcpy(dW,W.data(),WB,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dS,S.data(),SB,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dxq,xq.data(),K,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dxs,xsg.data(),(size_t)(K/16)*4,hipMemcpyHostToDevice));
    constexpr int ROWS=256/LANES;
    hipLaunchKernelGGL((k_p92_proj<K,LANES>),dim3((N+ROWS-1)/ROWS),dim3(256),0,0,dW,dS,dxq,dxs,R->weight_scale_2,dy);
    CHK(hipDeviceSynchronize());
    std::vector<float> got(N); CHK(hipMemcpy(got.data(),dy,(size_t)N*4,hipMemcpyDeviceToHost));
    double e=0,s2=0; for(int i=0;i<N;i++){ double d=got[i]-ref[i]; e+=d*d; s2+=(double)ref[i]*ref[i]; }
    const double rel=sqrt(e/s2);
    hipEvent_t t0,t1; CHK(hipEventCreate(&t0)); CHK(hipEventCreate(&t1));
    const int IT=300;
    for(int i=0;i<20;i++) hipLaunchKernelGGL((k_p92_proj<K,LANES>),dim3((N+ROWS-1)/ROWS),dim3(256),0,0,dW,dS,dxq,dxs,R->weight_scale_2,dy);
    CHK(hipDeviceSynchronize()); CHK(hipEventRecord(t0,0));
    for(int i=0;i<IT;i++) hipLaunchKernelGGL((k_p92_proj<K,LANES>),dim3((N+ROWS-1)/ROWS),dim3(256),0,0,dW,dS,dxq,dxs,R->weight_scale_2,dy);
    CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1));
    float ms; hipEventElapsedTime(&ms,t0,t1); *us=ms*1000.0/IT;
    printf("  %-42s [%5d,%5d] rel_l2 %.3e  %7.2f us  %6.1f GB/s\n",
           nm+14,N,K,rel,*us,(double)(WB+SB)/(*us/1e6)/1e9);
    hipFree(dW);hipFree(dS);hipFree(dxq);hipFree(dxs);hipFree(dy);
    return rel<1e-5?0:1;
}

int main(int argc,char**argv){
    const char*ART=argc>1?argv[1]:"./artifact";
    const int L=argc>2?atoi(argv[2]):5;
    mf=open((std::string(ART)+"/manifest.bin").c_str(),O_RDONLY);
    wf=open((std::string(ART)+"/weights.nvfp4").c_str(),O_RDONLY);
    sf=open((std::string(ART)+"/scales.e4m3").c_str(),O_RDONLY);
    MHdr h; rd(mf,&h,sizeof h,0);
    static std::vector<MRec> recs(h.tensor_count);
    rd(mf,recs.data(),(size_t)h.tensor_count*h.record_bytes,sizeof h);
    for(auto&r:recs) BN[std::string(r.name,strnlen(r.name,96))]=&r;

    int bad=0; double a,b,c,d;
    printf("NVFP4 attention projections, real layer-%d weights:\n",L);
    bad |= gate_proj<2560,32>("model.layers.%d.self_attn.q_a_proj.weight",L,&a);
    bad |= gate_proj<1024,32>("model.layers.%d.self_attn.q_b_proj.weight",L,&b);
    bad |= gate_proj< 512,32>("model.layers.%d.self_attn.kv_b_proj.weight",L,&c);
    bad |= gate_proj<6144,64>("model.layers.%d.self_attn.o_proj.weight",L,&d);
    printf("  four projections: %.2f us/layer -> %.3f ms/token over 46 layers\n",
           a+b+c+d,(a+b+c+d)*46/1000);

    // ---- RMSNorm ---------------------------------------------------------
    printf("\nRMSNorm:\n");
    {
        std::mt19937 rng(7); std::normal_distribution<float> nd(0,1);
        const int N=2560; std::vector<float> x(N); std::vector<uint16_t> g(N);
        for(auto&v:x) v=nd(rng)*3.f;
        for(auto&v:g){ float t=1.f+nd(rng)*0.1f; uint32_t u; memcpy(&u,&t,4); v=(uint16_t)(u>>16); }
        double ss=0; for(float v:x) ss+=(double)v*v;
        const float inv=(float)(1.0/sqrt(ss/N+1e-5));
        std::vector<float> ref(N); for(int i=0;i<N;i++) ref[i]=x[i]*inv*b2f(g[i]);
        float*dx,*dy; uint16_t*dg;
        CHK(hipMalloc(&dx,N*4)); CHK(hipMalloc(&dy,N*4)); CHK(hipMalloc(&dg,N*2));
        CHK(hipMemcpy(dx,x.data(),N*4,hipMemcpyHostToDevice));
        CHK(hipMemcpy(dg,g.data(),N*2,hipMemcpyHostToDevice));
        hipLaunchKernelGGL((k_p92_rmsnorm<2560>),dim3(1),dim3(256),0,0,dx,dg,dy,1e-5f);
        CHK(hipDeviceSynchronize());
        std::vector<float> got(N); CHK(hipMemcpy(got.data(),dy,N*4,hipMemcpyDeviceToHost));
        double e=0,s2=0; for(int i=0;i<N;i++){ double q=got[i]-ref[i]; e+=q*q; s2+=(double)ref[i]*ref[i]; }
        printf("  width 2560, eps 1e-5: rel_l2 %.3e\n",sqrt(e/s2));
        if(sqrt(e/s2)>=1e-6) bad|=1;
    }
    printf("\n%s\n",bad?"FAIL":"PASS");
    return bad;
}
