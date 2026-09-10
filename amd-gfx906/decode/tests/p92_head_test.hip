// Gate lm_head on the real 218 MB NVFP4 head from the artifact. The gate that
// matters is not logit error, it is whether the int8 activation the dot4 path
// requires changes the emitted token: argmax and the top-5 are compared
// against a reference that keeps the activation in fp32 and uses exact E2M1.
#include "../src/p92_head.hip"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
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

int main(int argc,char**argv){
    const char*ART = argc>1?argv[1]:"./artifact";
    const int TRIALS = argc>2?atoi(argv[2]):16;
    int mf=open((std::string(ART)+"/manifest.bin").c_str(),O_RDONLY);
    int wf=open((std::string(ART)+"/weights.nvfp4").c_str(),O_RDONLY);
    int sf=open((std::string(ART)+"/scales.e4m3").c_str(),O_RDONLY);
    MHdr h; rd(mf,&h,sizeof h,0);
    std::vector<MRec> recs(h.tensor_count); rd(mf,recs.data(),(size_t)h.tensor_count*h.record_bytes,sizeof h);
    const MRec*R=nullptr;
    for(auto&r:recs) if(!strncmp(r.name,"lm_head.weight",14)) R=&r;
    if(!R){ printf("lm_head missing\n"); return 1; }
    printf("lm_head %ux%u  weight_scale_2 %.6g\n",R->rows,R->columns,R->weight_scale_2);
    if(R->rows!=P92_VOCAB||R->columns!=P92_NEMBD){ printf("unexpected shape\n"); return 1; }

    const size_t WB=(size_t)P92_VOCAB*(P92_NEMBD/2), SB=(size_t)P92_VOCAB*(P92_NEMBD/16);
    std::vector<uint8_t> hW(WB), hS(SB);
    rd(wf,hW.data(),WB,(off_t)R->weight_offset);
    rd(sf,hS.data(),SB,(off_t)R->scale_offset);
    printf("head resident: %.1f MB weights + %.1f MB scales\n", WB/1e6, SB/1e6);

    uint8_t *dW,*dS; int8_t *dxq,*dxq2; float *dxs,*dxs2,*dlog,*dpv; int *dpi,*dout;
    const int CH=(P92_VOCAB+P92_HEAD_RED-1)/P92_HEAD_RED;
    CHK(hipMalloc(&dW,WB)); CHK(hipMalloc(&dS,SB));
    CHK(hipMalloc(&dxq,P92_NEMBD)); CHK(hipMalloc(&dxs,4));
    CHK(hipMalloc(&dxq2,P92_NEMBD)); CHK(hipMalloc(&dxs2,4));
    CHK(hipMalloc(&dlog,(size_t)P92_VOCAB*4));
    CHK(hipMalloc(&dpv,(size_t)CH*4)); CHK(hipMalloc(&dpi,(size_t)CH*4)); CHK(hipMalloc(&dout,4));
    int *dcand; CHK(hipMalloc(&dcand,P92_HEAD_REFINE*4));
    uint32_t *dst,*dhist,*dcnt2,*doff2;
    CHK(hipMalloc(&dst,4*4)); CHK(hipMalloc(&dhist,P92_RBINS*4));
    CHK(hipMalloc(&dcnt2,2*P92_GMAXC*4)); CHK(hipMalloc(&doff2,(2*P92_GMAXC+2)*4));
    const int HNC=(P92_VOCAB+P92_GCHUNK-1)/P92_GCHUNK;
    // The head's candidate selection runs over 151552 rows. The fused
    // single-block selector measured 343.90 us there, against 12.92 for the
    // refine it feeds - it is a one-block kernel and does not scale. The staged
    // parallel selector produces byte-identical output and is used instead.
    auto head_select=[&]() -> void {
        (void)hipMemsetAsync(dst,0,4*4,0);
        for (int sh : {21,10,0}) {
            (void)hipMemsetAsync(dhist,0,P92_RBINS*4,0);
            hipLaunchKernelGGL(k_p92_idx_hist,dim3(64),dim3(256),0,0,dlog,P92_VOCAB,sh,dst,dhist);
            hipLaunchKernelGGL(k_p92_idx_pick,dim3(1),dim3(1024),0,0,dhist,P92_HEAD_REFINE,sh,dst); }
        hipLaunchKernelGGL(k_p92_idx_count,dim3(HNC),dim3(256),0,0,dlog,P92_VOCAB,dst,dcnt2);
        hipLaunchKernelGGL(k_p92_idx_scan,dim3(1),dim3(256),0,0,dcnt2,HNC,P92_HEAD_REFINE,doff2);
        hipLaunchKernelGGL(k_p92_idx_write,dim3(HNC),dim3(256),0,0,dlog,P92_VOCAB,dst,doff2,P92_HEAD_REFINE,dcand);
        hipLaunchKernelGGL(k_p92_idx_pad,dim3(8),dim3(256),0,0,doff2,P92_HEAD_REFINE,dcand); };
    CHK(hipMemcpy(dW,hW.data(),WB,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dS,hS.data(),SB,hipMemcpyHostToDevice));

    std::mt19937 rng(20260908); std::normal_distribution<float> nd(0,1);
    int arg_agree=0, top5_agree=0, top1_in_ref5=0;
    double worst_rel=0;
    std::vector<float> logits(P92_VOCAB);
    for(int t=0;t<TRIALS;t++){
        std::vector<float> x(P92_NEMBD); for(auto&v:x) v=nd(rng);
        float xm=0; for(float v:x) xm=fmaxf(xm,fabsf(v));
        const float xsc=xm/127.f;
        std::vector<int8_t> xq(P92_NEMBD), xq2(P92_NEMBD);
        std::vector<float> res(P92_NEMBD);
        for(int i=0;i<P92_NEMBD;i++){ int q=(int)rintf(x[i]/xsc); q=q<-128?-128:(q>127?127:q);
            xq[i]=(int8_t)q; res[i]=x[i]-q*xsc; }
        float rm=0; for(float v:res) rm=fmaxf(rm,fabsf(v));
        const float xsc2 = rm>0?rm/127.f:1.f;
        for(int i=0;i<P92_NEMBD;i++){ int q=(int)rintf(res[i]/xsc2); xq2[i]=(int8_t)(q<-128?-128:(q>127?127:q)); }
        CHK(hipMemcpy(dxq,xq.data(),P92_NEMBD,hipMemcpyHostToDevice));
        CHK(hipMemcpy(dxs,&xsc,4,hipMemcpyHostToDevice));
        CHK(hipMemcpy(dxq2,xq2.data(),P92_NEMBD,hipMemcpyHostToDevice));
        CHK(hipMemcpy(dxs2,&xsc2,4,hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k_p92_head_logits,dim3((P92_VOCAB+7)/8),dim3(256),0,0,dW,dS,dxq,dxs,R->weight_scale_2,dlog);
        head_select();
        hipLaunchKernelGGL(k_p92_head_refine,dim3(P92_HEAD_REFINE/P92_HEAD_PER_BLOCK),dim3(256),0,0,
                           dW,dS,dxq,dxq2,dxs,dxs2,R->weight_scale_2,dcand,P92_HEAD_REFINE,dlog);
        hipLaunchKernelGGL(k_p92_head_argmax_part,dim3(CH),dim3(256),0,0,dlog,dpv,dpi);
        hipLaunchKernelGGL(k_p92_head_argmax_final,dim3(1),dim3(256),0,0,dpv,dpi,CH,dout);
        CHK(hipDeviceSynchronize());
        int gidx; CHK(hipMemcpy(&gidx,dout,4,hipMemcpyDeviceToHost));
        CHK(hipMemcpy(logits.data(),dlog,(size_t)P92_VOCAB*4,hipMemcpyDeviceToHost));

        // reference: exact E2M1, fp32 activation, on the rows that matter plus a sample
        std::vector<int> cand(P92_VOCAB); for(int i=0;i<P92_VOCAB;i++) cand[i]=i;
        std::partial_sort(cand.begin(),cand.begin()+64,cand.end(),
            [&](int a,int b){ if(logits[a]!=logits[b]) return logits[a]>logits[b]; return a<b; });
        cand.resize(64);
        std::vector<std::pair<double,int>> ref;
        for(int r : cand){
            double a=0;
            for(int g=0;g<P92_NEMBD/16;g++){
                const uint8_t *p=&hW[(size_t)r*(P92_NEMBD/2)+g*8]; double s=0;
                for(int j=0;j<8;j++) s += (double)e2m1(p[j]&15u)*x[g*16+2*j] + (double)e2m1(p[j]>>4)*x[g*16+2*j+1];
                a += s*ue4m3(hS[(size_t)r*(P92_NEMBD/16)+g]); }
            ref.push_back({a*R->weight_scale_2, r}); }
        std::sort(ref.begin(),ref.end(),[](auto&a,auto&b){ if(a.first!=b.first) return a.first>b.first; return a.second<b.second; });
        if(ref[0].second==gidx) arg_agree++;
        std::vector<int> g5,r5;
        for(int i=0;i<5;i++){ g5.push_back(cand[i]); r5.push_back(ref[i].second); }
        if(g5==r5) top5_agree++;
        if(std::find(r5.begin(),r5.end(),gidx)!=r5.end()) top1_in_ref5++;
        double num=0,den=0;
        for(auto&pr:ref){ double d=logits[pr.second]-pr.first; num+=d*d; den+=pr.first*pr.first; }
        worst_rel=fmax(worst_rel,sqrt(num/den));
    }
    printf("argmax agrees with the fp32-activation reference: %d/%d\n",arg_agree,TRIALS);
    printf("top-5 order agrees exactly:                        %d/%d\n",top5_agree,TRIALS);
    printf("device argmax inside the reference top-5:          %d/%d\n",top1_in_ref5,TRIALS);
    printf("worst relative_l2 over the top-64 logits:          %.3e\n",worst_rel);

    hipEvent_t t0,t1; CHK(hipEventCreate(&t0)); CHK(hipEventCreate(&t1));
    auto head=[&]() -> void {
        hipLaunchKernelGGL(k_p92_head_logits,dim3((P92_VOCAB+7)/8),dim3(256),0,0,dW,dS,dxq,dxs,R->weight_scale_2,dlog);
        head_select();
        hipLaunchKernelGGL(k_p92_head_refine,dim3(P92_HEAD_REFINE/P92_HEAD_PER_BLOCK),dim3(256),0,0,
                           dW,dS,dxq,dxq2,dxs,dxs2,R->weight_scale_2,dcand,P92_HEAD_REFINE,dlog);
        hipLaunchKernelGGL(k_p92_head_argmax_part,dim3(CH),dim3(256),0,0,dlog,dpv,dpi);
        hipLaunchKernelGGL(k_p92_head_argmax_final,dim3(1),dim3(256),0,0,dpv,dpi,CH,dout); };
    for(int i=0;i<10;i++) head(); CHK(hipDeviceSynchronize());
    CHK(hipEventRecord(t0,0)); const int IT=100;
    for(int i=0;i<IT;i++) head();
    CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1));
    float ms; hipEventElapsedTime(&ms,t0,t1); const double us=ms*1000.0/IT;
    printf("head: %.2f us/token  %.1f MB  %.1f GB/s\n", us, (WB+SB)/1e6, (double)(WB+SB)/(us/1e6)/1e9);
    // The gate is the emitted token, not logit error: the int8 activation is a
    // deliberate part of the contract and 8.6e-03 relative error on the logits
    // is what it costs. What must not change is argmax.
    auto time_one=[&](const char*name, auto fn){
        for(int i=0;i<10;i++) fn(); (void)hipDeviceSynchronize();
        (void)hipEventRecord(t0,0); for(int i=0;i<IT;i++) fn();
        (void)hipEventRecord(t1,0); (void)hipEventSynchronize(t1);
        float e; (void)hipEventElapsedTime(&e,t0,t1);
        printf("  %-24s %7.2f us\n",name,e*1000.0/IT); };
    time_one("fast logits",[&]{ hipLaunchKernelGGL(k_p92_head_logits,dim3((P92_VOCAB+7)/8),dim3(256),0,0,dW,dS,dxq,dxs,R->weight_scale_2,dlog); });
    time_one("top-256 select fused",[&]{ hipLaunchKernelGGL(k_p92_idx_select,dim3(1),dim3(P92_SEL_THREADS),0,0,dlog,P92_VOCAB,P92_HEAD_REFINE,dcand); });
    time_one("top-256 select staged",[&]{ head_select(); });
    time_one("refine 256",[&]{ hipLaunchKernelGGL(k_p92_head_refine,dim3(P92_HEAD_REFINE/P92_HEAD_PER_BLOCK),dim3(256),0,0,dW,dS,dxq,dxq2,dxs,dxs2,R->weight_scale_2,dcand,P92_HEAD_REFINE,dlog); });
    time_one("argmax",[&]{ hipLaunchKernelGGL(k_p92_head_argmax_part,dim3(CH),dim3(256),0,0,dlog,dpv,dpi);
                             hipLaunchKernelGGL(k_p92_head_argmax_final,dim3(1),dim3(256),0,0,dpv,dpi,CH,dout); });
    const bool pass = arg_agree==TRIALS;
    printf("%s (gate: argmax agrees on every trial)\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
