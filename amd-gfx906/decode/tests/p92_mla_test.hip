// Gate the absorbed MLA decode against a scalar CPU reference: sinks plus a
// padded selection list, rope term included, softmax over the union.
#include "../src/p92_mla_decode.hip"
#include "../src/p92_mla_split.hip"
#include "../src/p92_mla_g16.hip"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <cstdlib>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
static p92_bf f2bf(float f){ uint32_t u; memcpy(&u,&f,4); return (p92_bf)((u+0x8000u)>>16); }
static float  bf2f(p92_bf b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }

int main(int argc,char**argv){
    const int T    = argc>1?atoi(argv[1]):4096;    // cache positions
    const int NSEL = argc>2?atoi(argv[2]):2048;    // DSA top-k
    std::mt19937 rng(20260908); std::normal_distribution<float> nd(0,1);
    std::vector<float> ql((size_t)P92_NH*P92_KVLORA), qp((size_t)P92_NH*P92_PE);
    for(auto&v:ql) v=nd(rng)*0.05f;
    for(auto&v:qp) v=nd(rng)*0.05f;
    std::vector<p92_bf> lat((size_t)T*P92_KVLORA), kpe((size_t)T*P92_PE),
                        sl((size_t)P92_SINKS*P92_KVLORA), sp((size_t)P92_SINKS*P92_PE);
    for(auto&v:lat) v=f2bf(nd(rng)*0.1f);
    for(auto&v:kpe) v=f2bf(nd(rng)*0.1f);
    for(auto&v:sl)  v=f2bf(nd(rng)*0.1f);
    for(auto&v:sp)  v=f2bf(nd(rng)*0.1f);
    std::vector<int> sel(NSEL);
    // selection pattern: 0 random over the cache, 1 contiguous, 2 strided
    const int selmode = getenv("P92_SELMODE") ? atoi(getenv("P92_SELMODE")) : 0;
    for(int i=0;i<NSEL;i++)
        sel[i] = selmode==1 ? (i % T)
               : selmode==2 ? ((i * (T/NSEL)) % T)
                            : (int)(rng()%T);
    for(int i=NSEL-7;i<NSEL;i++) sel[i]=-1;                 // padding must be skipped

    // ---- CPU reference ---------------------------------------------------
    const float scale = 1.0f/sqrtf((float)P92_QK_DIM);
    std::vector<float> ref((size_t)P92_NH*P92_KVLORA,0.f);
    for(int h=0;h<P92_NH;h++){
        const float *q=&ql[(size_t)h*P92_KVLORA], *qe=&qp[(size_t)h*P92_PE];
        std::vector<float> sc; sc.reserve(P92_SINKS+NSEL);
        std::vector<int> src; src.reserve(P92_SINKS+NSEL);
        for(int j=0;j<P92_SINKS;j++){
            double s=0; const p92_bf*c=&sl[(size_t)j*P92_KVLORA];
            for(int d=0;d<P92_KVLORA;d++) s+=(double)q[d]*bf2f(c[d]);
            const p92_bf*e=&sp[(size_t)j*P92_PE];
            for(int d=0;d<P92_PE;d++) s+=(double)qe[d]*bf2f(e[d]);
            sc.push_back((float)(s*scale)); src.push_back(-1000-j); }
        for(int j=0;j<NSEL;j++){ int tk=sel[j]; if(tk<0) continue;
            double s=0; const p92_bf*c=&lat[(size_t)tk*P92_KVLORA];
            for(int d=0;d<P92_KVLORA;d++) s+=(double)q[d]*bf2f(c[d]);
            const p92_bf*e=&kpe[(size_t)tk*P92_PE];
            for(int d=0;d<P92_PE;d++) s+=(double)qe[d]*bf2f(e[d]);
            sc.push_back((float)(s*scale)); src.push_back(tk); }
        float m=-1e30f; for(float s:sc) m=fmaxf(m,s);
        double sum=0; std::vector<double> p(sc.size());
        for(size_t j=0;j<sc.size();j++){ p[j]=exp((double)sc[j]-m); sum+=p[j]; }
        for(size_t j=0;j<sc.size();j++){
            const double w=p[j]/sum;
            const p92_bf*c = src[j]<=-1000 ? &sl[(size_t)(-1000-src[j])*P92_KVLORA]
                                           : &lat[(size_t)src[j]*P92_KVLORA];
            for(int d=0;d<P92_KVLORA;d++) ref[(size_t)h*P92_KVLORA+d]+=(float)(w*bf2f(c[d])); }
    }

    // ---- GPU -------------------------------------------------------------
    float *dql,*dqp,*dctx; p92_bf *dlat,*dkpe,*dsl,*dsp; int *dsel;
    CHK(hipMalloc(&dql,ql.size()*4)); CHK(hipMalloc(&dqp,qp.size()*4));
    CHK(hipMalloc(&dlat,lat.size()*2)); CHK(hipMalloc(&dkpe,kpe.size()*2));
    CHK(hipMalloc(&dsl,sl.size()*2)); CHK(hipMalloc(&dsp,sp.size()*2));
    CHK(hipMalloc(&dsel,NSEL*4)); CHK(hipMalloc(&dctx,(size_t)P92_NH*P92_KVLORA*4));
    CHK(hipMemcpy(dql,ql.data(),ql.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dqp,qp.data(),qp.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dlat,lat.data(),lat.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dkpe,kpe.data(),kpe.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsl,sl.data(),sl.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsp,sp.data(),sp.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsel,sel.data(),NSEL*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_p92_mla_decode,dim3(P92_NH),dim3(256),0,0,
                       dql,dqp,dlat,dkpe,dsl,dsp,dsel,NSEL,P92_SINKS,dctx);
    CHK(hipDeviceSynchronize());
    std::vector<float> got((size_t)P92_NH*P92_KVLORA);
    CHK(hipMemcpy(got.data(),dctx,got.size()*4,hipMemcpyDeviceToHost));
    double es=0,rs=0,mx=0;
    for(size_t i=0;i<got.size();i++){ double d=got[i]-ref[i]; es+=d*d; rs+=(double)ref[i]*ref[i]; mx=fmax(mx,fabs(d)); }
    const double rel=sqrt(es/rs);
    printf("context %d, %d selected (7 padded) + %d sinks, %d heads\n",T,NSEL,P92_SINKS,P92_NH);
    printf("absorbed MLA decode vs CPU reference: relative_l2 %.3e  max_abs %.3e\n",rel,mx);

    // ---- split-K arm ------------------------------------------------------
    float *dpa,*dpm,*dpl,*dctx2;
    CHK(hipMalloc(&dpa,(size_t)P92_NH*P92_SPLITS*P92_KVLORA*4));
    CHK(hipMalloc(&dpm,P92_NH*P92_SPLITS*4)); CHK(hipMalloc(&dpl,P92_NH*P92_SPLITS*4));
    CHK(hipMalloc(&dctx2,(size_t)P92_NH*P92_KVLORA*4));
    auto splitk=[&]() -> void {
        hipLaunchKernelGGL(k_p92_mla_split,dim3(P92_NH,P92_SPLITS/4),dim3(256),0,0,
                           dql,dqp,dlat,dkpe,dsl,dsp,dsel,NSEL,P92_SINKS,dpa,dpm,dpl);
        hipLaunchKernelGGL(k_p92_mla_combine,dim3(P92_NH),dim3(256),0,0,dpa,dpm,dpl,dctx2); };
    splitk(); CHK(hipDeviceSynchronize());
    std::vector<float> got2((size_t)P92_NH*P92_KVLORA);
    CHK(hipMemcpy(got2.data(),dctx2,got2.size()*4,hipMemcpyDeviceToHost));
    double es2=0,mx2=0;
    for(size_t i=0;i<got2.size();i++){ double d=got2[i]-ref[i]; es2+=d*d; mx2=fmax(mx2,fabs(d)); }
    const double rel2=sqrt(es2/rs);
    printf("split-K MLA decode  vs CPU reference: relative_l2 %.3e  max_abs %.3e\n",rel2,mx2);

    hipEvent_t t0,t1; CHK(hipEventCreate(&t0)); CHK(hipEventCreate(&t1));
    for(int i=0;i<20;i++) hipLaunchKernelGGL(k_p92_mla_decode,dim3(P92_NH),dim3(256),0,0,
                       dql,dqp,dlat,dkpe,dsl,dsp,dsel,NSEL,P92_SINKS,dctx);
    CHK(hipDeviceSynchronize()); CHK(hipEventRecord(t0,0));
    const int IT=200;
    for(int i=0;i<IT;i++) hipLaunchKernelGGL(k_p92_mla_decode,dim3(P92_NH),dim3(256),0,0,
                       dql,dqp,dlat,dkpe,dsl,dsp,dsel,NSEL,P92_SINKS,dctx);
    CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1));
    float ms; hipEventElapsedTime(&ms,t0,t1);
    const double us=ms*1000.0/IT;
    const double by=(double)P92_NH*(NSEL+P92_SINKS)*(P92_KVLORA+P92_PE)*2;
    printf("timing: %.2f us/layer  %.1f GB/s   16 DSA layers -> %.2f ms/token\n",
           us, by/(us/1e6)/1e9, us*16/1000);
    auto g16=[&]() -> void {
        hipLaunchKernelGGL(k_p92_mla_g16,dim3(P92_NH,P92_SPLITS/4),dim3(256),0,0,
                           dql,dqp,dlat,dkpe,dsl,dsp,dsel,NSEL,P92_SINKS,0,0,dpa,dpm,dpl);
        hipLaunchKernelGGL(k_p92_mla_combine,dim3(P92_NH),dim3(256),0,0,dpa,dpm,dpl,dctx2); };
    g16(); CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(got2.data(),dctx2,got2.size()*4,hipMemcpyDeviceToHost));
    double es3=0,mx3=0;
    for(size_t i=0;i<got2.size();i++){ double d=got2[i]-ref[i]; es3+=d*d; mx3=fmax(mx3,fabs(d)); }
    const double rel3=sqrt(es3/rs);
    printf("g16 MLA decode      vs CPU reference: relative_l2 %.3e  max_abs %.3e\n",rel3,mx3);
    for(int i=0;i<20;i++) g16(); CHK(hipDeviceSynchronize()); CHK(hipEventRecord(t0,0));
    for(int i=0;i<IT;i++) g16();
    CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1));
    float ms3; hipEventElapsedTime(&ms3,t0,t1);
    const double us3=ms3*1000.0/IT;
    printf("g16    : %.2f us/layer  %.1f GB/s   16 DSA layers -> %.2f ms/token   speedup %.2fx\n",
           us3, by/(us3/1e6)/1e9, us3*16/1000, us/us3);

    for(int i=0;i<20;i++) splitk(); CHK(hipDeviceSynchronize()); CHK(hipEventRecord(t0,0));
    for(int i=0;i<IT;i++) splitk();
    CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1));
    float ms2; hipEventElapsedTime(&ms2,t0,t1);
    const double us2=ms2*1000.0/IT;
    printf("split-K: %.2f us/layer  %.1f GB/s   16 DSA layers -> %.2f ms/token   speedup %.2fx\n",
           us2, by/(us2/1e6)/1e9, us2*16/1000, us/us2);
    const bool pass = rel < 1e-5 && rel2 < 1e-5 && rel3 < 1e-5;
    printf("%s (gate: all arms relative_l2 < 1e-5)\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
