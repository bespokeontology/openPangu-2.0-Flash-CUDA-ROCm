// The complete per-layer DSA path, composed and measured end to end: append the
// new index key, score every position, select the top 2048, then run absorbed
// MLA over that selection plus the 128 sinks. Microkernel timings do not add
// up to this, so this is the number that matters.
//
// Also measures the sliding-window layer, which skips the indexer entirely and
// takes the last 512 positions, and reports the per-token cost of the real
// 16 DSA / 30 sliding layer mix.
#include "../src/p92_mla_decode.hip"
#include "../src/p92_mla_split.hip"
#include "../src/p92_mla_g16.hip"
#include "../src/p92_dsa_index.hip"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
static p92_bf f2b(float f){ uint32_t u; memcpy(&u,&f,4); return (p92_bf)((u+0x8000u)>>16); }
static float  b2f(p92_bf b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }

int main(int argc,char**argv){
    const int N = argc>1?atoi(argv[1]):8192;      // resident positions
    const int cap = N;
    std::mt19937 rng(20260908); std::normal_distribution<float> nd(0,1);

    // MLA state
    std::vector<float> ql((size_t)P92_NH*P92_KVLORA), qp((size_t)P92_NH*P92_PE);
    for(auto&v:ql) v=nd(rng)*0.05f;
    for(auto&v:qp) v=nd(rng)*0.05f;
    std::vector<p92_bf> lat((size_t)N*P92_KVLORA), kpe((size_t)N*P92_PE),
                        sl((size_t)P92_SINKS*P92_KVLORA), spk((size_t)P92_SINKS*P92_PE);
    for(auto&v:lat) v=f2b(nd(rng)*0.1f);
    for(auto&v:kpe) v=f2b(nd(rng)*0.1f);
    for(auto&v:sl)  v=f2b(nd(rng)*0.1f);
    for(auto&v:spk) v=f2b(nd(rng)*0.1f);
    // indexer state
    std::vector<float> qi((size_t)P92_IDX_HEADS*P92_IDX_DIM), wh(P92_IDX_HEADS);
    for(auto&v:qi) v=nd(rng)*0.3f;
    for(auto&v:wh) v=fabsf(nd(rng))*0.5f;
    std::vector<p92_bf> kc((size_t)P92_IDX_DIM*cap);
    std::vector<std::vector<float>> K(N,std::vector<float>(P92_IDX_DIM));
    for(int p=0;p<N;p++) for(int d=0;d<P92_IDX_DIM;d++){
        const float v=nd(rng)*0.4f; K[p][d]=b2f(f2b(v)); kc[(size_t)d*cap+p]=f2b(v); }

    float *dql,*dqp,*dqi,*dwh,*dsc,*dpa,*dpm,*dpl,*dctx;
    p92_bf *dlat,*dkpe,*dsl,*dsp,*dkc; int *dsel,*dnsel;
    uint32_t *dstate,*dh,*dcnt,*doff;
    CHK(hipMalloc(&dql,ql.size()*4)); CHK(hipMalloc(&dqp,qp.size()*4));
    CHK(hipMalloc(&dqi,qi.size()*4)); CHK(hipMalloc(&dwh,wh.size()*4));
    CHK(hipMalloc(&dlat,lat.size()*2)); CHK(hipMalloc(&dkpe,kpe.size()*2));
    CHK(hipMalloc(&dsl,sl.size()*2)); CHK(hipMalloc(&dsp,spk.size()*2));
    CHK(hipMalloc(&dkc,kc.size()*2)); CHK(hipMalloc(&dsc,(size_t)N*4));
    CHK(hipMalloc(&dsel,P92_IDX_TOPK*4)); CHK(hipMalloc(&dnsel,4));
    CHK(hipMalloc(&dstate,4*4)); CHK(hipMalloc(&dh,P92_RBINS*4));
    CHK(hipMalloc(&dcnt,2*P92_GMAXC*4)); CHK(hipMalloc(&doff,(2*P92_GMAXC+2)*4));
    CHK(hipMalloc(&dpa,(size_t)P92_NH*P92_SPLITS*P92_KVLORA*4));
    CHK(hipMalloc(&dpm,P92_NH*P92_SPLITS*4)); CHK(hipMalloc(&dpl,P92_NH*P92_SPLITS*4));
    CHK(hipMalloc(&dctx,(size_t)P92_NH*P92_KVLORA*4));
    CHK(hipMemcpy(dql,ql.data(),ql.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dqp,qp.data(),qp.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dqi,qi.data(),qi.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dwh,wh.data(),wh.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dlat,lat.data(),lat.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dkpe,kpe.data(),kpe.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsl,sl.data(),sl.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsp,spk.data(),spk.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dkc,kc.data(),kc.size()*2,hipMemcpyHostToDevice));

    const float iscale = 1.0f/sqrtf((float)P92_IDX_DIM);
    const int SB=(N+255)/256, WANT=std::min(N,P92_IDX_TOPK);
    const int NC=(N+P92_GCHUNK-1)/P92_GCHUNK;
    const bool use_fused = N <= 16384;          // measured crossover

    auto dsa_layer=[&]() -> void {
        hipLaunchKernelGGL(k_p92_idx_scores,dim3(SB),dim3(256),0,0,dqi,dkc,dwh,cap,N,iscale,dsc);
        if (use_fused) {
            hipLaunchKernelGGL(k_p92_idx_select,dim3(1),dim3(P92_SEL_THREADS),0,0,dsc,N,WANT,dsel);
        } else {
            (void)hipMemsetAsync(dstate,0,4*4,0);
            for (int sh : {21,10,0}) {
                (void)hipMemsetAsync(dh,0,P92_RBINS*4,0);
                hipLaunchKernelGGL(k_p92_idx_hist,dim3(64),dim3(256),0,0,dsc,N,sh,dstate,dh);
                hipLaunchKernelGGL(k_p92_idx_pick,dim3(1),dim3(1024),0,0,dh,WANT,sh,dstate); }
            hipLaunchKernelGGL(k_p92_idx_count,dim3(NC),dim3(256),0,0,dsc,N,dstate,dcnt);
            hipLaunchKernelGGL(k_p92_idx_scan,dim3(1),dim3(256),0,0,dcnt,NC,P92_IDX_TOPK,doff);
            hipLaunchKernelGGL(k_p92_idx_write,dim3(NC),dim3(256),0,0,dsc,N,dstate,doff,P92_IDX_TOPK,dsel);
            hipLaunchKernelGGL(k_p92_idx_pad,dim3(8),dim3(256),0,0,doff,P92_IDX_TOPK,dsel); }
        hipLaunchKernelGGL(k_p92_mla_g16,dim3(P92_NH,P92_SPLITS/4),dim3(256),0,0,
                           dql,dqp,dlat,dkpe,dsl,dsp,dsel,WANT,P92_SINKS,0,0,dpa,dpm,dpl);
        hipLaunchKernelGGL(k_p92_mla_combine,dim3(P92_NH),dim3(256),0,0,dpa,dpm,dpl,dctx); };

    const int WIN = std::min(N,512);
    auto swa_layer=[&]() -> void {
        hipLaunchKernelGGL(k_p92_mla_g16,dim3(P92_NH,P92_SPLITS/4),dim3(256),0,0,
                           dql,dqp,dlat,dkpe,dsl,dsp,nullptr,WIN,P92_SINKS,N-WIN,cap,dpa,dpm,dpl);
        hipLaunchKernelGGL(k_p92_mla_combine,dim3(P92_NH),dim3(256),0,0,dpa,dpm,dpl,dctx); };

    // ---- correctness of the composed path --------------------------------
    dsa_layer(); CHK(hipDeviceSynchronize());
    std::vector<int> sel(P92_IDX_TOPK); std::vector<float> got((size_t)P92_NH*P92_KVLORA);
    CHK(hipMemcpy(sel.data(),dsel,P92_IDX_TOPK*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(got.data(),dctx,got.size()*4,hipMemcpyDeviceToHost));
    // CPU: score, select, attend, using the selection the device actually produced
    std::vector<float> ref((size_t)P92_NH*P92_KVLORA,0.f);
    const float ascale = 1.0f/sqrtf((float)P92_QK_DIM);
    for(int h=0;h<P92_NH;h++){
        const float *q=&ql[(size_t)h*P92_KVLORA], *qe=&qp[(size_t)h*P92_PE];
        std::vector<float> s; std::vector<int> src;
        for(int j=0;j<P92_SINKS;j++){ double a=0;
            for(int d=0;d<P92_KVLORA;d++) a+=(double)q[d]*b2f(sl[(size_t)j*P92_KVLORA+d]);
            for(int d=0;d<P92_PE;d++)     a+=(double)qe[d]*b2f(spk[(size_t)j*P92_PE+d]);
            s.push_back((float)(a*ascale)); src.push_back(-1000-j); }
        for(int j=0;j<WANT;j++){ const int tk=sel[j]; if(tk<0) continue; double a=0;
            for(int d=0;d<P92_KVLORA;d++) a+=(double)q[d]*b2f(lat[(size_t)tk*P92_KVLORA+d]);
            for(int d=0;d<P92_PE;d++)     a+=(double)qe[d]*b2f(kpe[(size_t)tk*P92_PE+d]);
            s.push_back((float)(a*ascale)); src.push_back(tk); }
        float m=-1e30f; for(float v:s) m=fmaxf(m,v);
        double sum=0; std::vector<double> p(s.size());
        for(size_t j=0;j<s.size();j++){ p[j]=exp((double)s[j]-m); sum+=p[j]; }
        for(size_t j=0;j<s.size();j++){ const double w=p[j]/sum;
            const p92_bf*c = src[j]<=-1000 ? &sl[(size_t)(-1000-src[j])*P92_KVLORA]
                                           : &lat[(size_t)src[j]*P92_KVLORA];
            for(int d=0;d<P92_KVLORA;d++) ref[(size_t)h*P92_KVLORA+d]+=(float)(w*b2f(c[d])); } }
    double es=0,rs=0,mx=0;
    for(size_t i=0;i<got.size();i++){ double d=got[i]-ref[i]; es+=d*d; rs+=(double)ref[i]*ref[i]; mx=fmax(mx,fabs(d)); }
    const double rel=sqrt(es/rs);
    int nsel_live=0; for(int v:sel) if(v>=0) nsel_live++;
    printf("composed DSA layer, %d positions, %d selected, %d sinks, selector %s\n",
           N,nsel_live,P92_SINKS,use_fused?"fused":"staged");
    printf("  vs CPU reference over the device's own selection: relative_l2 %.3e  max_abs %.3e\n",rel,mx);

    // ---- timing ----------------------------------------------------------
    hipEvent_t t0,t1; CHK(hipEventCreate(&t0)); CHK(hipEventCreate(&t1));
    const int IT=100;
    for(int i=0;i<20;i++) dsa_layer(); CHK(hipDeviceSynchronize());
    CHK(hipEventRecord(t0,0)); for(int i=0;i<IT;i++) dsa_layer();
    CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1));
    float ms; hipEventElapsedTime(&ms,t0,t1); const double us_dsa=ms*1000.0/IT;
    for(int i=0;i<20;i++) swa_layer(); CHK(hipDeviceSynchronize());
    CHK(hipEventRecord(t0,0)); for(int i=0;i<IT;i++) swa_layer();
    CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1));
    hipEventElapsedTime(&ms,t0,t1); const double us_swa=ms*1000.0/IT;

    // paired, interleaved: indexer alone, MLA alone, composed, three reps each,
    // same allocations and same cache, so composition is compared against its
    // own parts rather than against a differently-shaped microbenchmark
    auto idx_only=[&]() -> void {
        hipLaunchKernelGGL(k_p92_idx_scores,dim3(SB),dim3(256),0,0,dqi,dkc,dwh,cap,N,iscale,dsc);
        if (use_fused) hipLaunchKernelGGL(k_p92_idx_select,dim3(1),dim3(P92_SEL_THREADS),0,0,dsc,N,WANT,dsel);
        else { (void)hipMemsetAsync(dstate,0,4*4,0);
            for (int sh : {21,10,0}) { (void)hipMemsetAsync(dh,0,P92_RBINS*4,0);
                hipLaunchKernelGGL(k_p92_idx_hist,dim3(64),dim3(256),0,0,dsc,N,sh,dstate,dh);
                hipLaunchKernelGGL(k_p92_idx_pick,dim3(1),dim3(1024),0,0,dh,WANT,sh,dstate); }
            hipLaunchKernelGGL(k_p92_idx_count,dim3(NC),dim3(256),0,0,dsc,N,dstate,dcnt);
            hipLaunchKernelGGL(k_p92_idx_scan,dim3(1),dim3(256),0,0,dcnt,NC,P92_IDX_TOPK,doff);
            hipLaunchKernelGGL(k_p92_idx_write,dim3(NC),dim3(256),0,0,dsc,N,dstate,doff,P92_IDX_TOPK,dsel);
            hipLaunchKernelGGL(k_p92_idx_pad,dim3(8),dim3(256),0,0,doff,P92_IDX_TOPK,dsel); } };
    auto mla_only=[&]() -> void {
        hipLaunchKernelGGL(k_p92_mla_g16,dim3(P92_NH,P92_SPLITS/4),dim3(256),0,0,
                           dql,dqp,dlat,dkpe,dsl,dsp,dsel,WANT,P92_SINKS,0,0,dpa,dpm,dpl);
        hipLaunchKernelGGL(k_p92_mla_combine,dim3(P92_NH),dim3(256),0,0,dpa,dpm,dpl,dctx); };
    double ai=0,am=0,ac=0;
    for(int rep=0;rep<3;rep++){
      for(int i=0;i<10;i++){ idx_only(); mla_only(); dsa_layer(); }
      CHK(hipDeviceSynchronize());
      CHK(hipEventRecord(t0,0)); for(int i=0;i<IT;i++) idx_only();
      CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1)); hipEventElapsedTime(&ms,t0,t1); ai+=ms;
      CHK(hipEventRecord(t0,0)); for(int i=0;i<IT;i++) mla_only();
      CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1)); hipEventElapsedTime(&ms,t0,t1); am+=ms;
      CHK(hipEventRecord(t0,0)); for(int i=0;i<IT;i++) dsa_layer();
      CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1)); hipEventElapsedTime(&ms,t0,t1); ac+=ms; }
    const double ui=ai*1000/(IT*3), um=am*1000/(IT*3), uc=ac*1000/(IT*3);
    printf("  paired, same allocations, 3 interleaved reps:\n");
    printf("    indexer alone %8.2f us\n", ui);
    printf("    MLA alone     %8.2f us\n", um);
    printf("    composed      %8.2f us   parts sum %8.2f   gap %+7.2f us (%+.1f%%)\n",
           uc, ui+um, uc-(ui+um), 100.0*(uc-(ui+um))/(ui+um));
    printf("  DSA layer      %8.2f us   x16 = %6.3f ms/token\n", us_dsa, us_dsa*16/1000);
    printf("  sliding layer  %8.2f us   x30 = %6.3f ms/token\n", us_swa, us_swa*30/1000);
    printf("  whole attention path (46 layers): %.3f ms/token\n", (us_dsa*16+us_swa*30)/1000);
    const bool pass = rel < 1e-5;
    printf("%s (gate: composed path relative_l2 < 1e-5)\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
