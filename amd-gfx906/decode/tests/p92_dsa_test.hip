// Gate the DSA indexer: scorer against a CPU reference, selected SET against a
// CPU top-2048, determinism across repeated runs, rope pairing, and the scan
// cost as a function of context - which is the whole reason this component is
// a risk.
#include "../src/p92_dsa_index.hip"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <set>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
static p92_bf f2b(float f){ uint32_t u; memcpy(&u,&f,4); return (p92_bf)((u+0x8000u)>>16); }
static float  b2f(p92_bf b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
static uint32_t flipc(float f){ uint32_t u; memcpy(&u,&f,4); return (u&0x80000000u)?~u:(u|0x80000000u); }

int main(int argc,char**argv){
    const int N = argc>1?atoi(argv[1]):8192;
    const int cap = N;
    std::mt19937 rng(20260908); std::normal_distribution<float> nd(0,1);
    std::vector<float> q((size_t)P92_IDX_HEADS*P92_IDX_DIM), wh(P92_IDX_HEADS);
    for(auto&v:q) v=nd(rng)*0.3f;
    for(auto&v:wh) v=fabsf(nd(rng))*0.5f;
    std::vector<p92_bf> cache((size_t)P92_IDX_DIM*cap);
    std::vector<std::vector<float>> K(N, std::vector<float>(P92_IDX_DIM));
    for(int p=0;p<N;p++) for(int d=0;d<P92_IDX_DIM;d++){
        const float v=nd(rng)*0.4f; K[p][d]=b2f(f2b(v)); cache[(size_t)d*cap+p]=f2b(v); }
    const float scale = 1.0f/sqrtf((float)P92_IDX_DIM);

    // ---- CPU reference ---------------------------------------------------
    std::vector<float> ref(N);
    for(int p=0;p<N;p++){ double s=0;
        for(int h=0;h<P92_IDX_HEADS;h++){ double a=0;
            for(int d=0;d<P92_IDX_DIM;d++) a+=(double)q[h*P92_IDX_DIM+d]*K[p][d];
            s += (double)wh[h]*std::max(a*scale,0.0); }
        ref[p]=(float)s; }

    float *dq,*dw,*dsc; p92_bf *dc; int *dsel; uint32_t *dh,*dbin,*dab;
    CHK(hipMalloc(&dq,q.size()*4)); CHK(hipMalloc(&dw,wh.size()*4));
    CHK(hipMalloc(&dc,cache.size()*2)); CHK(hipMalloc(&dsc,(size_t)N*4));
    CHK(hipMalloc(&dsel,P92_IDX_TOPK*4)); CHK(hipMalloc(&dh,P92_RBINS*4));
    CHK(hipMalloc(&dbin,4)); CHK(hipMalloc(&dab,4));
    uint32_t *dstate,*dcnt,*doff; CHK(hipMalloc(&dstate,4*4));
    CHK(hipMalloc(&dcnt,2*P92_GMAXC*4)); CHK(hipMalloc(&doff,(2*P92_GMAXC+2)*4));
    CHK(hipMemcpy(dq,q.data(),q.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dw,wh.data(),wh.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dc,cache.data(),cache.size()*2,hipMemcpyHostToDevice));

    const int SB=(N+255)/256;
    hipLaunchKernelGGL(k_p92_idx_scores,dim3(SB),dim3(256),0,0,dq,dc,dw,cap,N,scale,dsc);
    CHK(hipDeviceSynchronize());
    std::vector<float> gsc(N); CHK(hipMemcpy(gsc.data(),dsc,(size_t)N*4,hipMemcpyDeviceToHost));
    double es=0,rs=0,mx=0;
    for(int p=0;p<N;p++){ double d=gsc[p]-ref[p]; es+=d*d; rs+=(double)ref[p]*ref[p]; mx=fmax(mx,fabs(d)); }
    printf("scorer: relative_l2 %.3e  max_abs %.3e  (n=%d, 24 heads, 128 dims)\n",sqrt(es/rs),mx,N);

    // ---- selection -------------------------------------------------------
    const int WANT = std::min(N,P92_IDX_TOPK);
    auto select=[&](std::vector<int>&out)->void{
        (void)hipMemset(dstate,0,4*4);
        for (int sh : {21,10,0}) {
            (void)hipMemset(dh,0,P92_RBINS*4);
            hipLaunchKernelGGL(k_p92_idx_hist,dim3(64),dim3(256),0,0,dsc,N,sh,dstate,dh);
            hipLaunchKernelGGL(k_p92_idx_pick,dim3(1),dim3(1024),0,0,dh,WANT,sh,dstate);
        }
        const int NC=(N+P92_GCHUNK-1)/P92_GCHUNK;
        hipLaunchKernelGGL(k_p92_idx_count,dim3(NC),dim3(256),0,0,dsc,N,dstate,dcnt);
        hipLaunchKernelGGL(k_p92_idx_scan,dim3(1),dim3(256),0,0,dcnt,NC,P92_IDX_TOPK,doff);
        hipLaunchKernelGGL(k_p92_idx_write,dim3(NC),dim3(256),0,0,dsc,N,dstate,doff,P92_IDX_TOPK,dsel);
        hipLaunchKernelGGL(k_p92_idx_pad,dim3(8),dim3(256),0,0,doff,P92_IDX_TOPK,dsel);
        (void)hipDeviceSynchronize();
        out.resize(P92_IDX_TOPK);
        (void)hipMemcpy(out.data(),dsel,P92_IDX_TOPK*4,hipMemcpyDeviceToHost); };
    auto select1=[&](std::vector<int>&out)->void{
        hipLaunchKernelGGL(k_p92_idx_select,dim3(1),dim3(P92_SEL_THREADS),0,0,dsc,N,WANT,dsel);
        (void)hipDeviceSynchronize();
        out.resize(P92_IDX_TOPK);
        (void)hipMemcpy(out.data(),dsel,P92_IDX_TOPK*4,hipMemcpyDeviceToHost); };
    std::vector<int> s1,s2; select(s1); select(s2);
    std::vector<int> sf1,sf2; select1(sf1); select1(sf2);
    printf("fused vs staged: %s   fused determinism: %s\n",
           (sf1==s1)?"identical":"DIFFER", (sf1==sf2)?"byte identical":"DIFFER");
    const bool deterministic = (s1==s2);

    std::vector<int> idx(N); for(int i=0;i<N;i++) idx[i]=i;
    std::stable_sort(idx.begin(),idx.end(),[&](int a,int b){
        const uint32_t fa=flipc(gsc[a]), fb=flipc(gsc[b]);
        if(fa!=fb) return fa>fb; return a<b; });
    const int want=WANT;
    std::set<int> cpu(idx.begin(),idx.begin()+want), gpu;
    for(int v:s1) if(v>=0) gpu.insert(v);
    std::vector<int> diff; std::set_symmetric_difference(cpu.begin(),cpu.end(),
        gpu.begin(),gpu.end(),std::back_inserter(diff));
    printf("selection: %zu selected, symmetric difference vs CPU top-%d = %zu\n",gpu.size(),want,diff.size());
    printf("determinism: two runs %s\n", deterministic?"byte identical":"DIFFER");

    // ---- rope pairing ----------------------------------------------------
    std::vector<float> v(P92_IDX_DIM); for(int i=0;i<P92_IDX_DIM;i++) v[i]=(float)i;
    std::vector<float> cs(P92_IDX_ROPE/2,0.f), sn(P92_IDX_ROPE/2,1.f);   // 90 degrees
    float *dv,*dcs,*dsn;
    CHK(hipMalloc(&dv,P92_IDX_DIM*4)); CHK(hipMalloc(&dcs,P92_IDX_ROPE/2*4)); CHK(hipMalloc(&dsn,P92_IDX_ROPE/2*4));
    CHK(hipMemcpy(dv,v.data(),P92_IDX_DIM*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dcs,cs.data(),P92_IDX_ROPE/2*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsn,sn.data(),P92_IDX_ROPE/2*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_p92_idx_rope,dim3(1),dim3(64),0,0,dv,dcs,dsn);
    CHK(hipDeviceSynchronize());
    std::vector<float> gv(P92_IDX_DIM); CHK(hipMemcpy(gv.data(),dv,P92_IDX_DIM*4,hipMemcpyDeviceToHost));
    // with cos=0,sin=1: v[i] = -v[i+32], v[i+32] = v[i]; dims 64..127 untouched
    bool rope_ok = fabsf(gv[0]+32.f)<1e-5f && fabsf(gv[32]-0.f)<1e-5f
                && fabsf(gv[64]-64.f)<1e-5f && fabsf(gv[127]-127.f)<1e-5f;
    printf("rope: pairs (i,i+32) over the first 64 dims, 64..127 untouched -> %s\n", rope_ok?"correct":"WRONG");

    // ---- scan cost vs context -------------------------------------------
    hipEvent_t t0,t1; CHK(hipEventCreate(&t0)); CHK(hipEventCreate(&t1));
    for(int i=0;i<20;i++) hipLaunchKernelGGL(k_p92_idx_scores,dim3(SB),dim3(256),0,0,dq,dc,dw,cap,N,scale,dsc);
    CHK(hipDeviceSynchronize()); CHK(hipEventRecord(t0,0));
    const int IT=200;
    for(int i=0;i<IT;i++) hipLaunchKernelGGL(k_p92_idx_scores,dim3(SB),dim3(256),0,0,dq,dc,dw,cap,N,scale,dsc);
    CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1));
    float ms; hipEventElapsedTime(&ms,t0,t1);
    const double us=ms*1000.0/IT;
    printf("scorer cost: %.2f us at n=%d  ->  16 DSA layers = %.3f ms/token\n",us,N,us*16/1000);

    // time the kernels alone: no per-iteration sync, no device-to-host copy
    CHK(hipDeviceSynchronize()); CHK(hipEventRecord(t0,0));
    for(int i=0;i<IT;i++)
        hipLaunchKernelGGL(k_p92_idx_select,dim3(1),dim3(P92_SEL_THREADS),0,0,dsc,N,WANT,dsel);
    CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1));
    hipEventElapsedTime(&ms,t0,t1);
    const double us_f=ms*1000.0/IT;
    printf("fused selection : %.2f us at n=%d  ->  16 DSA layers = %.3f ms/token\n",
           us_f,N,us_f*16/1000);
    const int NC2=(N+P92_GCHUNK-1)/P92_GCHUNK;
    CHK(hipEventRecord(t0,0));
    for(int i=0;i<IT;i++){
        (void)hipMemsetAsync(dstate,0,4*4,0);
        for (int sh : {21,10,0}) {
            (void)hipMemsetAsync(dh,0,P92_RBINS*4,0);
            hipLaunchKernelGGL(k_p92_idx_hist,dim3(64),dim3(256),0,0,dsc,N,sh,dstate,dh);
            hipLaunchKernelGGL(k_p92_idx_pick,dim3(1),dim3(1024),0,0,dh,WANT,sh,dstate); }
        hipLaunchKernelGGL(k_p92_idx_count,dim3(NC2),dim3(256),0,0,dsc,N,dstate,dcnt);
        hipLaunchKernelGGL(k_p92_idx_scan,dim3(1),dim3(256),0,0,dcnt,NC2,P92_IDX_TOPK,doff);
        hipLaunchKernelGGL(k_p92_idx_write,dim3(NC2),dim3(256),0,0,dsc,N,dstate,doff,P92_IDX_TOPK,dsel);
        hipLaunchKernelGGL(k_p92_idx_pad,dim3(8),dim3(256),0,0,doff,P92_IDX_TOPK,dsel); }
    CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1));
    hipEventElapsedTime(&ms,t0,t1);
    const double us_s=ms*1000.0/IT;
    printf("staged selection: %.2f us at n=%d  ->  16 DSA layers = %.3f ms/token\n",
           us_s,N,us_s*16/1000);
    printf("indexer total (scorer + best select): %.2f us -> %.3f ms/token over 16 layers\n",
           us + fmin(us_f,us_s), (us + fmin(us_f,us_s))*16/1000);
    const bool pass = sqrt(es/rs)<1e-5 && diff.empty() && deterministic && rope_ok
                      && sf1==s1 && sf1==sf2;
    printf("%s\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
