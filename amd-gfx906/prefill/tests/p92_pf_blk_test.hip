// Gate for k_pf_attn_blk: window mode (full causal prefix and sliding window) and
// selection mode, each against an independent host reference (plain softmax, no
// online rescale) AND against the production kernels (k_pf_attn2<2>, k_pf_attn_spg2),
// plus a negative control (causal edge shifted). Not a benchmark.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include "../src/p92_pf_attn.hip"
#include "../src/p92_pf_attn_blk.hip"
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
static float b2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; __builtin_memcpy(&f,&u,4); return f; }

int main(){
    const int T=40, NH=48, KV=512, PE=64, SK=128, CB=37, P=CB+T, WANT=48;
    std::mt19937 r(11);
    auto rf=[&](float s){ return (float)((int)(r()%2000)-1000)/1000.f*s; };
    std::vector<float> qlat((size_t)T*NH*KV), qpe((size_t)T*NH*PE);
    std::vector<uint16_t> lat((size_t)P*KV), kpe((size_t)P*PE), sl((size_t)SK*KV), sp((size_t)SK*PE);
    for(auto&v:qlat) v=rf(0.25f);
    for(auto&v:qpe)  v=rf(0.25f);
    for(auto&v:lat)  v=(uint16_t)(0x3B00u+(r()%512));
    for(auto&v:kpe)  v=(uint16_t)(0x3B00u+(r()%512));
    for(auto&v:sl)   v=(uint16_t)(0x3B00u+(r()%512));
    for(auto&v:sp)   v=(uint16_t)(0x3B00u+(r()%512));
    { float m=0; for(float v:qlat) m=fmaxf(m,fabsf(v)); printf("  data range: |qlat|max %.4g\n",m); if(!(m<10.f)){ printf("  GENERATOR OUT OF RANGE\n"); return 1; } }
    // selection lists: variable count per query, positions <= the query's own
    std::vector<int32_t> sel((size_t)T*WANT,-1), ns(T,0);
    for(int q=0;q<T;q++){ const int n = 3 + (q*7)%(WANT-3); int c=0;
        for(int u=0;u<n;u++){ const int p=(u*11+q*3)%(CB+q+1); sel[(size_t)q*WANT+c++]=p; }
        ns[q]=c; }

    float *dq,*dqp,*out,*out2; uint16_t *dl,*dk,*dsl,*dsp; int32_t *dsel,*dns;
    CHK(hipMalloc(&dq,qlat.size()*4)); CHK(hipMalloc(&dqp,qpe.size()*4));
    CHK(hipMalloc(&out,(size_t)T*NH*KV*4)); CHK(hipMalloc(&out2,(size_t)T*NH*KV*4));
    CHK(hipMalloc(&dl,lat.size()*2)); CHK(hipMalloc(&dk,kpe.size()*2));
    CHK(hipMalloc(&dsl,sl.size()*2)); CHK(hipMalloc(&dsp,sp.size()*2));
    CHK(hipMalloc(&dsel,sel.size()*4)); CHK(hipMalloc(&dns,ns.size()*4));
    CHK(hipMemcpy(dq,qlat.data(),qlat.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dqp,qpe.data(),qpe.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dl,lat.data(),lat.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dk,kpe.data(),kpe.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsl,sl.data(),sl.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsp,sp.data(),sp.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsel,sel.data(),sel.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dns,ns.data(),ns.size()*4,hipMemcpyHostToDevice));

    const double sc = 1.0/sqrt(192.0);
    // reference over an explicit key list (sinks always first)
    auto ref=[&](int q,int h,const std::vector<int>&keys,std::vector<double>&o){
        std::vector<double> s; std::vector<const uint16_t*> kv;
        for(int k=0;k<SK;k++){ double d=0;
            for(int i=0;i<KV;i++) d+=(double)qlat[((size_t)q*NH+h)*KV+i]*b2f(sl[(size_t)k*KV+i]);
            for(int i=0;i<PE;i++) d+=(double)qpe[((size_t)q*NH+h)*PE+i]*b2f(sp[(size_t)k*PE+i]);
            s.push_back(d*sc); kv.push_back(&sl[(size_t)k*KV]); }
        for(int p:keys){ double d=0;
            for(int i=0;i<KV;i++) d+=(double)qlat[((size_t)q*NH+h)*KV+i]*b2f(lat[(size_t)p*KV+i]);
            for(int i=0;i<PE;i++) d+=(double)qpe[((size_t)q*NH+h)*PE+i]*b2f(kpe[(size_t)p*PE+i]);
            s.push_back(d*sc); kv.push_back(&lat[(size_t)p*KV]); }
        double m=-1e300; for(double v:s) m=fmax(m,v);
        double z=0; for(double&v:s){ v=exp(v-m); z+=v; }
        o.assign(KV,0.0);
        for(size_t j=0;j<s.size();j++) for(int i=0;i<KV;i++) o[i]+=s[j]*(double)b2f(kv[j][i]);
        for(int i=0;i<KV;i++) o[i]/=z; };
    auto compare=[&](const char*name,const std::vector<float>&got,auto keys_of)->int{
        size_t nf=0; for(float v:got) if(!std::isfinite(v)) nf++;
        double worst=0,mg=0,mr=0; int n=0;
        for(int q=0;q<T;q+=3) for(int h=0;h<NH;h+=7){
            std::vector<double> o; ref(q,h,keys_of(q),o);
            for(int i=0;i<KV;i+=13){ const double w=o[i], g=got[((size_t)q*NH+h)*KV+i];
                const double den=fabs(w)>1e-4?fabs(w):1e-4; worst=fmax(worst,fabs(w-g)/den);
                mg=fmax(mg,fabs(g)); mr=fmax(mr,fabs(w)); n++; } }
        const int ok = nf==0 && worst<5e-3 && mg>1e-12 && mr>1e-12;
        printf("  %-34s %5d points, worst relative %.3e  |k|max %.3e |ref|max %.3e nonfinite %zu%s\n",
               name,n,worst,mg,mr,nf,ok?"":"   <-- FAIL");
        return ok; };
    auto agree=[&](const char*name,const std::vector<float>&a,const std::vector<float>&b)->int{
        size_t d=0; double w=0;
        for(size_t i=0;i<a.size();i++){ const double den=fabs(b[i])>1e-4?fabs(b[i]):1e-4; const double rel=fabs(a[i]-b[i])/den; w=fmax(w,rel); if(rel>1e-3) d++; }
        printf("  %-34s %zu of %zu outputs differ > 1e-3, worst relative %.3e%s\n",name,d,a.size(),w,d?"   <-- FAIL":"");
        return d==0; };
    int pass=1;
    std::vector<float> got((size_t)T*NH*KV), got2((size_t)T*NH*KV);
    auto poison=[&](float*p){ std::vector<float> pz((size_t)T*NH*KV,-12345.f); return hipMemcpy(p,pz.data(),pz.size()*4,hipMemcpyHostToDevice); };

    // ---- window mode, full causal prefix (win = 0) ----
    CHK(poison(out));
    hipLaunchKernelGGL(k_pf_attn_blk,dim3(T,NH/PFB_HB),dim3(PFB_NT),0,0,dq,dqp,dl,dk,dsl,dsp,(const int32_t*)nullptr,(const int32_t*)nullptr,0,CB,T,0,out);
    CHK(hipDeviceSynchronize()); CHK(hipMemcpy(got.data(),out,got.size()*4,hipMemcpyDeviceToHost));
    pass &= compare("blk full prefix vs host",got,[&](int q){ std::vector<int> k; for(int p=0;p<=CB+q;p++) k.push_back(p); return k; });
    CHK(poison(out2));
    hipLaunchKernelGGL((k_pf_attn2<2>),dim3((T+1)/2,NH),dim3(64),0,0,dq,dqp,dl,dk,dsl,dsp,CB,T,0,out2);
    CHK(hipDeviceSynchronize()); CHK(hipMemcpy(got2.data(),out2,got2.size()*4,hipMemcpyDeviceToHost));
    pass &= agree("blk vs k_pf_attn2<2> (full prefix)",got,got2);

    // ---- window mode, sliding window 16 ----
    CHK(poison(out));
    hipLaunchKernelGGL(k_pf_attn_blk,dim3(T,NH/PFB_HB),dim3(PFB_NT),0,0,dq,dqp,dl,dk,dsl,dsp,(const int32_t*)nullptr,(const int32_t*)nullptr,0,CB,T,16,out);
    CHK(hipDeviceSynchronize()); CHK(hipMemcpy(got.data(),out,got.size()*4,hipMemcpyDeviceToHost));
    pass &= compare("blk window 16 vs host",got,[&](int q){ std::vector<int> k; for(int p=std::max(0,CB+q-15);p<=CB+q;p++) k.push_back(p); return k; });
    CHK(poison(out2));
    hipLaunchKernelGGL((k_pf_attn2<2>),dim3((T+1)/2,NH),dim3(64),0,0,dq,dqp,dl,dk,dsl,dsp,CB,T,16,out2);
    CHK(hipDeviceSynchronize()); CHK(hipMemcpy(got2.data(),out2,got2.size()*4,hipMemcpyDeviceToHost));
    pass &= agree("blk vs k_pf_attn2<2> (window 16)",got,got2);

    // ---- selection mode ----
    CHK(poison(out));
    hipLaunchKernelGGL(k_pf_attn_blk,dim3(T,NH/PFB_HB),dim3(PFB_NT),0,0,dq,dqp,dl,dk,dsl,dsp,dsel,dns,WANT,CB,T,0,out);
    CHK(hipDeviceSynchronize()); CHK(hipMemcpy(got.data(),out,got.size()*4,hipMemcpyDeviceToHost));
    pass &= compare("blk selection vs host",got,[&](int q){ std::vector<int> k; for(int u=0;u<ns[q];u++) k.push_back(sel[(size_t)q*WANT+u]); return k; });
    { uint16_t *dlp,*dpp; CHK(hipMalloc(&dlp,(size_t)T*WANT*KV*2)); CHK(hipMalloc(&dpp,(size_t)T*WANT*PE*2));
      hipLaunchKernelGGL(k_pf_sel_pack,dim3(T),dim3(256),0,0,dsel,dns,T,WANT,dl,dk,dlp,dpp);
      CHK(hipDeviceSynchronize()); CHK(poison(out2));
      hipLaunchKernelGGL(k_pf_attn_spg2,dim3(T,NH/2),dim3(64),0,0,dq,dqp,dlp,dpp,dsl,dsp,dns,WANT,T,out2);
      CHK(hipDeviceSynchronize()); CHK(hipMemcpy(got2.data(),out2,got2.size()*4,hipMemcpyDeviceToHost));
      pass &= agree("blk vs k_pf_attn_spg2 (selection)",got,got2);
      CHK(hipFree(dlp)); CHK(hipFree(dpp)); }

    // ---- negative control: shift the causal edge by one position ----
    { std::vector<float> keep=got;
      CHK(poison(out));
      hipLaunchKernelGGL(k_pf_attn_blk,dim3(T,NH/PFB_HB),dim3(PFB_NT),0,0,dq,dqp,dl,dk,dsl,dsp,(const int32_t*)nullptr,(const int32_t*)nullptr,0,CB,T,0,out);
      CHK(hipDeviceSynchronize()); CHK(hipMemcpy(got.data(),out,got.size()*4,hipMemcpyDeviceToHost));
      CHK(poison(out));
      hipLaunchKernelGGL(k_pf_attn_blk,dim3(T,NH/PFB_HB),dim3(PFB_NT),0,0,dq,dqp,dl,dk,dsl,dsp,(const int32_t*)nullptr,(const int32_t*)nullptr,0,CB+1,T,0,out);
      CHK(hipDeviceSynchronize()); CHK(hipMemcpy(got2.data(),out,got2.size()*4,hipMemcpyDeviceToHost));
      size_t moved=0; for(size_t i=0;i<got.size();i++) if(got[i]!=got2[i]) moved++;
      printf("  negative control (causal edge shifted): %zu of %zu changed%s\n",moved,got.size(),moved?"":"   <-- DETECTOR BLIND");
      if(!moved) pass=0; }
    printf("%s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
