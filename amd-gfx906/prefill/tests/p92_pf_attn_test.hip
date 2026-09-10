// Gate for chunked prefill attention. Semantic and layout check before
// composition, NOT a benchmark. TQ 2/4/8 must agree with each other and with an
// independent host reference; the tile is a throughput choice decided later by
// composed prefill tok/s, never here.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
// HIP non-RDC: TU-include the kernel so the launch lands in the same device image.
#include "../src/p92_pf_attn.hip"
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
#define G_NH 48
#define G_KV 512
#define G_PE 64
#define G_SK 128
typedef uint16_t pf_bf;
hipError_t pf_attn_launch(int kern, int tq, const float*, const float*, const uint16_t*,
    const uint16_t*, const uint16_t*, const uint16_t*, int, int, float*, hipStream_t);
static float b2f(pf_bf b){ return __builtin_bit_cast(float, (uint32_t)b << 16); }

int main(){
    const int KERN = getenv("PF_ATTN_KERNEL") ? atoi(getenv("PF_ATTN_KERNEL")) : 1;
    printf("  gating attention kernel %d\n", KERN);
    const int T = 32, CB = 0, P = CB + T;
    std::mt19937 r(5);
    std::vector<float> qlat((size_t)T*G_NH*G_KV), qpe((size_t)T*G_NH*G_PE);
    std::vector<pf_bf> lat((size_t)P*G_KV), kpe((size_t)P*G_PE),
                       sl((size_t)G_SK*G_KV), sp((size_t)G_SK*G_PE);
    for(auto&v:qlat) v=(float)((int)(r()%2000)-1000)/4000.f;
    for(auto&v:qpe)  v=(float)((int)(r()%2000)-1000)/4000.f;
    for(auto&v:lat)  v=(pf_bf)(0x3B00u+(r()%512));
    for(auto&v:kpe)  v=(pf_bf)(0x3B00u+(r()%512));
    for(auto&v:sl)   v=(pf_bf)(0x3B00u+(r()%512));
    for(auto&v:sp)   v=(pf_bf)(0x3B00u+(r()%512));

    float *dq,*dqp,*out; pf_bf *dl,*dk,*dsl,*dsp;
    CHK(hipMalloc(&dq,qlat.size()*4));   CHK(hipMalloc(&dqp,qpe.size()*4));
    CHK(hipMalloc(&dl,lat.size()*2));    CHK(hipMalloc(&dk,kpe.size()*2));
    CHK(hipMalloc(&dsl,sl.size()*2));    CHK(hipMalloc(&dsp,sp.size()*2));
    CHK(hipMalloc(&out,(size_t)T*G_NH*G_KV*4));
    CHK(hipMemcpy(dq,qlat.data(),qlat.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dqp,qpe.data(),qpe.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dl,lat.data(),lat.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dk,kpe.data(),kpe.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsl,sl.data(),sl.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsp,sp.data(),sp.size()*2,hipMemcpyHostToDevice));

    // ---- host reference: plain causal softmax, no online rescale ------------
    // Deliberately a DIFFERENT algorithm from the kernel's streaming max/sum.
    // Agreement then means the online formulation is right, not that two copies
    // of it match.
    const float sc = 1.0f/sqrtf(192.0f);
    auto ref=[&](int q,int h,std::vector<double>&o){
        std::vector<double> s; s.reserve(G_SK+P);
        for(int k=0;k<G_SK;k++){ double d=0;
            for(int i=0;i<G_KV;i++) d+=(double)qlat[((size_t)q*G_NH+h)*G_KV+i]*b2f(sl[(size_t)k*G_KV+i]);
            for(int i=0;i<G_PE;i++) d+=(double)qpe[((size_t)q*G_NH+h)*G_PE+i]*b2f(sp[(size_t)k*G_PE+i]);
            s.push_back(d*sc); }
        for(int k=0;k<=CB+q;k++){ double d=0;
            for(int i=0;i<G_KV;i++) d+=(double)qlat[((size_t)q*G_NH+h)*G_KV+i]*b2f(lat[(size_t)k*G_KV+i]);
            for(int i=0;i<G_PE;i++) d+=(double)qpe[((size_t)q*G_NH+h)*G_PE+i]*b2f(kpe[(size_t)k*G_PE+i]);
            s.push_back(d*sc); }
        double m=-1e300; for(double v:s) m=fmax(m,v);
        double z=0; for(double&v:s){ v=exp(v-m); z+=v; }
        o.assign(G_KV,0.0);
        for(size_t j=0;j<s.size();j++){
            const pf_bf*kv = j<(size_t)G_SK ? &sl[(size_t)j*G_KV] : &lat[(size_t)(j-G_SK)*G_KV];
            for(int i=0;i<G_KV;i++) o[i]+= s[j]*(double)b2f(kv[i]); }
        for(int i=0;i<G_KV;i++) o[i]/=z; };

    int pass=1;
    std::vector<float> got((size_t)T*G_NH*G_KV);
    std::vector<std::vector<float>> keep;
    for(int tq : {2,4,8}){
        CHK(hipMemset(out,0,(size_t)T*G_NH*G_KV*4));
        CHK(pf_attn_launch(KERN,tq,dq,dqp,dl,dk,dsl,dsp,CB,T,0,out,0));
        CHK(hipDeviceSynchronize());
        CHK(hipMemcpy(got.data(),out,got.size()*4,hipMemcpyDeviceToHost));
        { size_t nf=0; for(float v:got) if(!std::isfinite(v)) nf++;
          if(nf){ printf("  TQ=%d NON-FINITE OUTPUT: %zu of %zu\n",tq,nf,got.size());
                  pass=0; } }
        keep.push_back(got);
        double worst=0; int n=0;
        for(int q=0;q<T;q+=7) for(int h=0;h<G_NH;h+=17){
            std::vector<double> o; ref(q,h,o);
            for(int i=0;i<G_KV;i+=97){
                const double w=o[i], g=got[((size_t)q*G_NH+h)*G_KV+i];
                const double den=fabs(w)>1e-4?fabs(w):1e-4;
                worst=fmax(worst,fabs(w-g)/den); n++; } }
        // magnitude first: a gate comparing zero against zero reports a perfect
        // score. Report what was actually produced before reporting agreement.
        double mg=0, mr=0; { std::vector<double> o0; ref(0,0,o0);
            for(int i=0;i<G_KV;i++){ mg=fmax(mg,fabs((double)got[(size_t)0*G_NH*G_KV+i]));
                                     mr=fmax(mr,fabs(o0[i])); } }
        printf("  TQ=%d vs host reference: %4d points, worst relative %.3e"
               "   |kernel|max %.3e |ref|max %.3e%s\n",
               tq,n,worst,mg,mr,
               (mg<1e-12||mr<1e-12) ? "   <-- OUTPUT IS ZERO, GATE IS VACUOUS"
                                    : (worst<5e-3?"":"  <-- FAIL"));
        if (mg < 1e-12 || mr < 1e-12) pass = 0;
        if(!(worst<5e-3)) pass=0;
    }
    // all tiles must agree with each other: a tile-dependent result is a
    // masking or boundary bug, and the causal edge is where TQ actually differs
    size_t d24=0,d48=0;
    for(size_t i=0;i<got.size();i++){
        if(fabs(keep[0][i]-keep[1][i])>1e-4*fmax(1.f,fabs(keep[0][i]))) d24++;
        if(fabs(keep[1][i]-keep[2][i])>1e-4*fmax(1.f,fabs(keep[1][i]))) d48++; }
    printf("  TQ2 vs TQ4 differ %zu, TQ4 vs TQ8 differ %zu (of %zu)\n",d24,d48,got.size());
    if(d24||d48) pass=0;

    // negative control: break causality by one position. Must change the result,
    // or the comparison above is not sensitive to the masking it exists to test.
    CHK(hipMemset(out,0,(size_t)T*G_NH*G_KV*4));
    CHK(pf_attn_launch(KERN,4,dq,dqp,dl,dk,dsl,dsp,CB+1,T,0,out,0));
    CHK(hipDeviceSynchronize());
    std::vector<float> shifted(got.size());
    CHK(hipMemcpy(shifted.data(),out,shifted.size()*4,hipMemcpyDeviceToHost));
    size_t moved=0;
    for(size_t i=0;i<got.size();i++) if(shifted[i]!=keep[1][i]) moved++;
    printf("  negative control (causal edge shifted): %zu changed%s\n",
           moved, moved?"":"   <-- DETECTOR BLIND");
    if(!moved) pass=0;
    printf("%s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
