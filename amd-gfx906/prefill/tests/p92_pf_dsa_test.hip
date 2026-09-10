// Minimal gate for the prefill DSA selector.
//
// Gates the two semantically load-bearing things and nothing else: that the
// scores match an independent host reference, and that the selected set is
// genuinely the top-2048 of those scores. Also measures the ONE number that
// decides how sparse attention must be built - how much the selections of TQ
// adjacent queries overlap - because a per-query index list destroys the key
// sharing the dense kernel gets from TQ=4, and the union is the way back.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "../src/p92_pf_dsa.hip"

#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
static float b2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; __builtin_memcpy(&f,&u,4); return f; }
static uint16_t f2b(float f){ uint32_t u; __builtin_memcpy(&u,&f,4); u+=0x7FFFu+((u>>16)&1u); return (uint16_t)(u>>16); }

int main(){
    const int T = 64, BASE = 4000, CAP = 4096, WANT = PF_IDX_TOPK;
    const int HEADS = PF_IDX_HEADS, DIM = PF_IDX_DIM;
    const float scale = 1.0f / sqrtf((float)DIM);
    std::mt19937 r(20260909);
    std::uniform_real_distribution<float> d(-1.f, 1.f);

    std::vector<float> q((size_t)T*HEADS*DIM), wh((size_t)T*HEADS);
    std::vector<uint16_t> cache((size_t)DIM*CAP);
    for(auto&v:q)  v = d(r) * 0.3f;
    for(auto&v:wh) v = fabsf(d(r));                  // head weights are non-negative
    for(auto&v:cache) v = f2b(d(r) * 0.5f);
    // RANGE CHECK the generated data. An unsigned-underflow in a generator once
    // produced values up to 4.6e15 here and presented as a kernel bug.
    { float qm=0, cm=0, wm=0;
      for(float v:q) qm=fmaxf(qm,fabsf(v));
      for(uint16_t v:cache) cm=fmaxf(cm,fabsf(b2f(v)));
      for(float v:wh) wm=fmaxf(wm,fabsf(v));
      printf("  data range: |q|max %.4g  |cache|max %.4g  |wh|max %.4g\n", qm, cm, wm);
      if(!(qm<10.f && cm<10.f && wm<10.f)){ printf("  GENERATOR OUT OF RANGE\n"); return 1; } }

    float *dq,*dwh,*dsc; uint16_t *dc; int *dsel,*dn;
    CHK(hipMalloc(&dq,q.size()*4));   CHK(hipMalloc(&dwh,wh.size()*4));
    CHK(hipMalloc(&dc,cache.size()*2));
    CHK(hipMalloc(&dsc,(size_t)T*CAP*4));
    CHK(hipMalloc(&dsel,(size_t)T*WANT*4)); CHK(hipMalloc(&dn,(size_t)T*4));
    CHK(hipMemcpy(dq,q.data(),q.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dwh,wh.data(),wh.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dc,cache.data(),cache.size()*2,hipMemcpyHostToDevice));
    // poison: a kernel that writes only some rows must be detectable
    { std::vector<float> pz((size_t)T*CAP, -98765.f);
      CHK(hipMemcpy(dsc,pz.data(),pz.size()*4,hipMemcpyHostToDevice)); }

    const int maxn = BASE + T;
    int pass = 1;
    hipLaunchKernelGGL(k_pf_idx_scores,dim3((maxn+255)/256,T),dim3(256),0,0,
                       dq,dc,dwh,CAP,T,BASE,scale,dsc,CAP);
    CHK(hipDeviceSynchronize());
    hipLaunchKernelGGL(k_pf_idx_select,dim3(T),dim3(PF_SEL_THREADS),0,0,
                       dsc,T,BASE,WANT,CAP,dsel,dn);
    CHK(hipDeviceSynchronize());

    std::vector<float> sc((size_t)T*CAP);
    std::vector<int>   sel((size_t)T*WANT), nsel(T);
    CHK(hipMemcpy(sc.data(),dsc,sc.size()*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(sel.data(),dsel,sel.size()*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(nsel.data(),dn,T*4,hipMemcpyDeviceToHost));

    // v3 must compute the SAME scores from the dim-major query. Different
    // memory layout and a different issue path; identical arithmetic.
    { std::vector<float> qT(q.size());
      for(int t=0;t<T;t++) for(int h=0;h<HEADS;h++) for(int i=0;i<DIM;i++)
          qT[((size_t)t*DIM + i)*HEADS + h] = q[((size_t)t*HEADS+h)*DIM+i];
      float *dqT; CHK(hipMalloc(&dqT,qT.size()*4));
      CHK(hipMemcpy(dqT,qT.data(),qT.size()*4,hipMemcpyHostToDevice));
      float *dsc3; CHK(hipMalloc(&dsc3,(size_t)T*CAP*4));
      std::vector<float> pz3((size_t)T*CAP,-98765.f);
      CHK(hipMemcpy(dsc3,pz3.data(),pz3.size()*4,hipMemcpyHostToDevice));
      hipLaunchKernelGGL(k_pf_idx_scores3,dim3((maxn+255)/256,T),dim3(256),0,0,
                         dqT,dc,dwh,CAP,T,BASE,scale,dsc3,CAP);
      CHK(hipDeviceSynchronize());
      std::vector<float> s3((size_t)T*CAP);
      CHK(hipMemcpy(s3.data(),dsc3,s3.size()*4,hipMemcpyDeviceToHost));
      size_t nf3=0; double w3=0;
      for(int t=0;t<T;t++){ const int n=BASE+t+1;
        for(int p2=0;p2<n;p2++){ const float a=sc[(size_t)t*CAP+p2], b3=s3[(size_t)t*CAP+p2];
          if(!std::isfinite(b3)){ nf3++; continue; }
          const double den=fabs(a)>1e-6?fabs(a):1e-6; w3=std::max(w3,fabs((double)a-b3)/den); } }
      printf("  scorer v3 vs v1: worst relative %.3e over all scored positions%s\n",
             w3, (nf3||w3>=5e-3)?"   <-- FAIL":"");
      if(nf3){ printf("  v3 NON-FINITE: %zu\n",nf3); pass=0; }
      if(w3>=5e-3) pass=0;
      CHK(hipFree(dqT)); CHK(hipFree(dsc3)); }


    // NON-FINITE FIRST, before any magnitude or error reduction. fmax(m,fabs(NaN))
    // leaves m unchanged, so all-NaN output scores as a perfect match.
    { size_t nf=0; for(int t=0;t<T;t++){ const int n=BASE+t+1;
        for(int p=0;p<n;p++) if(!std::isfinite(sc[(size_t)t*CAP+p])) nf++; }
      if(nf){ printf("  NON-FINITE SCORES: %zu\n", nf); return 1; } }

    // GATE A: scores against an independent host reference, on 8 queries
    double worst = 0; int checked = 0;
    for(int t=0;t<T;t+=8){
        const int n = BASE+t+1;
        for(int p=0;p<n;p+=397){                       // prime stride
            double s=0;
            for(int h=0;h<HEADS;h++){
                double acc=0;
                for(int i=0;i<DIM;i++)
                    acc += (double)q[((size_t)t*HEADS+h)*DIM+i] * (double)b2f(cache[(size_t)i*CAP+p]);
                s += (double)wh[(size_t)t*HEADS+h] * std::max(acc*(double)scale, 0.0);
            }
            const double got = sc[(size_t)t*CAP+p];
            const double den = fabs(s)>1e-6?fabs(s):1e-6;
            worst = std::max(worst, fabs(s-got)/den); checked++;
        }
    }
    printf("  scores vs host reference: %d points, worst relative %.3e\n", checked, worst);
    if(!(worst < 5e-3)) pass = 0;

    // GATE B: the selected set really is the top-WANT of those scores
    int bad_sets = 0;
    for(int t=0;t<T;t+=8){
        const int n = BASE+t+1;
        std::vector<int> idx(n); for(int i=0;i<n;i++) idx[i]=i;
        std::sort(idx.begin(),idx.end(),[&](int a,int b){
            const float fa=sc[(size_t)t*CAP+a], fb=sc[(size_t)t*CAP+b];
            return fa!=fb ? fa>fb : a<b; });
        std::vector<int> want(idx.begin(), idx.begin()+std::min(n,WANT));
        std::vector<int> got(sel.begin()+(size_t)t*WANT, sel.begin()+(size_t)t*WANT+nsel[t]);
        std::sort(want.begin(),want.end()); std::sort(got.begin(),got.end());
        if(want!=got) bad_sets++;
    }
    printf("  top-%d selection: %d of 8 queries differ from the host top-k%s\n",
           WANT, bad_sets, bad_sets?"   <-- FAIL":"");
    if(bad_sets) pass = 0;

    // NEGATIVE CONTROL: perturb one score and the selection must respond
    { std::vector<float> one(1, 1e9f);
      CHK(hipMemcpy(dsc + (size_t)0*CAP + 7, one.data(), 4, hipMemcpyHostToDevice));
      hipLaunchKernelGGL(k_pf_idx_select,dim3(T),dim3(PF_SEL_THREADS),0,0,
                         dsc,T,BASE,WANT,CAP,dsel,dn);
      CHK(hipDeviceSynchronize());
      std::vector<int> s2((size_t)T*WANT);
      CHK(hipMemcpy(s2.data(),dsel,s2.size()*4,hipMemcpyDeviceToHost));
      std::vector<int> a(sel.begin(),sel.begin()+WANT), b(s2.begin(),s2.begin()+WANT);
      std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end());
      const bool moved = (a!=b) || std::find(b.begin(),b.end(),7)!=b.end();
      printf("  negative control (one score raised to 1e9): selection %s\n",
             moved?"changed":"UNCHANGED   <-- FAIL");
      if(!moved) pass = 0; }

    // THE DESIGN MEASUREMENT: how much do adjacent queries' selections overlap?
    // The dense kernel gets a 4x key-traffic discount from TQ=4 sharing one
    // sweep. A per-query index list destroys that unless a tile can sweep the
    // UNION of its queries' selections once, each query masking to its own
    // subset. Union near 2048 means sharing survives; near 4*2048 means it dies.
    for(int TQ : {2,4,8}){
        double sum_union = 0; int tiles = 0;
        for(int t0=0;t0+TQ<=T;t0+=TQ){
            std::vector<int> u;
            for(int t=t0;t<t0+TQ;t++)
                u.insert(u.end(), sel.begin()+(size_t)t*WANT, sel.begin()+(size_t)t*WANT+nsel[t]);
            std::sort(u.begin(),u.end());
            u.erase(std::unique(u.begin(),u.end()),u.end());
            sum_union += u.size(); tiles++;
        }
        const double mean = sum_union/tiles;
        printf("  TQ=%d tile: mean union %.0f positions = %.2fx of %d (worst case %dx)\n",
               TQ, mean, mean/WANT, WANT, TQ);
    }
    printf("%s\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
