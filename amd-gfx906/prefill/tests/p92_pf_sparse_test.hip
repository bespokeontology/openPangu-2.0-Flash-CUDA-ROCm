// Gate for k_pf_attn_sp: does it attend to EXACTLY the masked subset?
// The failure this exists to catch is a mask that is off by a bit or a tile,
// which would still produce plausible finite output and a plausible throughput.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include "../src/p92_pf_attn.hip"

#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
static float b2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; __builtin_memcpy(&f,&u,4); return f; }

int main(){
    const int T=8, NH=48, KV=512, PE=64, SK=128, P=96, TQ=4, UMAX=32;
    const int ntile=(T+TQ-1)/TQ;
    std::mt19937 r(7);
    auto rf=[&](float s){ return (float)((int)(r()%2000)-1000)/1000.f*s; };

    std::vector<float> qlat((size_t)T*NH*KV), qpe((size_t)T*NH*PE);
    std::vector<uint16_t> lat((size_t)P*KV), kpe((size_t)P*PE), sl((size_t)SK*KV), sp((size_t)SK*PE);
    for(auto&v:qlat) v=rf(0.25f);
    for(auto&v:qpe)  v=rf(0.25f);
    for(auto&v:lat)  v=(uint16_t)(0x3B00u+(r()%512));
    for(auto&v:kpe)  v=(uint16_t)(0x3B00u+(r()%512));
    for(auto&v:sl)   v=(uint16_t)(0x3B00u+(r()%512));
    for(auto&v:sp)   v=(uint16_t)(0x3B00u+(r()%512));
    { float m=0; for(float v:qlat) m=fmaxf(m,fabsf(v));
      printf("  data range: |qlat|max %.4g\n", m);
      if(!(m<10.f)){ printf("  GENERATOR OUT OF RANGE\n"); return 1; } }

    // deliberately IRREGULAR masks: different counts per query, gaps, and one
    // query in each tile that sees almost nothing
    std::vector<int32_t> uni((size_t)ntile*UMAX,0), ucnt(ntile);
    std::vector<uint8_t> umask((size_t)ntile*UMAX,0);
    for(int tl=0; tl<ntile; tl++){
        int n=0;
        for(int u=0; u<UMAX; u++){
            const int pos = (u*7 + tl*3) % P;
            uint8_t m=0;
            for(int t=0;t<TQ;t++) if((int)(r()%100) < (t==1?10:60)) m |= (1u<<t);
            if(!m) continue;
            uni[(size_t)tl*UMAX+n]=pos; umask[(size_t)tl*UMAX+n]=m; n++;
        }
        ucnt[tl]=n;
    }
    printf("  tile unions: "); for(int tl=0;tl<ntile;tl++) printf("%d ",ucnt[tl]); printf("\n");

    float *dq,*dqp,*out; uint16_t *dl,*dk,*dsl,*dsp; int32_t *du,*dc; uint8_t *dm;
    CHK(hipMalloc(&dq,qlat.size()*4));  CHK(hipMalloc(&dqp,qpe.size()*4));
    CHK(hipMalloc(&out,(size_t)T*NH*KV*4));
    CHK(hipMalloc(&dl,lat.size()*2));   CHK(hipMalloc(&dk,kpe.size()*2));
    CHK(hipMalloc(&dsl,sl.size()*2));   CHK(hipMalloc(&dsp,sp.size()*2));
    CHK(hipMalloc(&du,uni.size()*4));   CHK(hipMalloc(&dm,umask.size()));
    CHK(hipMalloc(&dc,ucnt.size()*4));
    CHK(hipMemcpy(dq,qlat.data(),qlat.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dqp,qpe.data(),qpe.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dl,lat.data(),lat.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dk,kpe.data(),kpe.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsl,sl.data(),sl.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dsp,sp.data(),sp.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(du,uni.data(),uni.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dm,umask.data(),umask.size(),hipMemcpyHostToDevice));
    CHK(hipMemcpy(dc,ucnt.data(),ucnt.size()*4,hipMemcpyHostToDevice));
    { std::vector<float> pz((size_t)T*NH*KV,-12345.f);
      CHK(hipMemcpy(out,pz.data(),pz.size()*4,hipMemcpyHostToDevice)); }

    hipLaunchKernelGGL((k_pf_attn_sp<4>),dim3(ntile,NH),dim3(64),0,0,
                       dq,dqp,dl,dk,dsl,dsp,du,dm,dc,UMAX,T,out);
    CHK(hipDeviceSynchronize());
    std::vector<float> got((size_t)T*NH*KV);
    CHK(hipMemcpy(got.data(),out,got.size()*4,hipMemcpyDeviceToHost));

    size_t nf=0; for(float v:got) if(!std::isfinite(v)) nf++;
    if(nf){ printf("  NON-FINITE OUTPUT: %zu of %zu\n",nf,got.size()); return 1; }

    const double sc = 1.0/sqrt(192.0);
    auto ref=[&](int q,int h,std::vector<double>&o){
        const int tl=q/TQ, t=q%TQ;
        std::vector<double> s; std::vector<const uint16_t*> kv;
        for(int k=0;k<SK;k++){ double d=0;
            for(int i=0;i<KV;i++) d+=(double)qlat[((size_t)q*NH+h)*KV+i]*b2f(sl[(size_t)k*KV+i]);
            for(int i=0;i<PE;i++) d+=(double)qpe[((size_t)q*NH+h)*PE+i]*b2f(sp[(size_t)k*PE+i]);
            s.push_back(d*sc); kv.push_back(&sl[(size_t)k*KV]); }
        for(int u=0;u<ucnt[tl];u++){
            if(!((umask[(size_t)tl*UMAX+u]>>t)&1)) continue;
            const int p=uni[(size_t)tl*UMAX+u]; double d=0;
            for(int i=0;i<KV;i++) d+=(double)qlat[((size_t)q*NH+h)*KV+i]*b2f(lat[(size_t)p*KV+i]);
            for(int i=0;i<PE;i++) d+=(double)qpe[((size_t)q*NH+h)*PE+i]*b2f(kpe[(size_t)p*PE+i]);
            s.push_back(d*sc); kv.push_back(&lat[(size_t)p*KV]); }
        double m=-1e300; for(double v:s) m=fmax(m,v);
        double z=0; for(double&v:s){ v=exp(v-m); z+=v; }
        o.assign(KV,0.0);
        for(size_t j=0;j<s.size();j++) for(int i=0;i<KV;i++) o[i]+=s[j]*(double)b2f(kv[j][i]);
        for(int i=0;i<KV;i++) o[i]/=z; };

    double worst=0, mg=0, mr=0; int checked=0;
    for(int q=0;q<T;q++) for(int h=0;h<NH;h+=11){
        std::vector<double> o; ref(q,h,o);
        for(int i=0;i<KV;i+=97){
            const double w=o[i], g=got[((size_t)q*NH+h)*KV+i];
            const double den=fabs(w)>1e-4?fabs(w):1e-4;
            worst=fmax(worst,fabs(w-g)/den);
            mg=fmax(mg,fabs(g)); mr=fmax(mr,fabs(w)); checked++; } }
    printf("  sparse vs host reference: %d points, worst relative %.3e  |k|max %.3e |ref|max %.3e%s\n",
           checked,worst,mg,mr,(mg<1e-12||mr<1e-12)?"   <-- OUTPUT IS ZERO, VACUOUS":"");
    int pass = (worst<5e-3) && mg>1e-12 && mr>1e-12;

    // PACKED PATH: build the consumer representation on the host exactly as
    // k_pf_union_pack would (lat 1024 B, pe 128 B, mask byte in a 16 B slot,
    // exclusive scan of tile counts) and require k_pf_attn_spk to reproduce the
    // index path BIT FOR BIT. Same keys, same order, same masks, same reduction
    // - there is no tolerance to hide behind.
    {
        std::vector<int32_t> po(ntile,0);
        for(int tl=1;tl<ntile;tl++) po[tl]=po[tl-1]+ucnt[tl-1];
        const int tot=po[ntile-1]+ucnt[ntile-1];
        std::vector<uint16_t> lp((size_t)tot*KV,0), ppk((size_t)tot*PE,0);
        std::vector<uint8_t> mp((size_t)tot*16,0);
        for(int tl=0;tl<ntile;tl++){
            const int off=po[tl];
            for(int u=0;u<ucnt[tl];u++){
                const int p=uni[(size_t)tl*UMAX+u];
                for(int i=0;i<KV;i++) lp[((size_t)off+u)*KV+i]=lat[(size_t)p*KV+i];
                for(int i=0;i<PE;i++) ppk[((size_t)off+u)*PE+i]=kpe[(size_t)p*PE+i];
                mp[((size_t)off+u)*16]=umask[(size_t)tl*UMAX+u];
            }
        }
        uint16_t *dlp,*dppk; uint8_t *dmp; int32_t *dpo;
        CHK(hipMalloc(&dlp,lp.size()*2)); CHK(hipMalloc(&dppk,ppk.size()*2));
        CHK(hipMalloc(&dmp,mp.size()));    CHK(hipMalloc(&dpo,po.size()*4));
        CHK(hipMemcpy(dlp,lp.data(),lp.size()*2,hipMemcpyHostToDevice));
        CHK(hipMemcpy(dppk,ppk.data(),ppk.size()*2,hipMemcpyHostToDevice));
        CHK(hipMemcpy(dmp,mp.data(),mp.size(),hipMemcpyHostToDevice));
        CHK(hipMemcpy(dpo,po.data(),po.size()*4,hipMemcpyHostToDevice));
        { std::vector<float> pz2((size_t)T*NH*KV,-12345.f);
          CHK(hipMemcpy(out,pz2.data(),pz2.size()*4,hipMemcpyHostToDevice)); }
        hipLaunchKernelGGL((k_pf_attn_spk<4>),dim3(ntile,NH),dim3(64),0,0,
                           dq,dqp,dlp,dppk,dmp,dsl,dsp,dc,dpo,T,out);
        CHK(hipDeviceSynchronize());
        std::vector<float> gpk((size_t)T*NH*KV);
        CHK(hipMemcpy(gpk.data(),out,gpk.size()*4,hipMemcpyHostToDevice));
        size_t diff=0, nf2=0;
        for(size_t i=0;i<got.size();i++){
            if(!std::isfinite(gpk[i])) nf2++;
            if(__builtin_memcmp(&gpk[i],&got[i],4)) diff++;
        }
        printf("  packed vs index path: %zu of %zu outputs bit-different, nonfinite %zu%s\n",
               diff,got.size(),nf2,(diff||nf2)?"   <-- FAIL, packed path diverged":"");
        if(diff||nf2) pass=0;
        CHK(hipFree(dlp)); CHK(hipFree(dppk)); CHK(hipFree(dmp)); CHK(hipFree(dpo));
    }

    // G2TQ1 PATH (k_pf_attn_spg2): TQ=1 means the selection IS the key list
    // and both heads of a block attend it fully. Variable nsel per query
    // exercises the walk clamp. Reference is the same softmax math, unmasked.
    {
        const int WANT = 24;                 // selection capacity in this test
        std::vector<int32_t> sel2((size_t)T*WANT,0), ns2(T,0);
        for(int q=0;q<T;q++){ const int n = 5 + (q*7)%(WANT-5);
            for(int u=0;u<n;u++) sel2[(size_t)q*WANT+u] = (u*11 + q*3) % P;
            ns2[q] = n; }
        int32_t *ds2, *dn2; uint16_t *dlp2,*dpp2;
        CHK(hipMalloc(&ds2,sel2.size()*4)); CHK(hipMalloc(&dn2,ns2.size()*4));
        CHK(hipMalloc(&dlp2,(size_t)T*WANT*KV*2)); CHK(hipMalloc(&dpp2,(size_t)T*WANT*PE*2));
        CHK(hipMemcpy(ds2,sel2.data(),sel2.size()*4,hipMemcpyHostToDevice));
        CHK(hipMemcpy(dn2,ns2.data(),ns2.size()*4,hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k_pf_sel_pack,dim3(T),dim3(256),0,0,
                           ds2,dn2,T,WANT,dl,dk,dlp2,dpp2);
        CHK(hipDeviceSynchronize());
        { std::vector<float> pz2((size_t)T*NH*KV,-12345.f);
          CHK(hipMemcpy(out,pz2.data(),pz2.size()*4,hipMemcpyHostToDevice)); }
        hipLaunchKernelGGL(k_pf_attn_spg2,dim3(T,NH/2),dim3(64),0,0,
                           dq,dqp,dlp2,dpp2,dsl,dsp,dn2,WANT,T,out);
        CHK(hipDeviceSynchronize());
        std::vector<float> gg((size_t)T*NH*KV);
        CHK(hipMemcpy(gg.data(),out,gg.size()*4,hipMemcpyHostToDevice));
        auto ref2=[&](int q,int h,std::vector<double>&o){
            std::vector<double> s; std::vector<const uint16_t*> kv;
            for(int k=0;k<SK;k++){ double d=0;
                for(int i=0;i<KV;i++) d+=(double)qlat[((size_t)q*NH+h)*KV+i]*b2f(sl[(size_t)k*KV+i]);
                for(int i=0;i<PE;i++) d+=(double)qpe[((size_t)q*NH+h)*PE+i]*b2f(sp[(size_t)k*PE+i]);
                s.push_back(d*sc); kv.push_back(&sl[(size_t)k*KV]); }
            for(int u=0;u<ns2[q];u++){ const int p=sel2[(size_t)q*WANT+u]; double d=0;
                for(int i=0;i<KV;i++) d+=(double)qlat[((size_t)q*NH+h)*KV+i]*b2f(lat[(size_t)p*KV+i]);
                for(int i=0;i<PE;i++) d+=(double)qpe[((size_t)q*NH+h)*PE+i]*b2f(kpe[(size_t)p*PE+i]);
                s.push_back(d*sc); kv.push_back(&lat[(size_t)p*KV]); }
            double m=-1e300; for(double v:s) m=fmax(m,v);
            double z=0; for(double&v:s){ v=exp(v-m); z+=v; }
            o.assign(KV,0.0);
            for(size_t j=0;j<s.size();j++) for(int i=0;i<KV;i++) o[i]+=s[j]*(double)b2f(kv[j][i]);
            for(int i=0;i<KV;i++) o[i]/=z; };
        double w2=0, mg2=0; int c2=0; size_t nf3=0;
        for(float v:gg) if(!std::isfinite(v)) nf3++;
        for(int q=0;q<T;q++) for(int h=0;h<NH;h+=11){
            std::vector<double> o; ref2(q,h,o);
            for(int i=0;i<KV;i+=97){
                const double w=o[i], g=gg[((size_t)q*NH+h)*KV+i];
                const double den=fabs(w)>1e-4?fabs(w):1e-4;
                w2=fmax(w2,fabs(w-g)/den); mg2=fmax(mg2,fabs(g)); c2++; } }
        printf("  spg2 vs host reference: %d points, worst relative %.3e  |k|max %.3e nonfinite %zu%s\n",
               c2,w2,mg2,nf3,(nf3||w2>=5e-3)?"   <-- FAIL":"");
        if(nf3||w2>=5e-3||mg2<1e-12) pass=0;
        CHK(hipFree(ds2)); CHK(hipFree(dn2)); CHK(hipFree(dlp2)); CHK(hipFree(dpp2));
    }

    // NEGATIVE CONTROL: clear one mask bit. The kernel must stop attending to
    // that position for that query, and the output must move.
    { std::vector<uint8_t> m2=umask; m2[0] &= ~1u;
      CHK(hipMemcpy(dm,m2.data(),m2.size(),hipMemcpyHostToDevice));
      hipLaunchKernelGGL((k_pf_attn_sp<4>),dim3(ntile,NH),dim3(64),0,0,
                         dq,dqp,dl,dk,dsl,dsp,du,dm,dc,UMAX,T,out);
      CHK(hipDeviceSynchronize());
      std::vector<float> g2((size_t)T*NH*KV);
      CHK(hipMemcpy(g2.data(),out,g2.size()*4,hipMemcpyDeviceToHost));
      size_t ch=0; for(size_t i=0;i<got.size();i++) if(fabs(got[i]-g2[i])>1e-6f*fmaxf(1.f,fabsf(got[i]))) ch++;
      printf("  negative control (one mask bit cleared): %zu outputs changed%s\n",
             ch, ch?"":"   <-- FAIL, the mask is not being read");
      if(!ch) pass=0; }

    printf("%s\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
