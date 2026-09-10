// Gate mHC against a scalar CPU reference of the contract: rmsnorm with gamma,
// phi projection to 24 logits, the split and Sinkhorn map, collapse, and
// re-expand. Real layer-5 weights where the local checkpoint has them.
#include "../src/p92_mhc.hip"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <fcntl.h>
#include <unistd.h>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
static float b2f(p92_mbf b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
static p92_mbf f2b(float f){ uint32_t u; memcpy(&u,&f,4); u+=0x7FFFu+((u>>16)&1u); return (p92_mbf)(u>>16); }

static std::vector<p92_mbf> load_bf(const char*dir,const char*name,size_t want){
    std::string ip=std::string(dir)+"/model.safetensors.index.json";
    int f=open(ip.c_str(),O_RDONLY); if(f<0) return {};
    std::string js; { char b[1<<16]; ssize_t k; while((k=read(f,b,sizeof b))>0) js.append(b,k); } close(f);
    std::string key=std::string("\"")+name+"\"";
    size_t p=js.find(key); if(p==std::string::npos) return {};
    size_t q=js.find('"',js.find(':',p)+1), r=js.find('"',q+1);
    std::string sp=std::string(dir)+"/"+js.substr(q+1,r-q-1);
    f=open(sp.c_str(),O_RDONLY); if(f<0) return {};
    uint64_t hl; if(pread(f,&hl,8,0)!=8){close(f);return {};}
    std::string hdr(hl,0); if(pread(f,&hdr[0],hl,8)!=(ssize_t)hl){close(f);return {};}
    p=hdr.find(key); if(p==std::string::npos){close(f);return {};}
    size_t off=hdr.find("data_offsets",p); off=hdr.find('[',off);
    long a=atol(hdr.c_str()+off+1), b=atol(hdr.c_str()+hdr.find(',',off)+1);
    if((size_t)(b-a)!=want*2){ printf("  %s: %ld bytes, want %zu\n",name,b-a,want*2); close(f); return {}; }
    std::vector<p92_mbf> out(want);
    if(pread(f,out.data(),b-a,8+hl+a)!=(b-a)){close(f);return {};}
    close(f); return out;
}

int main(int argc,char**argv){
    const char*CK = argc>1?argv[1]:"./checkpoint";
    const int L = argc>2?atoi(argv[2]):5;
    char n1[128],n2[128],n3[128],n4[128];
    snprintf(n1,sizeof n1,"model.layers.%d.attn_mhc_module.phi.weight",L);
    snprintf(n2,sizeof n2,"model.layers.%d.attn_mhc_module.norm_gamma",L);
    snprintf(n3,sizeof n3,"model.layers.%d.attn_mhc_module.branch_alpha",L);
    snprintf(n4,sizeof n4,"model.layers.%d.attn_mhc_module.branch_beta",L);
    auto phi=load_bf(CK,n1,(size_t)P92_MHC_MIX*P92_MHC_FLAT);
    auto gam=load_bf(CK,n2,P92_MHC_FLAT);
    auto alp=load_bf(CK,n3,3);
    auto bet=load_bf(CK,n4,P92_MHC_MIX);
    std::mt19937 rng(20260908); std::normal_distribution<float> nd(0,1);
    const bool real = !phi.empty()&&!gam.empty()&&!alp.empty()&&!bet.empty();
    if(!real){ printf("real mHC weights unavailable, using synthetic\n");
        phi.resize((size_t)P92_MHC_MIX*P92_MHC_FLAT); for(auto&v:phi) v=f2b(nd(rng)*0.02f);
        gam.resize(P92_MHC_FLAT); for(auto&v:gam) v=f2b(1.f+nd(rng)*0.1f);
        alp.resize(3); for(auto&v:alp) v=f2b(1.f+nd(rng)*0.2f);
        bet.resize(P92_MHC_MIX); for(auto&v:bet) v=f2b(nd(rng)*0.3f); }
    else printf("mHC weights: real, layer %d attn_mhc_module\n",L);

    std::vector<p92_mbf> x(P92_MHC_FLAT), m(P92_MHC_D);
    for(auto&v:x) v=f2b(nd(rng));
    for(auto&v:m) v=f2b(nd(rng)*0.5f);

    // ---- CPU reference ---------------------------------------------------
    double ss=0; for(int i=0;i<P92_MHC_FLAT;i++){ double v=b2f(x[i]); ss+=v*v; }
    const float inv=(float)(1.0/sqrt(ss/P92_MHC_FLAT+P92_MHC_RMS_EPS));
    std::vector<float> rlog(P92_MHC_MIX);
    for(int r=0;r<P92_MHC_MIX;r++){ double a=0;
        for(int i=0;i<P92_MHC_FLAT;i++)
            a += (double)b2f(phi[(size_t)r*P92_MHC_FLAT+i]) * ((double)b2f(x[i])*inv*b2f(gam[i]));
        rlog[r]=(float)a; }
    const float a0=b2f(alp[0]),a1=b2f(alp[1]),a2=b2f(alp[2]);
    float rpre[4],rpost[4],c[16];
    for(int h=0;h<4;h++){
        rpre[h]=1.f/(1.f+expf(-(rlog[h]*a0+b2f(bet[h]))))+P92_MHC_HC_EPS;
        rpost[h]=2.f/(1.f+expf(-(rlog[4+h]*a1+b2f(bet[4+h])))); }
    for(int i=0;i<16;i++) c[i]=rlog[8+i]*a2+b2f(bet[8+i]);
    for(int i=0;i<4;i++){ float mx=-3e38f;
        for(int j=0;j<4;j++) mx=fmaxf(mx,c[i*4+j]);
        float s=0,e[4]; for(int j=0;j<4;j++){ e[j]=expf(c[i*4+j]-mx); s+=e[j]; }
        for(int j=0;j<4;j++) c[i*4+j]=e[j]/s+P92_MHC_HC_EPS; }
    for(int j=0;j<4;j++){ float s=P92_MHC_HC_EPS;
        for(int i=0;i<4;i++) s+=c[i*4+j];
        for(int i=0;i<4;i++) c[i*4+j]/=s; }
    for(int it=0;it<P92_MHC_ITERS-1;it++){
        for(int i=0;i<4;i++){ float s=P92_MHC_HC_EPS;
            for(int j=0;j<4;j++) s+=c[i*4+j];
            for(int j=0;j<4;j++) c[i*4+j]/=s; }
        for(int j=0;j<4;j++){ float s=P92_MHC_HC_EPS;
            for(int i=0;i<4;i++) s+=c[i*4+j];
            for(int i=0;i<4;i++) c[i*4+j]/=s; } }
    std::vector<p92_mbf> rcol(P92_MHC_D), ry(P92_MHC_FLAT);
    for(int d=0;d<P92_MHC_D;d++){ float a=0;
        for(int h=0;h<4;h++) a+=rpre[h]*b2f(x[(size_t)h*P92_MHC_D+d]);
        rcol[d]=f2b(a); }
    for(int d=0;d<P92_MHC_D;d++) for(int h=0;h<4;h++){
        float a=rpost[h]*b2f(m[d]);
        for(int k=0;k<4;k++) a+=c[k*4+h]*b2f(x[(size_t)k*P92_MHC_D+d]);
        ry[(size_t)h*P92_MHC_D+d]=f2b(a); }

    // ---- GPU -------------------------------------------------------------
    p92_mbf *dx,*dphi,*dgam,*dalp,*dbet,*dm,*dy; float *dcol;
    float *dlog,*dpre,*dpost,*dcomb;
    CHK(hipMalloc(&dx,x.size()*2)); CHK(hipMalloc(&dphi,phi.size()*2));
    CHK(hipMalloc(&dgam,gam.size()*2)); CHK(hipMalloc(&dalp,3*2)); CHK(hipMalloc(&dbet,24*2));
    CHK(hipMalloc(&dm,m.size()*2));
    float *dmf; CHK(hipMalloc(&dmf,m.size()*4));
    { std::vector<float> mf(m.size()); for(size_t i=0;i<m.size();i++) mf[i]=b2f(m[i]);
      CHK(hipMemcpy(dmf,mf.data(),mf.size()*4,hipMemcpyHostToDevice)); } CHK(hipMalloc(&dcol,P92_MHC_D*4)); CHK(hipMalloc(&dy,P92_MHC_FLAT*2));
    CHK(hipMalloc(&dlog,24*P92_MHC_SPLIT*4)); float*dssq; CHK(hipMalloc(&dssq,P92_MHC_SPLIT*4)); CHK(hipMalloc(&dpre,4*4)); CHK(hipMalloc(&dpost,4*4)); CHK(hipMalloc(&dcomb,16*4));
    CHK(hipMemcpy(dx,x.data(),x.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dphi,phi.data(),phi.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dgam,gam.data(),gam.size()*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dalp,alp.data(),3*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dbet,bet.data(),24*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dm,m.data(),m.size()*2,hipMemcpyHostToDevice));

    auto mhc_pre=[&]() -> void {
        hipLaunchKernelGGL(k_p92_mhc_logits,dim3(P92_MHC_MIX,P92_MHC_SPLIT),dim3(256),0,0,dx,dphi,dgam,dlog,dssq);
        hipLaunchKernelGGL(k_p92_mhc_collapse,dim3(1),dim3(256),0,0,dlog,dssq,dalp,dbet,dx,dpre,dpost,dcomb,dcol); };
    auto mhc_post=[&]() -> void {
        hipLaunchKernelGGL(k_p92_mhc_reexpand,dim3((P92_MHC_D+255)/256),dim3(256),0,0,dpost,dcomb,dmf,dx,dy); };
    mhc_pre(); mhc_post(); CHK(hipDeviceSynchronize());

    std::vector<float> glog(24),gpre(4),gpost(4),gcomb(16);
    std::vector<float> gcolf(P92_MHC_D); std::vector<p92_mbf> gcol(P92_MHC_D), gy(P92_MHC_FLAT);
    { std::vector<float> pp(24*P92_MHC_SPLIT); CHK(hipMemcpy(pp.data(),dlog,pp.size()*4,hipMemcpyDeviceToHost));
      for(int r=0;r<24;r++){ float a=0; for(int sp=0;sp<P92_MHC_SPLIT;sp++) a+=pp[r*P92_MHC_SPLIT+sp]; glog[r]=a*inv; } }
    CHK(hipMemcpy(gpre.data(),dpre,4*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(gpost.data(),dpost,4*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(gcomb.data(),dcomb,16*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(gcolf.data(),dcol,P92_MHC_D*4,hipMemcpyDeviceToHost));
    for(int i=0;i<P92_MHC_D;i++) gcol[i]=f2b(gcolf[i]);
    CHK(hipMemcpy(gy.data(),dy,P92_MHC_FLAT*2,hipMemcpyDeviceToHost));

    auto rel=[&](const std::vector<float>&g,const float*r,int n){
        double e=0,s=0; for(int i=0;i<n;i++){ double d=g[i]-r[i]; e+=d*d; s+=(double)r[i]*r[i]; }
        return s>0?sqrt(e/s):0.0; };
    auto relb=[&](const std::vector<p92_mbf>&g,const std::vector<p92_mbf>&r){
        double e=0,s=0; for(size_t i=0;i<g.size();i++){ double d=b2f(g[i])-b2f(r[i]);
            e+=d*d; s+=(double)b2f(r[i])*b2f(r[i]); } return s>0?sqrt(e/s):0.0; };
    const double rl=rel(glog,rlog.data(),24), rp=rel(gpre,rpre,4),
                 rq=rel(gpost,rpost,4), rc=rel(gcomb,c,16),
                 rcl=relb(gcol,rcol), ryv=relb(gy,ry);
    // Sinkhorn output must be doubly stochastic
    double rmax=0,cmax=0;
    for(int i=0;i<4;i++){ double s=0; for(int j=0;j<4;j++) s+=gcomb[i*4+j]; rmax=fmax(rmax,fabs(s-1.0)); }
    for(int j=0;j<4;j++){ double s=0; for(int i=0;i<4;i++) s+=gcomb[i*4+j]; cmax=fmax(cmax,fabs(s-1.0)); }
    printf("logits    relative_l2 %.3e\n",rl);
    printf("pre/post  relative_l2 %.3e / %.3e\n",rp,rq);
    printf("comb      relative_l2 %.3e   Sinkhorn row dev %.2e col dev %.2e\n",rc,rmax,cmax);
    printf("collapse  relative_l2 %.3e\n",rcl);
    printf("reexpand  relative_l2 %.3e\n",ryv);

    hipEvent_t t0,t1; CHK(hipEventCreate(&t0)); CHK(hipEventCreate(&t1));
    const int IT=300;
    for(int i=0;i<30;i++){ mhc_pre(); mhc_post(); }
    CHK(hipDeviceSynchronize()); CHK(hipEventRecord(t0,0));
    for(int i=0;i<IT;i++){ mhc_pre(); mhc_post(); }
    CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1));
    float ms; hipEventElapsedTime(&ms,t0,t1);
    const double us=ms*1000.0/IT;
    printf("mHC site: %.2f us   92 sites (2 per layer x 46) = %.3f ms/token\n",us,us*92/1000);
    // where does it go
    auto time_one=[&](const char*name, auto fn){
        for(int i=0;i<30;i++) fn(); (void)hipDeviceSynchronize();
        (void)hipEventRecord(t0,0); for(int i=0;i<IT;i++) fn();
        (void)hipEventRecord(t1,0); (void)hipEventSynchronize(t1);
        float e; (void)hipEventElapsedTime(&e,t0,t1);
        printf("  %-26s %7.2f us\n",name,e*1000.0/IT); };
    time_one("logits+ssq (24x4 blocks)",[&]{ hipLaunchKernelGGL(k_p92_mhc_logits,dim3(P92_MHC_MIX,P92_MHC_SPLIT),dim3(256),0,0,dx,dphi,dgam,dlog,dssq); });
    time_one("collapse+Sinkhorn (1 blk)",[&]{ hipLaunchKernelGGL(k_p92_mhc_collapse,dim3(1),dim3(256),0,0,dlog,dssq,dalp,dbet,dx,dpre,dpost,dcomb,dcol); });
    time_one("reexpand",[&]{ hipLaunchKernelGGL(k_p92_mhc_reexpand,dim3((P92_MHC_D+255)/256),dim3(256),0,0,dpost,dcomb,dmf,dx,dy); });
    const bool pass = rl<1e-4 && rp<1e-5 && rq<1e-5 && rc<1e-5 && rcl<1e-3 && ryv<1e-3
                      && rmax<2e-6 && cmax<2e-6;
    printf("%s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
