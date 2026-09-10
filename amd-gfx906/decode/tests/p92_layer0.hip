// First assembled path: token 148899 through layer 0's attention prefix on the
// four-MI50 kernels, gated at every valid capture point from the CUDA oracle.
//
// The Spark capture is NOT usable as a boundary oracle and is not used here.
// p92_capture_layer0.cpp dumps with an unsynchronised cudaMemcpy against
// kernels running on a separate stream, and every dumped buffer is reused
// several times per pass - `normalized` alone is written at :226, :273, :284
// and :293. So each file holds whatever that buffer last held in the PREVIOUS
// pass, not the value at the named boundary. The capture README records the
// symptom without recognising it: "the SECOND set is the executed values, the
// first is the pre-execution warmup and is all zeros" is a description of the
// race itself.
//
// Proof rather than suspicion: the oracle's own source forces an invariant.
// repeat_streams replicates the embedding into four identical streams
// (nvfp4_ops.cu:53-65) and mhc_pre collapses them with a positive weighted sum
// (:729-737), so `mixed` is proportional to the embedding and the constant
// cancels in the RMSNorm. L0_INPUT_NORM must therefore equal
// rmsnorm(embedding) * input_layernorm exactly, for any mHC weights. It does
// not: cosine -0.178. Inverting the capture to recover the embedding row it
// implies matches no row in the vocabulary, best cosine 0.079 over 1600
// candidates.
//
// So layer 0 is gated against a scalar CPU reference transliterated from the
// oracle SOURCE, which is authoritative and is a genuinely independent path
// from the gfx906 kernels. The oracle keeps activations in fp32 where this port
// quantises them to int8 with per-16 scales, so the gate is cosine, not
// equality.
#include "../src/p92_mhc.hip"
#include "../src/p92_proj.hip"
#include "../src/p92_mome.hip"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <fcntl.h>
#include <unistd.h>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
#pragma pack(push,1)
struct MHdr { char magic[8]; uint32_t version, record_bytes, tensor_count, reserved;
              uint64_t weight_bytes, scale_bytes, source_payload_bytes; };
struct MRec { char name[96]; uint32_t rows, columns; uint64_t weight_offset, scale_offset;
              float weight_scale_2; uint32_t reserved; };
#pragma pack(pop)
static void rd(int fd,void*p,size_t n,off_t o){ size_t d=0; while(d<n){ ssize_t k=pread(fd,(char*)p+d,n-d,o+d); if(k<=0){printf("read fail\n");exit(1);} d+=k; } }
static float b2f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }

// ---- minimal safetensors reader -----------------------------------------
struct Ck {
    std::string dir, index;
    bool open(const char*d){ dir=d; int f=::open((dir+"/model.safetensors.index.json").c_str(),O_RDONLY);
        if(f<0) return false; char b[1<<16]; ssize_t k;
        while((k=read(f,b,sizeof b))>0) index.append(b,k); ::close(f); return true; }
    std::vector<uint8_t> get(const std::string&name, size_t*nbytes){
        std::string key="\""+name+"\"";
        size_t p=index.find(key); if(p==std::string::npos) return {};
        size_t q=index.find('"',index.find(':',p)+1), r=index.find('"',q+1);
        std::string sp=dir+"/"+index.substr(q+1,r-q-1);
        int f=::open(sp.c_str(),O_RDONLY); if(f<0) return {};
        uint64_t hl; if(pread(f,&hl,8,0)!=8){::close(f);return {};}
        std::string hdr(hl,0); if(pread(f,&hdr[0],hl,8)!=(ssize_t)hl){::close(f);return {};}
        p=hdr.find(key); if(p==std::string::npos){::close(f);return {};}
        size_t off=hdr.find("data_offsets",p); off=hdr.find('[',off);
        long a=atol(hdr.c_str()+off+1), b=atol(hdr.c_str()+hdr.find(',',off)+1);
        std::vector<uint8_t> out(b-a);
        if(pread(f,out.data(),b-a,8+hl+a)!=(b-a)){::close(f);return {};}
        ::close(f); *nbytes=b-a; return out; }
};

static std::vector<float> load_capture(const char*dir,const char*name,size_t n){
    char p[512]; snprintf(p,sizeof p,"%s/%s.f32",dir,name);
    int f=::open(p,O_RDONLY); if(f<0) return {};
    std::vector<float> v(n);
    if(pread(f,v.data(),n*4,0)!=(ssize_t)(n*4)){ ::close(f); return {}; }
    ::close(f); return v;
}


// ---- scalar CPU reference, transliterated from the CUDA oracle -----------
// nvfp4_ops.cu: mhc_pre_kernel :640-737, rmsnorm_kernel :74-90,
// mome_conv3_kernel :262-286. Activations stay fp32, as the oracle keeps them.
static float ue4m3_c(uint32_t b){ uint32_t e=b>>3,m=b&7u;
    if(e==0) return (float)m*0x1p-9f; if(e==15&&m==7) return INFINITY;
    return ldexpf(1.f+m/8.f,(int)e-7); }
static float e2m1_c(uint32_t n){ uint32_t m=n&7u,e=m>>1,f=m&1u;
    float v = e ? ldexpf(1.f+0.5f*f,(int)e-1) : (f?0.5f:0.f); return (n&8u)?-v:v; }
static std::vector<float> ref_rmsnorm(const std::vector<float>&x,const uint16_t*g,float eps){
    double ss=0; for(float v:x) ss+=(double)v*v;
    const float inv=(float)(1.0/sqrt(ss/x.size()+eps));
    std::vector<float> y(x.size());
    for(size_t i=0;i<x.size();i++) y[i]=x[i]*inv*b2f(g[i]);
    return y; }
static std::vector<float> ref_nvfp4(const std::vector<uint8_t>&W,const std::vector<uint8_t>&S,
                                    float sc2,const std::vector<float>&x,int N,int K){
    std::vector<float> y(N);
    for(int r=0;r<N;r++){ double a=0;
        for(int g=0;g<K/16;g++){ const uint8_t*p=&W[(size_t)r*(K/2)+g*8]; double t=0;
            for(int j=0;j<8;j++) t += (double)e2m1_c(p[j]&15u)*x[g*16+2*j] + (double)e2m1_c(p[j]>>4)*x[g*16+2*j+1];
            a += t*ue4m3_c(S[(size_t)r*(K/16)+g]); }
        y[r]=(float)(a*sc2); }
    return y; }
static std::vector<float> ref_mhc_pre(const std::vector<uint16_t>&streams,const uint16_t*phi,
        const uint16_t*gam,const uint16_t*alp,const uint16_t*bet,int HID){
    const int FLAT=HID*4, MIX=24;
    double ss=0; for(int i=0;i<FLAT;i++){ const double v=b2f(streams[i]); ss+=v*v; }
    const float inv=(float)(1.0/sqrt(ss/FLAT+1e-5));
    std::vector<float> mixes(MIX);
    for(int m=0;m<MIX;m++){ double a=0;
        for(int i=0;i<FLAT;i++) a += (double)(b2f(streams[i])*inv*b2f(gam[i]))*b2f(phi[(size_t)m*FLAT+i]);
        mixes[m]=(float)a; }
    const float a0=b2f(alp[0]);
    std::vector<float> pre(4);
    for(int h=0;h<4;h++) pre[h]=1.f/(1.f+expf(-(mixes[h]*a0+b2f(bet[h]))))+1e-6f;
    std::vector<float> mixed(HID);
    for(int d=0;d<HID;d++){ float v=0.f;
        for(int h=0;h<4;h++) v=fmaf(pre[h],b2f(streams[(size_t)h*HID+d]),v);
        mixed[d]=v; }
    return mixed; }
static std::vector<float> ref_mome(std::vector<float> v,const uint16_t*w,std::vector<float>&state,int C){
    std::vector<float> out(v.size());
    for(int c=0;c<C;c++){
        const float o=state[c*2], p=state[c*2+1], cur=v[c];
        const float cv=fmaf(o,b2f(w[(size_t)c*3]),fmaf(p,b2f(w[(size_t)c*3+1]),cur*b2f(w[(size_t)c*3+2])));
        state[c*2]=p; state[c*2+1]=cur; out[c]=cur+cv; }
    for(size_t i=C;i<v.size();i++) out[i]=v[i];
    return out; }
static std::vector<float> ref_bf16_gemv(const uint16_t*W,const std::vector<float>&x,int N,int K){
    std::vector<float> y(N);
    for(int r=0;r<N;r++){ double a=0;
        for(int i=0;i<K;i++) a += (double)b2f(W[(size_t)r*K+i])*x[i];
        y[r]=(float)a; }
    return y; }

struct Gate { const char*name; double cos, rel, rms_got, rms_ref; bool pass; };
static Gate compare(const char*name,const std::vector<float>&got,const std::vector<float>&ref,double cos_gate){
    double d=0,a=0,b=0,e=0,s=0;
    for(size_t i=0;i<got.size();i++){ d+=(double)got[i]*ref[i]; a+=(double)got[i]*got[i];
        b+=(double)ref[i]*ref[i]; const double q=got[i]-ref[i]; e+=q*q; s+=(double)ref[i]*ref[i]; }
    Gate g; g.name=name; g.cos=d/sqrt(a*b); g.rel=sqrt(e/s);
    g.rms_got=sqrt(a/got.size()); g.rms_ref=sqrt(b/ref.size()); g.pass=g.cos>=cos_gate;
    return g;
}

int main(int argc,char**argv){
    const char*CK  = argc>1?argv[1]:"./checkpoint";
    const char*ART = argc>2?argv[2]:"./artifact";
    const char*CAP = argc>3?argv[3]:"./data/openpangu-oracle-capture"; (void)CAP;
    const int TOKEN=148899;
    const int HID=2560, QLORA=1024, QUP=9216, KVLORA=512, ROPE=64, KVA=576;

    Ck ck; if(!ck.open(CK)){ printf("checkpoint index not readable\n"); return 1; }
    int mf=::open((std::string(ART)+"/manifest.bin").c_str(),O_RDONLY);
    int wf=::open((std::string(ART)+"/weights.nvfp4").c_str(),O_RDONLY);
    int sf=::open((std::string(ART)+"/scales.e4m3").c_str(),O_RDONLY);
    if(mf<0||wf<0||sf<0){ printf("artifact not readable\n"); return 1; }
    MHdr h; rd(mf,&h,sizeof h,0);
    std::vector<MRec> recs(h.tensor_count); rd(mf,recs.data(),(size_t)h.tensor_count*h.record_bytes,sizeof h);
    std::unordered_map<std::string,const MRec*> BN;
    for(auto&r:recs) BN[std::string(r.name,strnlen(r.name,96))]=&r;

    // ---- host weights ----------------------------------------------------
    size_t nb;
    auto need=[&](const char*n)->std::vector<uint8_t>{
        auto v=ck.get(n,&nb); if(v.empty()){ printf("MISSING %s\n",n); exit(1); } return v; };
    auto embed_row=[&]()->std::vector<uint16_t>{
        // read only the one row we need out of the embedding shard
        std::string key="\"model.embed_tokens.weight\"";
        size_t p=ck.index.find(key);
        size_t q=ck.index.find('"',ck.index.find(':',p)+1), r=ck.index.find('"',q+1);
        std::string sp=std::string(CK)+"/"+ck.index.substr(q+1,r-q-1);
        int f=::open(sp.c_str(),O_RDONLY);
        uint64_t hl; rd(f,&hl,8,0); std::string hdr(hl,0); rd(f,&hdr[0],hl,8);
        size_t pp=hdr.find(key); size_t off=hdr.find("data_offsets",pp); off=hdr.find('[',off);
        long a=atol(hdr.c_str()+off+1);
        std::vector<uint16_t> row(HID);
        rd(f,row.data(),HID*2,8+hl+a+(off_t)TOKEN*HID*2);
        ::close(f); return row; };

    auto emb   = embed_row();
    auto h_phi = need("model.layers.0.attn_mhc_module.phi.weight");
    auto h_gam = need("model.layers.0.attn_mhc_module.norm_gamma");
    auto h_alp = need("model.layers.0.attn_mhc_module.branch_alpha");
    auto h_bet = need("model.layers.0.attn_mhc_module.branch_beta");
    auto h_inn = need("model.layers.0.input_layernorm.weight");
    auto h_qcv = need("model.layers.0.self_attn.qa_conv.weight");
    auto h_qnm = need("model.layers.0.self_attn.q_a_layernorm.weight");
    auto h_kva = need("model.layers.0.self_attn.kv_a_proj_with_mqa.weight");
    auto h_kcv = need("model.layers.0.self_attn.compresskv_conv.weight");
    auto h_knm = need("model.layers.0.self_attn.kv_a_layernorm.weight");
    printf("layer-0 weights loaded from %s\n", CK);

    auto nv=[&](const char*n)->const MRec*{ auto it=BN.find(n);
        if(it==BN.end()){ printf("MISSING artifact %s\n",n); exit(1);} return it->second; };
    const MRec *Rqa=nv("model.layers.0.self_attn.q_a_proj.weight");
    const MRec *Rqb=nv("model.layers.0.self_attn.q_b_proj.weight");

    // ---- device ----------------------------------------------------------
    CHK(hipSetDevice(0));
    auto up=[&](const void*p,size_t n)->void*{ void*d; if(hipMalloc(&d,n)!=hipSuccess) exit(1);
        if(hipMemcpy(d,p,n,hipMemcpyHostToDevice)!=hipSuccess) exit(1); return d; };
    auto upnv=[&](const MRec*R,void**W,void**S){
        const size_t WB=(size_t)R->rows*(R->columns/2), SB=(size_t)R->rows*(R->columns/16);
        std::vector<uint8_t> w(WB),s(SB);
        rd(wf,w.data(),WB,(off_t)R->weight_offset); rd(sf,s.data(),SB,(off_t)R->scale_offset);
        *W=up(w.data(),WB); *S=up(s.data(),SB); };

    void *d_phi=up(h_phi.data(),h_phi.size()), *d_gam=up(h_gam.data(),h_gam.size());
    void *d_alp=up(h_alp.data(),h_alp.size()), *d_bet=up(h_bet.data(),h_bet.size());
    void *d_inn=up(h_inn.data(),h_inn.size()), *d_qcv=up(h_qcv.data(),h_qcv.size());
    void *d_qnm=up(h_qnm.data(),h_qnm.size()), *d_kva=up(h_kva.data(),h_kva.size());
    void *d_kcv=up(h_kcv.data(),h_kcv.size()), *d_knm=up(h_knm.data(),h_knm.size());
    void *d_Wqa,*d_Sqa,*d_Wqb,*d_Sqb; upnv(Rqa,&d_Wqa,&d_Sqa); upnv(Rqb,&d_Wqb,&d_Sqb);

    // four identical streams from the token embedding
    std::vector<uint16_t> streams((size_t)4*HID);
    for(int s=0;s<4;s++) memcpy(&streams[(size_t)s*HID],emb.data(),HID*2);
    void *d_str=up(streams.data(),streams.size()*2);

    uint16_t *d_col; float *d_mixf,*d_norm,*d_pre,*d_post,*d_comb,*d_part,*d_ssq;
    float *d_qa,*d_qnormf,*d_qb,*d_kvd,*d_kvn;
    int8_t *d_xq; float *d_xs;
    CHK(hipMalloc(&d_col,HID*2));
    CHK(hipMalloc(&d_mixf,HID*4)); CHK(hipMalloc(&d_norm,HID*4));
    CHK(hipMalloc(&d_pre,4*4)); CHK(hipMalloc(&d_post,4*4)); CHK(hipMalloc(&d_comb,16*4));
    CHK(hipMalloc(&d_part,24*P92_MHC_SPLIT*4)); CHK(hipMalloc(&d_ssq,P92_MHC_SPLIT*4));
    CHK(hipMalloc(&d_qa,QLORA*4)); CHK(hipMalloc(&d_qnormf,QLORA*4)); CHK(hipMalloc(&d_qb,QUP*4));
    CHK(hipMalloc(&d_kvd,KVA*4)); CHK(hipMalloc(&d_kvn,KVLORA*4));
    CHK(hipMalloc(&d_xq,HID)); CHK(hipMalloc(&d_xs,(HID/16)*4));
    uint16_t *d_qa_bf,*d_kvd_bf,*d_qstate,*d_kstate;
    CHK(hipMalloc(&d_qa_bf,QLORA*2)); CHK(hipMalloc(&d_kvd_bf,KVA*2));
    CHK(hipMalloc(&d_qstate,QLORA*2*2)); CHK(hipMalloc(&d_kstate,KVLORA*2*2));
    CHK(hipMemset(d_qstate,0,QLORA*2*2)); CHK(hipMemset(d_kstate,0,KVLORA*2*2));

    std::vector<Gate> gates;
    // build the whole reference chain up front, on the CPU, from the oracle source
    std::vector<float> R_norm, R_qa, R_qnorm, R_qb, R_kvn;
    {
        auto nvload=[&](const MRec*R){ std::pair<std::vector<uint8_t>,std::vector<uint8_t>> o;
            const size_t WB=(size_t)R->rows*(R->columns/2), SB=(size_t)R->rows*(R->columns/16);
            o.first.resize(WB); o.second.resize(SB);
            rd(wf,o.first.data(),WB,(off_t)R->weight_offset);
            rd(sf,o.second.data(),SB,(off_t)R->scale_offset); return o; };
        auto Wqa=nvload(Rqa); auto Wqb=nvload(Rqb);
        auto mixed = ref_mhc_pre(streams,(const uint16_t*)h_phi.data(),(const uint16_t*)h_gam.data(),
                                 (const uint16_t*)h_alp.data(),(const uint16_t*)h_bet.data(),HID);
        R_norm = ref_rmsnorm(mixed,(const uint16_t*)h_inn.data(),1e-5f);
        R_qa   = ref_nvfp4(Wqa.first,Wqa.second,Rqa->weight_scale_2,R_norm,QLORA,HID);
        std::vector<float> qs(QLORA*2,0.f);
        auto qc = ref_mome(R_qa,(const uint16_t*)h_qcv.data(),qs,QLORA);
        R_qnorm = ref_rmsnorm(qc,(const uint16_t*)h_qnm.data(),1e-5f);
        R_qb   = ref_nvfp4(Wqb.first,Wqb.second,Rqb->weight_scale_2,R_qnorm,QUP,QLORA);
        auto kvd = ref_bf16_gemv((const uint16_t*)h_kva.data(),R_norm,KVA,HID);
        std::vector<float> ks(KVLORA*2,0.f);
        auto kc = ref_mome(kvd,(const uint16_t*)h_kcv.data(),ks,KVLORA);
        kc.resize(KVLORA);
        R_kvn = ref_rmsnorm(kc,(const uint16_t*)h_knm.data(),1e-5f);
        printf("CPU reference from the oracle source: input_norm rms %.6g, q_a rms %.6g, q_b rms %.6g, kv_norm rms %.6g\n",
               sqrt(std::inner_product(R_norm.begin(),R_norm.end(),R_norm.begin(),0.0)/R_norm.size()),
               sqrt(std::inner_product(R_qa.begin(),R_qa.end(),R_qa.begin(),0.0)/R_qa.size()),
               sqrt(std::inner_product(R_qb.begin(),R_qb.end(),R_qb.begin(),0.0)/R_qb.size()),
               sqrt(std::inner_product(R_kvn.begin(),R_kvn.end(),R_kvn.begin(),0.0)/R_kvn.size()));
    }
    auto pull=[&](const float*d,size_t n){ std::vector<float> v(n);
        (void)hipMemcpy(v.data(),d,n*4,hipMemcpyDeviceToHost); return v; };

    // ---- 1. mHC pre, then input RMSNorm ---------------------------------
    hipLaunchKernelGGL(k_p92_mhc_logits,dim3(P92_MHC_MIX,P92_MHC_SPLIT),dim3(256),0,0,
                       (const p92_mbf*)d_str,(const p92_mbf*)d_phi,(const p92_mbf*)d_gam,d_part,d_ssq);
    hipLaunchKernelGGL(k_p92_mhc_collapse,dim3(1),dim3(256),0,0,
                       d_part,d_ssq,(const p92_mbf*)d_alp,(const p92_mbf*)d_bet,
                       (const p92_mbf*)d_str,d_pre,d_post,d_comb,(p92_mbf*)d_col);
    // collapsed is bf16; widen for the norm
    { std::vector<uint16_t> hc(HID); CHK(hipDeviceSynchronize());
      CHK(hipMemcpy(hc.data(),d_col,HID*2,hipMemcpyDeviceToHost));
      std::vector<float> f(HID); for(int i=0;i<HID;i++) f[i]=b2f(hc[i]);
      CHK(hipMemcpy(d_mixf,f.data(),HID*4,hipMemcpyHostToDevice)); }
    hipLaunchKernelGGL((k_p92_rmsnorm<2560>),dim3(1),dim3(256),0,0,
                       d_mixf,(const p92_pbf*)d_inn,d_norm,1e-5f);
    CHK(hipDeviceSynchronize());
    gates.push_back(compare("L0_INPUT_NORM",pull(d_norm,HID),R_norm,0.999));

    // ---- 2. q_a projection ----------------------------------------------
    hipLaunchKernelGGL((k_p92_quant<2560>),dim3(1),dim3(256),0,0,d_norm,d_xq,d_xs);
    hipLaunchKernelGGL((k_p92_proj<2560,32>),dim3(QLORA/8),dim3(256),0,0,
                       (const uint8_t*)d_Wqa,(const uint8_t*)d_Sqa,d_xq,d_xs,Rqa->weight_scale_2,d_qa);
    CHK(hipDeviceSynchronize());
    gates.push_back(compare("L0_Q_A",pull(d_qa,QLORA),R_qa,0.999));

    // ---- 3. q MoME conv, then q RMSNorm ---------------------------------
    { auto v=pull(d_qa,QLORA); std::vector<uint16_t> b(QLORA);
      for(int i=0;i<QLORA;i++){ uint32_t u; memcpy(&u,&v[i],4); u+=0x7FFFu+((u>>16)&1u); b[i]=(uint16_t)(u>>16); }
      CHK(hipMemcpy(d_qa_bf,b.data(),QLORA*2,hipMemcpyHostToDevice)); }
    hipLaunchKernelGGL(k_p92_mome_conv3<P92_MOME_QA>,dim3((QLORA+255)/256),dim3(256),0,0,
                       (p92_cbf*)d_qa_bf,(const p92_cbf*)d_qcv,(p92_cbf*)d_qstate);
    CHK(hipDeviceSynchronize());
    { std::vector<uint16_t> b(QLORA); CHK(hipMemcpy(b.data(),d_qa_bf,QLORA*2,hipMemcpyDeviceToHost));
      std::vector<float> f(QLORA); for(int i=0;i<QLORA;i++) f[i]=b2f(b[i]);
      CHK(hipMemcpy(d_qa,f.data(),QLORA*4,hipMemcpyHostToDevice)); }
    hipLaunchKernelGGL((k_p92_rmsnorm<1024>),dim3(1),dim3(256),0,0,
                       d_qa,(const p92_pbf*)d_qnm,d_qnormf,1e-5f);
    CHK(hipDeviceSynchronize());
    gates.push_back(compare("L0_Q_NORM",pull(d_qnormf,QLORA),R_qnorm,0.999));

    // ---- 4. q_b projection ----------------------------------------------
    int8_t *d_xq2; float *d_xs2;
    CHK(hipMalloc(&d_xq2,QLORA)); CHK(hipMalloc(&d_xs2,(QLORA/16)*4));
    hipLaunchKernelGGL((k_p92_quant<1024>),dim3(1),dim3(256),0,0,d_qnormf,d_xq2,d_xs2);
    hipLaunchKernelGGL((k_p92_proj<1024,32>),dim3(QUP/8),dim3(256),0,0,
                       (const uint8_t*)d_Wqb,(const uint8_t*)d_Sqb,d_xq2,d_xs2,Rqb->weight_scale_2,d_qb);
    CHK(hipDeviceSynchronize());
    gates.push_back(compare("L0_Q_B",pull(d_qb,QUP),R_qb,0.999));

    // ---- 5. kv_a bf16 GEMV, kv MoME, kv RMSNorm -------------------------
    hipLaunchKernelGGL((k_p92_bf16_gemv<2560>),dim3((KVA+3)/4),dim3(256),0,0,
                       (const p92_pbf*)d_kva,d_norm,d_kvd);
    CHK(hipDeviceSynchronize());
    { auto v=pull(d_kvd,KVA); std::vector<uint16_t> b(KVA);
      for(int i=0;i<KVA;i++){ uint32_t u; memcpy(&u,&v[i],4); u+=0x7FFFu+((u>>16)&1u); b[i]=(uint16_t)(u>>16); }
      CHK(hipMemcpy(d_kvd_bf,b.data(),KVA*2,hipMemcpyHostToDevice)); }
    hipLaunchKernelGGL(k_p92_mome_conv3<P92_MOME_KV>,dim3((KVLORA+255)/256),dim3(256),0,0,
                       (p92_cbf*)d_kvd_bf,(const p92_cbf*)d_kcv,(p92_cbf*)d_kstate);
    CHK(hipDeviceSynchronize());
    { std::vector<uint16_t> b(KVA); CHK(hipMemcpy(b.data(),d_kvd_bf,KVA*2,hipMemcpyDeviceToHost));
      std::vector<float> f(KVLORA); for(int i=0;i<KVLORA;i++) f[i]=b2f(b[i]);
      CHK(hipMemcpy(d_kvd,f.data(),KVLORA*4,hipMemcpyHostToDevice)); }
    hipLaunchKernelGGL((k_p92_rmsnorm<512>),dim3(1),dim3(256),0,0,
                       d_kvd,(const p92_pbf*)d_knm,d_kvn,1e-5f);
    CHK(hipDeviceSynchronize());
    gates.push_back(compare("L0_KV_NORM",pull(d_kvn,KVLORA),R_kvn,0.999));

    // ---- report ----------------------------------------------------------
    printf("\ntoken %d, position 0, layer 0, against a CPU reference from the CUDA oracle source:\n",TOKEN);
    printf("  %-16s %10s %12s %12s %12s  %s\n","boundary","cosine","rel_l2","rms got","rms ref","");
    bool ok=true; int first_bad=-1;
    for(size_t i=0;i<gates.size();i++){ const Gate&g=gates[i];
        printf("  %-16s %10.6f %12.3e %12.6g %12.6g  %s\n",
               g.name,g.cos,g.rel,g.rms_got,g.rms_ref,g.pass?"PASS":"<-- FIRST DIVERGENCE");
        if(!g.pass && first_bad<0){ first_bad=(int)i; ok=false; }
        if(!g.pass) break; }
    printf("\n%s\n", ok?"PASS (gate: cosine >= 0.999 at every layer-0 boundary)":"FAIL");
    return ok?0:1;
}
