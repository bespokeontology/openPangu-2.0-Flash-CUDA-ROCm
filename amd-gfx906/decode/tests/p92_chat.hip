// openPangu-2.0-Flash on 4x MI50. The whole decode step, composed from the
// accepted custom gfx906 kernels. No generic GEMM anywhere in the hot path.
//
// Layer order is the CUDA oracle's (model.cpp:185-228), executed with:
//   mHC pre/post + Sinkhorn        p92_mhc.hip
//   RMSNorm, projections, rope     p92_proj.hip
//   MoME convolutions, three sites p92_mome.hip
//   absorbed MLA + value up        p92_mla_prep.hip
//   MLA decode, 32-lane groups     p92_mla_g16.hip
//   DSA scorer + radix select      p92_dsa_index.hip
//   routed + shared experts        p92_nvfp4_dot4.hip on the 257-slot arena
//   dense MLP, layers 0 and 1      p92_dense.hip
//   head + top-256 refine          p92_head.hip
//   cross-card ring                p92_transport.cpp
#include "p92amd/p92_arena.h"
#include "p92amd/p92_transport.h"
#include "../src/p92_i8_expert.hip"
#include "../src/p92_mhc.hip"
#include "../src/p92_proj.hip"
#include "../src/p92_mome.hip"
#include "../src/p92_mla_prep.hip"
#include "../src/p92_mla_decode.hip"
#include "../src/p92_mla_split.hip"
#include "../src/p92_mla_g16.hip"
#include "../src/p92_dsa_index.hip"
#include "../src/p92_dense.hip"
#include "../src/p92_head.hip"
#include "../src/p92_router.hip"
#include "../src/p92_arena.cpp"
#include "../src/p92_transport.cpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include "p92/tokenizer.h"

#define CK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); exit(1);} }while(0)
#define HID 2560
#define QLORA 1024
#define QUP 9216
#define KVLORA 512
#define ROPE 64
#define KVA 576
#define AOUT 6144
#define NLAYER 46
#define NEXP 256
#define SINKS 128

#pragma pack(push,1)
struct MHdr { char magic[8]; uint32_t version, record_bytes, tensor_count, reserved;
              uint64_t weight_bytes, scale_bytes, source_payload_bytes; };
struct MRec { char name[96]; uint32_t rows, columns; uint64_t weight_offset, scale_offset;
              float weight_scale_2; uint32_t reserved; };
#pragma pack(pop)
static void rdf(int fd,void*p,size_t n,off_t o){ size_t d=0; while(d<n){ ssize_t k=pread(fd,(char*)p+d,n-d,o+d);
    if(k<=0){ printf("read fail at %ld\n",(long)o); exit(1);} d+=k; } }

// ---- host-side checkpoint ------------------------------------------------
struct Ck {
    std::string dir, index;
    bool open(const char*d){ dir=d; int f=::open((dir+"/model.safetensors.index.json").c_str(),O_RDONLY);
        if(f<0) return false; char b[1<<16]; ssize_t k;
        while((k=read(f,b,sizeof b))>0) index.append(b,k); ::close(f); return true; }
    std::vector<uint8_t> get(const std::string&name){
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
        ::close(f); return out; }
    // Resident embedding mapping. embed_row() below opens the shard, parses its
    // header and preads ONCE PER TOKEN, and it sat inside the timed region, so
    // per-token file I/O is inside every figure this project has banked. Harmless
    // to an A/B (both arms pay it, page-cached after the first token) but its cost
    // depends on page-cache state, a runtime variable inside the measurement. For
    // prefill a T-token chunk would pay T opens, which is why this exists.
    // Resolve and map once; the hot path is then a pointer.
    int emb_fd = -1; const uint8_t *emb_map = nullptr; size_t emb_len = 0; off_t emb_base = 0;
    bool embed_open(){
        if (emb_map) return true;
        std::string key="\"model.embed_tokens.weight\"";
        size_t p=index.find(key); if(p==std::string::npos) return false;
        size_t q=index.find('"',index.find(':',p)+1), r=index.find('"',q+1);
        std::string sp=dir+"/"+index.substr(q+1,r-q-1);
        emb_fd=::open(sp.c_str(),O_RDONLY); if(emb_fd<0) return false;
        uint64_t hl; if(pread(emb_fd,&hl,8,0)!=8) return false;
        std::string hdr(hl,0); if(pread(emb_fd,&hdr[0],hl,8)!=(ssize_t)hl) return false;
        size_t pp=hdr.find(key); if(pp==std::string::npos) return false;
        size_t off=hdr.find("data_offsets",pp); off=hdr.find('[',off);
        emb_base = (off_t)(8+hl+atol(hdr.c_str()+off+1));
        struct stat stt; if(fstat(emb_fd,&stt)) return false;
        emb_len=(size_t)stt.st_size;
        emb_map=(const uint8_t*)mmap(nullptr,emb_len,PROT_READ,MAP_PRIVATE,emb_fd,0);
        if(emb_map==MAP_FAILED){ emb_map=nullptr; return false; }
        return true; }
    // hot path: no syscall, no allocation, no copy
    const uint16_t* embed_ptr(int tok) const {
        return (const uint16_t*)(emb_map + emb_base + (size_t)tok*(size_t)HID*2); }

    std::vector<uint16_t> embed_row(int tok){
        std::string key="\"model.embed_tokens.weight\"";
        size_t p=index.find(key);
        size_t q=index.find('"',index.find(':',p)+1), r=index.find('"',q+1);
        std::string sp=dir+"/"+index.substr(q+1,r-q-1);
        int f=::open(sp.c_str(),O_RDONLY);
        uint64_t hl; rdf(f,&hl,8,0); std::string hdr(hl,0); rdf(f,&hdr[0],hl,8);
        size_t pp=hdr.find(key); size_t off=hdr.find("data_offsets",pp); off=hdr.find('[',off);
        long a=atol(hdr.c_str()+off+1);
        std::vector<uint16_t> row(HID);
        rdf(f,row.data(),HID*2,8+hl+a+(off_t)tok*HID*2);
        ::close(f); return row; }
};

static int mfd,wfd,sfd;
static std::unordered_map<std::string,const MRec*> BN;
static void* dev_up(const void*p,size_t n){ void*d; CK(hipMalloc(&d,n)); CK(hipMemcpy(d,p,n,hipMemcpyHostToDevice)); return d; }
struct Nv { uint8_t*w; uint8_t*s; float s2; int rows, cols; };
static Nv nv_up(const char*name){
    auto it=BN.find(name); if(it==BN.end()){ printf("MISSING artifact %s\n",name); exit(1); }
    const MRec*R=it->second;
    const size_t WB=(size_t)R->rows*(R->columns/2), SB=(size_t)R->rows*(R->columns/16);
    std::vector<uint8_t> w(WB), s(SB);
    rdf(wfd,w.data(),WB,(off_t)R->weight_offset); rdf(sfd,s.data(),SB,(off_t)R->scale_offset);
    Nv o; o.w=(uint8_t*)dev_up(w.data(),WB); o.s=(uint8_t*)dev_up(s.data(),SB);
    o.s2=R->weight_scale_2; o.rows=R->rows; o.cols=R->columns; return o; }

struct LayerW {
    Nv qa,qb,o, ix_wk,ix_wqb, dg,du,dd;
    uint16_t *kv_b,*kv_a,*in_norm,*post_attn,*pre_mlp,*post_mlp,*block_post;
    uint16_t *aphi,*agam,*aalp,*abet,*mphi,*mgam,*malp,*mbet;
    uint16_t *q_norm,*kv_norm,*qa_conv,*kv_conv,*o_conv;
    uint16_t *sink_kv_n,*sink_rope,*router_w,*ix_knorm,*ix_wproj;
    float *router_bias;
    uint16_t *mome_q,*mome_kv,*mome_o,*cache_kv,*cache_rope,*ix_cache;
    int cap; bool dense,dsa; int arena_slot_base;
};


// ---- small openPangu-shape kernels the executor needs --------------------
// Build the 9-wide dispatch: eight router selections plus the shared expert at
// arena slot 256, weight 1.0, which is how the reference combines it
// (final_hidden_states = final_hidden_states + shared_output).
__global__ void __launch_bounds__(16)
k_p92_build_dispatch(const int *__restrict__ ids, const float *__restrict__ wts,
                     int slot_base, int experts, int *__restrict__ slot,
                     float *__restrict__ w) {
    const int k = threadIdx.x;
    if (k < 8) { slot[k] = slot_base + ids[k]; w[k] = wts[k]; }
    else if (k == 8) { slot[8] = slot_base + P92_SHARED_INDEX; w[8] = 1.0f; }
}

// The merge module has phi [4, 10240]: four logits, pre weights only, no post
// and no comb. Sums the split partials, applies the hoisted rms, collapses.
__global__ void __launch_bounds__(256)
k_p92_merge_collapse(const float *__restrict__ partial, const float *__restrict__ ssq_partial,
                     const p92_mbf *__restrict__ alpha_pre, const p92_mbf *__restrict__ beta_pre,
                     const p92_mbf *__restrict__ x, p92_mbf *__restrict__ out) {
    __shared__ float spre[4]; __shared__ float s_inv;
    if (threadIdx.x == 0) {
        float ss = 0.f;
        for (int sp = 0; sp < P92_MHC_SPLIT; sp++) ss += ssq_partial[sp];
        s_inv = rsqrtf(ss / (float)P92_MHC_FLAT + P92_MHC_RMS_EPS);
    }
    __syncthreads();
    if (threadIdx.x < 4) {
        float a = 0.f;
        for (int sp = 0; sp < P92_MHC_SPLIT; sp++) a += partial[threadIdx.x * P92_MHC_SPLIT + sp];
        a *= s_inv;
        spre[threadIdx.x] = 1.f / (1.f + __expf(-(a * p92_mb2f(alpha_pre[0])
                                    + p92_mb2f(beta_pre[threadIdx.x])))) + P92_MHC_HC_EPS;
    }
    __syncthreads();
    for (int d = blockIdx.x * 256 + threadIdx.x; d < P92_MHC_D; d += gridDim.x * 256) {
        float acc = 0.f;
        #pragma unroll
        for (int hh = 0; hh < 4; hh++) acc = fmaf(spre[hh], p92_mb2f(x[(size_t)hh * P92_MHC_D + d]), acc);
        out[d] = p92_mf2b(acc);
    }
}

// indexer key: rope the first 64 of 128 dims as pairs (i, i+32), then append to
// the transposed [128][cap] cache. openPangu ropes only the first half and
// norms the KEY, both of which differ from Qwen.
__global__ void __launch_bounds__(128)
k_p92_ix_key(const float *__restrict__ k, const p92_pbf *__restrict__ knorm,
             const float *__restrict__ cosv, const float *__restrict__ sinv,
             p92_bf *__restrict__ cache, int cap, int slot, float eps) {
    // blockDim is exactly P92_IDX_DIM, so every lane participates and the
    // barrier below is reached by all of them. No early return.
    const int d = threadIdx.x;
    // RMS norm over the 128 index dims, FUSED so it costs no launch. w.ix_knorm
    // was loaded and never applied; the oracle norms the key between the
    // projection and the RoPE (model.cpp:1400 prefill, :1625 decode). The
    // comment on this kernel already said "norms the KEY" - it just never did.
    const float kd = k[d];
    float ss = kd * kd;
    #pragma unroll
    for (int o = 32; o; o >>= 1) ss += __shfl_down(ss, o, 64);
    __shared__ float red[2];
    if ((d & 63) == 0) red[d >> 6] = ss;
    __syncthreads();
    const float inv = rsqrtf((red[0] + red[1]) / (float)P92_IDX_DIM + eps);
    // the RoPE mixes d with d+-32, so a lane forms its partner's NORMALISED
    // value directly instead of exchanging it - inv is wave-uniform and the
    // two extra loads are L1 hits
    #define P92_NK(i) (k[(i)] * inv * p92_p2f(knorm[(i)]))
    float v = P92_NK(d);
    if (d < 32)      { const float a=P92_NK(d),    b=P92_NK(d+32); v = a*cosv[d] - b*sinv[d]; }
    else if (d < 64) { const float a=P92_NK(d-32), b=P92_NK(d);    v = b*cosv[d-32] + a*sinv[d-32]; }
    #undef P92_NK
    uint32_t u; __builtin_memcpy(&u,&v,4); u += 0x7FFFu + ((u>>16)&1u);
    cache[(size_t)d * cap + slot] = (p92_bf)(u >> 16);
}

// indexer query: same rope rule, applied per head over 24 heads of 128
__global__ void __launch_bounds__(128)
k_p92_ix_query(float *__restrict__ q, const float *__restrict__ cosv,
               const float *__restrict__ sinv) {
    const int h = blockIdx.x, i = threadIdx.x;
    if (i >= 32) return;
    float *v = q + (size_t)h * P92_IDX_DIM;
    const float a = v[i], b = v[i + 32];
    v[i]      = a * cosv[i] - b * sinv[i];
    v[i + 32] = b * cosv[i] + a * sinv[i];
}

// a contiguous position list, for a DSA layer below the top-k threshold
__global__ void __launch_bounds__(256)
k_p92_iota(int *__restrict__ sel, int n) {
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) sel[i] = i;
}

int main(int argc,char**argv){
    const char*CKDIR = argc>1?argv[1]:"./checkpoint";
    const char*ART   = argc>2?argv[2]:"./artifact";
    const char*ARENA = argc>3?argv[3]:"./arena";
    const int  NTOK  = argc>4?atoi(argv[4]):8;
    const int  MAXPOS= argc>5?atoi(argv[5]):512;
    const int  START = argc>6?atoi(argv[6]):148899;

    Ck ck; if(!ck.open(CKDIR)){ printf("no checkpoint index\n"); return 1; }
    mfd=::open((std::string(ART)+"/manifest.bin").c_str(),O_RDONLY);
    wfd=::open((std::string(ART)+"/weights.nvfp4").c_str(),O_RDONLY);
    sfd=::open((std::string(ART)+"/scales.e4m3").c_str(),O_RDONLY);
    if(mfd<0||wfd<0||sfd<0){ printf("no artifact\n"); return 1; }
    MHdr h; rdf(mfd,&h,sizeof h,0);
    static std::vector<MRec> recs(h.tensor_count);
    rdf(mfd,recs.data(),(size_t)h.tensor_count*h.record_bytes,sizeof h);
    for(auto&r:recs) BN[std::string(r.name,strnlen(r.name,96))]=&r;
    printf("artifact %u tensors; checkpoint %s\n", h.tensor_count, CKDIR);

    // ---- device ownership: 12/12/11/11 ----------------------------------
    auto lb=[](int d){ int b=NLAYER/4, r=NLAYER%4; return d*b+(d<r?d:r); };
    auto le=[](int d){ int b=NLAYER/4, r=NLAYER%4; return (d+1)*b+((d+1)<r?(d+1):r); };
    auto owner=[&](int L){ for(int d=0;d<4;d++) if(L>=lb(d)&&L<le(d)) return d; return -1; };

    P92Arena arena[4];
    for(int d=0;d<4;d++){
        CK(hipSetDevice(d));
        char p[512]; snprintf(p,sizeof p,"%s/arena_dev%d.bin",ARENA,d);
        if(p92_arena_load(p,&arena[d])){ printf("arena %d failed\n",d); return 1; }
        printf("dev%d layers %d..%d, arena %d slots\n",d,lb(d),le(d)-1,arena[d].slots);
    }

    std::vector<LayerW> LW(NLAYER);
    for(int L=0;L<NLAYER;L++){
        const int d=owner(L); CK(hipSetDevice(d));
        LayerW&w=LW[L];
        w.dense = (L<2); w.dsa = (L%3)==0;
        char n[192];
        auto bf=[&](const char*fmt)->uint16_t*{ snprintf(n,sizeof n,fmt,L);
            auto v=ck.get(n); if(v.empty()){ printf("MISSING %s\n",n); exit(1);} 
            return (uint16_t*)dev_up(v.data(),v.size()); };
        auto bf_opt=[&](const char*fmt)->uint16_t*{ snprintf(n,sizeof n,fmt,L);
            auto v=ck.get(n); if(v.empty()) return nullptr;
            return (uint16_t*)dev_up(v.data(),v.size()); };
        auto nvl=[&](const char*fmt)->Nv{ snprintf(n,sizeof n,fmt,L); return nv_up(n); };

        w.kv_b     = bf("model.layers.%d.self_attn.kv_b_proj.weight");
        w.kv_a     = bf("model.layers.%d.self_attn.kv_a_proj_with_mqa.weight");
        w.in_norm  = bf("model.layers.%d.input_layernorm.weight");
        w.post_attn= bf("model.layers.%d.post_attention_layernorm.weight");
        w.pre_mlp  = bf("model.layers.%d.pre_mlp_layernorm.weight");
        w.post_mlp = bf("model.layers.%d.post_mlp_layernorm.weight");
        w.block_post = bf_opt("model.layers.%d.block_post_layernorm.weight");
        w.aphi = bf("model.layers.%d.attn_mhc_module.phi.weight");
        w.agam = bf("model.layers.%d.attn_mhc_module.norm_gamma");
        w.aalp = bf("model.layers.%d.attn_mhc_module.branch_alpha");
        w.abet = bf("model.layers.%d.attn_mhc_module.branch_beta");
        w.mphi = bf("model.layers.%d.mlp_mhc_module.phi.weight");
        w.mgam = bf("model.layers.%d.mlp_mhc_module.norm_gamma");
        w.malp = bf("model.layers.%d.mlp_mhc_module.branch_alpha");
        w.mbet = bf("model.layers.%d.mlp_mhc_module.branch_beta");
        w.q_norm  = bf("model.layers.%d.self_attn.q_a_layernorm.weight");
        w.kv_norm = bf("model.layers.%d.self_attn.kv_a_layernorm.weight");
        w.qa_conv = bf("model.layers.%d.self_attn.qa_conv.weight");
        w.kv_conv = bf("model.layers.%d.self_attn.compresskv_conv.weight");
        w.o_conv  = bf("model.layers.%d.self_attn.o_conv.weight");
        w.sink_rope = bf("model.layers.%d.self_attn.param_sink_k_pe");
        w.qa = nvl("model.layers.%d.self_attn.q_a_proj.weight");
        w.qb = nvl("model.layers.%d.self_attn.q_b_proj.weight");
        w.o  = nvl("model.layers.%d.self_attn.o_proj.weight");
        if(w.dense){
            w.dg = nvl("model.layers.%d.mlp.gate_proj.weight");
            w.du = nvl("model.layers.%d.mlp.up_proj.weight");
            w.dd = nvl("model.layers.%d.mlp.down_proj.weight");
        } else {
            w.router_w = bf("model.layers.%d.mlp.gate.weight");
            snprintf(n,sizeof n,"model.layers.%d.mlp.e_score_correction_bias",L);
            auto v=ck.get(n); if(v.empty()){ printf("MISSING %s\n",n); exit(1);}
            w.router_bias=(float*)dev_up(v.data(),v.size());
            const int li = L - (lb(d)<2?2:lb(d));
            w.arena_slot_base = li * arena[d].h.experts;
        }
        if(w.dsa){
            w.ix_wk  = nvl("model.layers.%d.self_attn.indexer.wk.weight");
            w.ix_wqb = nvl("model.layers.%d.self_attn.indexer.wq_b.weight");
            w.ix_knorm = bf("model.layers.%d.self_attn.indexer.k_norm.weight");
            w.ix_wproj = bf("model.layers.%d.self_attn.indexer.weights_proj.weight");
        }
        // sinks: RMSNorm the 128 compressed-kv sinks once, at load
        snprintf(n,sizeof n,"model.layers.%d.self_attn.param_sink_compressed_kv",L);
        auto sk=ck.get(n); if(sk.empty()){ printf("MISSING %s\n",n); exit(1);}
        { uint16_t*raw=(uint16_t*)dev_up(sk.data(),sk.size());
          CK(hipMalloc(&w.sink_kv_n,(size_t)SINKS*KVLORA*2));
          float *tmp,*tn; CK(hipMalloc(&tmp,KVLORA*4)); CK(hipMalloc(&tn,KVLORA*4));
          for(int s=0;s<SINKS;s++){
            hipLaunchKernelGGL(k_p92_from_bf16,dim3((KVLORA+255)/256),dim3(256),0,0,
                               raw+(size_t)s*KVLORA,tmp,KVLORA);
            hipLaunchKernelGGL((k_p92_rmsnorm<512>),dim3(1),dim3(256),0,0,tmp,(const p92_pbf*)w.kv_norm,tn,1e-5f);
            hipLaunchKernelGGL(k_p92_to_bf16,dim3((KVLORA+255)/256),dim3(256),0,0,
                               tn,w.sink_kv_n+(size_t)s*KVLORA,KVLORA); }
          CK(hipDeviceSynchronize()); hipFree(raw); hipFree(tmp); hipFree(tn); }
        // state
        w.cap = w.dsa ? MAXPOS : 512;
        CK(hipMalloc(&w.cache_kv,(size_t)w.cap*KVLORA*2));
        CK(hipMalloc(&w.cache_rope,(size_t)w.cap*ROPE*2));
        CK(hipMalloc(&w.mome_q,QLORA*2*2)); CK(hipMemset(w.mome_q,0,QLORA*2*2));
        CK(hipMalloc(&w.mome_kv,KVLORA*2*2)); CK(hipMemset(w.mome_kv,0,KVLORA*2*2));
        CK(hipMalloc(&w.mome_o,AOUT*2*2)); CK(hipMemset(w.mome_o,0,AOUT*2*2));
        if(w.dsa){ CK(hipMalloc(&w.ix_cache,(size_t)P92_IDX_DIM*w.cap*2)); }
        if(L%8==0||L==NLAYER-1) printf("  layer %2d on dev%d %s%s loaded\n",L,d,
                                       w.dense?"dense ":"moe ",w.dsa?"dsa":"swa");
    }
    printf("all %d layers resident\n",NLAYER);

    // ---- globals ---------------------------------------------------------
    CK(hipSetDevice(3));
    auto gnorm = ck.get("model.norm.weight");
    uint16_t *d_gnorm=(uint16_t*)dev_up(gnorm.data(),gnorm.size());
    auto mgphi=ck.get("model.merge_mhc_module.phi.weight");
    auto mggam=ck.get("model.merge_mhc_module.norm_gamma");
    auto mgalp=ck.get("model.merge_mhc_module.branch_alpha_pre");
    auto mgbet=ck.get("model.merge_mhc_module.branch_beta_pre");
    uint16_t *d_mgphi=(uint16_t*)dev_up(mgphi.data(),mgphi.size());
    uint16_t *d_mggam=(uint16_t*)dev_up(mggam.data(),mggam.size());
    uint16_t *d_mgalp=(uint16_t*)dev_up(mgalp.data(),mgalp.size());
    uint16_t *d_mgbet=(uint16_t*)dev_up(mgbet.data(),mgbet.size());
    Nv head = nv_up("lm_head.weight");
    printf("head %dx%d, merge phi %zu B, final norm loaded\n",head.rows,head.cols,mgphi.size());

    P92Transport tx;
    if(p92_transport_init(&tx,0)){ printf("transport init failed\n"); return 1; }

    printf("\nready. generating %d tokens from %d\n\n", NTOK, START);

    // ---- per-device scratch ---------------------------------------------
    struct Sc {
        uint16_t *sa,*sb,*col,*qa_bf,*kvd_bf,*aout_bf,*norm_bf,*fin;
        float *mixf,*norm,*pre,*post,*comb,*part,*ssq;
        float *qlora,*qnorm,*qup,*kvdn,*curkv,*qabs,*qrope,*cosv,*sinv;
        float *ctx,*pacc,*pm,*pl,*aout,*hidden,*hid9,*hs9;
        float *xs2560,*xs1024,*xs6144,*dhid,*dhs;
        int8_t *xq2560,*xq1024,*xq6144,*hq9,*dhq;
        float *rlog,*rwts,*dw,*kidx,*qidx,*wh,*scores,*flat,*flat2;
        int *rids,*slot,*sel; uint32_t *ist,*ihist,*icnt,*ioff;
    } sc[4];
    for(int d=0;d<4;d++){ CK(hipSetDevice(d)); Sc&c=sc[d];
        auto A=[&](void**p,size_t n){ CK(hipMalloc(p,n)); };
        A((void**)&c.sa,4*HID*2); A((void**)&c.sb,4*HID*2); A((void**)&c.col,HID*2);
        A((void**)&c.qa_bf,QLORA*2); A((void**)&c.kvd_bf,KVA*2); A((void**)&c.aout_bf,AOUT*2);
        A((void**)&c.norm_bf,HID*2); A((void**)&c.fin,HID*2);
        A((void**)&c.mixf,HID*4); A((void**)&c.norm,HID*4);
        A((void**)&c.pre,4*4); A((void**)&c.post,4*4); A((void**)&c.comb,16*4);
        A((void**)&c.part,24*P92_MHC_SPLIT*4); A((void**)&c.ssq,P92_MHC_SPLIT*4);
        A((void**)&c.qlora,QLORA*4); A((void**)&c.qnorm,QLORA*4); A((void**)&c.qup,QUP*4);
        A((void**)&c.kvdn,KVA*4); A((void**)&c.curkv,KVLORA*4);
        A((void**)&c.qabs,(size_t)48*KVLORA*4); A((void**)&c.qrope,(size_t)48*ROPE*4);
        A((void**)&c.cosv,32*4); A((void**)&c.sinv,32*4);
        A((void**)&c.ctx,(size_t)48*KVLORA*4);
        A((void**)&c.pacc,(size_t)48*P92_SPLITS*KVLORA*4);
        A((void**)&c.pm,48*P92_SPLITS*4); A((void**)&c.pl,48*P92_SPLITS*4);
        A((void**)&c.aout,AOUT*4); A((void**)&c.hidden,HID*4);
        A((void**)&c.hid9,(size_t)P92_DISPATCH*P92_NFF*4);
        A((void**)&c.hq9,(size_t)P92_DISPATCH*P92_NFF); A((void**)&c.hs9,P92_DISPATCH*4);
        A((void**)&c.xq2560,HID); A((void**)&c.xs2560,(HID/16)*4);
        A((void**)&c.xq1024,QLORA); A((void**)&c.xs1024,(QLORA/16)*4);
        A((void**)&c.xq6144,AOUT); A((void**)&c.xs6144,(AOUT/16)*4);
        A((void**)&c.dhid,P92_DENSE_FF*4); A((void**)&c.dhq,P92_DENSE_FF); A((void**)&c.dhs,4);
        A((void**)&c.rlog,NEXP*4); A((void**)&c.rwts,8*4); A((void**)&c.rids,8*4);
        A((void**)&c.slot,P92_DISPATCH*4); A((void**)&c.dw,P92_DISPATCH*4);
        A((void**)&c.kidx,P92_IDX_DIM*4); A((void**)&c.qidx,3072*4); A((void**)&c.wh,24*4);
        A((void**)&c.scores,(size_t)MAXPOS*4); A((void**)&c.sel,(size_t)(MAXPOS>2048?MAXPOS:2048)*4);
        A((void**)&c.flat,4*HID*4); A((void**)&c.flat2,4*HID*4);
        A((void**)&c.ist,4*4); A((void**)&c.ihist,P92_RBINS*4);
        A((void**)&c.icnt,2*P92_GMAXC*4); A((void**)&c.ioff,(2*P92_GMAXC+2)*4);
    }
    // head scratch on dev3
    CK(hipSetDevice(3));
    float *hlog,*hpv,*hxs,*hxs2; int *hpi,*hcand,*hout; int8_t *hxq,*hxq2;
    const int HCH=(P92_VOCAB+P92_HEAD_RED-1)/P92_HEAD_RED;
    const int HNC=(P92_VOCAB+P92_GCHUNK-1)/P92_GCHUNK;
    CK(hipMalloc(&hlog,(size_t)P92_VOCAB*4)); CK(hipMalloc(&hpv,(size_t)HCH*4));
    CK(hipMalloc(&hpi,(size_t)HCH*4)); CK(hipMalloc(&hout,4));
    CK(hipMalloc(&hcand,P92_HEAD_REFINE*4));
    CK(hipMalloc(&hxq,HID)); CK(hipMalloc(&hxq2,HID));
    CK(hipMalloc(&hxs,4)); CK(hipMalloc(&hxs2,4));
    uint32_t *hst,*hhist,*hcnt,*hoff;
    CK(hipMalloc(&hst,4*4)); CK(hipMalloc(&hhist,P92_RBINS*4));
    CK(hipMalloc(&hcnt,2*P92_GMAXC*4)); CK(hipMalloc(&hoff,(2*P92_GMAXC+2)*4));

    // ---- one layer -------------------------------------------------------
    auto run_layer=[&](int L,int position)->void{
        const int d=owner(L); LayerW&w=LW[L]; Sc&c=sc[d];
        const int slotpos = w.dsa ? position : (position % w.cap);
        const int npos = position + 1;

        // attention mHC pre
        hipLaunchKernelGGL(k_p92_mhc_logits,dim3(P92_MHC_MIX,P92_MHC_SPLIT),dim3(256),0,0,
                           (const p92_mbf*)c.sa,(const p92_mbf*)w.aphi,(const p92_mbf*)w.agam,c.part,c.ssq);
        hipLaunchKernelGGL(k_p92_mhc_collapse,dim3(1),dim3(256),0,0,c.part,c.ssq,
                           (const p92_mbf*)w.aalp,(const p92_mbf*)w.abet,(const p92_mbf*)c.sa,
                           c.pre,c.post,c.comb,c.mixf);
        hipLaunchKernelGGL((k_p92_rmsnorm_quant<2560>),dim3(1),dim3(256),0,0,
                           c.mixf,(const p92_pbf*)w.in_norm,c.norm,c.xq2560,c.xs2560,1e-5f);

        // q path
        hipLaunchKernelGGL((k_p92_proj<2560,32>),dim3(QLORA/8),dim3(256),0,0,
                           w.qa.w,w.qa.s,c.xq2560,c.xs2560,w.qa.s2,c.qlora);
        hipLaunchKernelGGL(k_p92_mome_conv3_f32<P92_MOME_QA>,dim3((QLORA+255)/256),dim3(256),0,0,
                           c.qlora,(const p92_cbf*)w.qa_conv,(p92_cbf*)w.mome_q);
        hipLaunchKernelGGL((k_p92_rmsnorm_quant<1024>),dim3(1),dim3(256),0,0,
                           c.qlora,(const p92_pbf*)w.q_norm,c.qnorm,c.xq1024,c.xs1024,1e-5f);
        hipLaunchKernelGGL((k_p92_proj<1024,32>),dim3(QUP/8),dim3(256),0,0,
                           w.qb.w,w.qb.s,c.xq1024,c.xs1024,w.qb.s2,c.qup);

        // kv path
        hipLaunchKernelGGL((k_p92_bf16_gemv<2560>),dim3((KVA+3)/4),dim3(256),0,0,
                           (const p92_pbf*)w.kv_a,c.norm,c.kvdn);
        hipLaunchKernelGGL(k_p92_mome_conv3_f32<P92_MOME_KV>,dim3((KVLORA+255)/256),dim3(256),0,0,
                           c.kvdn,(const p92_cbf*)w.kv_conv,(p92_cbf*)w.mome_kv);
        hipLaunchKernelGGL((k_p92_rmsnorm<512>),dim3(1),dim3(256),0,0,c.kvdn,(const p92_pbf*)w.kv_norm,c.curkv,1e-5f);
        hipLaunchKernelGGL(k_p92_to_bf16,dim3((KVLORA+255)/256),dim3(256),0,0,
                           c.curkv,w.cache_kv+(size_t)slotpos*KVLORA,KVLORA);
        hipLaunchKernelGGL(k_p92_rope_table,dim3(1),dim3(64),0,0,position,c.cosv,c.sinv);
        hipLaunchKernelGGL(k_p92_key_rope,dim3(1),dim3(64),0,0,c.kvdn,c.cosv,c.sinv,
                           (p92_mbf16*)(w.cache_rope+(size_t)slotpos*ROPE));
        hipLaunchKernelGGL(k_p92_mla_absorb,dim3(48,2),dim3(256),0,0,c.qup,(const p92_mbf16*)w.kv_b,c.qabs);
        hipLaunchKernelGGL(k_p92_query_rope,dim3(48),dim3(64),0,0,c.qup,c.cosv,c.sinv,c.qrope);

        // attention body
        // The index cache must be populated at EVERY position on a DSA layer.
        // This projection and the cache write used to live inside the
        // npos>P92_IDX_TOPK branch below, so rows 0..2047 were NEVER written and
        // the selector scored raw hipMalloc contents the first time npos reached
        // 2049 - silently, since zeroed VRAM makes every score equal, the tie
        // band swallows the array, and the ascending gather returns the OLDEST
        // 2048 positions. The oracle stores the key unconditionally
        // (model.cpp:1637 decode, :1411 prefill); its kDsaTopK test at :1416
        // selects the attention path, not whether the key is cached.
        if(w.dsa){
            hipLaunchKernelGGL((k_p92_proj<2560,32>),dim3((128+7)/8),dim3(256),0,0,
                               w.ix_wk.w,w.ix_wk.s,c.xq2560,c.xs2560,w.ix_wk.s2,c.kidx);
            hipLaunchKernelGGL(k_p92_ix_key,dim3(1),dim3(128),0,0,c.kidx,
                               (const p92_pbf*)w.ix_knorm,c.cosv,c.sinv,
                               (p92_bf*)w.ix_cache,w.cap,slotpos,1e-5f);
        }
        int nsel; int win_first=0, win_cap=0;
        if(w.dsa && npos>P92_IDX_TOPK){
            hipLaunchKernelGGL((k_p92_proj<1024,32>),dim3(3072/8),dim3(256),0,0,
                               w.ix_wqb.w,w.ix_wqb.s,c.xq1024,c.xs1024,w.ix_wqb.s2,c.qidx);
            hipLaunchKernelGGL(k_p92_ix_query,dim3(24),dim3(128),0,0,c.qidx,c.cosv,c.sinv);
            hipLaunchKernelGGL((k_p92_bf16_gemv<2560>),dim3((24+3)/4),dim3(256),0,0,
                               (const p92_pbf*)w.ix_wproj,c.norm,c.wh);
            hipLaunchKernelGGL(k_p92_idx_scores,dim3((npos+255)/256),dim3(256),0,0,
                               c.qidx,(const p92_bf*)w.ix_cache,c.wh,w.cap,npos,
                               1.0f/sqrtf((float)P92_IDX_DIM),c.scores);
            if(npos<=16384)
                hipLaunchKernelGGL(k_p92_idx_select,dim3(1),dim3(P92_SEL_THREADS),0,0,
                                   c.scores,npos,P92_IDX_TOPK,c.sel);
            else {
                (void)hipMemsetAsync(c.ist,0,4*4,0);
                for(int sh2=0;sh2<3;sh2++){ const int shv=(sh2==0?21:(sh2==1?10:0));
                    (void)hipMemsetAsync(c.ihist,0,P92_RBINS*4,0);
                    hipLaunchKernelGGL(k_p92_idx_hist,dim3(64),dim3(256),0,0,c.scores,npos,shv,c.ist,c.ihist);
                    hipLaunchKernelGGL(k_p92_idx_pick,dim3(1),dim3(1024),0,0,c.ihist,P92_IDX_TOPK,shv,c.ist); }
                const int nc=(npos+P92_GCHUNK-1)/P92_GCHUNK;
                hipLaunchKernelGGL(k_p92_idx_count,dim3(nc),dim3(256),0,0,c.scores,npos,c.ist,c.icnt);
                hipLaunchKernelGGL(k_p92_idx_scan,dim3(1),dim3(256),0,0,c.icnt,nc,P92_IDX_TOPK,c.ioff);
                hipLaunchKernelGGL(k_p92_idx_write,dim3(nc),dim3(256),0,0,c.scores,npos,c.ist,c.ioff,P92_IDX_TOPK,c.sel);
                hipLaunchKernelGGL(k_p92_idx_pad,dim3(8),dim3(256),0,0,c.ioff,P92_IDX_TOPK,c.sel); }
            nsel=P92_IDX_TOPK;
        } else if(w.dsa){
            nsel=npos; win_first=0; win_cap=w.cap;
        } else {
            nsel = npos<512?npos:512; win_first=npos-nsel; win_cap=w.cap;
        }
        hipLaunchKernelGGL(k_p92_mla_g16,dim3(48,P92_SPLITS/4),dim3(256),0,0,
                           c.qabs,c.qrope,(const p92_bf*)w.cache_kv,(const p92_bf*)w.cache_rope,
                           (const p92_bf*)w.sink_kv_n,(const p92_bf*)w.sink_rope,
                           win_cap?nullptr:c.sel,nsel,SINKS,win_first,win_cap,c.pacc,c.pm,c.pl);
        hipLaunchKernelGGL(k_p92_mla_combine,dim3(48),dim3(256),0,0,c.pacc,c.pm,c.pl,c.ctx);
        hipLaunchKernelGGL(k_p92_mla_value_up,dim3((48*128+3)/4),dim3(256),0,0,
                           c.ctx,(const p92_mbf16*)w.kv_b,c.aout);
        hipLaunchKernelGGL(k_p92_mome_conv3_f32<P92_MOME_O>,dim3((AOUT+255)/256),dim3(256),0,0,
                           c.aout,(const p92_cbf*)w.o_conv,(p92_cbf*)w.mome_o);
        hipLaunchKernelGGL((k_p92_quant<6144>),dim3(1),dim3(256),0,0,c.aout,c.xq6144,c.xs6144);
        hipLaunchKernelGGL((k_p92_proj<6144,64>),dim3(HID/4),dim3(256),0,0,
                           w.o.w,w.o.s,c.xq6144,c.xs6144,w.o.s2,c.hidden);

        // attention mHC post -> sb
        hipLaunchKernelGGL((k_p92_rmsnorm<2560>),dim3(1),dim3(256),0,0,c.hidden,(const p92_pbf*)w.post_attn,c.norm,1e-5f);
        hipLaunchKernelGGL(k_p92_mhc_reexpand,dim3((HID+255)/256),dim3(256),0,0,
                           c.post,c.comb,c.norm,(const p92_mbf*)c.sa,(p92_mbf*)c.sb);

        // MLP mHC pre on sb
        hipLaunchKernelGGL(k_p92_mhc_logits,dim3(P92_MHC_MIX,P92_MHC_SPLIT),dim3(256),0,0,
                           (const p92_mbf*)c.sb,(const p92_mbf*)w.mphi,(const p92_mbf*)w.mgam,c.part,c.ssq);
        hipLaunchKernelGGL(k_p92_mhc_collapse,dim3(1),dim3(256),0,0,c.part,c.ssq,
                           (const p92_mbf*)w.malp,(const p92_mbf*)w.mbet,(const p92_mbf*)c.sb,
                           c.pre,c.post,c.comb,c.mixf);
        hipLaunchKernelGGL((k_p92_rmsnorm_quant<2560>),dim3(1),dim3(256),0,0,
                           c.mixf,(const p92_pbf*)w.pre_mlp,c.norm,c.xq2560,c.xs2560,1e-5f);

        if(w.dense){
            hipLaunchKernelGGL(k_p92_dense_gateup,dim3(P92_DENSE_FF/8),dim3(256),0,0,
                               w.dg.w,w.dg.s,w.du.w,w.du.s,c.xq2560,c.xs2560,w.dg.s2,w.du.s2,c.dhid);
            hipLaunchKernelGGL(k_p92_dense_requant,dim3(1),dim3(1024),0,0,c.dhid,c.dhq,c.dhs);
            hipLaunchKernelGGL(k_p92_dense_down,dim3(HID/4),dim3(256),0,0,
                               w.dd.w,w.dd.s,c.dhq,c.dhs,w.dd.s2,c.hidden);
        } else {
            hipLaunchKernelGGL(k_p92_route_gemv,dim3(NEXP/4),dim3(256),0,0,
                               (const uint16_t*)w.router_w,c.norm,c.rlog);
            hipLaunchKernelGGL(k_p92_route_top8,dim3(1),dim3(64),0,0,c.rlog,w.router_bias,c.rids,c.rwts);
            hipLaunchKernelGGL(k_p92_build_dispatch,dim3(1),dim3(16),0,0,
                               c.rids,c.rwts,w.arena_slot_base,arena[d].h.experts,c.slot,c.dw);
            hipLaunchKernelGGL(k_p92_f4_gateup,dim3(P92_NFF/8,P92_DISPATCH),dim3(256),0,0,
                               arena[d].gw,arena[d].gs,arena[d].uw,arena[d].us,
                               c.xq2560,c.xs2560,c.slot,arena[d].s2,c.hid9);
            hipLaunchKernelGGL(k_p92_f4_requant,dim3(P92_DISPATCH),dim3(256),0,0,c.hid9,c.slot,c.hq9,c.hs9);
            hipLaunchKernelGGL(k_p92_f4_downacc,dim3(320),dim3(256),0,0,
                               arena[d].dw,arena[d].ds,c.hq9,c.hs9,c.dw,c.slot,arena[d].s2,c.hidden);
            if(getenv("P92_MOE") && L==2 && position==0){
                (void)hipDeviceSynchronize();
                std::vector<float> nv(HID), h9((size_t)P92_DISPATCH*P92_NFF), hs(P92_DISPATCH), ww(P92_DISPATCH), s2v(3);
                std::vector<int> sl(P92_DISPATCH), ids(8);
                (void)hipMemcpy(nv.data(),c.norm,HID*4,hipMemcpyDeviceToHost);
                (void)hipMemcpy(h9.data(),c.hid9,h9.size()*4,hipMemcpyDeviceToHost);
                (void)hipMemcpy(hs.data(),c.hs9,P92_DISPATCH*4,hipMemcpyDeviceToHost);
                (void)hipMemcpy(ww.data(),c.dw,P92_DISPATCH*4,hipMemcpyDeviceToHost);
                (void)hipMemcpy(sl.data(),c.slot,P92_DISPATCH*4,hipMemcpyDeviceToHost);
                (void)hipMemcpy(ids.data(),c.rids,8*4,hipMemcpyDeviceToHost);
                (void)hipMemcpy(s2v.data(),arena[d].s2+(size_t)sl[0]*3,3*4,hipMemcpyDeviceToHost);
                std::vector<float> hh(HID); (void)hipMemcpy(hh.data(),c.hidden,HID*4,hipMemcpyDeviceToHost);
                auto rms=[](const std::vector<float>&v,size_t n){ double a=0; for(size_t i=0;i<n;i++) a+=(double)v[i]*v[i]; return sqrt(a/n); };
                printf("      MOE L2: norm rms %.5g | ids", rms(nv,HID));
                for(int k2=0;k2<8;k2++) printf(" %d",ids[k2]);
                printf(" | wts"); for(int k2=0;k2<P92_DISPATCH;k2++) printf(" %.3f",ww[k2]);
                printf("\n      slot0 %d s2 gate %.4g up %.4g down %.4g\n",sl[0],s2v[0],s2v[1],s2v[2]);
                printf("      gate/up+silu rms(expert0) %.5g | requant scales",rms(h9,P92_NFF));
                for(int k2=0;k2<P92_DISPATCH;k2++) printf(" %.4g",hs[k2]);
                printf("\n      moe out rms %.5g\n",rms(hh,HID));
            }
        }

        // MLP mHC post -> sa
        hipLaunchKernelGGL((k_p92_rmsnorm<2560>),dim3(1),dim3(256),0,0,c.hidden,(const p92_pbf*)w.post_mlp,c.norm,1e-5f);
        hipLaunchKernelGGL(k_p92_mhc_reexpand,dim3((HID+255)/256),dim3(256),0,0,
                           c.post,c.comb,c.norm,(const p92_mbf*)c.sb,(p92_mbf*)c.sa);
        // block_post_layernorm, on the layers listed in block_post_layernorm_idx,
        // normalises the whole four-stream block (kFlat = 10240), not one stream
        // (model.cpp:222-224).
        if(w.block_post){
            hipLaunchKernelGGL(k_p92_from_bf16,dim3((4*HID+255)/256),dim3(256),0,0,c.sa,c.flat,4*HID);
            hipLaunchKernelGGL((k_p92_rmsnorm<10240>),dim3(1),dim3(256),0,0,
                               c.flat,(const p92_pbf*)w.block_post,c.flat2,1e-5f);
            hipLaunchKernelGGL(k_p92_to_bf16,dim3((4*HID+255)/256),dim3(256),0,0,c.flat2,c.sa,4*HID);
        }
    };

    // ---- generation ------------------------------------------------------
    if(!ck.embed_open()){ printf("embed_open failed\n"); return 1; }   // outside the timed region
    std::vector<int> prompt;
    bool text_mode=false;
    p92::Tokenizer* tokp=nullptr;
    if(argc>7 && std::string(argv[7])=="-"){
        text_mode=true;
        static p92::Tokenizer tk(std::string(CKDIR)+"/tokenizer.json");
        tokp=&tk;
        std::string line, all;
        while(std::getline(std::cin,line)){ all+=line; all+="\n"; }
        const std::string tp=p92::Tokenizer::chat_prompt(all);
        for(int32_t t : tk.encode(tp)) prompt.push_back((int)t);
        fflush(stdout);
    } else if(argc>7){
        int pf=::open(argv[7],O_RDONLY);
        if(pf<0){ printf("no prompt file %s\n",argv[8]); return 1; }
        struct stat stt; if(fstat(pf,&stt)){ printf("prompt stat failed\n"); return 1; }
        std::vector<int64_t> pr((size_t)stt.st_size/8);
        if(!pr.empty()) rdf(pf,pr.data(),pr.size()*8,0);
        ::close(pf);
        for(int64_t t:pr) prompt.push_back((int)t);
        printf("prompt tokens: %zu\n",prompt.size());
    }
    std::vector<int> out_tokens;
    int token=START;
    double first_ms=0, total_ms=0;
    for(int step=0; step<NTOK; step++){
        auto t0=std::chrono::steady_clock::now();
        const uint16_t *emb=ck.embed_ptr(token);      // resident, no syscall
        CK(hipSetDevice(0));
        std::vector<uint16_t> st((size_t)4*HID);
        for(int k2=0;k2<4;k2++) memcpy(&st[(size_t)k2*HID],emb,HID*2);
        CK(hipMemcpy(sc[0].sa,st.data(),st.size()*2,hipMemcpyHostToDevice));
        int cur=0;
        for(int L=0;L<NLAYER;L++){
            const int d=owner(L);
            if(d!=cur){
                CK(hipSetDevice(cur)); CK(hipDeviceSynchronize());
                if(p92_transport_send(&tx,cur,sc[cur].sa)) exit(1);
                if(p92_transport_recv(&tx,cur,sc[d].sa)) exit(1);
                if(p92_transport_wait(&tx,d)) exit(1);
                cur=d;
            }
            CK(hipSetDevice(d));
            run_layer(L,step);
            if(getenv("P92_TRACE") && step==0){
                CK(hipDeviceSynchronize());
                std::vector<uint16_t> t(4*HID);
                CK(hipMemcpy(t.data(),sc[d].sa,4*HID*2,hipMemcpyDeviceToHost));
                double ss=0,mx=0; int nf=0;
                for(size_t i=0;i<t.size();i++){ uint32_t u=(uint32_t)t[i]<<16; float v; memcpy(&v,&u,4);
                    if(!std::isfinite(v)) nf++; ss+=(double)v*v; mx=fmax(mx,fabs((double)v)); }
                std::vector<float> hh(HID);
                CK(hipMemcpy(hh.data(),sc[d].hidden,HID*4,hipMemcpyDeviceToHost));
                double hs2=0; for(float v:hh) hs2+=(double)v*v;
                printf("    L%-2d dev%d %-5s %-3s  streams rms %10.4g max %10.4g nonfinite %d | mlp-out rms %10.4g\n",
                       L,d,LW[L].dense?"dense":"moe",LW[L].dsa?"dsa":"swa",
                       sqrt(ss/t.size()),mx,nf,sqrt(hs2/HID));
            }
        }
        CK(hipSetDevice(3)); CK(hipDeviceSynchronize());
        Sc&c3=sc[3];
        hipLaunchKernelGGL(k_p92_mhc_logits,dim3(4,P92_MHC_SPLIT),dim3(256),0,0,
                           (const p92_mbf*)c3.sa,(const p92_mbf*)d_mgphi,(const p92_mbf*)d_mggam,c3.part,c3.ssq);
        hipLaunchKernelGGL(k_p92_merge_collapse,dim3(1),dim3(256),0,0,c3.part,c3.ssq,
                           (const p92_mbf*)d_mgalp,(const p92_mbf*)d_mgbet,(const p92_mbf*)c3.sa,(p92_mbf*)c3.fin);
        hipLaunchKernelGGL(k_p92_from_bf16,dim3((HID+255)/256),dim3(256),0,0,c3.fin,c3.mixf,HID);
        hipLaunchKernelGGL((k_p92_rmsnorm<2560>),dim3(1),dim3(256),0,0,c3.mixf,(const p92_pbf*)d_gnorm,c3.norm,1e-5f);
        // head: two int8 activation components, fast pass then top-256 refine
        {
            std::vector<float> hx(HID); CK(hipDeviceSynchronize());
            CK(hipMemcpy(hx.data(),c3.norm,HID*4,hipMemcpyDeviceToHost));
            float m=0; for(float v:hx) m=fmaxf(m,fabsf(v));
            const float s1=m>0?m/127.f:1.f;
            std::vector<int8_t> q1(HID),q2(HID); std::vector<float> res(HID);
            for(int i=0;i<HID;i++){ int q=(int)rintf(hx[i]/s1); q=q<-128?-128:(q>127?127:q);
                q1[i]=(int8_t)q; res[i]=hx[i]-q*s1; }
            float rm=0; for(float v:res) rm=fmaxf(rm,fabsf(v));
            const float s2=rm>0?rm/127.f:1.f;
            for(int i=0;i<HID;i++){ int q=(int)rintf(res[i]/s2); q2[i]=(int8_t)(q<-128?-128:(q>127?127:q)); }
            CK(hipMemcpy(hxq,q1.data(),HID,hipMemcpyHostToDevice));
            CK(hipMemcpy(hxq2,q2.data(),HID,hipMemcpyHostToDevice));
            CK(hipMemcpy(hxs,&s1,4,hipMemcpyHostToDevice));
            CK(hipMemcpy(hxs2,&s2,4,hipMemcpyHostToDevice));
        }
        hipLaunchKernelGGL(k_p92_head_logits,dim3((P92_VOCAB+7)/8),dim3(256),0,0,
                           head.w,head.s,hxq,hxs,head.s2,hlog);
        (void)hipMemsetAsync(hst,0,4*4,0);
        for(int sh2=0;sh2<3;sh2++){ const int shv=(sh2==0?21:(sh2==1?10:0));
            (void)hipMemsetAsync(hhist,0,P92_RBINS*4,0);
            hipLaunchKernelGGL(k_p92_idx_hist,dim3(64),dim3(256),0,0,hlog,P92_VOCAB,shv,hst,hhist);
            hipLaunchKernelGGL(k_p92_idx_pick,dim3(1),dim3(1024),0,0,hhist,P92_HEAD_REFINE,shv,hst); }
        hipLaunchKernelGGL(k_p92_idx_count,dim3(HNC),dim3(256),0,0,hlog,P92_VOCAB,hst,hcnt);
        hipLaunchKernelGGL(k_p92_idx_scan,dim3(1),dim3(256),0,0,hcnt,HNC,P92_HEAD_REFINE,hoff);
        hipLaunchKernelGGL(k_p92_idx_write,dim3(HNC),dim3(256),0,0,hlog,P92_VOCAB,hst,hoff,P92_HEAD_REFINE,hcand);
        hipLaunchKernelGGL(k_p92_idx_pad,dim3(8),dim3(256),0,0,hoff,P92_HEAD_REFINE,hcand);
        hipLaunchKernelGGL(k_p92_head_refine,dim3(P92_HEAD_REFINE/P92_HEAD_PER_BLOCK),dim3(256),0,0,
                           head.w,head.s,hxq,hxq2,hxs,hxs2,head.s2,hcand,P92_HEAD_REFINE,hlog);
        hipLaunchKernelGGL(k_p92_head_argmax_part,dim3(HCH),dim3(256),0,0,hlog,hpv,hpi);
        hipLaunchKernelGGL(k_p92_head_argmax_final,dim3(1),dim3(256),0,0,hpv,hpi,HCH,hout);
        CK(hipDeviceSynchronize());
        int next; CK(hipMemcpy(&next,hout,4,hipMemcpyDeviceToHost));
        // ---- host sampling: temperature / top-k / top-p over the full logits ----
        { const char* ST=getenv("P92_TEMP"); const char* SK=getenv("P92_TOPK");
          const char* SP=getenv("P92_TOPP"); const char* SS=getenv("P92_SEED");
          const float T=ST?atof(ST):0.6f; int K=SK?atoi(SK):50; const float PP=SP?atof(SP):0.95f;
          if(K<1)K=1; if(K>P92_VOCAB)K=P92_VOCAB;
          if(T>0.f){
            static std::vector<float> lg;
            if((int)lg.size()!=P92_VOCAB) lg.resize(P92_VOCAB);
            CK(hipMemcpy(lg.data(),hlog,(size_t)P92_VOCAB*4,hipMemcpyDeviceToHost));
            static std::vector<int> idx;
            if((int)idx.size()!=P92_VOCAB){ idx.resize(P92_VOCAB); for(int i=0;i<P92_VOCAB;i++) idx[i]=i; }
            std::partial_sort(idx.begin(),idx.begin()+K,idx.end(),
                              [&](int a,int b){return lg[a]>lg[b];});
            const float mx=lg[idx[0]];
            std::vector<float> pr(K); double sum=0;
            for(int i=0;i<K;i++){ pr[i]=expf((lg[idx[i]]-mx)/T); sum+=pr[i]; }
            for(int i=0;i<K;i++) pr[i]=(float)(pr[i]/sum);
            double acc=0; int K2=K;
            for(int i=0;i<K;i++){ acc+=pr[i]; if(acc>=PP){ K2=i+1; break; } }
            static std::mt19937 rng(SS?(unsigned)atoi(SS):1234u);
            const double r=std::uniform_real_distribution<double>(0.0,1.0)(rng);
            double c=0; int pick=idx[0];
            for(int i=0;i<K2;i++){ c+=pr[i]; if(r<=c){ pick=idx[i]; break; } }
            next=pick;
          } }
        auto t1=std::chrono::steady_clock::now();
        const double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
        if(step==0) first_ms=ms; else total_ms+=ms;
        if(text_mode && tokp && (size_t)step >= prompt.size()){
            const std::string piece = tokp->decode_token(next);
            fputs(piece.c_str(), stdout); fflush(stdout);
        } else if(!text_mode){
            printf("  step %2d  position %2d  %6d -> %6d   %8.2f ms\n",step,step,token,next,ms);
            printf("TOK %d\n", next);
        }
        fflush(stdout);
        out_tokens.push_back(next);
        token = ((size_t)(step+1) < prompt.size()) ? prompt[step+1] : next;
    }
    printf("\ntokens:"); for(int t:out_tokens) printf(" %d",t); printf("\n");
    if(NTOK>1) printf("first token %.2f ms, steady %.2f ms/token = %.2f tok/s\n",
                      first_ms,total_ms/(NTOK-1),1000.0*(NTOK-1)/total_ms);
    p92_transport_free(&tx);
    return 0;
}

