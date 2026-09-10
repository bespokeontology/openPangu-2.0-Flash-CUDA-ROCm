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
#include <thread>
#include <atomic>
#include <mutex>
#include "p92amd/p92_pf.h"
#include "p92amd/p92_pf_moe.h"
#include "p92amd/p92_pf_drive.h"
#include "p92amd/p92_p2p.h"
__global__ void k_pf_shuffle_scales_gu(const uint8_t*, uint8_t*, int);
__global__ void k_pf_shuffle_scales_dn(const uint8_t*, uint8_t*, int);
__global__ void k_pf_shuffle_check(const uint8_t*, const uint8_t*, int, int, int, uint32_t*);
#include "../src/p92_pf_moe.hip"
#include "../src/p92_pf_down.hip"
#include "../src/p92_pf_quant.hip"
#include "../src/p92_pf_proj.hip"
#include "../src/p92_pf_body.hip"
#include "../src/p92_pf_attn.hip"
#include "../src/p92_pf_mla.hip"
#define PF_UMAX (8 * PF_IDX_TOPK)
#include "../src/p92_pf_dsa.hip"
#include <chrono>
#include <thread>
#include <atomic>

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
        // The arena arrives in SPEC order from the decode loader, but
        // k_pf_gateup and k_pf_down read their scales LANE-CONTIGUOUS - the
        // layout k_pf_shuffle_scales_* produces. Nothing shuffled the
        // production arena, so every routed expert paired its nibbles with
        // another group's scale. The MoE gate could not see it: the gate
        // preshuffles its own upload, so it tested the consumer against a
        // layout the producer never emitted. Same defect as k_pf_proj's
        // weight scales, one buffer further along.
        {
            const size_t GU = (size_t)arena[d].slots * PF_NFF * (PF_NEMBD/16);
            const size_t DN = (size_t)arena[d].slots * PF_NEMBD * (PF_NFF/16);
            const int gurows = arena[d].slots * PF_NFF;
            const int dnrows = arena[d].slots * PF_NEMBD;
            uint8_t *tmp = nullptr; CK(hipMalloc(&tmp, GU > DN ? GU : DN));
            uint32_t *bad = nullptr; CK(hipMalloc(&bad, 4)); CK(hipMemset(bad, 0, 4));
            auto gu = [&](uint8_t *buf) {
                hipLaunchKernelGGL(k_pf_shuffle_scales_gu,dim3((gurows+7)/8),dim3(256),0,0,
                                   buf,tmp,gurows);
                hipLaunchKernelGGL(k_pf_shuffle_check,dim3((gurows+7)/8),dim3(256),0,0,
                                   buf,tmp,gurows,PF_NEMBD/16,5,bad);
                CK(hipMemcpy(buf,tmp,GU,hipMemcpyDeviceToDevice)); };
            gu(arena[d].gs); gu(arena[d].us);
            hipLaunchKernelGGL(k_pf_shuffle_scales_dn,dim3((dnrows+7)/8),dim3(256),0,0,
                               arena[d].ds,tmp,dnrows);
            hipLaunchKernelGGL(k_pf_shuffle_check,dim3((dnrows+7)/8),dim3(256),0,0,
                               arena[d].ds,tmp,dnrows,PF_NFF/16,2,bad);
            CK(hipMemcpy(arena[d].ds,tmp,DN,hipMemcpyDeviceToDevice));
            CK(hipDeviceSynchronize());
            uint32_t nbad = 0; CK(hipMemcpy(&nbad,bad,4,hipMemcpyDeviceToHost));
            if (nbad) { printf("arena %d scale shuffle is not a permutation: %u\n",d,nbad); return 1; }
            CK(hipFree(tmp)); CK(hipFree(bad));
        }
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
        w.cap = MAXPOS;   // absolute addressing; the window is a kernel bound
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
    // ================= PREFILL: T-row scratch and chunk pipeline ============
    // Loading above is reused verbatim from the decode harness. Everything from
    // here is prefill: the scratch is T rows wide, and the layer body calls the
    // prefill kernels rather than the M=1 ones.
    const PfConfig cfg = pf_config();
    const int T = cfg.chunk;
    printf("prefill: chunk %d, TQ %d, P%d/D4\n", T, cfg.attn_tq, cfg.prefill_cards);

    struct Pf {
        uint16_t *state, *flat;          // [T][4][HID] bf16, [T][4*HID] bf16
        float *norm, *mixf, *comb, *logits;
        int8_t *xq; float *xs;
        float *qlora, *qnorm, *qup, *kvdn, *curkv, *qabs, *qrope, *ctx, *aout, *hidden;
        float *rlog; int32_t *rids; float *rwts;
        PfGroup *groups; int32_t *rows, *order, *ngroups; float *rwt;
        float *hid9, *hs9; int8_t *hq9;
        float *cosv, *sinv;
        uint16_t *state2; float *pre, *post;
        float *kidx, *qidx, *qidxT, *whd, *scores; int32_t *sel, *nsel;
        int32_t *uni, *ucnt; uint8_t *umask;
        uint16_t *lat_pack, *pe_pack; uint8_t *mask_pack; int32_t *packoff;
        int8_t *xq1024, *xq6144, *dhq; float *xs1024, *xs6144, *dhid, *dhs;
    } pf[4];
    const size_t GMAX = (size_t)T * (PF_TOPK + 1);
    for (int d = 0; d < 4; d++) {
        CK(hipSetDevice(d)); Pf &c = pf[d];
        auto A = [&](void **p, size_t n) { CK(hipMalloc(p, n)); };
        A((void**)&c.state,(size_t)T*4*HID*2);   A((void**)&c.flat,(size_t)T*4*HID*2);
        A((void**)&c.norm,(size_t)T*HID*4);      A((void**)&c.mixf,(size_t)T*HID*4);
        A((void**)&c.comb,(size_t)T*16*4);       A((void**)&c.logits,(size_t)T*24*4);
        A((void**)&c.xq,(size_t)T*HID);          A((void**)&c.xs,(size_t)T*(HID/16)*4);
        A((void**)&c.qlora,(size_t)T*QLORA*4);   A((void**)&c.qnorm,(size_t)T*QLORA*4);
        A((void**)&c.qup,(size_t)T*QUP*4);       A((void**)&c.kvdn,(size_t)T*KVA*4);
        A((void**)&c.curkv,(size_t)T*KVLORA*4);
        A((void**)&c.qabs,(size_t)T*48*KVLORA*4);A((void**)&c.qrope,(size_t)T*48*ROPE*4);
        A((void**)&c.ctx,(size_t)T*48*KVLORA*4); A((void**)&c.aout,(size_t)T*AOUT*4);
        A((void**)&c.hidden,(size_t)T*HID*4);
        A((void**)&c.rlog,(size_t)T*NEXP*4);
        A((void**)&c.rids,(size_t)T*PF_TOPK*4);  A((void**)&c.rwts,(size_t)T*PF_TOPK*4);
        A((void**)&c.groups,sizeof(PfGroup)*(PF_GROUP_MAX+1));  // +1 dead slot
        A((void**)&c.rows,GMAX*4);               A((void**)&c.rwt,GMAX*4);
        A((void**)&c.order,PF_GROUP_MAX*4);      A((void**)&c.ngroups,4);
        A((void**)&c.hid9,GMAX*P92_NFF*4);       A((void**)&c.hq9,GMAX*P92_NFF);
        A((void**)&c.hs9,GMAX*4);
        A((void**)&c.cosv,(size_t)MAXPOS*32*4);  A((void**)&c.sinv,(size_t)MAXPOS*32*4);
        A((void**)&c.state2,(size_t)T*4*HID*2);
        A((void**)&c.pre,(size_t)T*4*4);         A((void**)&c.post,(size_t)T*4*4);
        A((void**)&c.xq1024,(size_t)T*QLORA);    A((void**)&c.xs1024,(size_t)T*(QLORA/16)*4);
        A((void**)&c.xq6144,(size_t)T*AOUT);     A((void**)&c.xs6144,(size_t)T*(AOUT/16)*4);
        A((void**)&c.dhid,(size_t)T*PM_DENSE_FF*4); A((void**)&c.dhq,(size_t)T*PM_DENSE_FF);
        A((void**)&c.dhs,(size_t)T*(PM_DENSE_FF/16)*4);
        // DSA selector scratch. scores is the big one: T x MAXPOS floats, 4 MB
        // at 4096 and 16 MB at 16384. It is materialised because the radix
        // select makes three passes over it; streaming would mean recomputing
        // the scores three times, which costs more than storing them once.
        A((void**)&c.kidx,(size_t)T*PF_IDX_DIM*4);
        A((void**)&c.qidx,(size_t)T*PF_IDX_HEADS*PF_IDX_DIM*4);
        A((void**)&c.qidxT,(size_t)T*PF_IDX_HEADS*PF_IDX_DIM*4);
        A((void**)&c.whd,(size_t)T*PF_IDX_HEADS*4);
        A((void**)&c.scores,(size_t)T*MAXPOS*4);
        A((void**)&c.sel,(size_t)T*PF_IDX_TOPK*4);
        A((void**)&c.nsel,(size_t)T*4);
        // Union scratch. umax is the worst case TQ*2048; measured mean union is
        // 3794, so this is roughly 2x over-allocated and costs 2 MB.
        A((void**)&c.uni,(size_t)((T+1)/2)*PF_UMAX*4);
        A((void**)&c.umask,(size_t)((T+1)/2)*PF_UMAX);
        A((void**)&c.ucnt,(size_t)((T+1)/2)*4);
        // Packed-union streams. Worst case is every query's top-k disjoint:
        // T*PF_IDX_TOPK slots a layer; a slot is 1024 B lat + 128 B pe + 16 B
        // mask. 612 MB a card at chunk 256 - the price of turning the scattered
        // union walk into three sequential streams.
        A((void**)&c.lat_pack,(size_t)T*PF_IDX_TOPK*KVLORA*2);
        A((void**)&c.pe_pack,(size_t)T*PF_IDX_TOPK*ROPE*2);
        A((void**)&c.mask_pack,(size_t)T*PF_IDX_TOPK*16);
        A((void**)&c.packoff,(size_t)((T+1)/2)*4);
        // the RoPE table covers every position in the run and is built once
        hipLaunchKernelGGL(k_pf_rope_table,dim3(MAXPOS),dim3(64),0,0,c.cosv,c.sinv,MAXPOS);
    }

    // ---- THE INPUT -------------------------------------------------------
    // Card 0's state for a chunk IS the prompt: each token's embedding
    // replicated across the four mHC streams, exactly as decode seeds c.sa.
    // Nothing wrote this before, so the body ran on whatever hipMalloc last
    // left in that page - correct shapes, correct traffic, correct kernel
    // count, and values that meant nothing. A throughput figure taken over
    // uninitialised input measures the machine, not the model.
    if (!ck.embed_open()) { printf("embed_open failed\n"); return 1; }
    std::vector<int> tokens(NTOK);
    for (int i = 0; i < NTOK; i++) tokens[i] = (START + i) % 151552;
    std::vector<uint16_t> seedbuf((size_t)T * 4 * HID);
    auto pf_seed = [&](int base, int rows) {
        for (int r = 0; r < rows; r++) {
            const uint16_t *emb = ck.embed_ptr(tokens[base + r]);
            for (int k2 = 0; k2 < 4; k2++)
                memcpy(&seedbuf[((size_t)r * 4 + k2) * HID], emb, (size_t)HID * 2);
        }
        CK(hipSetDevice(0));
        CK(hipMemcpy(pf[0].state, seedbuf.data(), (size_t)rows * 4 * HID * 2,
                     hipMemcpyHostToDevice));
    };
    const bool pf_trace = getenv("PF_TRACE") != nullptr;
    // Union statistics. OFF by default and costing nothing when off, because it
    // adds a device sync per DSA layer per chunk and would perturb the very wall
    // it sits next to. Stats runs and timing runs are therefore different runs.
    const bool pf_ustats = getenv("PF_UNION_STATS") != nullptr;
    std::vector<int> ustat_u, ustat_n;
    std::mutex ustat_mx;
    // trace an ABSOLUTE position, so a row past the DSA threshold can be
    // compared against the decode engine at the same position
    const int pf_trace_pos = getenv("PF_TRACE_POS") ? atoi(getenv("PF_TRACE_POS")) : 0;

    PfPipeline pipe;
    if (pf_pipeline_init(&pipe, &cfg, NTOK)) { printf("pipeline init failed\n"); return 1; }

    // ---- WALL SENSITIVITY ------------------------------------------------
    // Profiled kernel milliseconds are not recoverable milliseconds. Work that
    // sits before a rendezvous can be improved to no effect at all, so the only
    // way to know what a bucket is worth is to add a known quantity of the exact
    // work and watch the wall: d(wall)/d(stage) near 1.0 is exposed, near 0.0 is
    // hidden in slack.
    //
    // The probe names a KERNEL, not a stage. Duplicating a stage measures what
    // you think only if re-running it is a no-op, and several stages here are
    // not: k_pf_mhc_scale multiplies the mixing logits in place, k_pf_mome_st
    // advances a chunk carry. k_pf_attn writes c.ctx from unchanged sources and
    // repeats cleanly.
    //
    // PF_SENS_STAGE=1 duplicates it WHERE IT RUNS.
    // PF_SENS_STAGE=2 duplicates the SAME kernel, the same number of times, at
    // the tail of the card's work where nothing can hide it. That is the
    // mandatory positive control: if mode 2 does not move the wall nearly
    // proportionally, the instrument is broken and mode 1's number means
    // nothing. Same kernel in both, so the only variable is position.
    //
    // Both count launches. A flat wall and a probe that never issued look
    // identical otherwise, and this session has already had one profile that
    // measured the last layer of eleven and read as an idle machine.
    std::atomic<long> sens_fired(0);

    // one chunk through the layers this card owns
    auto pf_layers = [&](int d, int base, int rows_in_chunk) {
        CK(hipSetDevice(d)); Pf &c = pf[d]; PfTimers &tm = pipe.timers[d];
        for (int L = pipe.layer_begin[d]; L < pipe.layer_end[d]; L++) {
            LayerW &w = LW[L];
            // DSA layers see the whole prefix. That is NOT what the model does above
            // 2048 positions - decode runs the indexer and attends to the top-2048 -
            // but prefill launches no indexer, so this diverges from the reference
            // engine on any prompt longer than 2048 and does work the model does not
            // ask for. PF_DSA_WIN bounds it with a trailing window purely to measure
            // the ceiling; a window is not a selection and is not a correctness fix.
            const int win = w.dsa ? cfg.dsa_win : 512;
            const int RC = rows_in_chunk;

            // ---- attention mHC: logits, hoisted RMS scale, gates, Sinkhorn, collapse
            pf_stage_begin(&tm, PF_S_MHC, 0);
            hipLaunchKernelGGL(k_pf_mhc_logits,dim3(24,RC),dim3(256),0,0,
                               c.state,(const uint16_t*)w.aphi,(const uint16_t*)w.agam,RC,c.logits);
            hipLaunchKernelGGL(k_pf_mhc_scale,dim3(RC),dim3(256),0,0,
                               c.state,(const uint16_t*)w.agam,RC,c.logits);
            hipLaunchKernelGGL(k_pf_mhc_prepost,dim3(RC),dim3(64),0,0,
                               c.logits,(const uint16_t*)w.aalp,(const uint16_t*)w.abet,RC,c.pre,c.post);
            hipLaunchKernelGGL(k_pf_mhc_collapse,dim3(RC),dim3(64),0,0,
                               c.logits,(const uint16_t*)w.aalp,(const uint16_t*)w.abet,RC,c.comb);
            hipLaunchKernelGGL(k_pf_mhc_mix,dim3((HID+255)/256,RC),dim3(256),0,0,
                               c.state,c.pre,RC,c.mixf);
            pf_stage_end(&tm, PF_S_MHC, 0);

            pf_stage_begin(&tm, PF_S_NORM, 0);
            hipLaunchKernelGGL((k_pf_rmsnorm_k<HID>),dim3(RC),dim3(256),0,0,
                               c.mixf,(const uint16_t*)w.in_norm,c.norm,RC,1e-5f);
            hipLaunchKernelGGL((k_pf_quant_k<HID>),dim3(RC),dim3(256),0,0,c.norm,RC,c.xq,c.xs);
            pf_stage_end(&tm, PF_S_NORM, 0);

            // ---- query path: q_a, MoME, q_norm, q_b
            pf_stage_begin(&tm, PF_S_PROJ, 0);
            hipLaunchKernelGGL((k_pf_proj<HID>),dim3((QLORA+7)/8),dim3(256),0,0,
                               w.qa.w,w.qa.s,c.xq,c.xs,w.qa.s2,RC,QLORA,c.qlora);
            hipLaunchKernelGGL(k_pf_mome_st,dim3((QLORA+255)/256),dim3(256),0,0,
                               c.qlora,(const uint16_t*)w.qa_conv,w.mome_q,RC,QLORA,QLORA);
            hipLaunchKernelGGL((k_pf_rmsnorm_k<QLORA>),dim3(RC),dim3(256),0,0,
                               c.qlora,(const uint16_t*)w.q_norm,c.qnorm,RC,1e-5f);
            hipLaunchKernelGGL((k_pf_quant_k<QLORA>),dim3(RC),dim3(256),0,0,
                               c.qnorm,RC,c.xq1024,c.xs1024);
            hipLaunchKernelGGL((k_pf_proj<QLORA>),dim3((QUP+7)/8),dim3(256),0,0,
                               w.qb.w,w.qb.s,c.xq1024,c.xs1024,w.qb.s2,RC,QUP,c.qup);
            // ---- key/value path: kv_a, MoME, kv_norm, publish into the cache
            hipLaunchKernelGGL((k_pf_bf16_gemv_rows<HID>),dim3((KVA+3)/4,RC),dim3(256),0,0,
                               (const uint16_t*)w.kv_a,c.norm,RC,KVA,c.kvdn);
            hipLaunchKernelGGL(k_pf_mome_st,dim3((KVLORA+255)/256),dim3(256),0,0,
                               c.kvdn,(const uint16_t*)w.kv_conv,w.mome_kv,RC,KVLORA,KVA);
            hipLaunchKernelGGL(k_pf_kv_cache,dim3(RC),dim3(256),0,0,
                               c.kvdn,(const uint16_t*)w.kv_norm,c.cosv,c.sinv,RC,base,KVA,
                               w.cache_kv,w.cache_rope);
            hipLaunchKernelGGL(k_pf_absorb,dim3(48,2,RC),dim3(256),0,0,
                               c.qup,(const uint16_t*)w.kv_b,RC,c.qabs);
            pf_stage_end(&tm, PF_S_PROJ, 0);

            // ---- DSA: indexer key always, selection only above the threshold
            // The key cache must be written at EVERY position on a DSA layer,
            // not only when selection runs. Decode carries a scar comment about
            // exactly this: gating the cache write on the threshold left rows
            // 0..2047 unwritten, and the selector then scored raw hipMalloc
            // contents - silently, because equal scores make the tie band
            // swallow the array and the ascending gather returns the OLDEST
            // 2048 positions.
            if (w.dsa) {
                pf_stage_begin(&tm, PF_S_DSA, 0);
                hipLaunchKernelGGL((k_pf_proj<HID>),dim3((PF_IDX_DIM+7)/8),dim3(256),0,0,
                                   w.ix_wk.w,w.ix_wk.s,c.xq,c.xs,w.ix_wk.s2,RC,PF_IDX_DIM,c.kidx);
                hipLaunchKernelGGL(k_pf_ix_key,dim3(RC),dim3(PF_IDX_DIM),0,0,
                                   c.kidx,(const uint16_t*)w.ix_knorm,c.cosv,c.sinv,
                                   RC,base,w.cap,1e-5f,w.ix_cache);
                pf_stage_end(&tm, PF_S_DSA, 0);
                if (cfg.dsa_sel && base + RC > PF_IDX_TOPK) {
                    pf_stage_begin(&tm, PF_S_DSA_SCORE, 0);
                    hipLaunchKernelGGL((k_pf_proj<QLORA>),dim3((PF_IDX_HEADS*PF_IDX_DIM+7)/8),dim3(256),0,0,
                                       w.ix_wqb.w,w.ix_wqb.s,c.xq1024,c.xs1024,w.ix_wqb.s2,
                                       RC,PF_IDX_HEADS*PF_IDX_DIM,c.qidx);
                    if (cfg.dsa_scorer == 3)
                        hipLaunchKernelGGL(k_pf_ix_query_t,dim3(PF_IDX_HEADS,RC),dim3(64),0,0,
                                           c.qidx,c.qidxT,c.cosv,c.sinv,RC,base);
                    else
                        hipLaunchKernelGGL(k_pf_ix_query,dim3(PF_IDX_HEADS,RC),dim3(32),0,0,
                                           c.qidx,c.cosv,c.sinv,RC,base);
                    hipLaunchKernelGGL((k_pf_bf16_gemv_rows<HID>),dim3((PF_IDX_HEADS+3)/4,RC),dim3(256),0,0,
                                       (const uint16_t*)w.ix_wproj,c.norm,RC,PF_IDX_HEADS,c.whd);
                    const int maxn = base + RC;
                    if (cfg.dsa_scorer == 3)
                        hipLaunchKernelGGL(k_pf_idx_scores3,dim3((maxn+255)/256,RC),dim3(256),0,0,
                                           c.qidxT,(const uint16_t*)w.ix_cache,c.whd,w.cap,RC,base,
                                           1.0f/sqrtf((float)PF_IDX_DIM),c.scores,MAXPOS);
                    else
                        hipLaunchKernelGGL(k_pf_idx_scores,dim3((maxn+255)/256,RC),dim3(256),0,0,
                                           c.qidx,(const uint16_t*)w.ix_cache,c.whd,w.cap,RC,base,
                                           1.0f/sqrtf((float)PF_IDX_DIM),c.scores,MAXPOS);
                    pf_stage_end(&tm, PF_S_DSA_SCORE, 0);
                    pf_stage_begin(&tm, PF_S_DSA_SEL, 0);
                    hipLaunchKernelGGL(k_pf_idx_select,dim3(RC),dim3(PF_SEL_THREADS),0,0,
                                       c.scores,RC,base,PF_IDX_TOPK,MAXPOS,c.sel,c.nsel);
                    pf_stage_end(&tm, PF_S_DSA_SEL, 0);
                    pf_stage_begin(&tm, PF_S_DSA_UNI, 0);
                    if (cfg.sp_g2) {
                        // G=2 head-shared path: pack each query's selection
                        // directly. No union, no masks, constant strides.
                        hipLaunchKernelGGL(k_pf_sel_pack,dim3(RC),dim3(256),0,0,
                                           c.sel,c.nsel,RC,PF_IDX_TOPK,
                                           (const uint16_t*)w.cache_kv,
                                           (const uint16_t*)w.cache_rope,
                                           c.lat_pack,c.pe_pack);
                    } else {
                    // one shared key set per TQ=4 tile, each query masked to
                    // its own top-2048 within it
                    { const int TQ = cfg.attn_tq;
                      const int nt = (RC + TQ - 1) / TQ;
                      if (TQ == 8)
                        hipLaunchKernelGGL((k_pf_union<8>),dim3(nt),dim3(256),
                                           (size_t)(base+RC),0,c.sel,c.nsel,RC,PF_IDX_TOPK,
                                           base+RC,c.uni,c.umask,c.ucnt,PF_UMAX);
                      else if (TQ == 2)
                        hipLaunchKernelGGL((k_pf_union<2>),dim3(nt),dim3(256),
                                           (size_t)(base+RC),0,c.sel,c.nsel,RC,PF_IDX_TOPK,
                                           base+RC,c.uni,c.umask,c.ucnt,PF_UMAX);
                      else
                        hipLaunchKernelGGL((k_pf_union<4>),dim3(nt),dim3(256),
                                           (size_t)(base+RC),0,c.sel,c.nsel,RC,PF_IDX_TOPK,
                                           base+RC,c.uni,c.umask,c.ucnt,PF_UMAX);
                      if (cfg.sp_pack) {
                          // consumer-designed emission: ordered union keys ->
                          // contiguous per-tile streams, so the sweep needs no
                          // index chain at all
                          hipLaunchKernelGGL(k_pf_pack_scan,dim3(1),dim3(256),0,0,
                                             c.ucnt,nt,c.packoff);
                          hipLaunchKernelGGL(k_pf_union_pack,dim3(nt),dim3(256),0,0,
                                             c.uni,c.umask,c.ucnt,c.packoff,PF_UMAX,
                                             (const uint16_t*)w.cache_kv,
                                             (const uint16_t*)w.cache_rope,
                                             c.lat_pack,c.pe_pack,c.mask_pack);
                      } } }
                    pf_stage_end(&tm, PF_S_DSA_UNI, 0);
                    if (pf_ustats && d == 0) {
                        CK(hipDeviceSynchronize());
                        const int nt = (RC + 3) / 4;
                        std::vector<int> u(nt), ns(RC);
                        CK(hipMemcpy(u.data(), c.ucnt, (size_t)nt*4, hipMemcpyDeviceToHost));
                        CK(hipMemcpy(ns.data(), c.nsel, (size_t)RC*4, hipMemcpyDeviceToHost));
                        std::lock_guard<std::mutex> lk(ustat_mx);
                        ustat_u.insert(ustat_u.end(), u.begin(), u.end());
                        ustat_n.insert(ustat_n.end(), ns.begin(), ns.end());
                    }
                }
            }

            pf_stage_begin(&tm, PF_S_ROPE, 0);
            hipLaunchKernelGGL(k_pf_qrope,dim3(48,RC),dim3(32),0,0,
                               c.qup,c.cosv,c.sinv,RC,base,c.qrope);
            pf_stage_end(&tm, PF_S_ROPE, 0);

            pf_stage_begin(&tm, PF_S_ATTN, 0);
            const bool sparse_ok = w.dsa && cfg.dsa_sel && (base + RC > PF_IDX_TOPK);
            if (sparse_ok) {
                const int TQ = cfg.attn_tq;
                const int nt = (RC + TQ - 1) / TQ;
                if (cfg.sp_g2) {
                    hipLaunchKernelGGL(k_pf_attn_spg2,dim3(RC,24),dim3(64),0,0,
                                       c.qabs,c.qrope,c.lat_pack,c.pe_pack,
                                       (const uint16_t*)w.sink_kv_n,(const uint16_t*)w.sink_rope,
                                       c.nsel,PF_IDX_TOPK,RC,c.ctx);
                } else if (cfg.sp_pack) {
                    if (TQ == 8)
                        hipLaunchKernelGGL((k_pf_attn_spk<8>),dim3(nt,48),dim3(64),0,0,
                                           c.qabs,c.qrope,c.lat_pack,c.pe_pack,c.mask_pack,
                                           (const uint16_t*)w.sink_kv_n,(const uint16_t*)w.sink_rope,
                                           c.ucnt,c.packoff,RC,c.ctx);
                    else if (TQ == 2)
                        hipLaunchKernelGGL((k_pf_attn_spk<2>),dim3(nt,48),dim3(64),0,0,
                                           c.qabs,c.qrope,c.lat_pack,c.pe_pack,c.mask_pack,
                                           (const uint16_t*)w.sink_kv_n,(const uint16_t*)w.sink_rope,
                                           c.ucnt,c.packoff,RC,c.ctx);
                    else
                        hipLaunchKernelGGL((k_pf_attn_spk<4>),dim3(nt,48),dim3(64),0,0,
                                           c.qabs,c.qrope,c.lat_pack,c.pe_pack,c.mask_pack,
                                           (const uint16_t*)w.sink_kv_n,(const uint16_t*)w.sink_rope,
                                           c.ucnt,c.packoff,RC,c.ctx);
                } else if (TQ == 8)
                    hipLaunchKernelGGL((k_pf_attn_sp<8>),dim3(nt,48),dim3(64),0,0,
                                       c.qabs,c.qrope,(const uint16_t*)w.cache_kv,
                                       (const uint16_t*)w.cache_rope,
                                       (const uint16_t*)w.sink_kv_n,(const uint16_t*)w.sink_rope,
                                       c.uni,c.umask,c.ucnt,PF_UMAX,RC,c.ctx);
                else if (TQ == 2)
                    hipLaunchKernelGGL((k_pf_attn_sp<2>),dim3(nt,48),dim3(64),0,0,
                                       c.qabs,c.qrope,(const uint16_t*)w.cache_kv,
                                       (const uint16_t*)w.cache_rope,
                                       (const uint16_t*)w.sink_kv_n,(const uint16_t*)w.sink_rope,
                                       c.uni,c.umask,c.ucnt,PF_UMAX,RC,c.ctx);
                else
                    hipLaunchKernelGGL((k_pf_attn_sp<4>),dim3(nt,48),dim3(64),0,0,
                                       c.qabs,c.qrope,(const uint16_t*)w.cache_kv,
                                       (const uint16_t*)w.cache_rope,
                                       (const uint16_t*)w.sink_kv_n,(const uint16_t*)w.sink_rope,
                                       c.uni,c.umask,c.ucnt,PF_UMAX,RC,c.ctx);
            }
            else
            pf_attn_launch(cfg.attn_kernel,cfg.attn_tq,c.qabs,c.qrope,
                           (const uint16_t*)w.cache_kv,(const uint16_t*)w.cache_rope,
                           (const uint16_t*)w.sink_kv_n,(const uint16_t*)w.sink_rope,
                           base,RC,win,c.ctx,0);
            if (pipe.sens_stage == 1)
                for (int r = 0; r < pipe.sens_reps; r++) {
                    pf_attn_launch(cfg.attn_kernel,cfg.attn_tq,c.qabs,c.qrope,
                                   (const uint16_t*)w.cache_kv,(const uint16_t*)w.cache_rope,
                                   (const uint16_t*)w.sink_kv_n,(const uint16_t*)w.sink_rope,
                                   base,RC,win,c.ctx,0);
                    sens_fired.fetch_add(1, std::memory_order_relaxed);
                }
            hipLaunchKernelGGL(k_pf_value_up,dim3((48*128+3)/4,RC),dim3(256),0,0,
                               c.ctx,(const uint16_t*)w.kv_b,RC,c.aout);
            hipLaunchKernelGGL(k_pf_mome_st,dim3((AOUT+255)/256),dim3(256),0,0,
                               c.aout,(const uint16_t*)w.o_conv,w.mome_o,RC,AOUT,AOUT);
            hipLaunchKernelGGL((k_pf_quant_k<AOUT>),dim3(RC),dim3(256),0,0,
                               c.aout,RC,c.xq6144,c.xs6144);
            hipLaunchKernelGGL((k_pf_proj<AOUT>),dim3((HID+7)/8),dim3(256),0,0,
                               w.o.w,w.o.s,c.xq6144,c.xs6144,w.o.s2,RC,HID,c.hidden);
            pf_stage_end(&tm, PF_S_ATTN, 0);
            if (pf_trace && L == 0 && pf_trace_pos >= base && pf_trace_pos < base + RC) {
                CK(hipDeviceSynchronize());
                auto rms = [&](const float *p, int n) { std::vector<float> h(n);
                    CK(hipMemcpy(h.data(), p, (size_t)n*4, hipMemcpyDeviceToHost));
                    double s2 = 0; for (float v : h) s2 += (double)v * v;
                    return sqrt(s2/(double)n); };
                printf("  DBG L0 qlora %.6g qnorm %.6g qup %.6g kvdn %.6g ctx %.6g aout %.6g attnout %.6g\n",
                       rms(c.qlora,QLORA),rms(c.qnorm,QLORA),rms(c.qup,QUP),rms(c.kvdn,KVA),
                       rms(c.ctx,48*512),rms(c.aout,AOUT),rms(c.hidden,HID));
            }

            // ---- attention mHC post: re-expand across the four streams -> state2
            pf_stage_begin(&tm, PF_S_MHC, 0);
            hipLaunchKernelGGL((k_pf_rmsnorm_k<HID>),dim3(RC),dim3(256),0,0,
                               c.hidden,(const uint16_t*)w.post_attn,c.norm,RC,1e-5f);
            hipLaunchKernelGGL(k_pf_reexpand,dim3((HID+255)/256,RC),dim3(256),0,0,
                               c.post,c.comb,c.norm,c.state,RC,c.state2);
            // ---- MLP mHC pre, on state2
            hipLaunchKernelGGL(k_pf_mhc_logits,dim3(24,RC),dim3(256),0,0,
                               c.state2,(const uint16_t*)w.mphi,(const uint16_t*)w.mgam,RC,c.logits);
            hipLaunchKernelGGL(k_pf_mhc_scale,dim3(RC),dim3(256),0,0,
                               c.state2,(const uint16_t*)w.mgam,RC,c.logits);
            hipLaunchKernelGGL(k_pf_mhc_prepost,dim3(RC),dim3(64),0,0,
                               c.logits,(const uint16_t*)w.malp,(const uint16_t*)w.mbet,RC,c.pre,c.post);
            hipLaunchKernelGGL(k_pf_mhc_collapse,dim3(RC),dim3(64),0,0,
                               c.logits,(const uint16_t*)w.malp,(const uint16_t*)w.mbet,RC,c.comb);
            hipLaunchKernelGGL(k_pf_mhc_mix,dim3((HID+255)/256,RC),dim3(256),0,0,
                               c.state2,c.pre,RC,c.mixf);
            pf_stage_end(&tm, PF_S_MHC, 0);

            pf_stage_begin(&tm, PF_S_NORM, 0);
            hipLaunchKernelGGL((k_pf_rmsnorm_k<HID>),dim3(RC),dim3(256),0,0,
                               c.mixf,(const uint16_t*)w.pre_mlp,c.norm,RC,1e-5f);
            hipLaunchKernelGGL((k_pf_quant_k<HID>),dim3(RC),dim3(256),0,0,c.norm,RC,c.xq,c.xs);
            pf_stage_end(&tm, PF_S_NORM, 0);

            if (w.dense) {
                pf_stage_begin(&tm, PF_S_GATEUP, 0);
                hipLaunchKernelGGL(k_pf_dense_gateup,dim3(PM_DENSE_FF/8),dim3(256),0,0,
                                   w.dg.w,w.dg.s,w.du.w,w.du.s,c.xq,c.xs,
                                   w.dg.s2,w.du.s2,RC,c.dhid);
                pf_stage_end(&tm, PF_S_GATEUP, 0);
                pf_stage_begin(&tm, PF_S_REQUANT, 0);
                hipLaunchKernelGGL((k_pf_quant_k<PM_DENSE_FF>),dim3(RC),dim3(256),0,0,
                                   c.dhid,RC,c.dhq,c.dhs);
                pf_stage_end(&tm, PF_S_REQUANT, 0);
                pf_stage_begin(&tm, PF_S_DOWN, 0);
                hipLaunchKernelGGL((k_pf_proj<PM_DENSE_FF>),dim3((HID+7)/8),dim3(256),0,0,
                                   w.dd.w,w.dd.s,c.dhq,c.dhs,w.dd.s2,RC,HID,c.hidden);
                pf_stage_end(&tm, PF_S_DOWN, 0);
            } else {
                pf_stage_begin(&tm, PF_S_ROUTE, 0);
                hipLaunchKernelGGL(k_pf_route_gemv,dim3(NEXP/4,RC),dim3(256),0,0,
                                   (const uint16_t*)w.router_w,c.norm,RC,c.rlog);
                hipLaunchKernelGGL(k_pf_router,dim3(RC),dim3(64),0,0,
                                   c.rlog,w.router_bias,RC,c.rids,c.rwts);
                if (pf_trace && base==0 && (L==2||L==4||L==6)) {
                    CK(hipDeviceSynchronize());
                    int ids[8]; float wt[8];
                    CK(hipMemcpy(ids,c.rids,32,hipMemcpyDeviceToHost));
                    CK(hipMemcpy(wt,c.rwts,32,hipMemcpyDeviceToHost));
                    printf("  RT L%d ids",L); for(int q=0;q<8;q++) printf(" %d",ids[q]);
                    printf(" wts"); for(int q=0;q<8;q++) printf(" %.4f",wt[q]); printf("\n");
                }
                hipLaunchKernelGGL(k_pf_group,dim3(1),dim3(256),0,0,
                                   c.rids,c.rwts,RC,w.arena_slot_base,NEXP,
                                   c.groups,c.rows,c.rwt,c.order,c.ngroups);
                pf_stage_end(&tm, PF_S_ROUTE, 0);
                // FIXED PF_GROUP_MAX GRID (09-09). The blocking D2H read of the
                // live-group count drained the queue and re-synced the host
                // every MoE layer (~2816 forced syncs a card at 64K) - the
                // drained-post-barrier launch measured at 10.8 us against
                // 1.25 us nominal. Dead grid slots read the pinned nrows==0
                // group and return, so the grids size themselves out of the
                // group table with no host round-trip.
                pf_stage_begin(&tm, PF_S_GATEUP, 0);
                hipLaunchKernelGGL(k_pf_gateup,dim3(P92_NFF/8,PF_GROUP_MAX),dim3(256),0,0,
                                   arena[d].gw,arena[d].gs,arena[d].uw,arena[d].us,
                                   c.xq,c.xs,arena[d].s2,c.groups,c.order,c.rows,c.hid9);
                pf_stage_end(&tm, PF_S_GATEUP, 0);
                pf_stage_begin(&tm, PF_S_REQUANT, 0);
                hipLaunchKernelGGL(k_pf_requant,dim3((int)GMAX),dim3(256),0,0,
                                   c.hid9,(int)GMAX,c.hq9,c.hs9);
                pf_stage_end(&tm, PF_S_REQUANT, 0);
                pf_stage_begin(&tm, PF_S_DOWN, 0);
                // k_pf_down scatters with atomicAdd, so the destination must
                // start at zero every layer. It never did.
                CK(hipMemsetAsync(c.hidden,0,(size_t)RC*HID*4,0));
                hipLaunchKernelGGL(k_pf_down,dim3(HID/8,PF_GROUP_MAX),dim3(256),0,0,
                                   arena[d].dw,arena[d].ds,c.hq9,c.hs9,arena[d].s2,
                                   c.groups,c.order,c.rows,c.rwt,c.hidden);
                pf_stage_end(&tm, PF_S_DOWN, 0);
            }

            // ---- MLP mHC post -> state
            pf_stage_begin(&tm, PF_S_MHC, 0);
            hipLaunchKernelGGL((k_pf_rmsnorm_k<HID>),dim3(RC),dim3(256),0,0,
                               c.hidden,(const uint16_t*)w.post_mlp,c.norm,RC,1e-5f);
            hipLaunchKernelGGL(k_pf_reexpand,dim3((HID+255)/256,RC),dim3(256),0,0,
                               c.post,c.comb,c.norm,c.state2,RC,c.state);
            if (w.block_post)
                hipLaunchKernelGGL(k_pf_block_post,dim3(RC),dim3(256),0,0,
                                   c.state,(const uint16_t*)w.block_post,RC);
            pf_stage_end(&tm, PF_S_MHC, 0);
            if (pf_trace && pf_trace_pos >= base && pf_trace_pos < base + RC) {
                const int trow = pf_trace_pos - base;
                CK(hipDeviceSynchronize());
                std::vector<uint16_t> t4((size_t)4 * HID);
                CK(hipMemcpy(t4.data(), c.state + (size_t)trow*4*HID, (size_t)4*HID*2, hipMemcpyDeviceToHost));
                double ss = 0, mx = 0; int nf = 0;
                for (size_t i = 0; i < t4.size(); i++) {
                    uint32_t u = (uint32_t)t4[i] << 16; float v; memcpy(&v, &u, 4);
                    if (!std::isfinite(v)) nf++;
                    ss += (double)v * v; mx = fmax(mx, fabs((double)v));
                }
                std::vector<float> hh(HID);
                CK(hipMemcpy(hh.data(), c.hidden + (size_t)trow*HID, (size_t)HID*4, hipMemcpyDeviceToHost));
                double hs2 = 0; for (float v : hh) hs2 += (double)v * v;
                printf("    L%-2d dev%d %-5s %-3s  streams rms %10.4g max %10.4g nonfinite %d | mlp-out rms %10.4g\n",
                       L, d, w.dense?"dense":"moe", w.dsa?"dsa":"swa",
                       sqrt(ss/(double)t4.size()), mx, nf, sqrt(hs2/(double)HID));
            }
        }
        if (pipe.sens_stage == 2)
            for (int L = pipe.layer_begin[d]; L < pipe.layer_end[d]; L++) {
                LayerW &w = LW[L];
                const int win2 = w.dsa ? cfg.dsa_win : 512;
                for (int r = 0; r < pipe.sens_reps; r++) {
                    pf_attn_launch(cfg.attn_kernel,cfg.attn_tq,c.qabs,c.qrope,
                                   (const uint16_t*)w.cache_kv,(const uint16_t*)w.cache_rope,
                                   (const uint16_t*)w.sink_kv_n,(const uint16_t*)w.sink_rope,
                                   base,rows_in_chunk,win2,c.ctx,0);
                    sens_fired.fetch_add(1, std::memory_order_relaxed);
                }
            }
        CK(hipDeviceSynchronize());
        pf_stage_accumulate(&tm);
    };

    // ---- the pipeline ------------------------------------------------------
    // PF_PIPE_DEPTH=1 runs it SERIALLY: card 0 completes every layer it owns,
    // synchronises, pushes, then card 1 begins. Overlap is 1.0 BY CONSTRUCTION,
    // not as a finding. That is the right shape for first light because a
    // wiring defect in eleven never-executed kernels is far easier to locate
    // without four threads in flight.
    //
    // PF_PIPE_DEPTH>1 runs the architecture the design actually calls for: one
    // thread per card, each consuming chunks as they arrive and starting chunk
    // k+1 the moment it has pushed chunk k, giving C/(C+3) utilisation. The
    // difference is not the transport - producer-push is identical in both -
    // it is whether a card waits for its consumer before continuing.
    //
    // The per-card hipDeviceSynchronize inside pf_layers is the host-side
    // rendezvous that makes depth 1 serial, and it is also the barrier that
    // hides pre-barrier work from the wall. Work removed from a non-straggler
    // before a barrier does not reach the token at all.
    const int nchunks = (NTOK + T - 1) / T;
    auto t0 = std::chrono::steady_clock::now();
    double ttft = 0.0;
    std::atomic<int> ready[4];
    for (int d = 0; d < 4; d++) ready[d].store(d == 0 ? 0 : -1);

    if (cfg.pipe_depth <= 1) {
        for (int k = 0; k < nchunks; k++) {
            const int base = k * T;
            const int rows = (base + T <= NTOK) ? T : (NTOK - base);
            for (int d = 0; d < 4; d++) {
                if (d == 0) pf_seed(base, rows);
                pf_layers(d, base, rows);
                if (d + 1 < 4) {
                    PfTimers &tm = pipe.timers[d];
                    pf_stage_begin(&tm, PF_S_HANDOFF, 0);
                    p92_p2p_push(&pipe.p2p, d, d + 1, pf[d].state, 0,
                                 (size_t)rows * 4 * HID * 2, 0, (uint32_t)(k + 1));
                    CK(hipSetDevice(d)); CK(hipStreamSynchronize(pipe.p2p.push[d]));
                    pf_stage_end(&tm, PF_S_HANDOFF, 0);
                }
            }
            if (k == 0) ttft = std::chrono::duration<double,std::milli>(
                                  std::chrono::steady_clock::now() - t0).count();
        }
    } else {
        // one thread a card. ready[d] is the highest chunk index card d has
        // finished; card d+1 may start chunk k once ready[d] >= k.
        std::vector<std::thread> th;
        std::atomic<double> ttft_a(0.0);
        for (int d = 0; d < 4; d++) th.emplace_back([&, d]{
            for (int k = 0; k < nchunks; k++) {
                if (d > 0) while (ready[d-1].load(std::memory_order_acquire) < k)
                    std::this_thread::yield();
                const int base = k * T;
                const int rows = (base + T <= NTOK) ? T : (NTOK - base);
                if (d == 0) pf_seed(base, rows);
                pf_layers(d, base, rows);
                if (d + 1 < 4) {
                    PfTimers &tm = pipe.timers[d];
                    pf_stage_begin(&tm, PF_S_HANDOFF, 0);
                    p92_p2p_push(&pipe.p2p, d, d + 1, pf[d].state, 0,
                                 (size_t)rows * 4 * HID * 2, 0, (uint32_t)(k + 1));
                    hipSetDevice(d); hipStreamSynchronize(pipe.p2p.push[d]);
                    pf_stage_end(&tm, PF_S_HANDOFF, 0);
                }
                ready[d].store(k, std::memory_order_release);
                if (d == 3 && k == 0)
                    ttft_a.store(std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now() - t0).count());
            }
        });
        for (auto &t : th) t.join();
        ttft = ttft_a.load();
    }
    const double ms = std::chrono::duration<double,std::milli>(
                          std::chrono::steady_clock::now() - t0).count();

    PfResult res{};
    res.prompt_tokens = NTOK; res.chunk = T;
    res.prefill_ms = ms; res.prefill_toks = NTOK / (ms / 1000.0);
    res.ttft_ms = ttft;
    if (pipe.sens_stage > 0)
        printf("  sensitivity: mode %d, %d extra reps, %ld duplicate launches ISSUED\n",
               pipe.sens_stage, pipe.sens_reps, sens_fired.load());
    if (pf_ustats && !ustat_u.empty()) {
        auto pct=[&](std::vector<int> v,double q){ std::sort(v.begin(),v.end());
            return v[(size_t)(q*(v.size()-1))]; };
        double su=0; for(int v:ustat_u) su+=v;
        double sn=0; for(int v:ustat_n) sn+=v;
        const double mu = su/ustat_u.size(), mn = sn/ustat_n.size();
        printf("\n  UNION STATS (TQ=4), context %d, %zu tiles, %zu queries\n",
               NTOK, ustat_u.size(), ustat_n.size());
        printf("    union keys   mean %.0f  p50 %d  p95 %d  max %d\n",
               mu, pct(ustat_u,0.50), pct(ustat_u,0.95), pct(ustat_u,1.0));
        printf("    selected set mean %.0f  p50 %d  p95 %d\n",
               mn, pct(ustat_n,0.50), pct(ustat_n,0.95));
        printf("    union / context        %.4f\n", mu/(double)NTOK);
        printf("    union / (4 x 2048)     %.4f   (1.0 = no overlap at all)\n", mu/(4.0*PF_IDX_TOPK));
        printf("    union / 2048           %.4f   (1.0 = four queries agree exactly)\n", mu/(double)PF_IDX_TOPK);
        printf("    keys per query         %.0f   vs dense %.0f\n", mu/4.0, (double)NTOK/2.0/4.0);
    }
    pf_report(&pipe, &res);
    pf_pipeline_free(&pipe);
    return 0;
}
