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
#include <deque>
#include <thread>
#include <atomic>
#include <mutex>
// ---- chunked prefill, IN-PROCESS. The prefill engine's kernels and its four-card chunk pipeline
// are compiled into the server so a prompt is ingested by the chunk pipeline (hundreds of tok/s)
// instead of being walked one decode step per token (~50-70 tok/s). The caches are the same
// buffers, at absolute positions; decode continues from them. (receipts 21-23 named this build.)
#include "p92amd/p92_pf.h"
#include "p92amd/p92_pf_moe.h"
#include "p92amd/p92_pf_drive.h"
#include "p92amd/p92_p2p.h"
__global__ void k_pf_shuffle_scales_gu(const uint8_t*, uint8_t*, int);
__global__ void k_pf_shuffle_scales_dn(const uint8_t*, uint8_t*, int);
__global__ void k_pf_shuffle_check(const uint8_t*, const uint8_t*, int, int, int, uint32_t*);
#include "../../prefill/src/p92_pf_moe.hip"
#include "../../prefill/src/p92_pf_down.hip"
#include "../../prefill/src/p92_pf_quant.hip"
#include "../../prefill/src/p92_pf_proj.hip"
#include "../../prefill/src/p92_pf_body.hip"
#include "../../prefill/src/p92_pf_attn.hip"
#include "../../prefill/src/p92_pf_attn_blk.hip"
#include "../../prefill/src/p92_pf_attn_blk2.hip"
#include "../../prefill/src/p92_pf_rb.hip"
#include "../../prefill/src/p92_pf_downdet.hip"
#include "../../prefill/src/p92_pf_mla.hip"
#define PF_UMAX (8 * PF_IDX_TOPK)
#include "../../prefill/src/p92_pf_dsa.hip"

// Decode crossing as a device-side chain: the producer card STORES the 20 KB four-stream state
// straight into the consumer card's buffer (peer write, the fast direction) and records an event;
// the consumer's stream waits on that event. No host synchronisation anywhere in the token loop
// until the head, so no drained-queue launches at every crossing (3 on the contiguous map, 11 on
// the ring map).
__global__ void __launch_bounds__(256)
k_pe_copy16(const uint4 *__restrict__ src, uint4 *__restrict__ dst, int n4) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n4) dst[i] = src[i];
}

// Inverse of k_pf_shuffle_scales_gu/dn: the per-16 expert scales back to the arena's own
// order, which the decode expert kernels read. per_row 160 / fan 5 for gate,up; 64 / 2 for down.
__global__ void __launch_bounds__(256)
k_pe_unshuffle_scales(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst,
                      int rows_total, int per_row, int fan) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows_total) return;
    const int l = threadIdx.x & 31;
    const uint8_t *s = src + (size_t)row * per_row;
    uint8_t *d = dst + (size_t)row * per_row;
    for (int i = 0; i < fan; i++) d[i * 32 + l] = s[l * fan + i];
}

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
    // rocBLAS projection mirrors (PF_PROJ_RB): int8 per-row + row scale for the NVFP4 projections,
    // fp32 copies of the bf16 ones
    int8_t *qa8,*qb8,*o8,*ixwk8,*ixwqb8; float *qa8s,*qb8s,*o8s,*ixwk8s,*ixwqb8s;
    float *kva_f,*router_f,*wproj_f;
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
    // P92_PREFILL=1 (default): a prompt of >= P92_PREFILL_MIN tokens is ingested by the chunked
    // prefill pipeline and the decoder continues from its caches. 0 = the token-walk ingest (one
    // decode step per prompt token), kept as the comparison arm.
    const bool use_prefill = !getenv("P92_PREFILL") || atoi(getenv("P92_PREFILL")) != 0;
    const int prefill_min = getenv("P92_PREFILL_MIN") ? atoi(getenv("P92_PREFILL_MIN")) : 32;
    // PF_PROJ_RB=1: the prefill's dense projections on rocBLAS (int8 per-row mirrors of the NVFP4
    // weights, fp32 sgemm for the bf16 ones) instead of the hand-written row-resident kernels.
    const bool pf_rb = getenv("PF_PROJ_RB") ? atoi(getenv("PF_PROJ_RB")) != 0 : true;    // default on (measured: identical tokens, -27 % proj at 8K, 1006 tok/s at 1K)
    // PF_DOWN_DET=1 (default): deterministic MoE down projection (per-row partials + fixed-order
    // combine) instead of the fp32 atomicAdd scatter, whose run-to-run order flips near-tied tokens.
    const bool pf_det = getenv("PF_DOWN_DET") ? atoi(getenv("PF_DOWN_DET")) != 0 : true;
    // P92_CHAIN=1 (default): decode crossings as a device-side chain (peer store + cross-device
    // event wait) instead of a host sync + pinned staging per crossing.
    const bool dec_chain = getenv("P92_CHAIN") ? atoi(getenv("P92_CHAIN")) != 0 : true;
    // PF_ATTN_BLK=1: the blocked attention (one query x 24 heads a block, LDS key tiles) on every
    // layer; the DSA layers then gather straight from the cache and skip the selection pack (and
    // the 584 MB a card of packed streams is not allocated).
    const int  pf_blk_mode = getenv("PF_ATTN_BLK") ? atoi(getenv("PF_ATTN_BLK")) : 2;   // 2 = v2 (fp16 tiles, dot2; default, measured), 1 = v1 (bf16 tiles), 0 = the per-head-pair kernels
    const bool pf_blk = pf_blk_mode != 0;

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
    // OWNERSHIP COMES FROM THE ARENA. A contiguous arena (header pad = 0) is the 12/12/11/11 map;
    // a ring arena (pad = block size B) gives card b % 4 the layer block b of B layers, so a
    // prefill chunk hops the ring 0 -> 1 -> 2 -> 3 -> 0 ... and the pipeline fills in 3 BLOCK-times
    // instead of 3 chunk-times. Decode crosses cards wherever the owner changes (3 or 11 times).
    int ring_blk = 0;
    auto owner=[&](int L){ if (ring_blk > 0) return (L / ring_blk) % 4; for(int d=0;d<4;d++) if(L>=lb(d)&&L<le(d)) return d; return -1; };

    P92Arena arena[4];
    for(int d=0;d<4;d++){
        CK(hipSetDevice(d));
        char p[512]; snprintf(p,sizeof p,"%s/arena_dev%d.bin",ARENA,d);
        if(p92_arena_load(p,&arena[d])){ printf("arena %d failed\n",d); return 1; }
        if(d==0) ring_blk=(int)arena[0].h.pad;
        else if((int)arena[d].h.pad!=ring_blk){ printf("arena %d ownership (ring %u) differs from dev0 (ring %d)\n",d,arena[d].h.pad,ring_blk); return 1; }
        printf("dev%d layers %d..%d, arena %d slots%s\n",d,(int)arena[d].h.layer_begin,(int)arena[d].h.layer_end-1,arena[d].slots,
               ring_blk>0?" (ring)":"");
    }
    if(ring_blk>0){ printf("ring ownership, block %d:",ring_blk); for(int L=0;L<NLAYER;L++) if(L%ring_blk==0) printf(" [%d..%d]->dev%d",L,(L+ring_blk<NLAYER?L+ring_blk:NLAYER)-1,owner(L)); printf("\n"); }
    if(owner(NLAYER-1)!=3){ printf("the last layer must live on dev3 (head placement)\n"); return 1; }

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
            int li = 0; for (int L2 = 2; L2 < L; L2++) if (owner(L2) == d) li++;   // rank among this card's MoE layers
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
        // absolute addressing everywhere under prefill: the chunk kernels write slot = position,
        // and decode's window on a sliding layer is (win_first + b) % cap = the absolute position
        // when cap = MAXPOS. Without prefill the sliding layers keep their 512-slot ring.
        w.cap = (use_prefill || w.dsa) ? MAXPOS : 512;
        CK(hipMalloc(&w.cache_kv,(size_t)w.cap*KVLORA*2));
        CK(hipMalloc(&w.cache_rope,(size_t)w.cap*ROPE*2));
        CK(hipMalloc(&w.mome_q,QLORA*2*2)); CK(hipMemset(w.mome_q,0,QLORA*2*2));
        CK(hipMalloc(&w.mome_kv,KVLORA*2*2)); CK(hipMemset(w.mome_kv,0,KVLORA*2*2));
        CK(hipMalloc(&w.mome_o,AOUT*2*2)); CK(hipMemset(w.mome_o,0,AOUT*2*2));
        if(w.dsa){ CK(hipMalloc(&w.ix_cache,(size_t)P92_IDX_DIM*w.cap*2)); }
        if(use_prefill && pf_rb){   // rocBLAS mirrors, built once on the card that owns the layer
            auto mirror=[&](const Nv& nv,int K,int8_t** w8,float** s8){
                if(nv.cols!=K){ printf("mirror: K %d != %d\n",nv.cols,K); exit(1); }
                CK(hipMalloc(w8,(size_t)nv.rows*K)); CK(hipMalloc(s8,(size_t)nv.rows*4));
                if(K==HID)        hipLaunchKernelGGL((k_pf_rb_mirror<HID>),  dim3(nv.rows),dim3(256),0,0,nv.w,nv.s,nv.s2,nv.rows,*w8,*s8);
                else if(K==QLORA) hipLaunchKernelGGL((k_pf_rb_mirror<QLORA>),dim3(nv.rows),dim3(256),0,0,nv.w,nv.s,nv.s2,nv.rows,*w8,*s8);
                else if(K==AOUT)  hipLaunchKernelGGL((k_pf_rb_mirror<AOUT>), dim3(nv.rows),dim3(256),0,0,nv.w,nv.s,nv.s2,nv.rows,*w8,*s8);
                else { printf("mirror: unsupported K %d\n",K); exit(1); } };
            auto f32=[&](const uint16_t* src,size_t n,float** dst){ CK(hipMalloc(dst,n*4));
                hipLaunchKernelGGL(k_pf_rb_bf16_f32,dim3((unsigned)((n+255)/256)),dim3(256),0,0,src,*dst,n); };
            mirror(w.qa,HID,&w.qa8,&w.qa8s); mirror(w.qb,QLORA,&w.qb8,&w.qb8s); mirror(w.o,AOUT,&w.o8,&w.o8s);
            if(w.dsa){ mirror(w.ix_wk,HID,&w.ixwk8,&w.ixwk8s); mirror(w.ix_wqb,QLORA,&w.ixwqb8,&w.ixwqb8s);
                       f32(w.ix_wproj,(size_t)PF_IDX_HEADS*HID,&w.wproj_f); }
            f32(w.kv_a,(size_t)KVA*HID,&w.kva_f);
            if(!w.dense) f32(w.router_w,(size_t)NEXP*HID,&w.router_f);
            CK(hipDeviceSynchronize());
        }
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
    hipEvent_t xev[4];
    for(int a=0;a<4;a++){ CK(hipSetDevice(a)); CK(hipEventCreateWithFlags(&xev[a],hipEventDisableTiming));
        for(int b=0;b<4;b++) if(a!=b){ hipError_t pe=hipDeviceEnablePeerAccess(b,0); if(pe!=hipSuccess&&pe!=hipErrorPeerAccessAlreadyEnabled){ printf("peer access %d->%d: %s\n",a,b,hipGetErrorString(pe)); return 1; } (void)hipGetLastError(); } }

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

    // ======================= CHUNKED PREFILL (in-process) =======================
    PfConfig cfg = pf_config();
    if (!getenv("PF_CHUNK")) cfg.chunk = 256;                 // the banked production chunk
    const int T = cfg.chunk;
    if (use_prefill) printf("prefill: chunk %d, TQ %d, P%d/D4, pipeline depth %d\n", T, cfg.attn_tq, cfg.prefill_cards, cfg.pipe_depth);
    struct Pf {
        uint16_t *state, *flat;          // [T][4][HID] bf16, [T][4*HID] bf16
        float *norm, *mixf, *comb, *logits;
        int8_t *xq; float *xs;
        float *qlora, *qnorm, *qup, *kvdn, *curkv, *qabs, *qrope, *ctx, *aout, *hidden;
        float *rlog; int32_t *rids; float *rwts;
        PfGroup *groups; int32_t *rows, *order, *ngroups; float *rwt;
        float *hid9, *hs9; int8_t *hq9;
        int32_t *tokslot; float *dpart;           // deterministic down: token -> grouped rows, per-row partials
        float *cosv, *sinv;
        uint16_t *state2; float *pre, *post;
        float *kidx, *qidx, *qidxT, *whd, *scores; int32_t *sel, *nsel;
        int32_t *uni, *ucnt; uint8_t *umask;
        uint16_t *lat_pack, *pe_pack; uint8_t *mask_pack; int32_t *packoff;
        int8_t *xq1024, *xq6144, *dhq; float *xs1024, *xs6144, *dhid, *dhs;
    } pf[4];
    const size_t GMAX = (size_t)T * (PF_TOPK + 1);
    if (use_prefill) for (int d = 0; d < 4; d++) {
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
        A((void**)&c.tokslot,(size_t)T*PF_TOKSLOTS*4); A((void**)&c.dpart,GMAX*PF_NEMBD*4);
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
        if (!pf_blk) {
        A((void**)&c.lat_pack,(size_t)T*PF_IDX_TOPK*KVLORA*2);
        A((void**)&c.pe_pack,(size_t)T*PF_IDX_TOPK*ROPE*2);
        A((void**)&c.mask_pack,(size_t)T*PF_IDX_TOPK*16);
        } else { c.lat_pack = nullptr; c.pe_pack = nullptr; c.mask_pack = nullptr; }
        A((void**)&c.packoff,(size_t)((T+1)/2)*4);
        // the RoPE table covers every position in the run and is built once
        hipLaunchKernelGGL(k_pf_rope_table,dim3(MAXPOS),dim3(64),0,0,c.cosv,c.sinv,MAXPOS);
    }

    // ---- arena scale layout. The prefill expert kernels read the per-16 group scales
    // LANE-CONTIGUOUS (k_pf_shuffle_scales_*); the decode expert kernels read the arena's own
    // order. Both kernel families are the frozen authorities, so the layout is flipped IN PLACE at
    // the phase boundary instead of touching either: ~1.5 GB of scales a card, a few ms each way,
    // verified as a permutation every time.
    int arena_layout[4] = {0,0,0,0};   // 0 = decode (arena) order, 1 = prefill (lane-contiguous)
    auto arena_layout_set = [&](int d, int want) {
        if (arena_layout[d] == want) return;
        CK(hipSetDevice(d));
        const size_t GU = (size_t)arena[d].slots * PF_NFF * (PF_NEMBD/16);
        const size_t DN = (size_t)arena[d].slots * PF_NEMBD * (PF_NFF/16);
        const int gurows = arena[d].slots * PF_NFF, dnrows = arena[d].slots * PF_NEMBD;
        uint8_t *tmp = nullptr; CK(hipMalloc(&tmp, GU > DN ? GU : DN));
        uint32_t *bad = nullptr; CK(hipMalloc(&bad, 4)); CK(hipMemset(bad, 0, 4));
        auto flip = [&](uint8_t *buf, int rows, size_t bytes, int per_row, int fan) {
            if (want) {
                if (fan == 5) hipLaunchKernelGGL(k_pf_shuffle_scales_gu,dim3((rows+7)/8),dim3(256),0,0,buf,tmp,rows);
                else          hipLaunchKernelGGL(k_pf_shuffle_scales_dn,dim3((rows+7)/8),dim3(256),0,0,buf,tmp,rows);
                hipLaunchKernelGGL(k_pf_shuffle_check,dim3((rows+7)/8),dim3(256),0,0,buf,tmp,rows,per_row,fan,bad);
            } else {
                hipLaunchKernelGGL(k_pe_unshuffle_scales,dim3((rows+7)/8),dim3(256),0,0,buf,tmp,rows,per_row,fan);
                hipLaunchKernelGGL(k_pf_shuffle_check,dim3((rows+7)/8),dim3(256),0,0,tmp,buf,rows,per_row,fan,bad);
            }
            CK(hipMemcpyAsync(buf,tmp,bytes,hipMemcpyDeviceToDevice,0));
        };
        flip(arena[d].gs, gurows, GU, PF_NEMBD/16, 5);
        flip(arena[d].us, gurows, GU, PF_NEMBD/16, 5);
        flip(arena[d].ds, dnrows, DN, PF_NFF/16, 2);
        CK(hipDeviceSynchronize());
        uint32_t nbad = 0; CK(hipMemcpy(&nbad,bad,4,hipMemcpyDeviceToHost));
        if (nbad) { printf("arena %d scale layout switch is not a permutation: %u\n",d,nbad); exit(1); }
        CK(hipFree(tmp)); CK(hipFree(bad));
        arena_layout[d] = want;
    };

    // ---- the block schedule. A block is a contiguous layer range owned by one card; a chunk
    // visits blocks 0..nblk-1 in order. Contiguous arena: 4 blocks = the cards' ranges. Ring
    // arena: blocks of ring_blk layers, card b % 4 owns block b.
    const int nblk = ring_blk > 0 ? (NLAYER + ring_blk - 1) / ring_blk : 4;
    auto blk_owner = [&](int b) { return ring_blk > 0 ? (b % 4) : b; };
    auto blk_L0 = [&](int b) { return ring_blk > 0 ? b * ring_blk : lb(b); };
    auto blk_L1 = [&](int b) { const int e = ring_blk > 0 ? (b + 1) * ring_blk : le(b); return e < NLAYER ? e : NLAYER; };
    const int PF_NSLOT = 4;
    PfPipeline pipe; memset(&pipe, 0, sizeof pipe);
    size_t land_bytes = (size_t)T * 4 * HID * 2;             // one chunk of four-stream state
    if (use_prefill) {
        if (pf_pipeline_init(&pipe, &cfg, MAXPOS)) { printf("pipeline init failed\n"); return 1; }
        // PF_NSLOT landing slots per card (the driver allocates one): a producer may run ahead of
        // its consumer by up to PF_NSLOT jobs before it has to wait for a slot to be released.
        p92_p2p_free(&pipe.p2p);
        if (p92_p2p_init(&pipe.p2p, (size_t)PF_NSLOT * land_bytes)) { printf("p2p re-init (%d slots) failed\n", PF_NSLOT); return 1; }
    }
    PfRb rb; memset(&rb, 0, sizeof rb);
    if (use_prefill && pf_rb) {
        if (pf_rb_init(&rb, T, QUP)) { printf("rocBLAS init failed\n"); return 1; }
        printf("prefill projections: rocBLAS int8 (per-row mirrors) + sgemm\n");
    }
    const bool pf_trace = getenv("PF_TRACE") != nullptr;
    const bool pf_ustats = getenv("PF_UNION_STATS") != nullptr;
    std::vector<int> ustat_u, ustat_n; std::mutex ustat_mx;
    const int pf_trace_pos = getenv("PF_TRACE_POS") ? atoi(getenv("PF_TRACE_POS")) : 0;
    std::atomic<long> sens_fired(0);
    std::vector<uint16_t> seedbuf((size_t)T * 4 * HID);

    // one chunk through the layers this card owns
    auto pf_layers = [&](int d, int base, int rows_in_chunk, int L0, int L1) {
        CK(hipSetDevice(d)); Pf &c = pf[d]; PfTimers &tm = pipe.timers[d];
        for (int L = L0; L < L1; L++) {
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
            if (pf_rb) {   // per-token int8 of the normed input (reused by the indexer wk), q_a on Tensile
                hipLaunchKernelGGL((k_pf_rb_quant_tok<HID>),dim3(RC),dim3(256),0,0,c.norm,RC,rb.xq2560[d],rb.xs2560[d]);
                if (pf_rb_gemm_i8(rb.h[d],w.qa8,QLORA,HID,rb.xq2560[d],RC,rb.C[d])) exit(1);
                hipLaunchKernelGGL(k_pf_rb_epi,dim3((RC*QLORA+255)/256),dim3(256),0,0,rb.C[d],w.qa8s,rb.xs2560[d],RC,QLORA,c.qlora);
            } else
            hipLaunchKernelGGL((k_pf_proj<HID>),dim3((QLORA+7)/8),dim3(256),0,0,
                               w.qa.w,w.qa.s,c.xq,c.xs,w.qa.s2,RC,QLORA,c.qlora);
            hipLaunchKernelGGL(k_pf_mome_st,dim3((QLORA+255)/256),dim3(256),0,0,
                               c.qlora,(const uint16_t*)w.qa_conv,w.mome_q,RC,QLORA,QLORA);
            hipLaunchKernelGGL((k_pf_rmsnorm_k<QLORA>),dim3(RC),dim3(256),0,0,
                               c.qlora,(const uint16_t*)w.q_norm,c.qnorm,RC,1e-5f);
            hipLaunchKernelGGL((k_pf_quant_k<QLORA>),dim3(RC),dim3(256),0,0,
                               c.qnorm,RC,c.xq1024,c.xs1024);
            if (pf_rb) {
                hipLaunchKernelGGL((k_pf_rb_quant_tok<QLORA>),dim3(RC),dim3(256),0,0,c.qnorm,RC,rb.xq1024[d],rb.xs1024[d]);
                if (pf_rb_gemm_i8(rb.h[d],w.qb8,QUP,QLORA,rb.xq1024[d],RC,rb.C[d])) exit(1);
                hipLaunchKernelGGL(k_pf_rb_epi,dim3((RC*QUP+255)/256),dim3(256),0,0,rb.C[d],w.qb8s,rb.xs1024[d],RC,QUP,c.qup);
            } else
            hipLaunchKernelGGL((k_pf_proj<QLORA>),dim3((QUP+7)/8),dim3(256),0,0,
                               w.qb.w,w.qb.s,c.xq1024,c.xs1024,w.qb.s2,RC,QUP,c.qup);
            // ---- key/value path: kv_a, MoME, kv_norm, publish into the cache
            if (pf_rb) { if (pf_rb_gemm_f32(rb.h[d],w.kva_f,KVA,HID,c.norm,RC,c.kvdn)) exit(1); }
            else
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
                if (pf_rb) {
                    if (pf_rb_gemm_i8(rb.h[d],w.ixwk8,PF_IDX_DIM,HID,rb.xq2560[d],RC,rb.C[d])) exit(1);
                    hipLaunchKernelGGL(k_pf_rb_epi,dim3((RC*PF_IDX_DIM+255)/256),dim3(256),0,0,rb.C[d],w.ixwk8s,rb.xs2560[d],RC,PF_IDX_DIM,c.kidx);
                } else
                hipLaunchKernelGGL((k_pf_proj<HID>),dim3((PF_IDX_DIM+7)/8),dim3(256),0,0,
                                   w.ix_wk.w,w.ix_wk.s,c.xq,c.xs,w.ix_wk.s2,RC,PF_IDX_DIM,c.kidx);
                hipLaunchKernelGGL(k_pf_ix_key,dim3(RC),dim3(PF_IDX_DIM),0,0,
                                   c.kidx,(const uint16_t*)w.ix_knorm,c.cosv,c.sinv,
                                   RC,base,w.cap,1e-5f,w.ix_cache);
                pf_stage_end(&tm, PF_S_DSA, 0);
                if (cfg.dsa_sel && base + RC > PF_IDX_TOPK) {
                    pf_stage_begin(&tm, PF_S_DSA_SCORE, 0);
                    if (pf_rb) {
                        if (pf_rb_gemm_i8(rb.h[d],w.ixwqb8,PF_IDX_HEADS*PF_IDX_DIM,QLORA,rb.xq1024[d],RC,rb.C[d])) exit(1);
                        hipLaunchKernelGGL(k_pf_rb_epi,dim3((RC*PF_IDX_HEADS*PF_IDX_DIM+255)/256),dim3(256),0,0,rb.C[d],w.ixwqb8s,rb.xs1024[d],RC,PF_IDX_HEADS*PF_IDX_DIM,c.qidx);
                    } else
                    hipLaunchKernelGGL((k_pf_proj<QLORA>),dim3((PF_IDX_HEADS*PF_IDX_DIM+7)/8),dim3(256),0,0,
                                       w.ix_wqb.w,w.ix_wqb.s,c.xq1024,c.xs1024,w.ix_wqb.s2,
                                       RC,PF_IDX_HEADS*PF_IDX_DIM,c.qidx);
                    if (cfg.dsa_scorer == 3)
                        hipLaunchKernelGGL(k_pf_ix_query_t,dim3(PF_IDX_HEADS,RC),dim3(64),0,0,
                                           c.qidx,c.qidxT,c.cosv,c.sinv,RC,base);
                    else
                        hipLaunchKernelGGL(k_pf_ix_query,dim3(PF_IDX_HEADS,RC),dim3(32),0,0,
                                           c.qidx,c.cosv,c.sinv,RC,base);
                    if (pf_rb) { if (pf_rb_gemm_f32(rb.h[d],w.wproj_f,PF_IDX_HEADS,HID,c.norm,RC,c.whd)) exit(1); }
                    else
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
                    if (pf_blk) {
                        // blocked attention gathers the selection from the cache itself
                    } else if (cfg.sp_g2) {
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
            if (pf_blk_mode == 2) {
                hipLaunchKernelGGL(k_pf_attn_blk2,dim3(RC,PF_NH/PFB2_HB),dim3(PFB2_NT),0,0,
                                   c.qabs,c.qrope,(const uint16_t*)w.cache_kv,(const uint16_t*)w.cache_rope,
                                   (const uint16_t*)w.sink_kv_n,(const uint16_t*)w.sink_rope,
                                   sparse_ok ? (const int32_t*)c.sel : (const int32_t*)nullptr,
                                   sparse_ok ? (const int32_t*)c.nsel : (const int32_t*)nullptr,
                                   PF_IDX_TOPK, base, RC, sparse_ok ? 0 : win, c.ctx);
            } else if (pf_blk) {
                hipLaunchKernelGGL(k_pf_attn_blk,dim3(RC,PF_NH/PFB_HB),dim3(PFB_NT),0,0,
                                   c.qabs,c.qrope,(const uint16_t*)w.cache_kv,(const uint16_t*)w.cache_rope,
                                   (const uint16_t*)w.sink_kv_n,(const uint16_t*)w.sink_rope,
                                   sparse_ok ? (const int32_t*)c.sel : (const int32_t*)nullptr,
                                   sparse_ok ? (const int32_t*)c.nsel : (const int32_t*)nullptr,
                                   PF_IDX_TOPK, base, RC, sparse_ok ? 0 : win, c.ctx);
            } else
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
            if (pf_rb) {
                hipLaunchKernelGGL((k_pf_rb_quant_tok<AOUT>),dim3(RC),dim3(256),0,0,c.aout,RC,rb.xq6144[d],rb.xs6144[d]);
                if (pf_rb_gemm_i8(rb.h[d],w.o8,HID,AOUT,rb.xq6144[d],RC,rb.C[d])) exit(1);
                hipLaunchKernelGGL(k_pf_rb_epi,dim3((RC*HID+255)/256),dim3(256),0,0,rb.C[d],w.o8s,rb.xs6144[d],RC,HID,c.hidden);
            } else {
            hipLaunchKernelGGL((k_pf_quant_k<AOUT>),dim3(RC),dim3(256),0,0,
                               c.aout,RC,c.xq6144,c.xs6144);
            hipLaunchKernelGGL((k_pf_proj<AOUT>),dim3((HID+7)/8),dim3(256),0,0,
                               w.o.w,w.o.s,c.xq6144,c.xs6144,w.o.s2,RC,HID,c.hidden);
            }
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
                if (pf_rb) { if (pf_rb_gemm_f32(rb.h[d],w.router_f,NEXP,HID,c.norm,RC,c.rlog)) exit(1); }
                else
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
                if (pf_det)
                hipLaunchKernelGGL(k_pf_group_det,dim3(1),dim3(256),0,0,
                                   c.rids,c.rwts,RC,w.arena_slot_base,NEXP,
                                   c.groups,c.rows,c.rwt,c.order,c.ngroups,c.tokslot);
                else
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
                if (pf_det) {
                    hipLaunchKernelGGL(k_pf_down_part,dim3(HID/8,PF_GROUP_MAX),dim3(256),0,0,
                                       arena[d].dw,arena[d].ds,c.hq9,c.hs9,arena[d].s2,
                                       c.groups,c.order,c.rwt,c.dpart);
                    hipLaunchKernelGGL(k_pf_down_combine,dim3(RC),dim3(256),0,0,c.dpart,c.tokslot,RC,c.hidden);
                } else {
                CK(hipMemsetAsync(c.hidden,0,(size_t)RC*HID*4,0));
                hipLaunchKernelGGL(k_pf_down,dim3(HID/8,PF_GROUP_MAX),dim3(256),0,0,
                                   arena[d].dw,arena[d].ds,c.hq9,c.hs9,arena[d].s2,
                                   c.groups,c.order,c.rows,c.rwt,c.hidden);
                }
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
            for (int L = L0; L < L1; L++) {
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

    // Runs the chunk pipeline over toks[0..n) at absolute positions pos0.., writing every layer's
    // caches and MoME carries exactly as the decoder would have. Returns the wall in ms.
    //
    // THE HANDOFF IS A COPY OUT OF THE LANDING SLOT. The producer pushes chunk k into the consumer's
    // landing slot (k & 1) and raises ready[d]; the consumer copies the slot into its own state
    // buffer LOCALLY (the pushed bytes already sit in its HBM) and raises consumed[d], which is the
    // producer's permission to reuse that slot two chunks later. The frozen bench pushed and then
    // computed on a buffer nothing had written - cards 1-3 ran every chunk on uninitialised state.
    auto run_prefill = [&](int pos0, const int *toks, int n) -> double {
        if (n <= 0) return 0.0;
        for (int d = 0; d < 4; d++) arena_layout_set(d, 1);
        const int nchunks = (n + T - 1) / T;
        // per-card job lists (chunk k, block b) in READINESS order: key = (k + b) * nchunks + k.
        // A card's producer is always one card (the owner of b-1), and the producer's job order maps
        // onto the consumer's job order monotonically, so landing slots can be assigned by position.
        std::vector<std::vector<std::pair<int,int>>> jobs(4);
        for (int b = 0; b < nblk; b++) for (int k = 0; k < nchunks; k++) jobs[blk_owner(b)].push_back({k, b});
        for (int g = 0; g < 4; g++) std::sort(jobs[g].begin(), jobs[g].end(),
            [&](const std::pair<int,int>& a, const std::pair<int,int>& c) {
                const long ka = (long)(a.first + a.second) * nchunks + a.first, kc = (long)(c.first + c.second) * nchunks + c.first;
                return ka < kc; });
        size_t maxjobs = 0; for (int g = 0; g < 4; g++) maxjobs = std::max(maxjobs, jobs[g].size());
        auto jpos = [&](int g, int k, int b) { const auto& J = jobs[g];
            for (size_t i = 0; i < J.size(); i++) if (J[i].first == k && J[i].second == b) return (int)i; return -1; };
        std::vector<std::atomic<int>> done((size_t)4 * maxjobs);   // [g * maxjobs + pos] = 1 once pushed on
        for (auto& x : done) x.store(0);
        std::atomic<int> consumed[4];
        for (int d = 0; d < 4; d++) consumed[d].store(-1);
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> th;
        for (int g = 0; g < 4; g++) th.emplace_back([&, g]{
            CK(hipSetDevice(g));
            const auto& J = jobs[g];
            for (size_t pos = 0; pos < J.size(); pos++) {
                const int k = J[pos].first, b = J[pos].second;
                const int base = k * T;
                const int rows = (base + T <= n) ? T : (n - base);
                const size_t bytes = (size_t)rows * 4 * HID * 2;
                if (b == 0) {                                     // the prompt: embeddings replicated across the four streams
                    for (int r = 0; r < rows; r++) {
                        const uint16_t *emb = ck.embed_ptr(toks[base + r]);
                        for (int k2 = 0; k2 < 4; k2++) memcpy(&seedbuf[((size_t)r * 4 + k2) * HID], emb, (size_t)HID * 2);
                    }
                    CK(hipMemcpy(pf[g].state, seedbuf.data(), bytes, hipMemcpyHostToDevice));
                } else {                                          // block b-1's output, pushed into this card's landing slot
                    const int src = blk_owner(b - 1), spos = jpos(src, k, b - 1);
                    while (done[(size_t)src * maxjobs + spos].load(std::memory_order_acquire) == 0) std::this_thread::yield();
                    CK(hipMemcpy(pf[g].state, (const char*)pipe.p2p.land[g] + (size_t)(pos % PF_NSLOT) * land_bytes,
                                 bytes, hipMemcpyDeviceToDevice));
                }
                // consumed[] is a POSITION counter, advanced by every job: a seed job frees the slot its
                // position maps to just like a copied job does. (Counting only copied jobs deadlocked the
                // ring: card 0's first jobs are seeds, so card 3 waited for a slot card 0 would never
                // release while card 0 waited for card 3's push.)
                consumed[g].store((int)pos, std::memory_order_release);
                pf_layers(g, pos0 + base, rows, blk_L0(b), blk_L1(b));
                if (b + 1 < nblk) {
                    const int dst = blk_owner(b + 1), dpos = jpos(dst, k, b + 1);
                    while (consumed[dst].load(std::memory_order_acquire) < dpos - PF_NSLOT) std::this_thread::yield();
                    PfTimers &tm = pipe.timers[g];
                    pf_stage_begin(&tm, PF_S_HANDOFF, 0);
                    p92_p2p_push(&pipe.p2p, g, dst, pf[g].state, (size_t)(dpos % PF_NSLOT) * land_bytes, bytes, 0, (uint32_t)(k + 1));
                    CK(hipSetDevice(g)); CK(hipStreamSynchronize(pipe.p2p.push[g]));
                    pf_stage_end(&tm, PF_S_HANDOFF, 0);
                    pf_stage_accumulate(&tm);
                }
                done[(size_t)g * maxjobs + pos].store(1, std::memory_order_release);
            }
        });
        for (auto &t : th) t.join();
        for (int d = 0; d < 4; d++) { CK(hipSetDevice(d)); CK(hipDeviceSynchronize()); }
        const double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - t0).count();
        for (int d = 0; d < 4; d++) arena_layout_set(d, 0);   // the decoder's order again (outside the timed region)
        if (cfg.verbose) {
            PfResult res{}; res.prompt_tokens = n; res.chunk = T; res.prefill_ms = ms; res.prefill_toks = n / (ms / 1000.0);
            pipe.nchunks = nchunks; pf_report(&pipe, &res);
            for (int d = 0; d < 4; d++) for (int st = 0; st < PF_S_COUNT; st++) pipe.timers[d].ms[st] = 0.0;
        }
        return ms;
    };
    // ===================== end CHUNKED PREFILL ===================================

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
    // ---- persistent serving (P92_SERVE) ---------------------------------
    // Keep the ingested context resident and answer further requests from stdin
    // without re-ingesting it. force holds tokens to feed verbatim; when it
    // drains, the loop generates. Positions advance across requests.
    const bool serve = getenv("P92_SERVE") != nullptr;
    const int maxnew = getenv("P92_MAXNEW") ? atoi(getenv("P92_MAXNEW")) : 128;
    p92::Tokenizer *serve_tok = tokp;
    if (serve) {
        static p92::Tokenizer stk(std::string(CKDIR) + "/tokenizer.json");
        if (!serve_tok) serve_tok = &stk;
    }
    std::deque<int> force;
    std::vector<int> out_tokens;
    int token = prompt.empty() ? START : prompt[0];
    double first_ms=0, total_ms=0; int steps_timed = 0;
    int base_pos = 0;
    double prefill_ms = 0.0; int prefill_n = 0;
    if (use_prefill && (int)prompt.size() >= prefill_min) {
        // all but the last prompt token go through the chunk pipeline; the decoder consumes the
        // last one and produces the first generated token
        prefill_n = (int)prompt.size() - 1;
        prefill_ms = run_prefill(0, prompt.data(), prefill_n);
        base_pos = prefill_n; token = prompt.back();
        printf("[prefill] %d tokens in %.1f ms = %.1f tok/s (chunk %d)\n", prefill_n, prefill_ms,
               1000.0 * prefill_n / prefill_ms, T);
        fflush(stdout);
    } else for (size_t i = 1; i < prompt.size(); i++) force.push_back(prompt[i]);
    int budget = NTOK - base_pos;
    for(;;){
    for(int step=0; step<budget; step++){
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
                if(dec_chain){
                    CK(hipSetDevice(cur));
                    hipLaunchKernelGGL(k_pe_copy16,dim3((4*HID*2/16+255)/256),dim3(256),0,0,
                                       (const uint4*)sc[cur].sa,(uint4*)sc[d].sa,4*HID*2/16);
                    CK(hipEventRecord(xev[cur],0));
                    CK(hipSetDevice(d));
                    CK(hipStreamWaitEvent(0,xev[cur],0));
                } else {
                    CK(hipSetDevice(cur)); CK(hipDeviceSynchronize());
                    if(p92_transport_send(&tx,cur,sc[cur].sa)) exit(1);
                    if(p92_transport_recv_to(&tx,cur,d,sc[d].sa)) exit(1);
                    if(p92_transport_wait(&tx,d)) exit(1);
                }
                cur=d;
            }
            CK(hipSetDevice(d));
            run_layer(L,base_pos+step);
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
        if(step==0) first_ms=ms; else { total_ms+=ms; steps_timed++; }
        if(text_mode && tokp && force.empty()){
            const std::string piece = tokp->decode_token(next);
            fputs(piece.c_str(), stdout); fflush(stdout);
        } else if(!text_mode){
            printf("  step %2d  position %2d  %6d -> %6d   %8.2f ms\n",step,step,token,next,ms);
            printf("TOK %d\n", next);
        }
        fflush(stdout);
        out_tokens.push_back(next);
        if(!force.empty()){ token = force.front(); force.pop_front(); }
        else token = next;
    }
    base_pos += budget;
    if(!serve) break;
    // ---- next request: append it to the context and keep going ----------
    {
        std::string line; bool got=false;
        while(std::getline(std::cin,line)){ if(!line.empty()){ got=true; break; } }
        if(!got) break;
        if(text_mode) fputs("\n", stdout);
        const std::string tq = p92::Tokenizer::chat_prompt(line);
        for(int32_t t : serve_tok->encode(tq)){ force.push_back((int)t); prompt.push_back((int)t); }
        out_tokens.clear(); first_ms=0; total_ms=0; steps_timed=0;
        printf("[serve] request %d tokens, at position %d\n", (int)force.size(), base_pos);
        if (use_prefill && (int)force.size() >= prefill_min) {
            // the previous answer's last token (not yet fed) + the request, all but its last token
            std::vector<int> req; req.push_back(token); req.insert(req.end(), force.begin(), force.end()); force.clear();
            const int m = (int)req.size() - 1;
            const double pms = run_prefill(base_pos, req.data(), m);
            printf("[serve] prefill %d tokens at position %d in %.1f ms = %.1f tok/s\n", m, base_pos, pms, 1000.0 * m / pms);
            base_pos += m; token = req.back(); budget = maxnew;
        } else budget = (int)force.size() + maxnew;
        fflush(stdout);
    }
    }   // for(;;) serve loop
    printf("\ntokens:"); for(int t:out_tokens) printf(" %d",t); printf("\n");
    if(steps_timed>0) printf("first token %.2f ms, steady %.2f ms/token = %.2f tok/s over %d steps\n",
                      first_ms,total_ms/steps_timed,1000.0*steps_timed/total_ms,steps_timed);
    if(prefill_n) printf("TTFT %.1f ms = prefill %d tokens %.1f ms + first decode step %.2f ms\n",
                         prefill_ms+first_ms, prefill_n, prefill_ms, first_ms);
    if (use_prefill) pf_pipeline_free(&pipe);
    p92_transport_free(&tx);
    return 0;
}

