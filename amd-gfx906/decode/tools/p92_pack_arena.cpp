// Build a per-device resident routed-expert arena from the NVFP4 artifact.
//
// The engine loads one file per GPU with six contiguous reads instead of
// walking 33,792 manifest records at boot. Layout matches the kernel indexing
// exactly: a plane per projection, addressed by slot = local_moe_index * 256 +
// expert, so k_p92_f4_gateup finds gate row r of slot s at
// gate_w + s * P92_GU_W_BYTES + r * (NEMBD/2).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#define NEMBD 2560
#define NFF   1024
#define NEXP  256
#define NSHARED 1
#define NSLOT (NEXP + NSHARED)   // slot 256 is the shared expert
#define TRUNK 46
#define NGPU  4
#define GU_W  ((size_t)NFF   * (NEMBD/2))    // 1,310,720
#define GU_S  ((size_t)NFF   * (NEMBD/16))   //   163,840
#define DN_W  ((size_t)NEMBD * (NFF/2))      // 1,310,720
#define DN_S  ((size_t)NEMBD * (NFF/16))     //   163,840

#pragma pack(push,1)
struct Hdr { char magic[8]; uint32_t version, record_bytes, tensor_count, reserved;
             uint64_t weight_bytes, scale_bytes, source_payload_bytes; };
struct Rec { char name[96]; uint32_t rows, columns; uint64_t weight_offset, scale_offset;
             float weight_scale_2; uint32_t reserved; };
struct ArenaHdr { char magic[8]; uint32_t version, device, layer_begin, layer_end;
                  uint32_t moe_layers, experts, pad;
                  uint64_t off_gw, off_uw, off_dw, off_gs, off_us, off_ds, off_s2, total; };
#pragma pack(pop)

static int lb(int d){ int b=TRUNK/NGPU, r=TRUNK%NGPU; return d*b+(d<r?d:r); }
static int le(int d){ int b=TRUNK/NGPU, r=TRUNK%NGPU; return (d+1)*b+((d+1)<r?(d+1):r); }
// RING OWNERSHIP (optional third argument BLK > 0): card d owns the layer blocks
// b = d, d+NGPU, d+2*NGPU, ... of BLK layers each, so a chunk hops the ring
// 0 -> 1 -> 2 -> 3 -> 0 ... and the prefill pipeline fills in (NGPU-1) BLOCK-times
// instead of (NGPU-1) chunk-times (the short-prompt lever). The arena header
// records BLK in `pad` (0 = the contiguous 12/12/11/11 layout), so the engine
// reads its ownership map from the arena it loads rather than from a switch.
static int g_blk = 0;
static int owner_of(int L){ return g_blk > 0 ? (L / g_blk) % NGPU : -1; }

static void rd(int fd, void*p, size_t n, off_t o){
    size_t d=0; while(d<n){ ssize_t k=pread(fd,(char*)p+d,n-d,o+d);
        if(k<=0){ fprintf(stderr,"pread fail %zd\n",k); exit(1);} d+=k; } }
static void wr(int fd, const void*p, size_t n, off_t o){
    size_t d=0; while(d<n){ ssize_t k=pwrite(fd,(const char*)p+d,n-d,o+d);
        if(k<=0){ fprintf(stderr,"pwrite fail %zd\n",k); exit(1);} d+=k; } }

int main(int argc,char**argv){
    if(argc<3){ printf("usage: %s ARTIFACT_DIR OUT_DIR [RING_BLK]\n",argv[0]); return 2; }
    const std::string A=argv[1], O=argv[2];
    g_blk = argc>3 ? atoi(argv[3]) : 0;
    if(g_blk>0) printf("ring ownership: block %d layers, card d owns blocks d, d+%d, ...\n", g_blk, NGPU);
    int mf=open((A+"/manifest.bin").c_str(),O_RDONLY);
    int wf=open((A+"/weights.nvfp4").c_str(),O_RDONLY);
    int sf=open((A+"/scales.e4m3").c_str(),O_RDONLY);
    if(mf<0||wf<0||sf<0){ perror("open artifact"); return 1; }
    Hdr h; rd(mf,&h,sizeof h,0);
    if(memcmp(h.magic,"P92FP41",7)){ fprintf(stderr,"bad magic\n"); return 1; }
    std::vector<Rec> recs(h.tensor_count);
    rd(mf,recs.data(),(size_t)h.tensor_count*h.record_bytes,sizeof h);
    std::unordered_map<std::string,const Rec*> byname;
    byname.reserve(h.tensor_count*2);
    for(auto&r:recs) byname[std::string(r.name,strnlen(r.name,96))]=&r;
    printf("manifest: %u tensors, %.2f GB weights, %.2f GB scales\n",
           h.tensor_count, h.weight_bytes/1e9, h.scale_bytes/1e9);

    std::vector<uint8_t> wbuf(GU_W), sbuf(GU_S);
    for(int d=0; d<NGPU; d++){
        std::vector<int> ml;
        if(g_blk>0){ for(int L=2; L<TRUNK; L++) if(owner_of(L)==d) ml.push_back(L); }   // 0,1 are dense
        else for(int L=lb(d); L<le(d); L++) if(L>=2) ml.push_back(L);
        const size_t n=(size_t)ml.size()*NSLOT;
        ArenaHdr ah{}; memcpy(ah.magic,"P92AREN",8); ah.version=1; ah.device=d;
        ah.layer_begin=g_blk>0?0:lb(d); ah.layer_end=g_blk>0?TRUNK:le(d); ah.moe_layers=(uint32_t)ml.size(); ah.experts=NSLOT;
        ah.pad=(uint32_t)g_blk;   // ring block size, 0 = contiguous
        size_t o=(sizeof(ArenaHdr)+4095)&~4095ull;
        ah.off_gw=o; o+=n*GU_W;  ah.off_uw=o; o+=n*GU_W;  ah.off_dw=o; o+=n*DN_W;
        ah.off_gs=o; o+=n*GU_S;  ah.off_us=o; o+=n*GU_S;  ah.off_ds=o; o+=n*DN_S;
        ah.off_s2=o; o+=n*3*sizeof(float); ah.total=o;
        char path[512]; snprintf(path,sizeof path,"%s/arena_dev%d.bin",O.c_str(),d);
        int of=open(path,O_WRONLY|O_CREAT|O_TRUNC,0644);
        if(of<0){ perror("open out"); return 1; }
        if(ftruncate(of,(off_t)ah.total)){ perror("ftruncate"); return 1; }
        wr(of,&ah,sizeof ah,0);
        std::vector<float> s2(n*3);
        printf("device %d: layers %d..%d, %zu MoE layers, arena %.2f GB -> %s\n",
               d,(int)ah.layer_begin,(int)ah.layer_end-1,ml.size(),ah.total/1e9,path);
        if(g_blk>0){ printf("  MoE layers:"); for(int L:ml) printf(" %d",L); printf("\n"); }
        for(size_t li=0; li<ml.size(); li++){
            for(int e=0;e<NSLOT;e++){
                const size_t slot=li*NSLOT+e;
                const bool shared = (e == NEXP);
                const char* proj[3]={"gate_proj","up_proj","down_proj"};
                const uint64_t ow[3]={ah.off_gw,ah.off_uw,ah.off_dw};
                const uint64_t os[3]={ah.off_gs,ah.off_us,ah.off_ds};
                const size_t  bw[3]={GU_W,GU_W,DN_W}, bs[3]={GU_S,GU_S,DN_S};
                for(int p=0;p<3;p++){
                    char nm[160];
                    if (shared)
                        snprintf(nm,sizeof nm,"model.layers.%d.mlp.shared_experts.%s.weight",ml[li],proj[p]);
                    else
                        snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.%s.weight",ml[li],e,proj[p]);
                    auto it=byname.find(nm);
                    if(it==byname.end()){ fprintf(stderr,"missing %s\n",nm); return 1; }
                    const Rec* r=it->second;
                    if((size_t)r->rows*r->columns/2 != bw[p]){
                        fprintf(stderr,"shape %s %ux%u\n",nm,r->rows,r->columns); return 1; }
                    rd(wf,wbuf.data(),bw[p],(off_t)r->weight_offset);
                    wr(of,wbuf.data(),bw[p],(off_t)(ow[p]+slot*bw[p]));
                    rd(sf,sbuf.data(),bs[p],(off_t)r->scale_offset);
                    wr(of,sbuf.data(),bs[p],(off_t)(os[p]+slot*bs[p]));
                    s2[slot*3+p]=r->weight_scale_2;
                }
            }
            printf("  layer %d done (%zu/%zu)\n",ml[li],li+1,ml.size()); fflush(stdout);
        }
        wr(of,s2.data(),s2.size()*sizeof(float),(off_t)ah.off_s2);
        close(of);
    }
    printf("packed.\n");
    return 0;
}
