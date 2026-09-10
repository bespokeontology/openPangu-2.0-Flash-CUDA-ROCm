// What does a layer handoff between two MI50s actually cost on this box?
//
// The recorded figure was 70-90 us a crossing, but that came from a run whose
// per-stage residue bundled a blocking H2D push, a blocking D2H pull,
// hipSetDevice and synchronous-memcpy driver overhead. Before building the
// engine around host staging, measure the paths that actually exist.
//
// openPangu payloads that cross: the 2560-wide residual (10,240 B fp32, or
// 5,120 B bf16) and the four-stream mHC block (10,240 bf16 = 20,480 B).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)

static const size_t SIZES[] = {5120, 10240, 20480, 262144, 4194304};
static const char*  NAMES[] = {"residual bf16 5 KB","residual fp32 10 KB","mHC 4-stream 20 KB","256 KB","4 MB"};
static const int NS = 5;

int main(){
    int n=0; CHK(hipGetDeviceCount(&n));
    printf("devices: %d\n", n);
    if(n<2){ printf("need 2+\n"); return 1; }

    // ---- 1. does peer copy actually move bytes? -------------------------
    printf("\n=== peer capability, and whether it moves data ===\n");
    int enabled=0, moved=0, pairs=0;
    for(int a=0;a<n;a++) for(int b=0;b<n;b++){
        if(a==b) continue; pairs++;
        int can=0; CHK(hipDeviceCanAccessPeer(&can,a,b));
        CHK(hipSetDevice(a));
        const hipError_t e=hipDeviceEnablePeerAccess(b,0);
        const bool ok = (e==hipSuccess||e==hipErrorPeerAccessAlreadyEnabled);
        if(ok) enabled++;
        // integrity: fill src with 0xAB, dst with 0x11, copy, read back
        CHK(hipSetDevice(a)); void*sa; CHK(hipMalloc(&sa,4096)); CHK(hipMemset(sa,0xAB,4096));
        CHK(hipSetDevice(b)); void*db; CHK(hipMalloc(&db,4096)); CHK(hipMemset(db,0x11,4096));
        CHK(hipDeviceSynchronize());
        const hipError_t ce = hipMemcpyPeer(db,b,sa,a,4096);
        CHK(hipSetDevice(b)); CHK(hipDeviceSynchronize());
        unsigned char host[16]; CHK(hipMemcpy(host,db,16,hipMemcpyDeviceToHost));
        const bool got = (host[0]==0xAB);
        if(ce==hipSuccess && got) moved++;
        if(a==0&&b==1) printf("  0->1 canAccessPeer=%d enable=%s memcpyPeer=%s first byte=0x%02X (source 0xAB, dest pre-fill 0x11)\n",
                              can, ok?"ok":hipGetErrorString(e), hipGetErrorString(ce), host[0]);
        hipFree(sa); hipFree(db);
    }
    printf("  %d/%d ordered pairs enable peer access; %d/%d actually transfer data\n",
           enabled,pairs,moved,pairs);

    // ---- 2. staging through pinned host memory --------------------------
    printf("\n=== host staging, pinned, device 0 -> device 1 ===\n");
    printf("  %-22s %10s %10s %10s %10s\n","payload","blocking","async","async GB/s","overlapped");
    for(int s=0;s<NS;s++){
        const size_t B=SIZES[s];
        void *d0,*d1,*hp;
        CHK(hipSetDevice(0)); CHK(hipMalloc(&d0,B));
        CHK(hipSetDevice(1)); CHK(hipMalloc(&d1,B));
        CHK(hipHostMalloc(&hp,B,hipHostMallocDefault));
        hipStream_t s0,s1;
        CHK(hipSetDevice(0)); CHK(hipStreamCreate(&s0));
        CHK(hipSetDevice(1)); CHK(hipStreamCreate(&s1));
        const int IT=500;

        // blocking, the way the old number was taken
        CHK(hipDeviceSynchronize());
        hipEvent_t a,b2; CHK(hipSetDevice(0)); CHK(hipEventCreate(&a)); CHK(hipEventCreate(&b2));
        auto now=[]{ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; };
        double t0=now();
        for(int i=0;i<IT;i++){
            CHK(hipSetDevice(0)); CHK(hipMemcpy(hp,d0,B,hipMemcpyDeviceToHost));
            CHK(hipSetDevice(1)); CHK(hipMemcpy(d1,hp,B,hipMemcpyHostToDevice));
        }
        const double blocking=(now()-t0)/IT;

        // async on per-device streams, one sync at the end of each crossing
        t0=now();
        for(int i=0;i<IT;i++){
            CHK(hipSetDevice(0)); CHK(hipMemcpyAsync(hp,d0,B,hipMemcpyDeviceToHost,s0));
            CHK(hipStreamSynchronize(s0));
            CHK(hipSetDevice(1)); CHK(hipMemcpyAsync(d1,hp,B,hipMemcpyHostToDevice,s1));
            CHK(hipStreamSynchronize(s1));
        }
        const double async_=(now()-t0)/IT;

        // pipelined: issue many crossings, sync once - the achievable rate when
        // the engine has other work to overlap with
        t0=now();
        for(int i=0;i<IT;i++){
            CHK(hipSetDevice(0)); CHK(hipMemcpyAsync(hp,d0,B,hipMemcpyDeviceToHost,s0));
            CHK(hipSetDevice(1)); CHK(hipMemcpyAsync(d1,hp,B,hipMemcpyHostToDevice,s1));
        }
        CHK(hipSetDevice(0)); CHK(hipStreamSynchronize(s0));
        CHK(hipSetDevice(1)); CHK(hipStreamSynchronize(s1));
        const double overlapped=(now()-t0)/IT;

        printf("  %-22s %9.2fus %9.2fus %9.2f %9.2fus\n",
               NAMES[s], blocking, async_, (double)B/(async_/1e6)/1e9, overlapped);
        hipFree(d0); hipFree(d1); hipHostFree(hp);
        hipStreamDestroy(s0); hipStreamDestroy(s1);
    }

    // ---- 3. pageable, for contrast --------------------------------------
    printf("\n=== pinned vs pageable, 20 KB mHC payload ===\n");
    {
        const size_t B=20480; void*d0,*d1,*hp; std::vector<char> pg(B);
        CHK(hipSetDevice(0)); CHK(hipMalloc(&d0,B));
        CHK(hipSetDevice(1)); CHK(hipMalloc(&d1,B));
        CHK(hipHostMalloc(&hp,B,hipHostMallocDefault));
        auto now=[]{ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; };
        const int IT=500;
        CHK(hipDeviceSynchronize());
        double t0=now();
        for(int i=0;i<IT;i++){ CHK(hipSetDevice(0)); CHK(hipMemcpy(hp,d0,B,hipMemcpyDeviceToHost));
                               CHK(hipSetDevice(1)); CHK(hipMemcpy(d1,hp,B,hipMemcpyHostToDevice)); }
        const double pin=(now()-t0)/IT;
        t0=now();
        for(int i=0;i<IT;i++){ CHK(hipSetDevice(0)); CHK(hipMemcpy(pg.data(),d0,B,hipMemcpyDeviceToHost));
                               CHK(hipSetDevice(1)); CHK(hipMemcpy(d1,pg.data(),B,hipMemcpyHostToDevice)); }
        const double page=(now()-t0)/IT;
        printf("  pinned %.2f us   pageable %.2f us\n", pin, page);
        hipFree(d0); hipFree(d1); hipHostFree(hp);
    }

    // ---- 4. what a token actually needs ---------------------------------
    printf("\n=== per-token transport, 12/12/11/11 split = 3 crossings ===\n");
    printf("  the residual and the mHC block cross together at each boundary\n");
    return 0;
}
