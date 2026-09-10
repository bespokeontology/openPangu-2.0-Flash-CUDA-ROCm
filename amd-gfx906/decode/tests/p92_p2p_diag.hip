// Bounded P2P diagnosis. hipMemcpyPeer returns success and moves nothing, which
// is a broken path rather than absent hardware. The discriminator: have a
// kernel ON device A dereference an allocation OWNED by device B, bypassing the
// copy API entirely. If that works, the copy API is the broken piece.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
#define SOFT(c) do{ hipError_t e=(c); if(e) printf("    [%s]\n",hipGetErrorString(e)); }while(0)

__global__ void k_remote_write(unsigned int *remote, unsigned int pat) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1024) remote[i] = pat + i;
}
__global__ void k_remote_read(const unsigned int *remote, unsigned int *out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1024) out[i] = remote[i];
}

int main(){
    int n=0; CHK(hipGetDeviceCount(&n));
    const size_t N=1024, B=N*sizeof(unsigned int);
    printf("=== 1. remote STORE: a kernel on A writes into B-owned VRAM ===\n");
    int store_ok=0, load_ok=0, pairs=0;
    for(int a=0;a<n;a++) for(int b=0;b<n;b++){
        if(a==b) continue; pairs++;
        CHK(hipSetDevice(b)); unsigned int *db; CHK(hipMalloc(&db,B)); CHK(hipMemset(db,0x11,B));
        CHK(hipDeviceSynchronize());
        CHK(hipSetDevice(a));
        const hipError_t pe=hipDeviceEnablePeerAccess(b,0);
        if(pe!=hipSuccess && pe!=hipErrorPeerAccessAlreadyEnabled){
            if(a==0&&b==1) printf("  0->1 enablePeerAccess: %s\n",hipGetErrorString(pe));
            hipFree(db); continue; }
        hipLaunchKernelGGL(k_remote_write,dim3(4),dim3(256),0,0,db,0xAB000000u);
        const hipError_t le=hipGetLastError();
        const hipError_t se=hipDeviceSynchronize();
        CHK(hipSetDevice(b));
        std::vector<unsigned int> h(N); CHK(hipMemcpy(h.data(),db,B,hipMemcpyDeviceToHost));
        const bool ok = (le==hipSuccess && se==hipSuccess && h[0]==0xAB000000u && h[7]==0xAB000007u);
        if(ok) store_ok++;
        if(a==0&&b==1) printf("  0->1 launch=%s sync=%s  B[0]=0x%08X B[7]=0x%08X (expect 0xAB000000 / 0xAB000007)\n",
                              hipGetErrorString(le),hipGetErrorString(se),h[0],h[7]);
        hipFree(db);
    }
    printf("  remote store works on %d/%d ordered pairs\n",store_ok,pairs);

    printf("\n=== 2. remote LOAD: a kernel on A reads B-owned VRAM ===\n");
    for(int a=0;a<n;a++) for(int b=0;b<n;b++){
        if(a==b) continue;
        CHK(hipSetDevice(b)); unsigned int *db; CHK(hipMalloc(&db,B));
        std::vector<unsigned int> seed(N); for(size_t i=0;i<N;i++) seed[i]=0xCD000000u+i;
        CHK(hipMemcpy(db,seed.data(),B,hipMemcpyHostToDevice)); CHK(hipDeviceSynchronize());
        CHK(hipSetDevice(a));
        const hipError_t pe=hipDeviceEnablePeerAccess(b,0);
        if(pe!=hipSuccess && pe!=hipErrorPeerAccessAlreadyEnabled){ hipFree(db); continue; }
        unsigned int *out; CHK(hipMalloc(&out,B)); CHK(hipMemset(out,0,B));
        hipLaunchKernelGGL(k_remote_read,dim3(4),dim3(256),0,0,db,out);
        const hipError_t se=hipDeviceSynchronize();
        std::vector<unsigned int> h(N); CHK(hipMemcpy(h.data(),out,B,hipMemcpyDeviceToHost));
        const bool ok = (se==hipSuccess && h[0]==0xCD000000u && h[7]==0xCD000007u);
        if(ok) load_ok++;
        if(a==0&&b==1) printf("  0->1 sync=%s  read[0]=0x%08X read[7]=0x%08X (expect 0xCD000000 / 0xCD000007)\n",
                              hipGetErrorString(se),h[0],h[7]);
        hipFree(db); hipFree(out);
    }
    printf("  remote load works on %d/%d ordered pairs\n",load_ok,pairs);

    printf("\n=== 3. hipMemcpyPeerAsync and hipMemcpy kind=DeviceToDevice ===\n");
    {
        CHK(hipSetDevice(0)); unsigned int *s0; CHK(hipMalloc(&s0,B));
        std::vector<unsigned int> seed(N,0xEE00EE00u);
        CHK(hipMemcpy(s0,seed.data(),B,hipMemcpyHostToDevice));
        CHK(hipSetDevice(1)); unsigned int *d1; CHK(hipMalloc(&d1,B)); CHK(hipMemset(d1,0x22,B));
        CHK(hipDeviceSynchronize());
        hipStream_t st; CHK(hipStreamCreate(&st));
        SOFT(hipMemcpyPeerAsync(d1,1,s0,0,B,st)); SOFT(hipStreamSynchronize(st));
        std::vector<unsigned int> h(N); CHK(hipMemcpy(h.data(),d1,B,hipMemcpyDeviceToHost));
        printf("  memcpyPeerAsync  -> 0x%08X %s\n",h[0],h[0]==0xEE00EE00u?"MOVED":"did not move");
        CHK(hipMemset(d1,0x22,B)); CHK(hipDeviceSynchronize());
        SOFT(hipMemcpy(d1,s0,B,hipMemcpyDeviceToDevice)); CHK(hipDeviceSynchronize());
        CHK(hipMemcpy(h.data(),d1,B,hipMemcpyDeviceToHost));
        printf("  memcpy D2D       -> 0x%08X %s\n",h[0],h[0]==0xEE00EE00u?"MOVED":"did not move");
        CHK(hipMemset(d1,0x22,B)); CHK(hipDeviceSynchronize());
        SOFT(hipMemcpy(d1,s0,B,hipMemcpyDefault)); CHK(hipDeviceSynchronize());
        CHK(hipMemcpy(h.data(),d1,B,hipMemcpyDeviceToHost));
        printf("  memcpy Default   -> 0x%08X %s\n",h[0],h[0]==0xEE00EE00u?"MOVED":"did not move");
    }
    return 0;
}
