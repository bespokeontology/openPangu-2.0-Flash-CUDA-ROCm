// Gate the transport ring on the thing peer copy fails: does the data arrive.
// A distinct pattern is written on device 0 and ringed 0->1->2->3; device 3
// must hold it byte-for-byte. Then the per-token cost of the real 3 crossings
// is measured, and checked against a kernel running concurrently to see whether
// the crossing can hide behind compute.
#include "p92amd/p92_transport.h"
#include "../src/p92_transport.cpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)

__global__ void k_busy(float *p, int iters) {
    float a = p[threadIdx.x & 255];
    for (int i = 0; i < iters; i++) a = fmaf(a, 1.0000001f, 1e-7f);
    if (a == 12345.678f) p[threadIdx.x & 255] = a;      // never taken, defeats DCE
}

int main(){
    P92Transport tx;
    if (p92_transport_init(&tx, 0)) { printf("transport init FAILED\n"); return 1; }
    printf("transport: %d devices, %zu B a crossing, %d crossings a token\n",
           tx.devices, tx.bytes, P92_TX_BOUNDS);

    std::vector<unsigned char> pattern(tx.bytes), back(tx.bytes);
    std::mt19937 rng(20260908);
    for (size_t i=0;i<tx.bytes;i++) pattern[i]=(unsigned char)(rng()&0xFF);

    void *buf[P92_TX_NGPU];
    for(int d=0;d<P92_TX_NGPU;d++){ CHK(hipSetDevice(d)); CHK(hipMalloc(&buf[d],tx.bytes));
        CHK(hipMemset(buf[d], d==0?0:0x11, tx.bytes)); }
    CHK(hipSetDevice(0)); CHK(hipMemcpy(buf[0],pattern.data(),tx.bytes,hipMemcpyHostToDevice));
    for(int d=0;d<P92_TX_NGPU;d++){ CHK(hipSetDevice(d)); CHK(hipDeviceSynchronize()); }

    // ring it
    for(int b=0;b<P92_TX_BOUNDS;b++){
        if(p92_transport_send(&tx,b,buf[b])) return 1;
        if(p92_transport_recv(&tx,b,buf[b+1])) return 1;
        if(p92_transport_wait(&tx,b+1)) return 1;
    }
    CHK(hipSetDevice(P92_TX_NGPU-1));
    CHK(hipMemcpy(back.data(),buf[P92_TX_NGPU-1],tx.bytes,hipMemcpyDeviceToHost));
    size_t bad=0; for(size_t i=0;i<tx.bytes;i++) if(back[i]!=pattern[i]) bad++;
    printf("integrity 0->1->2->3: %zu/%zu bytes differ  (peer copy fails this: 0/12 pairs move data)\n",
           bad, tx.bytes);

    // per-token cost of 3 crossings
    auto now=[]{ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; };
    const int IT=300;
    for(int i=0;i<20;i++) for(int b=0;b<P92_TX_BOUNDS;b++){
        p92_transport_send(&tx,b,buf[b]); p92_transport_recv(&tx,b,buf[b+1]); p92_transport_wait(&tx,b+1); }
    double t0=now();
    for(int i=0;i<IT;i++) for(int b=0;b<P92_TX_BOUNDS;b++){
        p92_transport_send(&tx,b,buf[b]); p92_transport_recv(&tx,b,buf[b+1]); p92_transport_wait(&tx,b+1); }
    const double serial=(now()-t0)/IT;

    // can it hide behind compute on the receiving card?
    float *busy; CHK(hipSetDevice(1)); CHK(hipMalloc(&busy,1024*4)); CHK(hipMemset(busy,0,1024*4));
    hipStream_t comp; CHK(hipStreamCreate(&comp));
    // calibrate a kernel to roughly the length of one crossing
    hipEvent_t e0,e1; CHK(hipEventCreate(&e0)); CHK(hipEventCreate(&e1));
    hipLaunchKernelGGL(k_busy,dim3(240),dim3(256),0,comp,busy,20000);
    CHK(hipStreamSynchronize(comp));
    CHK(hipEventRecord(e0,comp));
    for(int i=0;i<10;i++) hipLaunchKernelGGL(k_busy,dim3(240),dim3(256),0,comp,busy,20000);
    CHK(hipEventRecord(e1,comp)); CHK(hipEventSynchronize(e1));
    float kms; hipEventElapsedTime(&kms,e0,e1);
    const double kus=kms*1000.0/10;

    t0=now();
    for(int i=0;i<IT;i++){
        CHK(hipSetDevice(1)); hipLaunchKernelGGL(k_busy,dim3(240),dim3(256),0,comp,busy,20000);
        for(int b=0;b<P92_TX_BOUNDS;b++){
            p92_transport_send(&tx,b,buf[b]); p92_transport_recv(&tx,b,buf[b+1]); p92_transport_wait(&tx,b+1); }
        CHK(hipSetDevice(1)); CHK(hipStreamSynchronize(comp));
    }
    const double withcomp=(now()-t0)/IT;

    printf("3 crossings a token, serial            : %7.2f us  (%.4f ms/token)\n", serial, serial/1000);
    printf("a compute kernel alone                 : %7.2f us\n", kus);
    printf("3 crossings concurrent with that kernel: %7.2f us  (sum would be %.2f)\n", withcomp, serial+kus);
    printf("hidden behind compute                  : %7.2f us of %.2f\n",
           (serial+kus)-withcomp, serial);

    const bool pass = (bad==0);
    printf("%s (gate: every byte arrives across the whole ring)\n", pass?"PASS":"FAIL");
    p92_transport_free(&tx);
    return pass?0:1;
}
