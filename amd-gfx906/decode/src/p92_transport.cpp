#include "p92amd/p92_transport.h"
#include <cstdio>
#include <cstring>

#define TCHK(c) do{ hipError_t e_=(c); if(e_!=hipSuccess){ \
    fprintf(stderr,"transport: %s @%d\n",hipGetErrorString(e_),__LINE__); return -1; } }while(0)

int p92_transport_init(P92Transport *t, size_t bytes) {
    memset(t, 0, sizeof *t);
    t->bytes = bytes ? bytes : P92_TX_BYTES;
    TCHK(hipGetDeviceCount(&t->devices));
    if (t->devices < P92_TX_NGPU) {
        fprintf(stderr, "transport: %d devices, need %d\n", t->devices, P92_TX_NGPU);
        return -1;
    }
    for (int b = 0; b < P92_TX_BOUNDS; b++)
        TCHK(hipHostMalloc(&t->stage[b], t->bytes, hipHostMallocDefault));
    for (int d = 0; d < P92_TX_NGPU; d++) {
        TCHK(hipSetDevice(d));
        TCHK(hipStreamCreateWithFlags(&t->copy[d], hipStreamNonBlocking));
    }
    TCHK(hipSetDevice(0));
    return 0;
}

void p92_transport_free(P92Transport *t) {
    for (int b = 0; b < P92_TX_BOUNDS; b++) if (t->stage[b]) hipHostFree(t->stage[b]);
    for (int d = 0; d < P92_TX_NGPU; d++) if (t->copy[d]) { hipSetDevice(d); hipStreamDestroy(t->copy[d]); }
    memset(t, 0, sizeof *t);
}

int p92_transport_send(P92Transport *t, int from, const void *src) {
    if (from < 0 || from >= P92_TX_BOUNDS) return -1;
    TCHK(hipSetDevice(from));
    TCHK(hipMemcpyAsync(t->stage[from], src, t->bytes, hipMemcpyDeviceToHost, t->copy[from]));
    return 0;
}

int p92_transport_recv_to(P92Transport *t, int from, int to, void *dst) {
    if (from < 0 || from >= P92_TX_BOUNDS || to < 0 || to >= P92_TX_NGPU || to == from) return -1;
    // the sending device's copy must have landed in host memory first
    TCHK(hipSetDevice(from));
    TCHK(hipStreamSynchronize(t->copy[from]));
    TCHK(hipSetDevice(to));
    TCHK(hipMemcpyAsync(dst, t->stage[from], t->bytes, hipMemcpyHostToDevice, t->copy[to]));
    return 0;
}

int p92_transport_recv(P92Transport *t, int from, void *dst) {
    return p92_transport_recv_to(t, from, (from + 1) % P92_TX_NGPU, dst);
}

int p92_transport_wait(P92Transport *t, int device) {
    TCHK(hipSetDevice(device));
    TCHK(hipStreamSynchronize(t->copy[device]));
    return 0;
}
