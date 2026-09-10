#include "p92amd/p92_p2p.h"
#include <cstdio>
#include <vector>
#include <chrono>

// The push kernel. Reads LOCAL memory, stores into PEER memory - the fast
// direction, 23.1 GB/s. uint4 stores; the tail is handled by a byte loop.
__global__ void k_p92_push(const uint4 *__restrict__ src, uint4 *__restrict__ dst, int n4) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n4) dst[i] = src[i];
}
// Raises the arrival flag AFTER the payload stores retire on this stream.
__global__ void k_p92_flag(uint32_t *__restrict__ flag, int slot, uint32_t gen) {
    if (threadIdx.x == 0) __atomic_store_n(&flag[slot], gen, __ATOMIC_RELEASE);
}
__global__ void k_fill32(uint32_t *p, uint32_t v, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x; if (i < n) p[i] = v;
}

int p92_p2p_push(P92P2P *p, int from, int to, const void *src,
                 size_t off, size_t n, int slot, uint32_t generation) {
    if (!p->enabled) return -1;
    if (hipSetDevice(from) != hipSuccess) return -1;
    uint4 *d = (uint4 *)((char *)p->dst[from][to] + off);
    const int n4 = (int)(n / 16);
    hipLaunchKernelGGL(k_p92_push, dim3((n4 + 255) / 256), dim3(256), 0, p->push[from],
                       (const uint4 *)src, d, n4);
    // flag lives on `to`; writing it from `from` is also a peer store, and it is
    // ordered after the payload because both are on the same stream.
    hipLaunchKernelGGL(k_p92_flag, dim3(1), dim3(64), 0, p->push[from],
                       (uint32_t *)p->dst[from][to] + (p->bytes / 4) + slot, slot, generation);
    return 0;
}

int p92_p2p_wait(P92P2P *p, int dev, int slot, uint32_t generation) {
    if (!p->enabled) return -1;
    if (hipSetDevice(dev) != hipSuccess) return -1;
    // host-side spin on a device-visible flag; the consumer then reads its own
    // landing buffer locally at full HBM speed.
    // WALL-CLOCK bounded. A spin count is not a timeout: the first version
    // bounded this at 200,000,000 iterations of a hipMemcpy, about 16 minutes,
    // and the gate's own "missing push must be detected" negative control hung
    // on it holding the machine lock. A negative control has to fail FAST.
    uint32_t v = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        if (hipMemcpy((void *)&v, (const void *)(p->flag[dev] + slot), 4,
                      hipMemcpyDeviceToHost) != hipSuccess) return -1;
        if (v == generation) return 0;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count() > P92_P2P_WAIT_MS) break;
    }
    fprintf(stderr, "p92_p2p_wait: dev %d slot %d timed out after %d ms at gen %u (saw %u)\n",
            dev, slot, P92_P2P_WAIT_MS, generation, v);
    return -1;
}

int p92_p2p_init(P92P2P *p, size_t bytes) {
    int nd = 0;
    if (hipGetDeviceCount(&nd) != hipSuccess) return -1;
    if (nd > P92_P2P_NGPU) nd = P92_P2P_NGPU;
    p->devices = nd; p->bytes = bytes; p->enabled = 0;
    const size_t FLAGS = 64 * sizeof(uint32_t);
    for (int d = 0; d < nd; d++) {
        if (hipSetDevice(d) != hipSuccess) return -1;
        if (hipMalloc(&p->land[d], bytes + FLAGS) != hipSuccess) return -1;
        if (hipMemset(p->land[d], 0, bytes + FLAGS) != hipSuccess) return -1;
        p->flag[d] = (uint32_t *)((char *)p->land[d] + bytes);
        if (hipStreamCreate(&p->push[d]) != hipSuccess) return -1;
    }
    for (int a = 0; a < nd; a++) {
        if (hipSetDevice(a) != hipSuccess) return -1;
        for (int b = 0; b < nd; b++) {
            if (a == b) { p->dst[a][b] = p->land[a]; continue; }
            hipDeviceEnablePeerAccess(b, 0); hipGetLastError();
            p->dst[a][b] = p->land[b];
        }
    }
    // ---- INTEGRITY GATE, never a capability bit -------------------------
    const int N = (int)(bytes / 4);
    std::vector<uint32_t> h(N);
    int ok = 0, tot = 0;
    for (int a = 0; a < nd; a++) for (int b = 0; b < nd; b++) {
        if (a == b) continue; tot++;
        // pattern encodes the ORDERED PAIR, so a sender-local alias, a wrong
        // source and a stale value are all distinguishable from success.
        const uint32_t PAT = 0xC0DE0000u | (uint32_t)(a << 8) | (uint32_t)b;
        uint32_t *tmp = nullptr;
        hipSetDevice(a);
        if (hipMalloc(&tmp, bytes) != hipSuccess) return -1;
        hipLaunchKernelGGL(k_fill32, dim3((N + 255) / 256), dim3(256), 0, 0, tmp, PAT, N);
        hipDeviceSynchronize();
        hipSetDevice(b);
        hipLaunchKernelGGL(k_fill32, dim3((N + 255) / 256), dim3(256), 0, 0,
                           (uint32_t *)p->land[b], 0xBADBAD00u, N);
        hipDeviceSynchronize();
        p->enabled = 1;                       // provisional, for the gate only
        p92_p2p_push(p, a, b, tmp, 0, bytes, 0, 1u);
        hipSetDevice(a); hipStreamSynchronize(p->push[a]);
        p->enabled = 0;
        hipSetDevice(b);
        hipMemcpy(h.data(), p->land[b], bytes, hipMemcpyDeviceToHost);
        bool good = true;
        for (int i = 0; i < N; i++) if (h[i] != PAT) { good = false; break; }
        if (good) ok++;
        else fprintf(stderr, "  p2p %d->%d FAILED: got 0x%08X want 0x%08X%s\n",
                     a, b, h[0], PAT, h[0] == 0xBADBAD00u ? "  (no data moved)" : "");
        hipSetDevice(a); hipFree(tmp);
    }
    if (ok != tot) { fprintf(stderr, "p92_p2p: integrity gate %d/%d, refusing to enable\n", ok, tot); return -1; }
    printf("p92_p2p: integrity gate %d/%d directed pairs, enabled\n", ok, tot);
    p->enabled = 1;
    return 0;
}

void p92_p2p_free(P92P2P *p) {
    for (int d = 0; d < p->devices; d++) {
        hipSetDevice(d);
        if (p->land[d]) hipFree(p->land[d]);
        if (p->push[d]) hipStreamDestroy(p->push[d]);
    }
    p->enabled = 0;
}
