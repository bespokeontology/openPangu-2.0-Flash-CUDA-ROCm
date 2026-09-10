// Gate for the producer-push transport. A gate that cannot fail proves nothing,
// so every fault this transport can suffer gets a hostile negative control that
// MUST be detected: no push, a stale generation, a wrong source, and a partial
// push. Each is distinguishable from the others by construction - the payload
// encodes (generation, source, index), so a stale value cannot be mistaken for
// a fresh one and a wrong source cannot be mistaken for a missing one.
#include "p92amd/p92_p2p.h"
#include <cstdio>
#include <vector>
#include <cstdint>
__global__ void k_gen(uint32_t *p, uint32_t gen, uint32_t src, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) p[i] = (gen << 24) | (src << 20) | (uint32_t)(i & 0xFFFFF);
}
static bool verify(const std::vector<uint32_t> &h, uint32_t gen, uint32_t src) {
    for (size_t i = 0; i < h.size(); i++)
        if (h[i] != ((gen << 24) | (src << 20) | (uint32_t)(i & 0xFFFFF))) return false;
    return true;
}
int main() {
    const size_t B = 20480;                       // the decode-shaped payload
    P92P2P p{};
    if (p92_p2p_init(&p, B)) { printf("FAIL: init/integrity gate\n"); return 1; }
    const int N = (int)(B / 4), nd = p.devices;
    std::vector<uint32_t> h(N);
    uint32_t *src0 = nullptr;
    hipSetDevice(0); hipMalloc(&src0, B);
    int pass = 1;

    // ---- POSITIVE: a correct push must verify -------------------------
    hipSetDevice(0);
    hipLaunchKernelGGL(k_gen, dim3((N+255)/256), dim3(256), 0, 0, src0, 7u, 0u, N);
    hipDeviceSynchronize();
    p92_p2p_push(&p, 0, 1, src0, 0, B, 0, 7u);
    hipSetDevice(0); hipStreamSynchronize(p.push[0]);
    if (p92_p2p_wait(&p, 1, 0, 7u)) { printf("FAIL: wait did not observe gen 7\n"); pass = 0; }
    hipSetDevice(1); hipMemcpy(h.data(), p.land[1], B, hipMemcpyDeviceToHost);
    const bool pos = verify(h, 7u, 0u);
    printf("positive  correct push 0->1                     %s\n", pos ? "verified" : "WRONG DATA");
    if (!pos) pass = 0;

    // ---- NEGATIVE 1: no push at all. Flag must not advance. ------------
    {
        hipSetDevice(1);
        uint32_t z = 0; hipMemcpy(p.flag[1], &z, 4, hipMemcpyHostToDevice);
        const int rc = p92_p2p_wait(&p, 1, 0, 99u);   // nothing will ever send gen 99
        printf("negative  missing push detected                 %s\n", rc ? "yes" : "NO - DETECTOR BLIND");
        if (!rc) pass = 0;
    }
    // ---- NEGATIVE 2: stale generation. Old payload must not verify. ----
    {
        hipSetDevice(0);
        hipLaunchKernelGGL(k_gen, dim3((N+255)/256), dim3(256), 0, 0, src0, 3u, 0u, N);
        hipDeviceSynchronize();
        p92_p2p_push(&p, 0, 1, src0, 0, B, 0, 3u);
        hipSetDevice(0); hipStreamSynchronize(p.push[0]);
        hipSetDevice(1); hipMemcpy(h.data(), p.land[1], B, hipMemcpyDeviceToHost);
        const bool stale_passes_as_fresh = verify(h, 7u, 0u);
        printf("negative  stale gen 3 rejected as gen 7        %s\n",
               stale_passes_as_fresh ? "NO - INDISTINGUISHABLE" : "yes");
        if (stale_passes_as_fresh) pass = 0;
    }
    // ---- NEGATIVE 3: wrong source. dev2 pushes where dev0 should. ------
    {
        uint32_t *src2 = nullptr;
        hipSetDevice(2); hipMalloc(&src2, B);
        hipLaunchKernelGGL(k_gen, dim3((N+255)/256), dim3(256), 0, 0, src2, 8u, 2u, N);
        hipDeviceSynchronize();
        p92_p2p_push(&p, 2, 1, src2, 0, B, 0, 8u);
        hipSetDevice(2); hipStreamSynchronize(p.push[2]);
        hipSetDevice(1); hipMemcpy(h.data(), p.land[1], B, hipMemcpyDeviceToHost);
        const bool wrong_src_passes = verify(h, 8u, 0u);   // expecting source 0
        printf("negative  wrong source (dev2 not dev0) detected %s\n",
               wrong_src_passes ? "NO - INDISTINGUISHABLE" : "yes");
        if (wrong_src_passes) pass = 0;
        hipSetDevice(2); hipFree(src2);
    }
    // ---- NEGATIVE 4: partial push. Half the payload must not verify. ---
    {
        hipSetDevice(1);
        hipLaunchKernelGGL(k_gen, dim3((N+255)/256), dim3(256), 0, 0,
                           (uint32_t *)p.land[1], 0u, 0u, N);
        hipDeviceSynchronize();
        hipSetDevice(0);
        hipLaunchKernelGGL(k_gen, dim3((N+255)/256), dim3(256), 0, 0, src0, 9u, 0u, N);
        hipDeviceSynchronize();
        p92_p2p_push(&p, 0, 1, src0, 0, B / 2, 1, 9u);     // HALF
        hipSetDevice(0); hipStreamSynchronize(p.push[0]);
        hipSetDevice(1); hipMemcpy(h.data(), p.land[1], B, hipMemcpyDeviceToHost);
        const bool partial_passes = verify(h, 9u, 0u);
        printf("negative  partial push detected                 %s\n",
               partial_passes ? "NO - INDISTINGUISHABLE" : "yes");
        if (partial_passes) pass = 0;
    }
    // ---- all directed pairs carry a distinguishable payload ------------
    int pairs = 0, bad = 0;
    for (int a = 0; a < nd; a++) for (int b = 0; b < nd; b++) {
        if (a == b) continue; pairs++;
        uint32_t *s = nullptr;
        hipSetDevice(a); hipMalloc(&s, B);
        hipLaunchKernelGGL(k_gen, dim3((N+255)/256), dim3(256), 0, 0, s, 5u, (uint32_t)a, N);
        hipDeviceSynchronize();
        hipSetDevice(b);
        hipLaunchKernelGGL(k_gen, dim3((N+255)/256), dim3(256), 0, 0,
                           (uint32_t *)p.land[b], 0u, 15u, N);
        hipDeviceSynchronize();
        p92_p2p_push(&p, a, b, s, 0, B, 0, 5u);
        hipSetDevice(a); hipStreamSynchronize(p.push[a]);
        hipSetDevice(b); hipMemcpy(h.data(), p.land[b], B, hipMemcpyDeviceToHost);
        if (!verify(h, 5u, (uint32_t)a)) { bad++;
            printf("  pair %d->%d carried 0x%08X\n", a, b, h[0]); }
        hipSetDevice(a); hipFree(s);
    }
    printf("all pairs source-tagged and verified            %d/%d\n", pairs - bad, pairs);
    if (bad) pass = 0;
    hipSetDevice(0); hipFree(src0);
    p92_p2p_free(&p);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
