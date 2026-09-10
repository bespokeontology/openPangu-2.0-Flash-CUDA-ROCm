// Gate for the prefill grouped-MoE chain. Independent scalar references, then
// hostile negative controls that must fail distinguishably.
//
// The reference decodes NVFP4 FROM THE SPEC, not by calling p92_dq4. Reusing the
// device decoder would make this a self-consistency check, which proves the two
// copies agree and nothing about either being right.
#include "p92amd/p92_pf_moe.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)

// E2M1: sign(1) exp(2) mant(1). Magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6.
static float ref_e2m1(uint32_t nib) {
    static const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = mag[nib & 7u];
    return (nib & 8u) ? -v : v;
}
// UE4M3, unsigned: e=b>>3, m=b&7. e==0 subnormal m*2^-9, else (e+120)<<23 | m<<20.
static float ref_ue4m3(uint32_t b) {
    const uint32_t e = b >> 3, m = b & 7u;
    if (e == 0) return (float)m * 0.001953125f;
    if (b == 127u) return INFINITY;
    uint32_t u = ((e + 120u) << 23) | (m << 20);
    float f; __builtin_memcpy(&f, &u, 4); return f;
}
// One expert output row against one activation row, entirely on the host.
// Byte j of the packed row holds value 2j in the LOW nibble, 2j+1 in the HIGH.
static double ref_row(const uint8_t *w, const uint8_t *sc, const int8_t *xq,
                      const float *xs, int K) {
    double acc = 0.0;
    for (int g = 0; g < K / 16; g++) {
        const float gs = ref_ue4m3(sc[g]) * xs[g];
        double part = 0.0;
        for (int j = 0; j < 16; j++) {
            const uint8_t byte = w[g * 8 + (j >> 1)];
            const uint32_t nib = (j & 1) ? (byte >> 4) : (byte & 0xF);
            // the producer wrote the group even-then-odd
            const int slot = (j & 1) ? 8 + (j >> 1) : (j >> 1);
            part += (double)ref_e2m1(nib) * (double)xq[g * 16 + slot];
        }
        acc += part * (double)gs;
    }
    return acc;
}

int main() {
    const int T = 64, E = 8;                 // small, exhaustively checkable
    std::mt19937 r(11);
    // Sg/Su are uploaded PRESHUFFLED too (k_pf_shuffle_scales_gu), lane l's five
    // groups contiguous at [l*5..l*5+4]. Same reason: the gate must present the
    // layout the kernel consumes while the reference stays written from the spec.
    // The expert arena is indexed by ARENA SLOT, not by e<E. k_pf_group puts the
    // shared expert at PF_SHARED_SLOT (256) and k_pf_gateup reads its weights at
    // grp.eid, so sizing these by E made the shared group read 248 blocks past
    // the end and fault the GPU. Only routed slots are randomised; the rest stay
    // zero, which is a valid NVFP4 encoding and keeps the gate cheap.
    const size_t WB = (size_t)PF_NFF*PF_NEMBD/2, SB = (size_t)PF_NFF*PF_NEMBD/16;
    std::vector<uint8_t> Wg((size_t)PF_GROUP_MAX*WB), Sg((size_t)PF_GROUP_MAX*SB);
    std::vector<uint8_t> Wu(Wg.size()), Su(Sg.size());
    { std::vector<int> used;
      for (int e = 0; e < E; e++) used.push_back(e);
      used.push_back(PF_SHARED_SLOT);
      for (int e : used) {
        for (size_t i = 0; i < WB; i++) { Wg[(size_t)e*WB+i] = (uint8_t)r();
                                          Wu[(size_t)e*WB+i] = (uint8_t)r(); }
        for (size_t i = 0; i < SB; i++) { Sg[(size_t)e*SB+i] = (uint8_t)(32 + r() % 80);
                                          Su[(size_t)e*SB+i] = (uint8_t)(32 + r() % 80); } } }
    std::vector<int8_t> xq((size_t)T*PF_NEMBD);
    std::vector<float>  xs((size_t)T*(PF_NEMBD/16)), s2((size_t)PF_GROUP_MAX*3, 1e-4f);
    for (auto &v : xq) v = (int8_t)(r() % 255 - 127);
    for (auto &v : xs) v = 0.01f + 0.001f * (r() % 17);
    // routing: every token to topk experts, ids in [0,E)
    std::vector<int32_t> ids((size_t)T*PF_TOPK);
    std::vector<float>   wts((size_t)T*PF_TOPK);
    for (size_t i = 0; i < ids.size(); i++) { ids[i] = r() % E; wts[i] = 0.05f + 0.01f*(r()%20); }

    uint8_t *dWg,*dSg,*dWu,*dSu; int8_t *dxq; float *dxs,*ds2,*dhid; int32_t *dids;
    float *dwts; PfGroup *dgrp; int32_t *drows,*dorder,*dng; float *drwt;
    CHK(hipMalloc(&dWg,Wg.size())); CHK(hipMalloc(&dSg,Sg.size()));
    CHK(hipMalloc(&dWu,Wu.size())); CHK(hipMalloc(&dSu,Su.size()));
    CHK(hipMalloc(&dxq,xq.size())); CHK(hipMalloc(&dxs,xs.size()*4));
    CHK(hipMalloc(&ds2,s2.size()*4)); CHK(hipMalloc(&dids,ids.size()*4));
    CHK(hipMalloc(&dwts,wts.size()*4));
    CHK(hipMalloc(&dgrp,sizeof(PfGroup)*PF_GROUP_MAX));
    CHK(hipMalloc(&drows,(size_t)T*(PF_TOPK+1)*4)); CHK(hipMalloc(&drwt,(size_t)T*(PF_TOPK+1)*4));
    CHK(hipMalloc(&dorder,PF_GROUP_MAX*4)); CHK(hipMalloc(&dng,4));
    CHK(hipMalloc(&dhid,(size_t)T*(PF_TOPK+1)*PF_NFF*4));
    CHK(hipMemcpy(dWg,Wg.data(),Wg.size(),hipMemcpyHostToDevice));
    { auto shuf=[&](const std::vector<uint8_t>&src){ std::vector<uint8_t> o(src.size());
        const int G=PF_NEMBD/16, R=PF_GROUP_MAX*PF_NFF;
        for(int rr=0;rr<R;rr++) for(int g2=0;g2<G;g2++)
          o[(size_t)rr*G + (g2%32)*5 + (g2/32)] = src[(size_t)rr*G+g2];
        return o; };
      auto sgl=shuf(Sg), sul=shuf(Su);
      CHK(hipMemcpy(dSg,sgl.data(),sgl.size(),hipMemcpyHostToDevice));
      CHK(hipMemcpy(dSu,sul.data(),sul.size(),hipMemcpyHostToDevice)); }
    CHK(hipMemcpy(dWu,Wu.data(),Wu.size(),hipMemcpyHostToDevice));
    CHK(hipMemcpy(dxq,xq.data(),xq.size(),hipMemcpyHostToDevice));
    // k_pf_gateup reads the activation scales LANE-CONTIGUOUS, as k_pf_quant now
    // writes them: group g lives at (g%32)*5 + (g/32). The gate builds xs on the
    // host and does not run the quantiser, so it must apply the same permutation
    // or it would compare against a layout the kernel never sees.
    { std::vector<float> xsl(xs.size());
      for (int t2 = 0; t2 < T; t2++)
        for (int g2 = 0; g2 < PF_NEMBD/16; g2++)
          xsl[(size_t)t2*(PF_NEMBD/16) + (g2%32)*5 + (g2/32)] = xs[(size_t)t2*(PF_NEMBD/16)+g2];
      CHK(hipMemcpy(dxs,xsl.data(),xsl.size()*4,hipMemcpyHostToDevice)); }
    CHK(hipMemcpy(ds2,s2.data(),s2.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dids,ids.data(),ids.size()*4,hipMemcpyHostToDevice));
    CHK(hipMemcpy(dwts,wts.data(),wts.size()*4,hipMemcpyHostToDevice));

    hipLaunchKernelGGL(k_pf_group,dim3(1),dim3(256),0,0,dids,dwts,T,0,E,
                       dgrp,drows,drwt,dorder,dng);
    CHK(hipDeviceSynchronize());
    int32_t ng=0; CHK(hipMemcpy(&ng,dng,4,hipMemcpyDeviceToHost));
    std::vector<PfGroup> grp(PF_GROUP_MAX); std::vector<int32_t> rows((size_t)T*(PF_TOPK+1)), order(PF_GROUP_MAX);
    CHK(hipMemcpy(grp.data(),dgrp,sizeof(PfGroup)*PF_GROUP_MAX,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(rows.data(),drows,rows.size()*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(order.data(),dorder,order.size()*4,hipMemcpyDeviceToHost));

    // GATE A: the grouping is a permutation. Every (token, expert) pair appears
    // exactly once, and the shared expert receives every token exactly once.
    std::vector<int> seen((size_t)T*PF_GROUP_MAX, 0);   // indexed by ARENA SLOT, not by e<E
    long placed = 0;
    for (int gi = 0; gi < ng; gi++) {
        const PfGroup &G = grp[order[gi]];
        for (int i = 0; i < G.nrows; i++) { seen[(size_t)rows[G.first+i]*PF_GROUP_MAX + order[gi]]++; placed++; }
    }
    long dup=0, missing=0;
    for (int t=0;t<T;t++) for (int e=0;e<PF_GROUP_MAX;e++) {
        int want=0;
        if (e==PF_SHARED_SLOT) want=1;                     // shared expert slot
        else if (e<E) for (int k=0;k<PF_TOPK;k++) if (ids[(size_t)t*PF_TOPK+k]==e) want++;
        const int got=seen[(size_t)t*PF_GROUP_MAX+e];
        if (got>want) dup+=got-want; if (got<want) missing+=want-got;
    }
    printf("grouping: %d live groups, %ld placements, %ld duplicated, %ld missing\n",
           ng, placed, dup, missing);
    const bool gA = (dup==0 && missing==0);

    // GATE B: gate/up against the independent host reference.
    // POISON FIRST. hipMalloc does not zero, and this gate samples three
    // groups by four rows - a kernel writing only SOME rows could pass. A
    // sentinel makes wrote-nowhere and wrote-partially both detectable, and
    // coverage below is measured rather than assumed. The Qwen lane had a
    // gate PASS on broken code by reading a buffer the new path never wrote.
    { std::vector<float> pz((size_t)T*(PF_TOPK+1)*PF_NFF, -12345.678f);
      CHK(hipMemcpy(dhid,pz.data(),pz.size()*4,hipMemcpyHostToDevice)); }
    hipLaunchKernelGGL(k_pf_gateup,dim3(PF_NFF/8,ng),dim3(256),0,0,
                       dWg,dSg,dWu,dSu,dxq,dxs,ds2,dgrp,dorder,drows,dhid);
    CHK(hipDeviceSynchronize());
    std::vector<float> hid((size_t)T*(PF_TOPK+1)*PF_NFF);
    CHK(hipMemcpy(hid.data(),dhid,hid.size()*4,hipMemcpyDeviceToHost));
    double worst = 0.0; int checked = 0;
    for (int gi = 0; gi < ng && gi < 3; gi++) {            // 3 groups x 4 rows is enough
        const int e = order[gi]; const PfGroup &G = grp[e];
        for (int i = 0; i < G.nrows && i < 4; i++) {
            const int tok = rows[G.first+i];
            for (int n = 0; n < PF_NFF; n += 251) {        // prime stride
                const double gv = ref_row(&Wg[(size_t)e*PF_NFF*PF_NEMBD/2 + (size_t)n*(PF_NEMBD/2)],
                                          &Sg[(size_t)e*PF_NFF*PF_NEMBD/16 + (size_t)n*(PF_NEMBD/16)],
                                          &xq[(size_t)tok*PF_NEMBD], &xs[(size_t)tok*(PF_NEMBD/16)], PF_NEMBD)
                                 * (double)s2[(size_t)e*3+0];
                const double uv = ref_row(&Wu[(size_t)e*PF_NFF*PF_NEMBD/2 + (size_t)n*(PF_NEMBD/2)],
                                          &Su[(size_t)e*PF_NFF*PF_NEMBD/16 + (size_t)n*(PF_NEMBD/16)],
                                          &xq[(size_t)tok*PF_NEMBD], &xs[(size_t)tok*(PF_NEMBD/16)], PF_NEMBD)
                                 * (double)s2[(size_t)e*3+1];
                const double want = gv / (1.0 + exp(-gv)) * uv;
                const double got  = hid[(size_t)(G.first+i)*PF_NFF + n];
                const double den  = fabs(want) > 1e-6 ? fabs(want) : 1e-6;
                worst = fmax(worst, fabs(want-got)/den);
                checked++;
            }
        }
    }
    printf("gate/up vs host reference: %d points, worst relative %.3e\n", checked, worst);
    size_t live=0, unwritten=0;
    for (int gi=0; gi<ng; gi++) { const PfGroup &G = grp[order[gi]];
        for (int i2=0;i2<G.nrows;i2++) for (int n2=0;n2<PF_NFF;n2++) {
            live++;
            if (hid[(size_t)(G.first+i2)*PF_NFF+n2] == -12345.678f) unwritten++; } }
    printf("coverage: %zu live outputs, %zu still poisoned%s\n", live, unwritten,
           unwritten ? "   <-- KERNEL SKIPPED WORK" : "");
    const bool gB = (worst < 2e-2) && (unwritten == 0);

    // NEGATIVE CONTROL: corrupt one row-list entry. The reference must diverge,
    // or the comparison above is not actually reading what the kernel computed.
    int32_t bad = (rows[grp[order[0]].first] + 1) % T;
    CHK(hipMemcpy(drows + grp[order[0]].first, &bad, 4, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_pf_gateup,dim3(PF_NFF/8,ng),dim3(256),0,0,
                       dWg,dSg,dWu,dSu,dxq,dxs,ds2,dgrp,dorder,drows,dhid);
    CHK(hipDeviceSynchronize());
    std::vector<float> hid2(hid.size());
    CHK(hipMemcpy(hid2.data(),dhid,hid2.size()*4,hipMemcpyDeviceToHost));
    size_t moved = 0;
    for (size_t i = 0; i < (size_t)PF_NFF; i++)
        if (hid[(size_t)grp[order[0]].first*PF_NFF+i] != hid2[(size_t)grp[order[0]].first*PF_NFF+i]) moved++;
    printf("negative control (row list perturbed): %zu of %d outputs changed%s\n",
           moved, PF_NFF, moved ? "" : "   <-- DETECTOR BLIND");
    const bool gC = (moved > 0);

    printf("%s\n", (gA&&gB&&gC) ? "PASS" : "FAIL");
    return (gA&&gB&&gC) ? 0 : 1;
}
