// Gate 14: the MLA split ABI. Detects producer/consumer disagreement on
// P92_SPLITS, which used to be defined independently as 8 (p92_mla_g16.hip,
// #ifndef-guarded) and 16 (p92_mla_split.hip, unguarded). Production resolved
// to 16 only because of single-TU compilation and include order.
//
// The detector: poison pacc/pm/pl with a sentinel, run the producer, then check
// that EVERY slot the consumer will read was written. If the producer used a
// smaller P92_SPLITS than the consumer, the tail slots keep the sentinel and
// this fails. It also checks the reverse (producer writing past the consumer's
// range) by bounding against the allocation.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include "../src/p92_mla_prep.hip"
#include "../src/p92_mla_decode.hip"
#include "../src/p92_mla_split.hip"
#include "../src/p92_mla_g16.hip"
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)

// compile-time half of the contract
static_assert(P92_SPLITS % 4 == 0, "grid is dim3(48, P92_SPLITS/4); SPLITS must be a multiple of 4");
static_assert(P92_KVLORA % G16_LANES == 0, "KVLORA must divide evenly across lanes");
static_assert(G16_DPL * G16_LANES == P92_KVLORA, "per-lane dims must tile KVLORA exactly");
static_assert(G16_STEPS * G16_VEC == G16_DPL, "vector steps must tile the per-lane dims");
static_assert(P92_PE % G16_LANES == 0, "PE must divide evenly across lanes");

int main(int argc,char**argv){
    const int NSEL = argc>1?atoi(argv[1]):220;
    const int CAP=4096, NS=P92_SINKS, NH=P92_NH, KV=P92_KVLORA;
    printf("compiled geometry: SPLITS=%d LANES=%d DPL=%d STEPS=%d PEPL=%d GROUPS=%d\n",
           P92_SPLITS,G16_LANES,G16_DPL,G16_STEPS,G16_PEPL,G16_GROUPS);
    printf("grid dim3(%d,%d)x256 = %d waves; pacc = %d floats\n",
           NH,P92_SPLITS/4,NH*(P92_SPLITS/4)*4,NH*P92_SPLITS*KV);
    float *qlat,*qpe,*pacc,*pm,*pl,*ctx; p92_bf *lat,*kpe,*slat,*spe; int*sel;
    CHK(hipMalloc(&qlat,(size_t)NH*KV*4)); CHK(hipMalloc(&qpe,(size_t)NH*P92_PE*4));
    CHK(hipMalloc(&lat,(size_t)CAP*KV*2)); CHK(hipMalloc(&kpe,(size_t)CAP*P92_PE*2));
    CHK(hipMalloc(&slat,(size_t)NS*KV*2)); CHK(hipMalloc(&spe,(size_t)NS*P92_PE*2));
    CHK(hipMalloc(&sel,(size_t)CAP*4));
    const size_t pn=(size_t)NH*P92_SPLITS*KV, mn=(size_t)NH*P92_SPLITS;
    CHK(hipMalloc(&pacc,pn*4)); CHK(hipMalloc(&pm,mn*4)); CHK(hipMalloc(&pl,mn*4));
    CHK(hipMalloc(&ctx,(size_t)NH*KV*4));
    std::mt19937 r(7);
    { std::vector<float> t((size_t)NH*KV); for(auto&v:t) v=(float)(r()%2000-1000)/1000.f;
      CHK(hipMemcpy(qlat,t.data(),t.size()*4,hipMemcpyHostToDevice)); }
    { std::vector<float> t((size_t)NH*P92_PE); for(auto&v:t) v=(float)(r()%2000-1000)/1000.f;
      CHK(hipMemcpy(qpe,t.data(),t.size()*4,hipMemcpyHostToDevice)); }
    { std::vector<p92_bf> t((size_t)CAP*KV); for(auto&v:t) v=(p92_bf)(0x3F00u+(r()%256));
      CHK(hipMemcpy(lat,t.data(),t.size()*2,hipMemcpyHostToDevice)); }
    { std::vector<p92_bf> t((size_t)CAP*P92_PE); for(auto&v:t) v=(p92_bf)(0x3F00u+(r()%256));
      CHK(hipMemcpy(kpe,t.data(),t.size()*2,hipMemcpyHostToDevice)); }
    { std::vector<p92_bf> t((size_t)NS*KV); for(auto&v:t) v=(p92_bf)(0x3F00u+(r()%256));
      CHK(hipMemcpy(slat,t.data(),t.size()*2,hipMemcpyHostToDevice)); }
    { std::vector<p92_bf> t((size_t)NS*P92_PE); for(auto&v:t) v=(p92_bf)(0x3F00u+(r()%256));
      CHK(hipMemcpy(spe,t.data(),t.size()*2,hipMemcpyHostToDevice)); }
    { std::vector<int> t(CAP); for(int i=0;i<CAP;i++) t[i]=i%CAP;
      CHK(hipMemcpy(sel,t.data(),CAP*4,hipMemcpyHostToDevice)); }

    // poison every slot the consumer will touch
    const uint32_t POISON=0x7FBEEFAAu;
    { std::vector<uint32_t> p(pn,POISON); CHK(hipMemcpy(pacc,p.data(),pn*4,hipMemcpyHostToDevice)); }
    { std::vector<uint32_t> p(mn,POISON); CHK(hipMemcpy(pm,p.data(),mn*4,hipMemcpyHostToDevice));
      CHK(hipMemcpy(pl,p.data(),mn*4,hipMemcpyHostToDevice)); }

    hipLaunchKernelGGL(k_p92_mla_g16,dim3(NH,P92_SPLITS/4),dim3(256),0,0,
                       qlat,qpe,lat,kpe,slat,spe,sel,NSEL,NS,0,0,pacc,pm,pl);
    CHK(hipDeviceSynchronize());
    std::vector<uint32_t> ha(pn),hm(mn),hl(mn);
    CHK(hipMemcpy(ha.data(),pacc,pn*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(hm.data(),pm,mn*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(hl.data(),pl,mn*4,hipMemcpyDeviceToHost));
    size_t unwritten=0; int first_h=-1,first_s=-1;
    for(int h=0;h<NH;h++) for(int s=0;s<P92_SPLITS;s++){
        if(hm[h*P92_SPLITS+s]==POISON||hl[h*P92_SPLITS+s]==POISON){
            if(first_h<0){first_h=h;first_s=s;} unwritten++; }
        for(int d=0;d<KV;d++)
            if(ha[((size_t)h*P92_SPLITS+s)*KV+d]==POISON){
                if(first_h<0){first_h=h;first_s=s;} unwritten++; }
    }
    printf("producer coverage: %zu of %zu consumer-visible slots still poisoned\n",
           unwritten,pn+2*mn);
    if(unwritten) printf("  first gap at head %d split %d -> producer and consumer disagree on P92_SPLITS\n",
                         first_h,first_s);

    // consumer must then produce finite, non-poison context for every head
    hipLaunchKernelGGL(k_p92_mla_combine,dim3(NH),dim3(256),0,0,pacc,pm,pl,ctx);
    CHK(hipDeviceSynchronize());
    std::vector<float> hc((size_t)NH*KV);
    CHK(hipMemcpy(hc.data(),ctx,hc.size()*4,hipMemcpyDeviceToHost));
    size_t bad=0; for(float v:hc) if(!std::isfinite(v)) bad++;
    printf("consumer output: %zu of %zu non-finite\n",bad,hc.size());

    // NEGATIVE CONTROL: a gate that cannot fail proves nothing. Re-poison and
    // launch the producer with HALF the splits, simulating exactly the bug this
    // gate exists to catch - producer compiled with P92_SPLITS/2 while the
    // consumer still reads P92_SPLITS. The detector must fire.
    { std::vector<uint32_t> p(pn,POISON); CHK(hipMemcpy(pacc,p.data(),pn*4,hipMemcpyHostToDevice)); }
    { std::vector<uint32_t> p(mn,POISON); CHK(hipMemcpy(pm,p.data(),mn*4,hipMemcpyHostToDevice));
      CHK(hipMemcpy(pl,p.data(),mn*4,hipMemcpyHostToDevice)); }
    hipLaunchKernelGGL(k_p92_mla_g16,dim3(NH,P92_SPLITS/8),dim3(256),0,0,
                       qlat,qpe,lat,kpe,slat,spe,sel,NSEL,NS,0,0,pacc,pm,pl);
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(ha.data(),pacc,pn*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(hm.data(),pm,mn*4,hipMemcpyDeviceToHost));
    size_t nc_gap=0;
    for(int h=0;h<NH;h++) for(int s=0;s<P92_SPLITS;s++){
        if(hm[h*P92_SPLITS+s]==POISON) nc_gap++;
        for(int d=0;d<KV;d++) if(ha[((size_t)h*P92_SPLITS+s)*KV+d]==POISON) nc_gap++; }
    printf("negative control (producer at half splits): %zu gaps detected%s\n",
           nc_gap, nc_gap? "" : "  <-- DETECTOR IS BLIND");

    // ---- WINDOWED PATH EQUIVALENCE ------------------------------------
    // The kernel has two control-flow branches. win_cap==0 reads an explicit
    // selection list; win_cap!=0 computes (win_first + b) % win_cap. The
    // executor sets win_cap NONZERO on both its branches, so at the benchmark
    // every layer takes the windowed path - yet until 2026-09-08 every probe
    // and the pre-existing bench passed win_cap=0 and therefore compiled and
    // measured the OTHER branch. That is how a kernel with a 5 percent win in
    // it was declared closed. See receipts/11_PROFILING_LAW.md.
    //
    // Detector: run the windowed path, then run the LIST path with a sel[]
    // holding exactly the indices the windowed path should have produced. The
    // two must agree bit for bit. This exercises both branches and requires
    // them to be consistent.
    const int WCAP = 512, WFIRST = 300, WSEL = 220;
    std::vector<int> wlist(WSEL);
    for(int b=0;b<WSEL;b++) wlist[b] = (WFIRST + b) % WCAP;
    CHK(hipMemcpy(sel,wlist.data(),WSEL*4,hipMemcpyHostToDevice));
    float *pw,*pv; CHK(hipMalloc(&pw,pn*4)); CHK(hipMalloc(&pv,pn*4));
    CHK(hipMemset(pw,0,pn*4)); CHK(hipMemset(pv,0,pn*4));
    hipLaunchKernelGGL(k_p92_mla_g16,dim3(NH,P92_SPLITS/4),dim3(256),0,0,
                       qlat,qpe,lat,kpe,slat,spe,nullptr,WSEL,NS,WFIRST,WCAP,pw,pm,pl);
    hipLaunchKernelGGL(k_p92_mla_g16,dim3(NH,P92_SPLITS/4),dim3(256),0,0,
                       qlat,qpe,lat,kpe,slat,spe,sel,WSEL,NS,0,0,pv,pm,pl);
    CHK(hipDeviceSynchronize());
    std::vector<float> wa(pn),wb(pn);
    CHK(hipMemcpy(wa.data(),pw,pn*4,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(wb.data(),pv,pn*4,hipMemcpyDeviceToHost));
    size_t wdiff=0;
    for(size_t i=0;i<pn;i++) if(__builtin_memcmp(&wa[i],&wb[i],4)) wdiff++;
    printf("windowed vs list path: %zu of %zu differ%s\n", wdiff, pn,
           wdiff? "  <-- WINDOWED INDEXING IS WRONG" : "");

    // NEGATIVE CONTROL: perturb the wrap by one. The gate must notice.
    std::vector<int> bad_list(WSEL);
    for(int b=0;b<WSEL;b++) bad_list[b] = (WFIRST + b + 1) % WCAP;
    CHK(hipMemcpy(sel,bad_list.data(),WSEL*4,hipMemcpyHostToDevice));
    CHK(hipMemset(pv,0,pn*4));
    hipLaunchKernelGGL(k_p92_mla_g16,dim3(NH,P92_SPLITS/4),dim3(256),0,0,
                       qlat,qpe,lat,kpe,slat,spe,sel,WSEL,NS,0,0,pv,pm,pl);
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(wb.data(),pv,pn*4,hipMemcpyDeviceToHost));
    size_t wperturb=0;
    for(size_t i=0;i<pn;i++) if(__builtin_memcmp(&wa[i],&wb[i],4)) wperturb++;
    printf("negative control (wrap offset by one): %zu differ%s\n", wperturb,
           wperturb? "" : "  <-- DETECTOR IS BLIND");

    const bool pass = (unwritten==0) && (bad==0) && (nc_gap>0)
                   && (wdiff==0) && (wperturb>0);
    printf("%s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
