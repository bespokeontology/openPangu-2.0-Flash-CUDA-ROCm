// Gate 13: producer/consumer representation contracts.
//
// Every other gate builds its own quantised activation by hand, which silently
// bypasses the producer contract it is supposed to validate. That let a real
// defect through: k_p92_quant was changed to emit the even-then-odd order, and
// k_p92_dense_gateup still stage-permuted it, permuting an already-permuted
// vector. Layers 0 and 1 were corrupted, the engine emitted degenerate text,
// and all twelve gates passed.
//
// This gate invokes the REAL producers and feeds their actual output into the
// REAL consumers. The reference is built by reading the producer's output back,
// undoing the documented order, and reconstructing the activation - so a
// consumer that misreads the order cannot agree with it.
#include "../src/p92_proj.hip"
#include "../src/p92_i8_expert.hip"
#include "../src/p92_nvfp4_dot4.hip"
#include "../src/p92_dense.hip"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
static float e2m1c(uint32_t n){ uint32_t m=n&7u,e=m>>1,f=m&1u;
    float v = e ? ldexpf(1.f+0.5f*f,(int)e-1) : (f?0.5f:0.f); return (n&8u)?-v:v; }
static float ue4m3c(uint32_t b){ uint32_t e=b>>3,m=b&7u;
    if(e==0) return (float)m*0x1p-9f; if(e==15&&m==7) return INFINITY;
    return ldexpf(1.f+m/8.f,(int)e-7); }

struct Res { const char*edge; double cos; bool pass; };
static Res cmp(const char*edge,const std::vector<float>&g,const std::vector<float>&r){
    double d=0,a=0,b=0;
    for(size_t i=0;i<g.size();i++){ d+=(double)g[i]*r[i]; a+=(double)g[i]*g[i]; b+=(double)r[i]*r[i]; }
    Res o; o.edge=edge; o.cos=(a>0&&b>0)? d/sqrt(a*b) : 0.0; o.pass=o.cos>=0.9999; return o; }

int main(){
    std::mt19937 rng(20260908); std::normal_distribution<float> nd(0,1);
    std::vector<Res> R;

    // ---- edge: k_p92_quant<K> -> k_p92_proj<K,LANES> and -> f4_gateup ----
    auto edge_K = [&](int K,int LANES,int N,const char*name)->int{
        std::vector<float> x(K); for(auto&v:x) v=nd(rng)*(1.f+3.f*(rng()%7==0));  // with outliers
        std::vector<uint8_t> W((size_t)N*(K/2)), S((size_t)N*(K/16));
        for(auto&v:W) v=(uint8_t)(rng()&0xFF);
        for(auto&v:S) v=(uint8_t)(56+(rng()%16));
        float *dx,*dxs,*dy; int8_t *dxq; uint8_t *dW,*dS;
        CHK(hipMalloc(&dx,K*4)); CHK(hipMalloc(&dxq,K)); CHK(hipMalloc(&dxs,(size_t)(K/16)*4));
        CHK(hipMalloc(&dW,W.size())); CHK(hipMalloc(&dS,S.size())); CHK(hipMalloc(&dy,(size_t)N*4));
        CHK(hipMemcpy(dx,x.data(),K*4,hipMemcpyHostToDevice));
        CHK(hipMemcpy(dW,W.data(),W.size(),hipMemcpyHostToDevice));
        CHK(hipMemcpy(dS,S.data(),S.size(),hipMemcpyHostToDevice));
        // THE REAL PRODUCER
        if(K==2560)      hipLaunchKernelGGL((k_p92_quant<2560>),dim3(1),dim3(256),0,0,dx,dxq,dxs);
        else if(K==1024) hipLaunchKernelGGL((k_p92_quant<1024>),dim3(1),dim3(256),0,0,dx,dxq,dxs);
        else if(K==6144) hipLaunchKernelGGL((k_p92_quant<6144>),dim3(1),dim3(256),0,0,dx,dxq,dxs);
        else if(K==512)  hipLaunchKernelGGL((k_p92_quant<512>), dim3(1),dim3(256),0,0,dx,dxq,dxs);
        CHK(hipDeviceSynchronize());
        // read the producer's output back and undo the documented order
        std::vector<int8_t> hq(K); std::vector<float> hs(K/16);
        CHK(hipMemcpy(hq.data(),dxq,K,hipMemcpyDeviceToHost));
        CHK(hipMemcpy(hs.data(),dxs,(size_t)(K/16)*4,hipMemcpyDeviceToHost));
        std::vector<float> xr(K);
        for(int g=0;g<K/16;g++) for(int i=0;i<16;i++)
            xr[g*16+i] = (float)hq[g*16 + ((i&1)? 8+(i>>1) : (i>>1))] * hs[g];
        // reference from the reconstructed activation
        std::vector<float> ref(N);
        for(int r=0;r<N;r++){ double a=0;
            for(int g=0;g<K/16;g++){ const uint8_t*p=&W[(size_t)r*(K/2)+g*8]; double t=0;
                for(int j=0;j<8;j++) t += (double)e2m1c(p[j]&15u)*xr[g*16+2*j] + (double)e2m1c(p[j]>>4)*xr[g*16+2*j+1];
                a += t*ue4m3c(S[(size_t)r*(K/16)+g]); }
            ref[r]=(float)a; }
        const int ROWS=256/LANES;
        if(K==2560)      hipLaunchKernelGGL((k_p92_proj<2560,32>),dim3((N+ROWS-1)/ROWS),dim3(256),0,0,dW,dS,dxq,dxs,1.f,dy);
        else if(K==1024) hipLaunchKernelGGL((k_p92_proj<1024,32>),dim3((N+ROWS-1)/ROWS),dim3(256),0,0,dW,dS,dxq,dxs,1.f,dy);
        else if(K==6144) hipLaunchKernelGGL((k_p92_proj<6144,64>),dim3((N+ROWS-1)/ROWS),dim3(256),0,0,dW,dS,dxq,dxs,1.f,dy);
        else if(K==512)  hipLaunchKernelGGL((k_p92_proj<512,32>), dim3((N+ROWS-1)/ROWS),dim3(256),0,0,dW,dS,dxq,dxs,1.f,dy);
        CHK(hipDeviceSynchronize());
        std::vector<float> got(N); CHK(hipMemcpy(got.data(),dy,(size_t)N*4,hipMemcpyDeviceToHost));
        R.push_back(cmp(name,got,ref));
        hipFree(dx);hipFree(dxq);hipFree(dxs);hipFree(dW);hipFree(dS);hipFree(dy);
        return 0; };

    if(edge_K(2560,32,1024,"quant<2560> -> proj<2560,32>")) return 1;
    if(edge_K(1024,32, 512,"quant<1024> -> proj<1024,32>")) return 1;
    if(edge_K( 512,32, 768,"quant<512>  -> proj<512,32>")) return 1;
    if(edge_K(6144,64, 512,"quant<6144> -> proj<6144,64>")) return 1;

    // ---- edge: k_p92_quant<2560> -> k_p92_dense_gateup -------------------
    // the edge that actually broke, and that no other gate exercises
    {
        const int K=P92_NEMBD, N=P92_DENSE_FF;
        std::vector<float> x(K); for(auto&v:x) v=nd(rng)*(1.f+3.f*(rng()%7==0));
        std::vector<uint8_t> Wg((size_t)N*(K/2)),Wu((size_t)N*(K/2)),Sg((size_t)N*(K/16)),Su((size_t)N*(K/16));
        for(auto&v:Wg) v=(uint8_t)(rng()&0xFF); for(auto&v:Wu) v=(uint8_t)(rng()&0xFF);
        for(auto&v:Sg) v=(uint8_t)(56+(rng()%16)); for(auto&v:Su) v=(uint8_t)(56+(rng()%16));
        float *dx,*dxs,*dh; int8_t *dxq; uint8_t *dWg,*dWu,*dSg,*dSu;
        CHK(hipMalloc(&dx,K*4)); CHK(hipMalloc(&dxq,K)); CHK(hipMalloc(&dxs,(size_t)(K/16)*4));
        CHK(hipMalloc(&dWg,Wg.size())); CHK(hipMalloc(&dWu,Wu.size()));
        CHK(hipMalloc(&dSg,Sg.size())); CHK(hipMalloc(&dSu,Su.size())); CHK(hipMalloc(&dh,(size_t)N*4));
        CHK(hipMemcpy(dx,x.data(),K*4,hipMemcpyHostToDevice));
        CHK(hipMemcpy(dWg,Wg.data(),Wg.size(),hipMemcpyHostToDevice));
        CHK(hipMemcpy(dWu,Wu.data(),Wu.size(),hipMemcpyHostToDevice));
        CHK(hipMemcpy(dSg,Sg.data(),Sg.size(),hipMemcpyHostToDevice));
        CHK(hipMemcpy(dSu,Su.data(),Su.size(),hipMemcpyHostToDevice));
        hipLaunchKernelGGL((k_p92_quant<2560>),dim3(1),dim3(256),0,0,dx,dxq,dxs);
        CHK(hipDeviceSynchronize());
        std::vector<int8_t> hq(K); std::vector<float> hs(K/16);
        CHK(hipMemcpy(hq.data(),dxq,K,hipMemcpyDeviceToHost));
        CHK(hipMemcpy(hs.data(),dxs,(size_t)(K/16)*4,hipMemcpyDeviceToHost));
        std::vector<float> xr(K);
        for(int g=0;g<K/16;g++) for(int i=0;i<16;i++)
            xr[g*16+i] = (float)hq[g*16 + ((i&1)? 8+(i>>1) : (i>>1))] * hs[g];
        std::vector<float> ref(N);
        for(int r=0;r<N;r++){ double ag=0,au=0;
            for(int g=0;g<K/16;g++){
                const uint8_t*pg=&Wg[(size_t)r*(K/2)+g*8], *pu=&Wu[(size_t)r*(K/2)+g*8];
                double tg=0,tu=0;
                for(int j=0;j<8;j++){
                    tg += (double)e2m1c(pg[j]&15u)*xr[g*16+2*j] + (double)e2m1c(pg[j]>>4)*xr[g*16+2*j+1];
                    tu += (double)e2m1c(pu[j]&15u)*xr[g*16+2*j] + (double)e2m1c(pu[j]>>4)*xr[g*16+2*j+1]; }
                ag += tg*ue4m3c(Sg[(size_t)r*(K/16)+g]); au += tu*ue4m3c(Su[(size_t)r*(K/16)+g]); }
            const float gg=(float)ag, uu=(float)au;
            ref[r]= gg/(1.f+expf(-gg))*uu; }
        hipLaunchKernelGGL(k_p92_dense_gateup,dim3(P92_DENSE_FF/8),dim3(256),0,0,
                           dWg,dSg,dWu,dSu,dxq,dxs,1.f,1.f,dh);
        CHK(hipDeviceSynchronize());
        std::vector<float> got(N); CHK(hipMemcpy(got.data(),dh,(size_t)N*4,hipMemcpyDeviceToHost));
        R.push_back(cmp("quant<2560> -> dense_gateup",got,ref));
    }

    printf("producer/consumer representation contracts, real producers invoked:\n");
    bool ok=true;
    for(auto&r:R){ printf("  %-34s cosine %.8f  %s\n", r.edge, r.cos, r.pass?"PASS":"FAIL"); ok &= r.pass; }
    printf("%s (gate: every producer/consumer edge cosine >= 0.9999)\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
