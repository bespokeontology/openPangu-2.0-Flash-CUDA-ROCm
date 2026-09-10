// Gate MoME three ways: the decode step against a scalar CPU reference, the
// rows form against T serial decode steps bit-exactly (the oracle's own
// acceptance criterion in mome_rows_test.cu, which compares raw uint16 with
// != on both values and state), and real conv weights from the checkpoint.
#include "../src/p92_mome.hip"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <fcntl.h>
#include <unistd.h>
#define CHK(c) do{ hipError_t e=(c); if(e){ printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); return 1; } }while(0)
static float b2f(p92_cbf b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
static p92_cbf f2b(float f){ uint32_t u; memcpy(&u,&f,4); u+=0x7FFFu+((u>>16)&1u); return (p92_cbf)(u>>16); }

static std::vector<p92_cbf> load_bf(const char*dir,const char*name,size_t want){
    std::string ip=std::string(dir)+"/model.safetensors.index.json";
    int f=open(ip.c_str(),O_RDONLY); if(f<0) return {};
    std::string js; { char b[1<<16]; ssize_t k; while((k=read(f,b,sizeof b))>0) js.append(b,k); } close(f);
    std::string key=std::string("\"")+name+"\"";
    size_t p=js.find(key); if(p==std::string::npos) return {};
    size_t q=js.find('"',js.find(':',p)+1), r=js.find('"',q+1);
    std::string sp=std::string(dir)+"/"+js.substr(q+1,r-q-1);
    f=open(sp.c_str(),O_RDONLY); if(f<0) return {};
    uint64_t hl; if(pread(f,&hl,8,0)!=8){close(f);return {};}
    std::string hdr(hl,0); if(pread(f,&hdr[0],hl,8)!=(ssize_t)hl){close(f);return {};}
    p=hdr.find(key); if(p==std::string::npos){close(f);return {};}
    size_t off=hdr.find("data_offsets",p); off=hdr.find('[',off);
    long a=atol(hdr.c_str()+off+1), b=atol(hdr.c_str()+hdr.find(',',off)+1);
    if((size_t)(b-a)!=want*2){ close(f); return {}; }
    std::vector<p92_cbf> out(want);
    if(pread(f,out.data(),b-a,8+hl+a)!=(b-a)){close(f);return {};}
    close(f); return out;
}

template<int C>
static int run_site(const char*label,const char*tensor,const char*CK,int L,double*us_out){
    char nm[160]; snprintf(nm,sizeof nm,tensor,L);
    auto w=load_bf(CK,nm,(size_t)C*3);
    std::mt19937 rng(20260908+C); std::normal_distribution<float> nd(0,1);
    const bool real=!w.empty();
    if(!real){ w.resize((size_t)C*3); for(auto&v:w) v=f2b(nd(rng)*0.3f); }

    const int T=64;
    std::vector<p92_cbf> x((size_t)T*C), st0((size_t)C*2);
    for(auto&v:x) v=f2b(nd(rng));
    for(auto&v:st0) v=f2b(nd(rng)*0.5f);

    // CPU reference for the first row only
    std::vector<p92_cbf> cref(C);
    for(int c=0;c<C;c++){
        const float o=b2f(st0[(size_t)c*2]), p=b2f(st0[(size_t)c*2+1]), cur=b2f(x[c]);
        const float cv=fmaf(o,b2f(w[(size_t)c*3]),fmaf(p,b2f(w[(size_t)c*3+1]),cur*b2f(w[(size_t)c*3+2])));
        cref[c]=f2b(cur+cv); }

    p92_cbf *dv,*dw,*ds,*dv2,*ds2;
    CHK(hipMalloc(&dv,(size_t)T*C*2)); CHK(hipMalloc(&dv2,(size_t)T*C*2));
    CHK(hipMalloc(&dw,(size_t)C*3*2)); CHK(hipMalloc(&ds,(size_t)C*2*2)); CHK(hipMalloc(&ds2,(size_t)C*2*2));
    CHK(hipMemcpy(dw,w.data(),(size_t)C*3*2,hipMemcpyHostToDevice));

    // arm A: T serial decode steps
    CHK(hipMemcpy(dv,x.data(),(size_t)T*C*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(ds,st0.data(),(size_t)C*2*2,hipMemcpyHostToDevice));
    for(int r=0;r<T;r++)
        hipLaunchKernelGGL(k_p92_mome_conv3<C>,dim3((C+255)/256),dim3(256),0,0,dv+(size_t)r*C,dw,ds);
    CHK(hipDeviceSynchronize());
    std::vector<p92_cbf> ga((size_t)T*C), sa((size_t)C*2);
    CHK(hipMemcpy(ga.data(),dv,(size_t)T*C*2,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(sa.data(),ds,(size_t)C*2*2,hipMemcpyDeviceToHost));

    // arm B: one rows call
    CHK(hipMemcpy(dv2,x.data(),(size_t)T*C*2,hipMemcpyHostToDevice));
    CHK(hipMemcpy(ds2,st0.data(),(size_t)C*2*2,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_p92_mome_conv3_rows<C>,dim3((C+255)/256),dim3(256),0,0,dv2,C,T,dw,ds2);
    CHK(hipDeviceSynchronize());
    std::vector<p92_cbf> gb((size_t)T*C), sb((size_t)C*2);
    CHK(hipMemcpy(gb.data(),dv2,(size_t)T*C*2,hipMemcpyDeviceToHost));
    CHK(hipMemcpy(sb.data(),ds2,(size_t)C*2*2,hipMemcpyDeviceToHost));

    int cpu_bad=0; for(int c=0;c<C;c++) if(ga[c]!=cref[c]) cpu_bad++;
    int vbad=0,sbad=0;
    for(size_t i=0;i<ga.size();i++) if(ga[i]!=gb[i]) vbad++;
    for(size_t i=0;i<sa.size();i++) if(sa[i]!=sb[i]) sbad++;

    hipEvent_t t0,t1; CHK(hipEventCreate(&t0)); CHK(hipEventCreate(&t1));
    const int IT=2000;
    for(int i=0;i<100;i++) hipLaunchKernelGGL(k_p92_mome_conv3<C>,dim3((C+255)/256),dim3(256),0,0,dv,dw,ds);
    CHK(hipDeviceSynchronize()); CHK(hipEventRecord(t0,0));
    for(int i=0;i<IT;i++) hipLaunchKernelGGL(k_p92_mome_conv3<C>,dim3((C+255)/256),dim3(256),0,0,dv,dw,ds);
    CHK(hipEventRecord(t1,0)); CHK(hipEventSynchronize(t1));
    float ms; hipEventElapsedTime(&ms,t0,t1); const double us=ms*1000.0/IT;
    *us_out=us;
    printf("  %-14s C=%-5d weights %-10s step vs CPU %d/%d differ   rows vs %d steps: values %d/%zu state %d/%zu   %.2f us\n",
           label,C,real?"real":"synthetic",cpu_bad,C,T,vbad,ga.size(),sbad,sa.size(),us);
    hipFree(dv);hipFree(dv2);hipFree(dw);hipFree(ds);hipFree(ds2);
    return (cpu_bad==0 && vbad==0 && sbad==0)?0:1;
}

int main(int argc,char**argv){
    const char*CK = argc>1?argv[1]:"./checkpoint";
    const int L = argc>2?atoi(argv[2]):5;
    printf("MoME depthwise causal conv, width 3, layer %d\n",L);
    double a=0,b=0,c=0; int bad=0;
    bad |= run_site<P92_MOME_QA>("qa_conv","model.layers.%d.self_attn.qa_conv.weight",CK,L,&a);
    bad |= run_site<P92_MOME_KV>("compresskv","model.layers.%d.self_attn.compresskv_conv.weight",CK,L,&b);
    bad |= run_site<P92_MOME_O >("o_conv","model.layers.%d.self_attn.o_conv.weight",CK,L,&c);
    const double per=a+b+c;
    printf("three sites: %.2f us/layer   x46 trunk layers = %.3f ms/token\n",per,per*46/1000);
    printf("%s\n",bad?"FAIL":"PASS");
    return bad;
}
