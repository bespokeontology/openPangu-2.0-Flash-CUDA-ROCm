import json, struct, os, sys, time
import urllib.request as U
OUT="<CHECKPOINT_DIR>"
BASE="https://huggingface.co/openpangu/openPangu-2.0-Flash/resolve/main/"
idx=json.load(open(f"{OUT}/index.json"))["weight_map"]
need=json.load(open(f"{OUT}/need.json"))["need"]
byshard={}
for n in need: byshard.setdefault(idx[n],[]).append(n)
def get(url,rng=None):
    r=U.Request(url); r.add_header("User-Agent","p92-fetch")
    if rng: r.add_header("Range","bytes=%d-%d"%rng)
    for attempt in range(5):
        try:
            with U.urlopen(r,timeout=120) as f: return f.read()
        except Exception as e:
            if attempt==4: raise
            time.sleep(2*(attempt+1))
tensors={}; blobs={}; total=0
for si,(shard,names) in enumerate(sorted(byshard.items())):
    url=BASE+shard
    hl=struct.unpack('<Q',get(url,(0,7)))[0]
    hdr=json.loads(get(url,(8,8+hl-1)).decode())
    base=8+hl
    for nm in names:
        e=hdr[nm]; s,en=e["data_offsets"]
        data=get(url,(base+s,base+en-1))
        assert len(data)==en-s, (nm,len(data),en-s)
        blobs[nm]=data; tensors[nm]={"dtype":e["dtype"],"shape":e["shape"]}
        total+=len(data)
    print(f"[{si+1}/{len(byshard)}] {shard} +{len(names)} tensors, {total/1e6:.0f} MB", flush=True)
# assemble one safetensors shard
order=sorted(blobs)
hdr={}; off=0
for nm in order:
    n=len(blobs[nm]); hdr[nm]={"dtype":tensors[nm]["dtype"],"shape":tensors[nm]["shape"],"data_offsets":[off,off+n]}; off+=n
hj=json.dumps(hdr,separators=(',',':')).encode()
pad=(8-((8+len(hj))%8))%8; hj+=b' '*pad
p=f"{OUT}/model-00001.safetensors"
with open(p,'wb') as o:
    o.write(struct.pack('<Q',len(hj))); o.write(hj)
    for nm in order: o.write(blobs[nm])
print("WROTE", p, os.path.getsize(p), "tensors", len(order))
# tokenizer.json comes from the same upstream revision
tok = get(BASE + "tokenizer.json")
open(os.path.join(OUT, "tokenizer.json"), "wb").write(tok)
print("WROTE tokenizer.json", len(tok))
