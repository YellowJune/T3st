#!/usr/bin/env python3
"""CUDA execution gate for DFC-ReLU30 and DFC-PERM.

Fail-closed: no empirical row is emitted without CUDA.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch


def cap(n): return math.factorial(n).bit_length()-1

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='results/kaggle/universal_cuda.json'); ap.add_argument('--n',type=int,default=1_048_576); ap.add_argument('--repeats',type=int,default=50); args=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable: no empirical row emitted')
    dev=torch.device('cuda'); g=torch.Generator(device=dev); g.manual_seed(4301)
    x=torch.randn(args.n,device=dev,dtype=torch.float32,generator=g)
    y=torch.relu(x); y=torch.where(y==0,torch.zeros_like(y),y)
    bits=y.view(torch.int32); zero=(bits==0); z=int(zero.sum().item())
    payload=torch.randint(0,1<<30,(z,),device=dev,dtype=torch.int64,generator=g)
    encbits=bits.clone(); encbits[zero]=(payload | 0x80000000).to(torch.int32)
    phys=encbits.view(torch.float32)
    pb=phys.view(torch.int32); decb=pb.clone(); decb[pb<0]=0; dec=decb.view(torch.float32)
    relu_exact=bool(torch.equal(dec.view(torch.int32),y.view(torch.int32)))
    recovered=(pb[pb<0].to(torch.int64) & 0x3fffffff); payload_exact=bool(torch.equal(recovered,payload))
    pages=128; page_bytes=4096
    logical=torch.randint(0,256,(pages,page_bytes),device=dev,dtype=torch.uint8,generator=g)
    perm=torch.randperm(pages,device=dev,generator=g); physical=logical.index_select(0,perm)
    table=torch.empty(pages,device=dev,dtype=torch.int64); table[perm]=torch.arange(pages,device=dev)
    decoded=physical.index_select(0,table); perm_exact=bool(torch.equal(decoded,logical))
    def event_time(fn):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record(); fn(); b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)
    relu_decode=[]; perm_decode=[]
    for _ in range(args.repeats):
        relu_decode.append(event_time(lambda: torch.where(phys<0,torch.zeros_like(phys),phys)))
        perm_decode.append(event_time(lambda: physical.index_select(0,table)))
    out={'protocol':'dfc-universal-cuda-v1','gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'relu30':{'coordinates':args.n,'zeros':z,'payload_bits':30*z,'semantic_exact':relu_exact,'payload_exact':payload_exact,'decode_ms_median':float(torch.tensor(relu_decode).median().item())},'perm':{'pages':pages,'capacity_bits':cap(pages),'page_bytes':page_bytes,'semantic_exact':perm_exact,'decode_ms_median':float(torch.tensor(perm_decode).median().item())}}
    if not (relu_exact and payload_exact and perm_exact): raise AssertionError(out)
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
