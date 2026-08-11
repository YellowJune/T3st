"""Independent Kaggle rerun of non-optimizer DFC families from v3."""
from __future__ import annotations
import argparse, hashlib, json, math, mmap, os, random, tempfile
from pathlib import Path
import numpy as np
import torch
SIGN=np.uint32(0x80000000);LOW30=np.uint32(0x3fffffff);EXP=np.uint32(0x7f800000)

def relu_run(seed,n):
 r=np.random.default_rng(seed);x=r.standard_normal(n,dtype=np.float32);y=x.copy();y[x<=0]=np.float32(0);yb=y.view(np.uint32);yb[yb==SIGN]=0;zero=yb==0;p=r.integers(0,1<<30,size=int(zero.sum()),dtype=np.uint32);pb=yb.copy();pb[zero]=SIGN|p
 if np.any((pb[zero]&EXP)==EXP):raise AssertionError('nonfinite code')
 decb=pb.copy();decb[(pb&SIGN)!=0]=0;rec=pb[(pb&SIGN)!=0]&LOW30;d=decb.view(np.float32);maskfail=int(np.count_nonzero((d>0)!=(y>0)));sem=int(np.count_nonzero(decb!=yb));pay=int(np.count_nonzero(rec!=p))
 m=(min(n,8192)//64)*64;w=r.standard_normal((64,32),dtype=np.float32);a=torch.from_numpy(d[:m].reshape(-1,64).copy())@torch.from_numpy(w.copy());b=torch.from_numpy(y[:m].reshape(-1,64).copy())@torch.from_numpy(w.copy());down=int(not torch.equal(a,b))
 return {'seed':seed,'coordinates':n,'zeros':int(zero.sum()),'payload_bits':int(30*zero.sum()),'semantic_failures':sem,'payload_failures':pay,'downstream_failures':down,'backward_mask_failures':maskfail}

def cap(n):return math.factorial(n).bit_length()-1

def unrank(n,rank):
 e=list(range(n));out=[]
 for k in range(n,0,-1):f=math.factorial(k-1);q,rank=divmod(rank,f);out.append(e.pop(q))
 return out

def rank_perm(p):
 e=list(range(len(p)));r=0;n=len(p)
 for i,v in enumerate(p):j=e.index(v);r+=j*math.factorial(n-i-1);e.pop(j)
 return r

def encode(logical,rank):
 p=unrank(len(logical),rank);physical=logical[np.asarray(p)].copy();table=np.empty(len(p),dtype=np.int32)
 for j,i in enumerate(p):table[i]=j
 return physical,table

def decode(physical,table):return np.asarray(physical)[np.asarray(table,dtype=np.int64)].copy()

def payload_rank(table):
 p=[None]*len(table)
 for logical,phys in enumerate(table.tolist()):p[phys]=logical
 return rank_perm(p)

def attention(q,k,v):
 qt=torch.from_numpy(q.copy());kt=torch.from_numpy(k.copy());vt=torch.from_numpy(v.copy());s=qt@kt.T;s-=s.max(-1,keepdim=True).values;return (torch.softmax(s,-1)@vt).numpy()

def perm_run(seed,pages,trials=8,page_bytes=4096):
 r=np.random.default_rng(seed);rr=random.Random(seed);logical=r.integers(0,256,size=(pages,page_bytes),dtype=np.uint8);c=cap(pages);tpp=2;hd=16;k=r.standard_normal((pages*tpp,hd),dtype=np.float32);v=r.standard_normal((pages*tpp,hd),dtype=np.float32);q=r.standard_normal((4,hd),dtype=np.float32);base=attention(q,k,v);kv=np.stack([k,v],axis=1).reshape(pages,tpp,2,hd);fails=0
 for _ in range(trials):
  rank=rr.getrandbits(c);ph,tab=encode(logical,rank);fails+=int(not np.array_equal(decode(ph,tab),logical));fails+=int(payload_rank(tab)!=rank);pk,tk=encode(kv,rank);dkv=decode(pk,tk).reshape(pages*tpp,2,hd);fails+=int(not np.array_equal(attention(q,dkv[:,0],dkv[:,1]).view(np.uint32),base.view(np.uint32)))
 return {'pages':pages,'capacity_bits':c,'trials':trials,'failures':fails}

def lfsr(x):
 x=(x&0xfff) or 1;b=((x>>0)^(x>>1)^(x>>4)^(x>>6))&1;return ((x>>1)|(b<<11))&0xfff

def automaton(seed,steps=100000):
 r=np.random.default_rng(seed);logical=r.integers(0,2**32,size=(8,16),dtype=np.uint32);sem=base=0x123456789abcdef0;h=(seed&0xfff) or 1;sf=hf=0
 for t in range(steps):
  base=(base*6364136223846793005+1442695040888963407+t)&((1<<64)-1);sem=(sem*6364136223846793005+1442695040888963407+t)&((1<<64)-1);h=lfsr(h);ph,tab=encode(logical,h);dec=decode(ph,tab);sf+=int(not np.array_equal(dec,logical) or sem!=base);hf+=int((payload_rank(tab)&0xfff)!=h);logical=dec
 return {'steps':steps,'semantic_failures':sf,'hidden_state_failures':hf,'capacity_bits':cap(8),'hidden_state_bits':12}

def hierarchy(seed,pages=64,page_bytes=4096,repeats=40):
 r=np.random.default_rng(seed);rr=random.Random(seed);logical=r.integers(0,256,size=(pages,page_bytes),dtype=np.uint8);c=cap(pages);fails=0
 for _ in range(repeats):rank=rr.getrandbits(c);ph,tab=encode(logical,rank);fails+=int(not np.array_equal(decode(ph,tab),logical) or payload_rank(tab)!=rank)
 size=pages*page_bytes+pages*4
 with tempfile.NamedTemporaryFile(delete=False) as f:path=f.name;f.truncate(size)
 try:
  with open(path,'r+b') as f:
   mm=mmap.mmap(f.fileno(),size,access=mmap.ACCESS_WRITE)
   for _ in range(repeats):
    rank=rr.getrandbits(c);ph,tab=encode(logical,rank);mm[:pages*page_bytes]=ph.tobytes();mm[pages*page_bytes:]=tab.tobytes();mm.flush();ph2=np.frombuffer(mm,dtype=np.uint8,count=pages*page_bytes).reshape(pages,page_bytes).copy();tab2=np.frombuffer(mm,dtype=np.int32,count=pages,offset=pages*page_bytes).copy();fails+=int(not np.array_equal(decode(ph2,tab2),logical) or payload_rank(tab2)!=rank)
   mm.close()
 finally:os.unlink(path)
 return {'pages':pages,'page_bytes':page_bytes,'repeats':repeats,'fixed_allocation_bytes':size,'failures':fails}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();relu=[relu_run(s,1_048_576) for s in [3101,3119,3137,3163,3181,3203,3221,3251]];perm=[perm_run(s,n) for s,n in zip([3301,3323,3347,3371,3391],[8,16,32,64,128])];auto=automaton(3457);hier=hierarchy(3511);fail=sum(sum(x[k] for k in ['semantic_failures','payload_failures','downstream_failures','backward_mask_failures']) for x in relu)+sum(x['failures'] for x in perm)+auto['semantic_failures']+auto['hidden_state_failures']+hier['failures'];out={'schema_version':1,'protocol':'dfc-universal-kaggle-rerun-v1','relu30':relu,'perm_kv':perm,'hidden_automaton':auto,'hierarchy':hier,'total_failures':fail};raw=json.dumps(out,sort_keys=True,separators=(',',':')).encode();out['result_sha256']=hashlib.sha256(raw).hexdigest();Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'total_failures':fail,'sha256':out['result_sha256']},indent=2));
 if fail:raise RuntimeError('universal rerun failure')
if __name__=='__main__':main()
