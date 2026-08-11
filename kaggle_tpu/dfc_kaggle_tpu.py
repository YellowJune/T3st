#!/usr/bin/env python3
"""Kaggle TPU cross-substrate execution gate for universal DFC fibers."""
from __future__ import annotations
import hashlib, json, math, time
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import lax

OUT=Path('/kaggle/working/dfc_tpu_results') if Path('/kaggle/working').exists() else Path('dfc_tpu_results')

def bitcast(x,dtype): return lax.bitcast_convert_type(x,dtype)
def main():
    OUT.mkdir(parents=True,exist_ok=True);devs=jax.devices();platforms=sorted(set(d.platform for d in devs));key=jax.random.PRNGKey(7301);n=4_194_304
    x=jax.random.normal(key,(n,),dtype=jnp.float32);y=jnp.maximum(x,jnp.float32(0.0));yb=bitcast(y,jnp.uint32);zero=(yb==jnp.uint32(0));z=int(jnp.sum(zero).block_until_ready())
    key,pkey=jax.random.split(key);payload=jax.random.randint(pkey,(n,),0,1<<30,dtype=jnp.uint32);enc=jnp.where(zero,jnp.uint32(0x80000000)|payload,yb);dec=jnp.where((enc&jnp.uint32(0x80000000))!=0,jnp.uint32(0),enc)
    sem_fail=int(jnp.sum(dec!=yb).block_until_ready());pay_fail=int(jnp.sum(jnp.where(zero,(enc&jnp.uint32(0x3fffffff))!=payload,False)).block_until_ready())
    pages=128;page_bytes=4096;key,lkey,pkey=jax.random.split(key,3);logical=jax.random.randint(lkey,(pages,page_bytes),0,256,dtype=jnp.uint8);perm=jax.random.permutation(pkey,pages);physical=logical[perm];table=jnp.empty((pages,),dtype=jnp.int32).at[perm].set(jnp.arange(pages,dtype=jnp.int32));decoded=physical[table];perm_fail=int(jnp.sum(decoded!=logical).block_until_ready())
    def t(fn,reps=20):
        vals=[]
        for _ in range(reps):
            t0=time.perf_counter();o=fn();jax.block_until_ready(o);vals.append((time.perf_counter()-t0)*1e3)
        vals.sort();return vals[len(vals)//2]
    result={'schema_version':1,'protocol':'dfc-universal-tpu-v1','jax':jax.__version__,'devices':[str(d) for d in devs],'platforms':platforms,'relu30':{'coordinates':n,'zeros':z,'payload_bits':30*z,'semantic_bit_failures':sem_fail,'payload_bit_failures':pay_fail,'decode_ms_median':t(lambda:jnp.where((enc&jnp.uint32(0x80000000))!=0,jnp.uint32(0),enc))},'perm':{'pages':pages,'page_bytes':page_bytes,'capacity_bits':math.factorial(pages).bit_length()-1,'semantic_byte_failures':perm_fail,'decode_ms_median':t(lambda:physical[table])},'claim_boundary':'Cross-substrate decoder-fiber execution only; this is not an optimizer/HBM-memory-saving row.'}
    raw=json.dumps(result,sort_keys=True,separators=(',',':')).encode();result['result_sha256']=hashlib.sha256(raw).hexdigest();(OUT/'tpu_universal.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    if sem_fail or pay_fail or perm_fail:raise RuntimeError(json.dumps(result))
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
