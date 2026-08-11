"""GPU memory/throughput benchmark for DFC-LOW16 error-feedback substitution.

This benchmark deliberately separates measured state-scale allocations from
large-model projections. It never labels a 7B/30B projection as an executed
model run.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, platform, time
from pathlib import Path
import torch
from dfc_ef_residual import pack_fp32_residual_, unpack_fp32_residual, semantic_high16_bits, ef_int8_tensor


def device_info(device: torch.device) -> dict:
    out={"device":str(device),"torch":torch.__version__,"python":platform.python_version(),"cuda":torch.version.cuda}
    if device.type=="cuda":
        prop=torch.cuda.get_device_properties(device)
        free,total=torch.cuda.mem_get_info(device)
        out.update({"gpu_name":prop.name,"compute_capability":[prop.major,prop.minor],"total_hbm_bytes":int(total),"free_hbm_bytes_at_start":int(free)})
    return out


def cleanup(device):
    gc.collect()
    if device.type=="cuda": torch.cuda.empty_cache(); torch.cuda.synchronize(device)


def allocation_case(n:int, external:bool, device:torch.device) -> dict:
    cleanup(device)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    before=torch.cuda.memory_allocated(device) if device.type=="cuda" else 0
    p=torch.empty(n,dtype=torch.float16,device=device)
    first=torch.empty(n,dtype=torch.float32,device=device)
    second=torch.empty(n,dtype=torch.float32,device=device)
    residual=torch.empty(n,dtype=torch.float32,device=device) if external else None
    if device.type=="cuda": torch.cuda.synchronize(device)
    allocated=(torch.cuda.memory_allocated(device)-before) if device.type=="cuda" else (p.numel()*2+first.numel()*4+second.numel()*4+(0 if residual is None else residual.numel()*4))
    peak=(torch.cuda.max_memory_allocated(device)-before) if device.type=="cuda" else allocated
    result={"coordinates":int(n),"external_residual":bool(external),"allocated_bytes":int(allocated),"peak_bytes":int(peak),"expected_bytes_per_coordinate":14 if external else 10}
    del p,first,second,residual; cleanup(device)
    return result


def exactness_and_timing(n:int,repeats:int,device:torch.device) -> dict:
    g=torch.Generator(device=device); g.manual_seed(2701)
    first=torch.randn(n,generator=g,device=device,dtype=torch.float32)
    second=torch.randn(n,generator=g,device=device,dtype=torch.float32)
    residual=torch.randn(n,generator=g,device=device,dtype=torch.float32)
    fsem=semantic_high16_bits(first).clone(); ssem=semantic_high16_bits(second).clone()
    pack_fp32_residual_(first,second,residual); decoded=unpack_fp32_residual(first,second)
    bit_fail=int(torch.count_nonzero(decoded.view(torch.int32)!=residual.view(torch.int32)).item())
    semantic_fail=int(torch.count_nonzero(semantic_high16_bits(first)!=fsem).item()+torch.count_nonzero(semantic_high16_bits(second)!=ssem).item())
    grad=torch.randn(n,generator=g,device=device,dtype=torch.float32)
    for _ in range(3):
        r=unpack_fp32_residual(first,second); _,nr,_=ef_int8_tensor(grad,r); pack_fp32_residual_(first,second,nr)
    if device.type=="cuda": torch.cuda.synchronize(device)
    samples=[]
    for _ in range(repeats):
        t0=time.perf_counter(); r=unpack_fp32_residual(first,second); _,nr,_=ef_int8_tensor(grad,r); pack_fp32_residual_(first,second,nr)
        if device.type=="cuda": torch.cuda.synchronize(device)
        samples.append((time.perf_counter()-t0)*1e3)
    samples.sort(); med=samples[len(samples)//2]
    return {"coordinates":int(n),"payload_bit_failures":bit_fail,"semantic_bit_failures":semantic_fail,"repeats":repeats,"median_roundtrip_compress_ms":med,"min_ms":min(samples),"max_ms":max(samples)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--coordinates',type=int,default=100_000_000); ap.add_argument('--timing-coordinates',type=int,default=4_000_000); ap.add_argument('--repeats',type=int,default=9); ap.add_argument('--device',default='cuda'); ap.add_argument('--output',required=True)
    a=ap.parse_args(); device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    info=device_info(device)
    ext=allocation_case(a.coordinates,True,device); dfc=allocation_case(a.coordinates,False,device)
    measured_saved=ext['allocated_bytes']-dfc['allocated_bytes']
    exact=exactness_and_timing(a.timing_coordinates,a.repeats,device)
    result={"schema_version":1,"protocol":"dfc-low16-ef-memory-v1","hardware":info,"external":ext,"dfc":dfc,"measured_eliminated_bytes":int(measured_saved),"measured_eliminated_bytes_per_coordinate":float(measured_saved/a.coordinates),"exactness_timing":exact,"projections":{"7B_external_fp32_residual_bytes":28_000_000_000,"30B_external_fp32_residual_bytes":120_000_000_000,"note":"projections from the exact 4-byte/coordinate auxiliary-state law; not executed 7B/30B model runs"}}
    canonical=json.dumps(result,sort_keys=True,separators=(',',':')).encode(); result['result_sha256']=hashlib.sha256(canonical).hexdigest()
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__': main()
