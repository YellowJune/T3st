#!/usr/bin/env python3
"""One-shot Kaggle GPU execution harness for the DFC final validation suite.

Designed to preserve useful evidence even when a later phase fails: every phase
writes status/log files and the harness exits normally after sealing whatever
completed. Scientific pass/fail is carried by aggregate JSON, not process exit.
"""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
REPO=WORK/'T3st_dfc_final'; OUT=WORK/'dfc_final_results'; LOG=OUT/'logs'; CK=OUT/'checkpoints'
BRANCH='dfc-kaggle-final'; URL='https://github.com/YellowJune/T3st.git'
SEEDS=[3203,3251,3301]


def write_json(path,obj): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def status_update(name,ok,seconds,cmd=None,note=None):
    p=OUT/'phase_status.json';d=json.loads(p.read_text()) if p.exists() else {'schema_version':1,'phases':{}}
    d['phases'][name]={'ok':bool(ok),'seconds':float(seconds),'cmd':cmd,'note':note,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};write_json(p,d)
def run(name,cmd,env=None):
    LOG.mkdir(parents=True,exist_ok=True);t=time.time();log=LOG/f'{name}.log';merged=os.environ.copy();merged.update(env or {})
    with log.open('w') as f:
        f.write('$ '+' '.join(map(str,cmd))+'\n');f.flush();r=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,env=merged,cwd=REPO)
    sec=time.time()-t;status_update(name,r.returncode==0,sec,cmd=list(map(str,cmd)));return r.returncode==0
def ensure_repo():
    if REPO.exists(): shutil.rmtree(REPO)
    subprocess.run(['git','clone','--depth','1','--branch',BRANCH,URL,str(REPO)],check=True)
def shasums():
    rows=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS.txt' and 'checkpoints' not in p.parts:
            h=hashlib.sha256(p.read_bytes()).hexdigest();rows.append(f'{h}  {p.relative_to(OUT)}')
    (OUT/'SHA256SUMS.txt').write_text('\n'.join(rows)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True);CK.mkdir(parents=True,exist_ok=True)
    try: ensure_repo()
    except Exception as e: status_update('clone',False,0,note=repr(e));return
    sys.path.insert(0,str(REPO/'remote'))
    import torch
    gpu_count=torch.cuda.device_count();gpu_names=[torch.cuda.get_device_name(i) for i in range(gpu_count)]
    if not gpu_count: status_update('hardware',False,0,note='No CUDA GPU visible');return
    props=[]
    for i in range(gpu_count):
        p=torch.cuda.get_device_properties(i);props.append({'index':i,'name':p.name,'total_memory':p.total_memory,'cc':[p.major,p.minor]})
    write_json(OUT/'hardware.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu_count':gpu_count,'gpus':props});status_update('hardware',True,0,note=str(gpu_names))
    py=sys.executable
    run('universal_cpu',[py,'remote/universal_fibers_kaggle.py','--out',str(OUT/'universal_cpu.json')])
    run('universal_cuda',[py,'remote/universal_fibers_cuda.py','--out',str(OUT/'universal_cuda.json'),'--n','4194304','--repeats','50'],{'CUDA_VISIBLE_DEVICES':'0'})
    run('ef_gpu_exactness',[py,'remote/dfc_ef_gpu_preflight.py','--coordinates','1000003','--updates','25','--output',str(OUT/'ef_gpu_preflight.json')],{'CUDA_VISIBLE_DEVICES':'0'})
    total=props[0]['total_memory'];coords=max(10_000_000,min(600_000_000,int(total*0.45/14)))
    run('ef_memory',[py,'remote/dfc_ef_memory_benchmark.py','--coordinates',str(coords),'--timing-coordinates','8000000','--repeats','9','--device','cuda','--output',str(OUT/'ef_memory.json')],{'CUDA_VISIBLE_DEVICES':'0'})
    if props[0]['cc'][0] >= 7:
        run('triton_fused',[py,'remote/benchmark_triton_dfc.py','--sizes','1048576,4194304,16777216,33554432','--repeats','9','--output',str(OUT/'triton.json')],{'CUDA_VISIBLE_DEVICES':'0'})
    else:
        write_json(OUT/'triton_skipped.json',{'reason':'compute capability below 7.0','gpu':props[0]});status_update('triton_fused',True,0,note='skipped: unsupported pre-Volta capability')
    for method in ['external_ef','dfc_ef']:
        run(f'qwen05b_{method}',[py,'remote/qwen05b_full_ef_kaggle.py','--method',method,'--steps','4','--seq-len','32','--gradient-checkpointing','--output',str(OUT/f'qwen05b_{method}.json')],{'CUDA_VISIBLE_DEVICES':'0','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
    partial=OUT/'partial';partial.mkdir(exist_ok=True)
    def spec(method,seed,gpu):
        out=partial/f'{method}_seed{seed}.json';ck=CK/f'{method}_seed{seed}.pt'
        cmd=[py,'remote/llm_continual_qwen_partial_resume.py','--method',method,'--seed',str(seed),'--steps-per-task','96','--train-last-layers','1','--checkpoint',str(ck),'--checkpoint-every','48','--resume','--output',str(out)]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True';return cmd,env,out,ck
    for seed in SEEDS:
        pair=[spec('derpp',seed,0),spec('dfc_sign_derpp',seed,1 if gpu_count>1 else 0)]
        if gpu_count>1:
            procs=[];started=time.time()
            for method,(cmd,env,out,ck) in zip(['derpp','dfc_sign_derpp'],pair):
                f=(LOG/f'partial_{method}_seed{seed}.log').open('w');f.write('$ '+' '.join(cmd)+'\n');f.flush();p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=REPO);procs.append((method,p,f,cmd,out,ck))
            for method,p,f,cmd,out,ck in procs:
                rc=p.wait();f.close();status_update(f'partial_{method}_seed{seed}',rc==0,time.time()-started,cmd=cmd)
                if rc==0 and out.exists() and json.loads(out.read_text()).get('complete'): ck.unlink(missing_ok=True)
        else:
            for method,(cmd,env,out,ck) in zip(['derpp','dfc_sign_derpp'],pair):
                ok=run(f'partial_{method}_seed{seed}',cmd,env)
                if ok and out.exists() and json.loads(out.read_text()).get('complete'): ck.unlink(missing_ok=True)
    required=[OUT/'ef_memory.json',OUT/'ef_gpu_preflight.json',OUT/'qwen05b_external_ef.json',OUT/'qwen05b_dfc_ef.json']
    if all(p.exists() for p in required):
        run('aggregate',[py,'remote/aggregate_kaggle_final.py','--memory',str(required[0]),'--gpu-preflight',str(required[1]),'--qwen-external',str(required[2]),'--qwen-dfc',str(required[3]),'--partial-dir',str(partial),'--output',str(OUT/'aggregate.json')])
    else: status_update('aggregate',False,0,note='one or more prerequisite result files missing')
    shasums();shutil.make_archive(str(WORK/'DFC_KAGGLE_FINAL_RESULTS'),'zip',root_dir=OUT)
    print('FINAL_OUTPUT_DIR',OUT);print('FINAL_ZIP',WORK/'DFC_KAGGLE_FINAL_RESULTS.zip')
if __name__=='__main__':main()
