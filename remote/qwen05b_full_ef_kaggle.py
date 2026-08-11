"""Real Qwen2.5-0.5B full-parameter DFC-LOW16 error-feedback experiment.

Both methods use the same BF16-high semantic AdamW transition and the same
per-tensor INT8 error-feedback compressor. `external_ef` allocates one FP32
residual per trainable coordinate; `dfc_ef` stores exactly the same residual
bits in the two LOW16 moment fibers, requiring zero additional model-sized
residual allocation.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch_fiber import DFCLow16AdamW
from dfc_ef_residual import ExternalErrorFeedback, DFCLow16ErrorFeedback

TEXTS=[
    "The blue river crosses the quiet valley under a clear morning sky.",
    "A small research system should preserve evidence after every experiment.",
    "Reliable computation requires both numerical semantics and physical state.",
    "Machine learning systems trade memory bandwidth against numerical precision.",
]


def sha_model(model):
    h=hashlib.sha256()
    with torch.no_grad():
        for name,p in model.named_parameters():
            h.update(name.encode()+b'\0'); t=p.detach().contiguous().view(torch.uint8).cpu(); h.update(memoryview(t.numpy()))
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',required=True,choices=['external_ef','dfc_ef']);ap.add_argument('--model',default='Qwen/Qwen2.5-0.5B');ap.add_argument('--revision',default='060db6499f32faf8b98477b0a26969ef7d8b9987');ap.add_argument('--steps',type=int,default=4);ap.add_argument('--seq-len',type=int,default=32);ap.add_argument('--lr',type=float,default=2e-5);ap.add_argument('--seed',type=int,default=3109);ap.add_argument('--gradient-checkpointing',action='store_true');ap.add_argument('--output',required=True);a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM,AutoTokenizer
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    try: torch.use_deterministic_algorithms(True,warn_only=True)
    except Exception: pass
    resolved=HfApi().model_info(a.model,revision=a.revision).sha
    tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,torch_dtype=torch.float16,low_cpu_mem_usage=True).cuda();model.config.use_cache=False
    if a.gradient_checkpointing:model.gradient_checkpointing_enable()
    model.train();params=[p for p in model.parameters() if p.requires_grad];P=sum(p.numel() for p in params)
    torch.cuda.synchronize();base_alloc=int(torch.cuda.memory_allocated());torch.cuda.reset_peak_memory_stats()
    opt=DFCLow16AdamW(params,lr=a.lr,weight_decay=0.0,enable_fiber=a.method=='dfc_ef')
    torch.cuda.synchronize();after_opt=int(torch.cuda.memory_allocated())
    ef=DFCLow16ErrorFeedback(opt,params) if a.method=='dfc_ef' else ExternalErrorFeedback(params)
    torch.cuda.synchronize();after_ef=int(torch.cuda.memory_allocated())
    losses=[];started=time.perf_counter()
    for step in range(a.steps):
        text=TEXTS[step%len(TEXTS)]
        enc=tok(text,return_tensors='pt',truncation=True,max_length=a.seq_len,padding='max_length')
        ids=enc['input_ids'].cuda();mask=enc['attention_mask'].cuda();labels=ids.clone();labels[mask==0]=-100
        opt.zero_grad(set_to_none=True);out=model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False);loss=out.loss;loss.backward();ef.compress_grads_();opt.step();losses.append(float(loss.detach()))
        del ids,mask,labels,out,loss
    torch.cuda.synchronize();wall=time.perf_counter()-started;peak=int(torch.cuda.max_memory_allocated());digest=sha_model(model)
    result={'schema_version':1,'protocol':'qwen05b-full-low16-ef-v1','method':a.method,'seed':a.seed,'model':a.model,'requested_revision':a.revision,'resolved_hub_revision':resolved,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'total_trainable_parameters':int(P),'model_dtype':'float16','optimizer_semantics':'BF16-high moments in FP32 containers','compressor':'per-tensor symmetric INT8 with FP32 error feedback','steps':a.steps,'seq_len':a.seq_len,'losses':losses,'base_model_allocated_bytes':base_alloc,'after_optimizer_allocated_bytes':after_opt,'after_ef_allocated_bytes':after_ef,'peak_hbm_bytes':peak,'external_ef_allocated_bytes':int(ef.allocated_bytes),'dfc_low16_capacity_bytes':int(4*P if a.method=='dfc_ef' else 0),'counterfactual_fp32_ef_bytes':int(4*P),'encoded_gradient_bytes':int(ef.stats.encoded_bytes),'model_sha256':digest,'wall_seconds_training':wall}
    canonical=json.dumps(result,sort_keys=True,separators=(',',':')).encode();result['result_sha256']=hashlib.sha256(canonical).hexdigest();path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
