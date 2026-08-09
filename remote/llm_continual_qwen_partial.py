"""Qwen2.5-0.5B partial-finetuning continual benchmark for DFC-SIGN.

Instead of LoRA, this protocol unfreezes the final transformer block(s) while
keeping ordinary FP32 AdamW moments. The sign fiber therefore scales with the
actual partially-finetuned state. External DER++ and DFC-SIGN+DER++ receive the
same 512-byte external envelope, two prompt slots, optimizer updates, and loss.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_fiber import DFCAdamW, TorchSignFiberChannel
from llm_continual_qwen import (
    QwenReplayCodec, CombinedByteChannel, ReservoirStore, make_stream,
    tokenize_prompt, target_id, last_logits, evaluate, metrics,
    TOPK, RECORD_BYTES, SEQ_LEN,
)


def run(args):
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device(args.device)
    resolved=HfApi().model_info(args.model, revision=args.revision).sha
    tok=AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tok.padding_side='left'
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.config.use_cache=False
    for p in model.parameters(): p.requires_grad_(False)
    layers=model.model.layers
    for layer in layers[-args.train_last_layers:]:
        for p in layer.parameters(): p.requires_grad_(True)
    model=model.to(device); model.train()
    trainable=[p for p in model.parameters() if p.requires_grad]
    trainable_n=int(sum(p.numel() for p in trainable))
    opt=DFCAdamW(trainable, lr=args.lr, betas=(0.9,0.999), eps=1e-8, weight_decay=args.weight_decay, enable_fiber=args.method=='dfc_sign_derpp')
    fiber=TorchSignFiberChannel(opt) if args.method=='dfc_sign_derpp' else None
    store=ReservoirStore(CombinedByteChannel(args.external_bytes,fiber), np.random.default_rng(args.seed+10003))
    tasks=make_stream(); label_ids={ex.target_text:target_id(tok,ex.target_text) for task in tasks for ex in task}
    if len(set(label_ids.values()))!=8: raise RuntimeError(f'target collision {label_ids}')
    matrix=np.full((len(tasks),len(tasks)),np.nan); rng=np.random.default_rng(args.seed+20003); losses=[]; updates=0; started=time.perf_counter()
    for ti,task in enumerate(tasks):
        for _ in range(args.steps_per_task):
            cur=task[int(rng.integers(0,len(task)))]; ci,cm=tokenize_prompt(tok,cur.text,device); ct=target_id(tok,cur.target_text)
            rep=store.sample() if args.method!='naive' else None
            if rep is None:
                ri,rm,rt=ci.clone(),cm.clone(),ct; ridx=rlog=None
            else:
                ri,rm,rt=rep['input_ids'].to(device),rep['attention_mask'].to(device),int(rep['target']); ridx=rep['topk_indices'].to(device); rlog=rep['topk_logits'].to(device)
            logits=last_logits(model,torch.stack([ci,ri]),torch.stack([cm,rm])); lc=F.cross_entropy(logits[0:1],torch.tensor([ct],device=device))
            if rep is None or args.method=='naive':
                loss=lc
            else:
                lrpl=F.cross_entropy(logits[1:2],torch.tensor([rt],device=device)); dark=F.mse_loss(logits[1,ridx].float(),rlog.float()); loss=lc+args.replay_ce_weight*lrpl+args.distill_weight*dark
            vals,idx=torch.topk(logits[0].detach(),k=TOPK); record=QwenReplayCodec.encode(ci,cm,ct,cur.task,idx,vals)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,args.grad_clip); opt.step(); updates+=1; losses.append(float(loss.detach()))
            if args.method!='naive': store.insert(record)
        matrix[ti]=np.asarray(evaluate(model,tok,tasks,device,ti))
    result={'schema_version':1,'protocol':'qwen-causal-partial-v1','method':args.method,'seed':args.seed,'model':args.model,'requested_revision':args.revision,'resolved_hub_revision':resolved,'device':str(device),'torch':torch.__version__,'trainable_parameters':trainable_n,'total_model_parameters':int(sum(p.numel() for p in model.parameters())),'train_last_layers':args.train_last_layers,'external_bytes':args.external_bytes,'sign_fiber_bytes':0 if fiber is None else fiber.byte_capacity,'record_bytes':RECORD_BYTES,'record_capacity':0 if args.method=='naive' else store.capacity_records,'records_final':0 if args.method=='naive' else store.count,'records_seen':0 if args.method=='naive' else store.seen,'store_sha256':None if args.method=='naive' else store.digest(),'batch_size':2,'steps_per_task':args.steps_per_task,'tasks':len(tasks),'updates':updates,'processed_prompt_slots':2*updates,'seq_len':SEQ_LEN,'distill_weight':args.distill_weight,'replay_ce_weight':args.replay_ce_weight,'lr':args.lr,'weight_decay':args.weight_decay,'accuracy_matrix':matrix.tolist(),**metrics(matrix),'mean_training_loss':float(np.mean(losses)),'wall_seconds':time.perf_counter()-started}
    raw=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=True).encode(); result['result_sha256']=hashlib.sha256(raw).hexdigest(); return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--method',required=True,choices=['derpp','dfc_sign_derpp']); p.add_argument('--seed',type=int,default=1889); p.add_argument('--model',default='Qwen/Qwen2.5-0.5B'); p.add_argument('--revision',default='060db6499f32faf8b98477b0a26969ef7d8b9987'); p.add_argument('--device',default='cpu'); p.add_argument('--steps-per-task',type=int,default=128); p.add_argument('--external-bytes',type=int,default=512); p.add_argument('--train-last-layers',type=int,default=1); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--weight-decay',type=float,default=0.0); p.add_argument('--distill-weight',type=float,default=.01); p.add_argument('--replay-ce-weight',type=float,default=.25); p.add_argument('--grad-clip',type=float,default=1.0); p.add_argument('--threads',type=int,default=4); p.add_argument('--output',required=True); a=p.parse_args(); r=run(a); path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=True)+'\n'); print(json.dumps({k:r[k] for k in ['protocol','method','seed','trainable_parameters','final_average_accuracy','average_forgetting','current_task_accuracy','record_capacity','sign_fiber_bytes','wall_seconds','result_sha256']},indent=2))
if __name__=='__main__': main()
