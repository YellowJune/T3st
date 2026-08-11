"""Crash-resumable Qwen continual adaptation for Kaggle GPU validation.

The frozen pretrained base is reloaded from a pinned Hugging Face revision.
Only trainable layers plus optimizer state are checkpointed. For DFC-SIGN,
the replay payload is already inside the FP32 second-moment words serialized by
optimizer.state_dict(); the 512-byte external envelope and RNG states are saved
separately. Checkpoints are written atomically and can resume mid-task.

Important: ordinary Optimizer.load_state_dict may cast floating optimizer state
to the parameter dtype. Because the trainable Qwen parameters are FP16 while
DFC's moments are intentionally FP32 physical containers, resume uses an
explicit bit-preserving FP32 state restore instead.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
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


def _store_without_init(channel, rng):
    store=ReservoirStore.__new__(ReservoirStore)
    store.channel=channel; store.rng=rng
    store.capacity_records=max(0,(channel.byte_capacity-32)//RECORD_BYTES)
    return store


def _atomic_torch_save(obj,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,tmp); os.replace(tmp,path)


def _named_trainable(model):
    return {n:p for n,p in model.named_parameters() if p.requires_grad}


def _rng_state(data_rng,store_rng,device):
    out={"python":random.getstate(),"numpy":np.random.get_state(),"torch_cpu":torch.get_rng_state(),"data_rng":data_rng.bit_generator.state,"store_rng":store_rng.bit_generator.state}
    if device.type=='cuda': out['torch_cuda']=torch.cuda.get_rng_state_all()
    return out


def _restore_rng(state,data_rng,store_rng,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu']);data_rng.bit_generator.state=state['data_rng'];store_rng.bit_generator.state=state['store_rng']
    if device.type=='cuda' and 'torch_cuda' in state: torch.cuda.set_rng_state_all(state['torch_cuda'])


def _load_optimizer_fp32_exact(opt, saved, params, device):
    """Restore DFCAdamW state without parameter-dtype casting.

    This intentionally supports the simple single/multi-group DFCAdamW layout
    used here. Saved FP32 moment words are copied as FP32, preserving payload
    sign bits exactly even when parameters themselves are FP16.
    """
    saved_groups=saved['param_groups']
    if len(saved_groups)!=len(opt.param_groups): raise RuntimeError('optimizer group count mismatch')
    flat=[]
    for group,sg in zip(opt.param_groups,saved_groups):
        if len(group['params'])!=len(sg['params']): raise RuntimeError('optimizer parameter count mismatch')
        for k,v in sg.items():
            if k!='params': group[k]=v
        flat.extend(zip(group['params'],sg['params']))
    if len(flat)!=len(params): raise RuntimeError('optimizer flattened parameter mismatch')
    for p,saved_id in flat:
        src=saved['state'].get(saved_id,{})
        dst=opt.state[p]
        if 'step' in src: dst['step']=int(src['step']) if not torch.is_tensor(src['step']) else int(src['step'].item())
        for key in ('exp_avg','exp_avg_sq'):
            if key not in src: raise RuntimeError(f'missing optimizer state {key}')
            tensor=src[key]
            if tensor.dtype!=torch.float32: raise RuntimeError(f'checkpoint {key} is not FP32')
            if tensor.shape!=p.shape: raise RuntimeError(f'checkpoint {key} shape mismatch')
            dst[key].copy_(tensor.to(device=device,dtype=torch.float32))
            if dst[key].dtype!=torch.float32: raise RuntimeError(f'restored {key} lost FP32 container')


def save_checkpoint(path,model,opt,channel,store,data_rng,store_rng,matrix,losses,progress,meta,device):
    trainable={n:p.detach().cpu().clone() for n,p in _named_trainable(model).items()}
    obj={"schema":2,"meta":meta,"trainable":trainable,"optimizer":opt.state_dict(),"external":bytes(channel.external),"matrix":matrix,"losses":losses,"progress":progress,"rng":_rng_state(data_rng,store_rng,device),"store_digest":store.digest(),"store_count":store.count,"store_seen":store.seen}
    _atomic_torch_save(obj,Path(path))


def run(args):
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM,AutoTokenizer
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device)
    resolved=HfApi().model_info(args.model,revision=args.revision).sha
    tok=AutoTokenizer.from_pretrained(args.model,revision=args.revision);tok.padding_side='left'
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    dtype=torch.float16 if args.model_dtype=='float16' else torch.float32
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,torch_dtype=dtype,low_cpu_mem_usage=True)
    model.config.use_cache=False
    for p in model.parameters():p.requires_grad_(False)
    for layer in model.model.layers[-args.train_last_layers:]:
        for p in layer.parameters():p.requires_grad_(True)
    model=model.to(device);model.train();named=_named_trainable(model);trainable=list(named.values());trainable_n=sum(p.numel() for p in trainable)
    opt=DFCAdamW(trainable,lr=args.lr,betas=(.9,.999),eps=1e-8,weight_decay=args.weight_decay,enable_fiber=args.method=='dfc_sign_derpp')
    fiber=TorchSignFiberChannel(opt) if args.method=='dfc_sign_derpp' else None
    channel=CombinedByteChannel(args.external_bytes,fiber)
    store_rng=np.random.default_rng(args.seed+10003);data_rng=np.random.default_rng(args.seed+20003)
    store=ReservoirStore(channel,store_rng)
    tasks=make_stream();label_ids={ex.target_text:target_id(tok,ex.target_text) for task in tasks for ex in task}
    if len(set(label_ids.values()))!=8:raise RuntimeError(f'target collision {label_ids}')
    matrix=np.full((len(tasks),len(tasks)),np.nan);losses=[];progress={"task":0,"step":0,"updates":0};resumed=False
    meta={"model":args.model,"revision":args.revision,"resolved":resolved,"method":args.method,"seed":args.seed,"train_last_layers":args.train_last_layers,"external_bytes":args.external_bytes,"steps_per_task":args.steps_per_task,"model_dtype":args.model_dtype}
    ckpath=Path(args.checkpoint) if args.checkpoint else None
    if ckpath and ckpath.exists() and args.resume:
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        if ck.get('schema') not in (1,2): raise RuntimeError('unsupported checkpoint schema')
        for k in ['model','revision','method','seed','train_last_layers','external_bytes','steps_per_task','model_dtype']:
            if ck['meta'][k]!=meta[k]:raise RuntimeError(f'checkpoint protocol mismatch: {k}')
        if ck['meta']['resolved']!=resolved:raise RuntimeError('resolved hub revision changed')
        for n,t in ck['trainable'].items():named[n].data.copy_(t.to(device=device,dtype=named[n].dtype))
        _load_optimizer_fp32_exact(opt,ck['optimizer'],trainable,device)
        fiber=TorchSignFiberChannel(opt) if args.method=='dfc_sign_derpp' else None;channel=CombinedByteChannel(args.external_bytes,fiber);channel.external[:]=ck['external']
        store_rng=np.random.default_rng();data_rng=np.random.default_rng();store=_store_without_init(channel,store_rng)
        matrix=np.asarray(ck['matrix'],dtype=np.float64);losses=list(ck['losses']);progress=dict(ck['progress']);_restore_rng(ck['rng'],data_rng,store_rng,device)
        if store.digest()!=ck['store_digest'] or store.count!=ck['store_count'] or store.seen!=ck['store_seen']:raise RuntimeError('checkpoint replay-store integrity failure')
        resumed=True
    started=time.perf_counter();updates=int(progress['updates']);stop=False
    for ti in range(int(progress['task']),len(tasks)):
        task=tasks[ti];start_step=int(progress['step']) if ti==int(progress['task']) else 0
        for si in range(start_step,args.steps_per_task):
            cur=task[int(data_rng.integers(0,len(task)))];ci,cm=tokenize_prompt(tok,cur.text,device);ct=target_id(tok,cur.target_text);rep=store.sample()
            if rep is None:ri,rm,rt=ci.clone(),cm.clone(),ct;ridx=rlog=None
            else:ri,rm,rt=rep['input_ids'].to(device),rep['attention_mask'].to(device),int(rep['target']);ridx=rep['topk_indices'].to(device);rlog=rep['topk_logits'].to(device)
            logits=last_logits(model,torch.stack([ci,ri]),torch.stack([cm,rm]));lc=F.cross_entropy(logits[0:1],torch.tensor([ct],device=device))
            if rep is None:loss=lc
            else:loss=lc+args.replay_ce_weight*F.cross_entropy(logits[1:2],torch.tensor([rt],device=device))+args.distill_weight*F.mse_loss(logits[1,ridx].float(),rlog.float())
            vals,idx=torch.topk(logits[0].detach(),k=TOPK);record=QwenReplayCodec.encode(ci,cm,ct,cur.task,idx,vals)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(trainable,args.grad_clip);opt.step();store.insert(record);updates+=1;losses.append(float(loss.detach()))
            progress={"task":ti,"step":si+1,"updates":updates}
            if progress['step']>=args.steps_per_task:progress={"task":ti+1,"step":0,"updates":updates}
            if ckpath and args.checkpoint_every>0 and updates%args.checkpoint_every==0:save_checkpoint(ckpath,model,opt,channel,store,data_rng,store_rng,matrix,losses,progress,meta,device)
            if args.max_updates and updates>=args.max_updates:stop=True;break
        if stop:break
        matrix[ti]=np.asarray(evaluate(model,tok,tasks,device,ti))
        if ckpath:save_checkpoint(ckpath,model,opt,channel,store,data_rng,store_rng,matrix,losses,{"task":ti+1,"step":0,"updates":updates},meta,device)
    complete=(updates>=args.steps_per_task*len(tasks))
    if not complete:return {"schema_version":2,"protocol":"qwen-partial-resume-v2","complete":False,"resumed":resumed,"updates":updates,"checkpoint":str(ckpath)}
    result={"schema_version":2,"protocol":"qwen-partial-resume-v2","complete":True,"resumed":resumed,"method":args.method,"seed":args.seed,"model":args.model,"requested_revision":args.revision,"resolved_hub_revision":resolved,"device":str(device),"model_dtype":args.model_dtype,"torch":torch.__version__,"trainable_parameters":int(trainable_n),"total_model_parameters":int(sum(p.numel() for p in model.parameters())),"train_last_layers":args.train_last_layers,"external_bytes":args.external_bytes,"sign_fiber_bytes":0 if fiber is None else fiber.byte_capacity,"record_bytes":RECORD_BYTES,"record_capacity":store.capacity_records,"records_final":store.count,"records_seen":store.seen,"store_sha256":store.digest(),"batch_size":2,"steps_per_task":args.steps_per_task,"tasks":len(tasks),"updates":updates,"accuracy_matrix":matrix.tolist(),**metrics(matrix),"mean_training_loss":float(np.mean(losses)),"wall_seconds_this_invocation":time.perf_counter()-started}
    canonical=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=True).encode();result['result_sha256']=hashlib.sha256(canonical).hexdigest();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['derpp','dfc_sign_derpp']);p.add_argument('--seed',type=int,default=2903);p.add_argument('--model',default='Qwen/Qwen2.5-1.5B');p.add_argument('--revision',default='8faed761d45a263340a0528343f099c05c9a4323');p.add_argument('--device',default='cuda');p.add_argument('--model-dtype',choices=['float16','float32'],default='float16');p.add_argument('--steps-per-task',type=int,default=96);p.add_argument('--external-bytes',type=int,default=512);p.add_argument('--train-last-layers',type=int,default=1);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--weight-decay',type=float,default=0.0);p.add_argument('--distill-weight',type=float,default=.01);p.add_argument('--replay-ce-weight',type=float,default=.25);p.add_argument('--grad-clip',type=float,default=1.0);p.add_argument('--checkpoint');p.add_argument('--checkpoint-every',type=int,default=48);p.add_argument('--resume',action='store_true');p.add_argument('--max-updates',type=int,default=0);p.add_argument('--output',required=True);a=p.parse_args();r=run(a);path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=True)+'\n');print(json.dumps(r,indent=2,sort_keys=True))
if __name__=='__main__':main()
