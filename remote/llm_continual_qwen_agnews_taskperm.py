"""Full-FP32 DFC-SIGN on a real Qwen continual task-adaptation stream.

Four sequential AG News tasks each contain all four news classes but use a fixed
permutation of the output labels and an explicit task tag in the input. This is a
language analogue of a permuted-task continual-learning stream: every task is
non-degenerate and balanced, so current-task accuracy remains a meaningful
plasticity metric. External DER++ and DFC-SIGN+DER++ receive the same model,
ordinary FP32 AdamW state allocation, 512-byte external envelope, batch shape,
update count, and replay objective. DFC-SIGN addresses only the sign fiber
already resident in the FP32 second moments of trainable LoRA/head parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch_fiber import DFCAdamW, TorchSignFiberChannel
from llm_continual_qwen_agnews import (
    N_LABELS, SEQ_LEN, RECORD_BYTES, ReplayCodec, CombinedByteChannel,
    ReservoirStore, tokenize, logits_of, metrics,
)

CURRENT_SLOTS = 3
BATCH_SIZE = 4
TASK_NAMES = ("alpha", "beta", "gamma", "delta")
PERMUTATIONS = ((0,1,2,3),(1,2,3,0),(2,3,0,1),(3,0,1,2))

@dataclass(frozen=True)
class Example:
    task: int
    text: str
    label: int
    source_label: int

def load_task_stream(seed: int, train_per_class: int, eval_per_class: int):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    dataset_id = "fancyzhx/ag_news"
    revision = HfApi().dataset_info(dataset_id).sha
    ds = load_dataset(dataset_id, revision=revision)
    rng = np.random.default_rng(seed + 41017)
    train_labels = np.asarray(ds["train"]["label"])
    test_labels = np.asarray(ds["test"]["label"])
    train_pools, eval_pools = {}, {}
    for label in range(N_LABELS):
        a = np.flatnonzero(train_labels == label); b = np.flatnonzero(test_labels == label)
        rng.shuffle(a); rng.shuffle(b); train_pools[label]=a; eval_pools[label]=b
    tasks=[]; canonical=[]
    for task,(task_name,perm) in enumerate(zip(TASK_NAMES,PERMUTATIONS)):
        train=[]; evaluation=[]
        for source_label in range(N_LABELS):
            t0=task*train_per_class; e0=task*eval_per_class
            target=int(perm[source_label])
            for index in train_pools[source_label][t0:t0+train_per_class]:
                raw=str(ds["train"][int(index)]["text"]); train.append(Example(task,f"task {task_name}: {raw}",target,source_label))
            for index in eval_pools[source_label][e0:e0+eval_per_class]:
                raw=str(ds["test"][int(index)]["text"]); evaluation.append(Example(task,f"task {task_name}: {raw}",target,source_label))
        rng.shuffle(train); rng.shuffle(evaluation); tasks.append((train,evaluation))
        canonical.append({"task":task,"task_name":task_name,"permutation":list(perm),"train":[{"text":x.text,"label":x.label} for x in train],"eval":[{"text":x.text,"label":x.label} for x in evaluation]})
    subset_sha=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return tasks,dataset_id,revision,subset_sha

@torch.inference_mode()
def evaluate(model,tokenizer,tasks,device,upto):
    model.eval(); result=[]
    for task_index,(_,evaluation) in enumerate(tasks):
        if task_index>upto: result.append(float("nan")); continue
        correct=0; total=0
        for start in range(0,len(evaluation),16):
            chunk=evaluation[start:start+16]; ids,masks=zip(*(tokenize(tokenizer,ex.text,device) for ex in chunk))
            pred=logits_of(model,torch.stack(ids),torch.stack(masks)).argmax(-1).cpu().numpy(); labels=np.asarray([ex.label for ex in chunk])
            correct+=int(np.sum(pred==labels)); total+=len(chunk)
        result.append(correct/total)
    model.train(); return result

def run(args):
    from huggingface_hub import HfApi
    from peft import LoraConfig,get_peft_model
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device(args.device); model_revision=HfApi().model_info(args.model,revision=args.revision).sha
    tasks,dataset_id,dataset_revision,subset_sha=load_task_stream(args.seed,args.train_per_class,args.eval_per_class)
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="left"
    model=AutoModelForSequenceClassification.from_pretrained(args.model,revision=args.revision,num_labels=N_LABELS,torch_dtype=torch.float32,low_cpu_mem_usage=True); model.config.pad_token_id=tokenizer.pad_token_id; model.config.use_cache=False
    n_layers=int(model.config.num_hidden_layers); first=max(0,n_layers-args.lora_last_layers)
    config=LoraConfig(r=args.lora_rank,lora_alpha=2*args.lora_rank,lora_dropout=0.0,bias="none",task_type="SEQ_CLS",target_modules=["q_proj","v_proj"],modules_to_save=["score"],layers_to_transform=list(range(first,n_layers)),layers_pattern="layers")
    model=get_peft_model(model,config).to(device); model.train(); trainable=[p for p in model.parameters() if p.requires_grad]; trainable_n=sum(p.numel() for p in trainable)
    optimizer=DFCAdamW(trainable,lr=args.lr,weight_decay=args.weight_decay,enable_fiber=args.method=="dfc_sign_derpp"); fiber=TorchSignFiberChannel(optimizer) if args.method=="dfc_sign_derpp" else None
    store=ReservoirStore(CombinedByteChannel(args.external_bytes,fiber),np.random.default_rng(args.seed+10003))
    T=len(tasks); matrix=np.full((T,T),np.nan,dtype=np.float64); rng=np.random.default_rng(args.seed+20003); updates=0; losses=[]; started=time.perf_counter()
    for task_index,(train_examples,_) in enumerate(tasks):
        for _ in range(args.steps_per_task):
            current=[train_examples[int(rng.integers(0,len(train_examples)))] for _ in range(CURRENT_SLOTS)]; current_tokens=[tokenize(tokenizer,ex.text,device) for ex in current]; replay=store.sample() if args.method!="naive" else None
            if replay is None:
                extra=train_examples[int(rng.integers(0,len(train_examples)))]; extra_ids,extra_mask=tokenize(tokenizer,extra.text,device); batch_ids=[x[0] for x in current_tokens]+[extra_ids]; batch_masks=[x[1] for x in current_tokens]+[extra_mask]; labels=[x.label for x in current]+[extra.label]; replay_logits=None
            else:
                batch_ids=[x[0] for x in current_tokens]+[replay["input_ids"].to(device)]; batch_masks=[x[1] for x in current_tokens]+[replay["attention_mask"].to(device)]; labels=[x.label for x in current]+[int(replay["label"])]; replay_logits=replay["logits"].to(device)
            logits=logits_of(model,torch.stack(batch_ids),torch.stack(batch_masks)); loss=F.cross_entropy(logits,torch.tensor(labels,device=device))
            if replay_logits is not None: loss=loss+args.distill_weight*F.mse_loss(logits[-1].float(),replay_logits.float())
            records=[ReplayCodec.encode(current_tokens[i][0],current_tokens[i][1],current[i].label,current[i].task,logits[i].detach()) for i in range(CURRENT_SLOTS)]
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,args.grad_clip); optimizer.step(); updates+=1; losses.append(float(loss.detach()))
            if args.method!="naive":
                for record in records: store.insert(record)
        matrix[task_index]=np.asarray(evaluate(model,tokenizer,tasks,device,task_index))
    result={"schema_version":1,"protocol":"qwen-agnews-taskperm-lora-v1","method":args.method,"seed":args.seed,"model":args.model,"requested_model_revision":args.revision,"resolved_model_revision":model_revision,"dataset":dataset_id,"resolved_dataset_revision":dataset_revision,"dataset_subset_sha256":subset_sha,"task_names":list(TASK_NAMES),"label_permutations":[list(p) for p in PERMUTATIONS],"torch":torch.__version__,"trainable_parameters":int(trainable_n),"total_model_parameters":int(sum(p.numel() for p in model.parameters())),"lora_rank":args.lora_rank,"lora_last_layers":args.lora_last_layers,"external_bytes":args.external_bytes,"sign_fiber_bytes":0 if fiber is None else fiber.byte_capacity,"record_bytes":RECORD_BYTES,"record_capacity":0 if args.method=="naive" else store.capacity_records,"records_final":0 if args.method=="naive" else store.count,"records_seen":0 if args.method=="naive" else store.seen,"store_sha256":None if args.method=="naive" else store.digest(),"batch_size":BATCH_SIZE,"current_slots_per_update":CURRENT_SLOTS,"replay_slots_per_update":1,"replay_fraction":0.25,"steps_per_task":args.steps_per_task,"tasks":T,"updates":updates,"processed_text_slots":BATCH_SIZE*updates,"seq_len":SEQ_LEN,"num_labels":N_LABELS,"train_per_class":args.train_per_class,"eval_per_class":args.eval_per_class,"distill_weight":args.distill_weight,"lr":args.lr,"weight_decay":args.weight_decay,"accuracy_matrix":matrix.tolist(),**metrics(matrix),"mean_training_loss":float(np.mean(losses)),"wall_seconds":time.perf_counter()-started}
    canonical=json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=True).encode(); result["result_sha256"]=hashlib.sha256(canonical).hexdigest(); return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--method",required=True,choices=["naive","derpp","dfc_sign_derpp"]); parser.add_argument("--seed",type=int,default=1501); parser.add_argument("--model",default="Qwen/Qwen2.5-0.5B"); parser.add_argument("--revision",default="060db6499f32faf8b98477b0a26969ef7d8b9987"); parser.add_argument("--device",default="cpu"); parser.add_argument("--steps-per-task",type=int,default=96); parser.add_argument("--external-bytes",type=int,default=512); parser.add_argument("--train-per-class",type=int,default=8); parser.add_argument("--eval-per-class",type=int,default=16); parser.add_argument("--lora-rank",type=int,default=4); parser.add_argument("--lora-last-layers",type=int,default=4); parser.add_argument("--lr",type=float,default=0.01); parser.add_argument("--weight-decay",type=float,default=0.0); parser.add_argument("--distill-weight",type=float,default=0.025); parser.add_argument("--grad-clip",type=float,default=1.0); parser.add_argument("--threads",type=int,default=4); parser.add_argument("--output",required=True); args=parser.parse_args(); result=run(args); path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=True)+"\n",encoding="utf-8"); print(json.dumps({k:result[k] for k in ["protocol","method","seed","final_average_accuracy","average_forgetting","current_task_accuracy","record_capacity","sign_fiber_bytes","wall_seconds","result_sha256"]},indent=2))
if __name__=="__main__": main()
