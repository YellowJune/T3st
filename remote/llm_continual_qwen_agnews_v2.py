"""Qwen-0.5B AG News continual gate with a 3-current:1-replay batch.

This v2 protocol is the post-falsifier correction of the initial 1:1 prompt-slot
pilot. It matches the 25% replay fraction used by the main strict DFC gate: each
update has three current AG News examples and one replay example, and replay is
one ordinary example in the cross-entropy mean. External DER++ and DFC-SIGN use
identical model, FP32 AdamW allocation, external bytes, batch shape and update
count. DFC-SIGN only addresses sign bits already resident in FP32 second moments.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_fiber import DFCAdamW, TorchSignFiberChannel
from llm_continual_qwen_agnews import N_LABELS,SEQ_LEN,RECORD_BYTES,ReplayCodec,CombinedByteChannel,ReservoirStore,load_stream,tokenize,logits_of,evaluate,metrics

CURRENT_SLOTS=3
BATCH_SIZE=4

def run(args):
 from huggingface_hub import HfApi
 from peft import LoraConfig,get_peft_model
 from transformers import AutoModelForSequenceClassification,AutoTokenizer
 torch.set_num_threads(args.threads); torch.set_num_interop_threads(1); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
 device=torch.device(args.device); model_revision=HfApi().model_info(args.model,revision=args.revision).sha
 tasks,dataset_id,dataset_revision,subset_sha=load_stream(args.seed,args.train_per_class,args.eval_per_class)
 tok=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
 if tok.pad_token_id is None: tok.pad_token=tok.eos_token
 tok.padding_side="left"
 model=AutoModelForSequenceClassification.from_pretrained(args.model,revision=args.revision,num_labels=N_LABELS,torch_dtype=torch.float32,low_cpu_mem_usage=True); model.config.pad_token_id=tok.pad_token_id; model.config.use_cache=False
 n_layers=int(model.config.num_hidden_layers); first=max(0,n_layers-args.lora_last_layers)
 cfg=LoraConfig(r=args.lora_rank,lora_alpha=2*args.lora_rank,lora_dropout=0.0,bias="none",task_type="SEQ_CLS",target_modules=["q_proj","v_proj"],modules_to_save=["score"],layers_to_transform=list(range(first,n_layers)),layers_pattern="layers")
 model=get_peft_model(model,cfg).to(device); model.train(); trainable=[p for p in model.parameters() if p.requires_grad]; trainable_n=sum(p.numel() for p in trainable)
 opt=DFCAdamW(trainable,lr=args.lr,weight_decay=args.weight_decay,enable_fiber=args.method=="dfc_sign_derpp"); fiber=TorchSignFiberChannel(opt) if args.method=="dfc_sign_derpp" else None; store=ReservoirStore(CombinedByteChannel(args.external_bytes,fiber),np.random.default_rng(args.seed+10003))
 T=len(tasks); matrix=np.full((T,T),np.nan,dtype=np.float64); rng=np.random.default_rng(args.seed+20003); updates=0; losses=[]; started=time.perf_counter()
 for task_index,(train_examples,_) in enumerate(tasks):
  for _ in range(args.steps_per_task):
   current=[train_examples[int(rng.integers(0,len(train_examples)))] for _ in range(CURRENT_SLOTS)]; current_tokens=[tokenize(tok,ex.text,device) for ex in current]; replay=store.sample() if args.method!="naive" else None
   if replay is None:
    extra=train_examples[int(rng.integers(0,len(train_examples)))]; ei,em=tokenize(tok,extra.text,device); batch_ids=[x[0] for x in current_tokens]+[ei]; batch_masks=[x[1] for x in current_tokens]+[em]; labels=[x.label for x in current]+[extra.label]; replay_logits=None
   else:
    batch_ids=[x[0] for x in current_tokens]+[replay["input_ids"].to(device)]; batch_masks=[x[1] for x in current_tokens]+[replay["attention_mask"].to(device)]; labels=[x.label for x in current]+[int(replay["label"])]; replay_logits=replay["logits"].to(device)
   logits=logits_of(model,torch.stack(batch_ids),torch.stack(batch_masks)); loss=F.cross_entropy(logits,torch.tensor(labels,device=device))
   if replay_logits is not None: loss=loss+args.distill_weight*F.mse_loss(logits[-1].float(),replay_logits.float())
   records=[ReplayCodec.encode(current_tokens[i][0],current_tokens[i][1],current[i].label,current[i].task,logits[i].detach()) for i in range(CURRENT_SLOTS)]
   opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,args.grad_clip); opt.step(); updates+=1; losses.append(float(loss.detach()))
   if args.method!="naive":
    for record in records: store.insert(record)
  matrix[task_index]=np.asarray(evaluate(model,tok,tasks,device,task_index))
 result={"schema_version":2,"protocol":"qwen-agnews-classinc-lora-v2","method":args.method,"seed":args.seed,"model":args.model,"requested_model_revision":args.revision,"resolved_model_revision":model_revision,"dataset":dataset_id,"resolved_dataset_revision":dataset_revision,"dataset_subset_sha256":subset_sha,"torch":torch.__version__,"trainable_parameters":int(trainable_n),"total_model_parameters":int(sum(p.numel() for p in model.parameters())),"lora_rank":args.lora_rank,"lora_last_layers":args.lora_last_layers,"external_bytes":args.external_bytes,"sign_fiber_bytes":0 if fiber is None else fiber.byte_capacity,"record_bytes":RECORD_BYTES,"record_capacity":0 if args.method=="naive" else store.capacity_records,"records_final":0 if args.method=="naive" else store.count,"records_seen":0 if args.method=="naive" else store.seen,"store_sha256":None if args.method=="naive" else store.digest(),"batch_size":BATCH_SIZE,"current_slots_per_update":CURRENT_SLOTS,"replay_slots_per_update":1,"replay_fraction":0.25,"steps_per_task":args.steps_per_task,"tasks":T,"updates":updates,"processed_text_slots":BATCH_SIZE*updates,"seq_len":SEQ_LEN,"num_labels":N_LABELS,"train_per_class":args.train_per_class,"eval_per_class":args.eval_per_class,"distill_weight":args.distill_weight,"lr":args.lr,"weight_decay":args.weight_decay,"accuracy_matrix":matrix.tolist(),**metrics(matrix),"mean_training_loss":float(np.mean(losses)),"wall_seconds":time.perf_counter()-started}
 canonical=json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=True).encode(); result["result_sha256"]=hashlib.sha256(canonical).hexdigest(); return result

def main():
 p=argparse.ArgumentParser(); p.add_argument("--method",required=True,choices=["naive","derpp","dfc_sign_derpp"]); p.add_argument("--seed",type=int,default=1201); p.add_argument("--model",default="Qwen/Qwen2.5-0.5B"); p.add_argument("--revision",default="060db6499f32faf8b98477b0a26969ef7d8b9987"); p.add_argument("--device",default="cpu"); p.add_argument("--steps-per-task",type=int,default=64); p.add_argument("--external-bytes",type=int,default=512); p.add_argument("--train-per-class",type=int,default=32); p.add_argument("--eval-per-class",type=int,default=64); p.add_argument("--lora-rank",type=int,default=4); p.add_argument("--lora-last-layers",type=int,default=4); p.add_argument("--lr",type=float,default=0.01); p.add_argument("--weight-decay",type=float,default=0.0); p.add_argument("--distill-weight",type=float,default=0.025); p.add_argument("--grad-clip",type=float,default=1.0); p.add_argument("--threads",type=int,default=4); p.add_argument("--output",required=True); args=p.parse_args(); result=run(args); path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=True)+"\n"); print(json.dumps({k:result[k] for k in ["protocol","method","seed","final_average_accuracy","average_forgetting","current_task_accuracy","record_capacity","sign_fiber_bytes","wall_seconds","result_sha256"]},indent=2))
if __name__=="__main__": main()
