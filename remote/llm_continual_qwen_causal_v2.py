"""Full-FP32 DFC-SIGN continual adaptation on a real Qwen causal LM.

Four sequential domain-conditioned associative tasks reuse eight keys but assign
four different key->token mappings. Qwen2.5-0.5B is adapted with LoRA only; the
LoRA AdamW moments remain ordinary FP32. Every update contains three current
prompts and one replay prompt (25% replay fraction). External DER++ and
DFC-SIGN+DER++ receive the same 512 external bytes, batch shape, updates, model,
optimizer state allocation, and replay loss. DFC-SIGN addresses only the sign
fiber already resident in FP32 second moments.
"""
from __future__ import annotations
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_fiber import DFCAdamW,TorchSignFiberChannel
from llm_continual_qwen import QwenReplayCodec,CombinedByteChannel,ReservoirStore,make_stream,_tokenize_prompt,_target_id,_last_logits,evaluate_matrix,accuracy_metrics,RECORD_BYTES,SEQ_LEN,TOPK
CURRENT_SLOTS=3
BATCH_SIZE=4

def run(args):
 from huggingface_hub import HfApi
 from peft import LoraConfig,get_peft_model
 from transformers import AutoModelForCausalLM,AutoTokenizer
 torch.set_num_threads(args.threads); torch.set_num_interop_threads(1); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); device=torch.device(args.device)
 resolved_revision=HfApi().model_info(args.model,revision=args.revision).sha; tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision); tokenizer.padding_side="left"
 if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
 model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,torch_dtype=torch.float32,low_cpu_mem_usage=True); model.config.use_cache=False
 n_layers=int(model.config.num_hidden_layers); first=max(0,n_layers-args.lora_last_layers); config=LoraConfig(r=args.lora_rank,lora_alpha=2*args.lora_rank,lora_dropout=0.0,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","v_proj"],layers_to_transform=list(range(first,n_layers)),layers_pattern="layers")
 model=get_peft_model(model,config).to(device); model.train(); trainable=[p for p in model.parameters() if p.requires_grad]; trainable_n=sum(p.numel() for p in trainable)
 optimizer=DFCAdamW(trainable,lr=args.lr,weight_decay=args.weight_decay,enable_fiber=args.method=="dfc_sign_derpp"); fiber=TorchSignFiberChannel(optimizer) if args.method=="dfc_sign_derpp" else None; store=ReservoirStore(CombinedByteChannel(args.external_bytes,fiber),np.random.default_rng(args.seed+10003))
 tasks=make_stream(); T=len(tasks); matrix=np.full((T,T),np.nan,dtype=np.float64); rng=np.random.default_rng(args.seed+20003); updates=0; losses=[]; started=time.perf_counter()
 for task_index,task in enumerate(tasks):
  for _ in range(args.steps_per_task):
   current=[task[int(rng.integers(0,len(task)))] for _ in range(CURRENT_SLOTS)]; current_tokens=[_tokenize_prompt(tokenizer,ex.text,device) for ex in current]; replay=store.sample()
   if replay is None:
    extra=task[int(rng.integers(0,len(task)))]; extra_ids,extra_mask=_tokenize_prompt(tokenizer,extra.text,device); batch_ids=[x[0] for x in current_tokens]+[extra_ids]; batch_masks=[x[1] for x in current_tokens]+[extra_mask]; targets=[_target_id(tokenizer,ex.target_text) for ex in current]+[_target_id(tokenizer,extra.target_text)]; replay_topk_indices=None; replay_topk_logits=None
   else:
    batch_ids=[x[0] for x in current_tokens]+[replay["input_ids"].to(device)]; batch_masks=[x[1] for x in current_tokens]+[replay["attention_mask"].to(device)]; targets=[_target_id(tokenizer,ex.target_text) for ex in current]+[int(replay["target"])]; replay_topk_indices=replay["topk_indices"].to(device); replay_topk_logits=replay["topk_logits"].to(device)
   logits=_last_logits(model,torch.stack(batch_ids),torch.stack(batch_masks)); loss=F.cross_entropy(logits,torch.tensor(targets,device=device))
   if replay_topk_indices is not None: loss=loss+args.distill_weight*F.mse_loss(logits[-1,replay_topk_indices].float(),replay_topk_logits.float())
   records=[]
   for i,ex in enumerate(current):
    vals,idx=torch.topk(logits[i].detach(),k=TOPK); records.append(QwenReplayCodec.encode(current_tokens[i][0],current_tokens[i][1],_target_id(tokenizer,ex.target_text),ex.task,idx,vals))
   optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,args.grad_clip); optimizer.step(); updates+=1; losses.append(float(loss.detach()))
   for record in records: store.insert(record)
  matrix[task_index]=np.asarray(evaluate_matrix(model,tokenizer,tasks,device,task_index))
 result={"schema_version":1,"protocol":"qwen-causal-domainmap-lora-v2","method":args.method,"seed":args.seed,"model":args.model,"requested_model_revision":args.revision,"resolved_model_revision":resolved_revision,"torch":torch.__version__,"trainable_parameters":int(trainable_n),"total_model_parameters":int(sum(p.numel() for p in model.parameters())),"lora_rank":args.lora_rank,"lora_last_layers":args.lora_last_layers,"external_bytes":args.external_bytes,"sign_fiber_bytes":0 if fiber is None else fiber.byte_capacity,"record_bytes":RECORD_BYTES,"record_capacity":store.capacity_records,"records_final":store.count,"records_seen":store.seen,"store_sha256":store.digest(),"batch_size":BATCH_SIZE,"current_slots_per_update":CURRENT_SLOTS,"replay_slots_per_update":1,"replay_fraction":0.25,"steps_per_task":args.steps_per_task,"tasks":T,"updates":updates,"processed_prompt_slots":BATCH_SIZE*updates,"seq_len":SEQ_LEN,"topk":TOPK,"distill_weight":args.distill_weight,"lr":args.lr,"weight_decay":args.weight_decay,"accuracy_matrix":matrix.tolist(),**accuracy_metrics(matrix),"mean_training_loss":float(np.mean(losses)),"wall_seconds":time.perf_counter()-started}
 canonical=json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=True).encode(); result["result_sha256"]=hashlib.sha256(canonical).hexdigest(); return result

def main():
 p=argparse.ArgumentParser(); p.add_argument("--method",required=True,choices=["derpp","dfc_sign_derpp"]); p.add_argument("--seed",type=int,default=1801); p.add_argument("--model",default="Qwen/Qwen2.5-0.5B"); p.add_argument("--revision",default="060db6499f32faf8b98477b0a26969ef7d8b9987"); p.add_argument("--device",default="cpu"); p.add_argument("--steps-per-task",type=int,default=64); p.add_argument("--external-bytes",type=int,default=512); p.add_argument("--lora-rank",type=int,default=4); p.add_argument("--lora-last-layers",type=int,default=4); p.add_argument("--lr",type=float,default=0.01); p.add_argument("--weight-decay",type=float,default=0.0); p.add_argument("--distill-weight",type=float,default=0.02); p.add_argument("--grad-clip",type=float,default=1.0); p.add_argument("--threads",type=int,default=4); p.add_argument("--output",required=True); args=p.parse_args(); result=run(args); path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=True)+"\n",encoding="utf-8"); print(json.dumps({k:result[k] for k in ["protocol","method","seed","final_average_accuracy","average_forgetting","current_task_accuracy","record_capacity","sign_fiber_bytes","wall_seconds","result_sha256"]},indent=2))
if __name__=="__main__": main()
