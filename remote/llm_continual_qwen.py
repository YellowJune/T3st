"""Resource-matched continual adaptation on a real Qwen causal LM.

DFC-SIGN stores replay records in the sign bits of the ordinary FP32 AdamW
second moment attached to trainable LoRA parameters. External DER++ and
DFC-SIGN receive the same external byte envelope and identical batch/update
shapes. Results include the full accuracy matrix, forgetting, byte ledger, and
resolved Hugging Face revision.
"""
from __future__ import annotations
import argparse,binascii,hashlib,json,random,struct,time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_fiber import DFCAdamW,TorchSignFiberChannel

HEADER_BYTES=32
RECORD_BYTES=104
RECORD_MAGIC=b"QLR1"
STORE_MAGIC=b"DFCLLM1\0"
TOPK=4
SEQ_LEN=16

class QwenReplayCodec:
 @staticmethod
 def encode(input_ids,attention_mask,target,task,topk_indices,topk_logits):
  ids=input_ids.detach().cpu().to(torch.int64).reshape(-1); mask=attention_mask.detach().cpu().to(torch.uint8).reshape(-1)
  if ids.numel()!=SEQ_LEN or mask.numel()!=SEQ_LEN: raise ValueError("fixed sequence length required")
  mask_bits=sum((int(b)&1)<<i for i,b in enumerate(mask.tolist()))
  idx=topk_indices.detach().cpu().to(torch.int64).reshape(-1); vals=topk_logits.detach().cpu().to(torch.float16).reshape(-1)
  body=bytearray(RECORD_MAGIC); body.extend(struct.pack("<BBH",int(task),0,mask_bits)); body.extend(np.asarray(ids,dtype="<u4").tobytes()); body.extend(struct.pack("<I",int(target))); body.extend(np.asarray(idx,dtype="<u4").tobytes()); body.extend(np.asarray(vals,dtype="<f2").tobytes())
  if len(body)!=RECORD_BYTES-4: raise AssertionError(len(body))
  body.extend(struct.pack("<I",binascii.crc32(body)&0xffffffff)); return bytes(body)
 @staticmethod
 def decode(raw):
  if len(raw)!=RECORD_BYTES: raise ValueError("bad record length")
  body=raw[:-4]; crc=struct.unpack("<I",raw[-4:])[0]
  if binascii.crc32(body)&0xffffffff!=crc: raise RuntimeError("record CRC mismatch")
  if body[:4]!=RECORD_MAGIC: raise RuntimeError("record magic mismatch")
  task,_,mask_bits=struct.unpack("<BBH",body[4:8]); c=8
  ids=np.frombuffer(body[c:c+64],dtype="<u4").astype(np.int64); c+=64
  target=struct.unpack("<I",body[c:c+4])[0]; c+=4
  idx=np.frombuffer(body[c:c+16],dtype="<u4").astype(np.int64); c+=16
  vals=np.frombuffer(body[c:c+8],dtype="<f2").astype(np.float32)
  mask=np.asarray([(mask_bits>>i)&1 for i in range(SEQ_LEN)],dtype=np.int64)
  return {"task":int(task),"input_ids":torch.from_numpy(ids.copy()),"attention_mask":torch.from_numpy(mask.copy()),"target":int(target),"topk_indices":torch.from_numpy(idx.copy()),"topk_logits":torch.from_numpy(vals.copy())}

class CombinedByteChannel:
 def __init__(self,external_bytes,fiber):
  if external_bytes<HEADER_BYTES: raise ValueError("external envelope too small")
  self.external=bytearray(external_bytes); self.fiber=fiber; self.fiber_bytes=0 if fiber is None else fiber.byte_capacity
 @property
 def byte_capacity(self): return len(self.external)+self.fiber_bytes
 def read(self,start,count):
  if start<0 or count<0 or start+count>self.byte_capacity: raise IndexError
  out=bytearray(); pos=start; rem=count
  if pos<len(self.external) and rem:
   take=min(rem,len(self.external)-pos); out.extend(self.external[pos:pos+take]); pos+=take; rem-=take
  if rem: out.extend(self.fiber.read_bytes(pos-len(self.external),rem))
  return bytes(out)
 def write(self,start,payload):
  raw=bytes(payload)
  if start<0 or start+len(raw)>self.byte_capacity: raise IndexError
  pos=start; cur=0; rem=len(raw)
  if pos<len(self.external) and rem:
   take=min(rem,len(self.external)-pos); self.external[pos:pos+take]=raw[cur:cur+take]; pos+=take; cur+=take; rem-=take
  if rem: self.fiber.write_bytes(pos-len(self.external),raw[cur:])

class ReservoirStore:
 def __init__(self,channel,rng):
  self.channel=channel; self.rng=rng; self.capacity_records=max(0,(channel.byte_capacity-HEADER_BYTES)//RECORD_BYTES); self._write_header(0,0)
 def _header(self,count,seen):
  body=bytearray(STORE_MAGIC); body.extend(struct.pack("<IIIII",1,RECORD_BYTES,self.capacity_records,count,seen)); body.extend(struct.pack("<I",binascii.crc32(body)&0xffffffff)); return bytes(body)
 def _write_header(self,count,seen): self.channel.write(0,self._header(count,seen))
 def _read_header(self):
  raw=self.channel.read(0,HEADER_BYTES); body=raw[:-4]
  if binascii.crc32(body)&0xffffffff!=struct.unpack("<I",raw[-4:])[0]: raise RuntimeError("header CRC mismatch")
  if body[:8]!=STORE_MAGIC: raise RuntimeError("header magic mismatch")
  ver,rb,cap,count,seen=struct.unpack("<IIIII",body[8:28])
  if (ver,rb,cap)!=(1,RECORD_BYTES,self.capacity_records): raise RuntimeError("header schema mismatch")
  return int(count),int(seen)
 def _offset(self,i): return HEADER_BYTES+i*RECORD_BYTES
 @property
 def count(self): return self._read_header()[0]
 @property
 def seen(self): return self._read_header()[1]
 def insert(self,record):
  count,seen=self._read_header(); seen+=1
  if self.capacity_records==0: self._write_header(count,seen); return
  if count<self.capacity_records: slot=count; count+=1
  else:
   slot=int(self.rng.integers(0,seen))
   if slot>=self.capacity_records: self._write_header(count,seen); return
  self.channel.write(self._offset(slot),record); self._write_header(count,seen)
 def sample(self):
  count,_=self._read_header()
  if count==0:return None
  i=int(self.rng.integers(0,count)); return QwenReplayCodec.decode(self.channel.read(self._offset(i),RECORD_BYTES))
 def digest(self):
  count,_=self._read_header(); return hashlib.sha256(self.channel.read(0,HEADER_BYTES+count*RECORD_BYTES)).hexdigest()

@dataclass(frozen=True)
class PromptExample:
 task:int; key:int; text:str; target_text:str

def make_stream():
 labels=[" red"," blue"," green"," yellow"," black"," white"," orange"," purple"]; shifts=[0,3,5,1]; domains=["alpha","beta","gamma","delta"]
 return [[PromptExample(t,k,f"domain {d} key {k} =",labels[(k+s)%8]) for k in range(8)] for t,(d,s) in enumerate(zip(domains,shifts))]

def tokenize_prompt(tok,text,device):
 e=tok(text,return_tensors="pt",add_special_tokens=False,truncation=True,max_length=SEQ_LEN,padding="max_length"); return e["input_ids"][0].to(device),e["attention_mask"][0].to(device)
def target_id(tok,text):
 ids=tok.encode(text,add_special_tokens=False)
 if len(ids)!=1: raise RuntimeError(f"target not single token: {text} -> {ids}")
 return int(ids[0])
def last_logits(model,ids,mask): return model(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,-1,:]

@torch.inference_mode()
def evaluate(model,tok,tasks,device,upto):
 model.eval(); out=[]
 for ti,task in enumerate(tasks):
  if ti>upto: out.append(float("nan")); continue
  ids=[]; masks=[]; targets=[]
  for ex in task:
   i,m=tokenize_prompt(tok,ex.text,device); ids.append(i); masks.append(m); targets.append(target_id(tok,ex.target_text))
  pred=last_logits(model,torch.stack(ids),torch.stack(masks)).argmax(-1).cpu().numpy(); out.append(float(np.mean(pred==np.asarray(targets))))
 model.train(); return out

def metrics(matrix):
 final=float(np.mean(matrix[-1])); current=float(np.mean(np.diag(matrix))); fg=[]
 for t in range(matrix.shape[0]-1):
  hist=matrix[t:,t]; fg.append(float(np.nanmax(hist[:-1])-hist[-1]))
 return {"final_average_accuracy":final,"current_task_accuracy":current,"average_forgetting":float(np.mean(fg))}

def run(args):
 from huggingface_hub import HfApi
 from peft import LoraConfig,get_peft_model
 from transformers import AutoModelForCausalLM,AutoTokenizer
 torch.set_num_threads(args.threads); torch.set_num_interop_threads(1); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
 device=torch.device(args.device); resolved=HfApi().model_info(args.model).sha
 tok=AutoTokenizer.from_pretrained(args.model,revision=args.revision); tok.padding_side="left"; tok.pad_token=tok.eos_token if tok.pad_token_id is None else tok.pad_token
 model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,torch_dtype=torch.float32,low_cpu_mem_usage=True); model.config.use_cache=False
 n=int(model.config.num_hidden_layers); first=max(0,n-args.lora_last_layers)
 cfg=LoraConfig(r=args.lora_rank,lora_alpha=2*args.lora_rank,lora_dropout=0.0,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","v_proj"],layers_to_transform=list(range(first,n)),layers_pattern="layers")
 model=get_peft_model(model,cfg).to(device); model.train(); trainable=[p for p in model.parameters() if p.requires_grad]; trainable_n=sum(p.numel() for p in trainable)
 opt=DFCAdamW(trainable,lr=args.lr,weight_decay=0.0,enable_fiber=args.method=="dfc_sign_derpp"); fiber=TorchSignFiberChannel(opt) if args.method=="dfc_sign_derpp" else None
 store=ReservoirStore(CombinedByteChannel(args.external_bytes,fiber),np.random.default_rng(args.seed+10003)); tasks=make_stream()
 token_map={ex.target_text:target_id(tok,ex.target_text) for task in tasks for ex in task}
 if len(set(token_map.values()))!=8: raise RuntimeError(f"target collision {token_map}")
 matrix=np.full((4,4),np.nan,dtype=np.float64); rng=np.random.default_rng(args.seed+20003); losses=[]; updates=0; started=time.perf_counter()
 for ti,task in enumerate(tasks):
  for _ in range(args.steps_per_task):
   cur=task[int(rng.integers(0,len(task)))]; ci,cm=tokenize_prompt(tok,cur.text,device); ct=target_id(tok,cur.target_text); rep=store.sample() if args.method!="naive" else None
   if rep is None: ri,rm,rt=ci.clone(),cm.clone(),ct; rix=rlog=None
   else: ri,rm,rt=rep["input_ids"].to(device),rep["attention_mask"].to(device),int(rep["target"]); rix,rlog=rep["topk_indices"].to(device),rep["topk_logits"].to(device)
   logits=last_logits(model,torch.stack([ci,ri]),torch.stack([cm,rm])); lc=F.cross_entropy(logits[0:1],torch.tensor([ct],device=device))
   if rep is None or args.method=="naive": loss=.5*(lc+F.cross_entropy(logits[1:2],torch.tensor([rt],device=device)))
   else: loss=lc+args.replay_ce_weight*F.cross_entropy(logits[1:2],torch.tensor([rt],device=device))+args.distill_weight*F.mse_loss(logits[1,rix].float(),rlog.float())
   vals,idx=torch.topk(logits[0].detach(),k=TOPK); record=QwenReplayCodec.encode(ci,cm,ct,cur.task,idx,vals)
   opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step(); updates+=1; losses.append(float(loss.detach()))
   if args.method!="naive": store.insert(record)
  matrix[ti]=np.asarray(evaluate(model,tok,tasks,device,ti))
 result={"schema_version":1,"method":args.method,"seed":args.seed,"model":args.model,"requested_revision":args.revision,"resolved_hub_revision":resolved,"torch":torch.__version__,"trainable_parameters":int(trainable_n),"total_model_parameters":int(sum(p.numel() for p in model.parameters())),"lora_rank":args.lora_rank,"lora_last_layers":args.lora_last_layers,"external_bytes":args.external_bytes,"sign_fiber_bytes":0 if fiber is None else fiber.byte_capacity,"record_bytes":RECORD_BYTES,"record_capacity":0 if args.method=="naive" else store.capacity_records,"records_final":0 if args.method=="naive" else store.count,"records_seen":0 if args.method=="naive" else store.seen,"store_sha256":None if args.method=="naive" else store.digest(),"batch_size":2,"steps_per_task":args.steps_per_task,"tasks":4,"updates":updates,"processed_prompt_slots":2*updates,"seq_len":SEQ_LEN,"distill_weight":args.distill_weight,"replay_ce_weight":args.replay_ce_weight,"lr":args.lr,"accuracy_matrix":matrix.tolist(),**metrics(matrix),"mean_training_loss":float(np.mean(losses)),"wall_seconds":time.perf_counter()-started}
 canonical=json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=True).encode(); result["result_sha256"]=hashlib.sha256(canonical).hexdigest(); return result

def main():
 p=argparse.ArgumentParser(); p.add_argument("--method",required=True,choices=["naive","derpp","dfc_sign_derpp"]); p.add_argument("--seed",type=int,default=701); p.add_argument("--model",default="Qwen/Qwen2.5-0.5B"); p.add_argument("--revision",default="main"); p.add_argument("--device",default="cpu"); p.add_argument("--steps-per-task",type=int,default=16); p.add_argument("--external-bytes",type=int,default=512); p.add_argument("--lora-rank",type=int,default=4); p.add_argument("--lora-last-layers",type=int,default=8); p.add_argument("--lr",type=float,default=3e-3); p.add_argument("--distill-weight",type=float,default=.2); p.add_argument("--replay-ce-weight",type=float,default=1.0); p.add_argument("--threads",type=int,default=4); p.add_argument("--output",required=True); args=p.parse_args(); result=run(args); path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=True)+"\n"); print(json.dumps({k:result[k] for k in ["method","seed","final_average_accuracy","average_forgetting","current_task_accuracy","record_capacity","sign_fiber_bytes","wall_seconds","result_sha256"]},indent=2))
if __name__=="__main__": main()
