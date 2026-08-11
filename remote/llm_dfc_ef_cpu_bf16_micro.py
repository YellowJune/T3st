"""Compute-bounded CPU BF16 actual-model placement gate.

This wrapper is for large-model placement validation on free CPU runners.  It
keeps the same pretrained model, trainable layer, compressor, optimizer, EF
state, and update path as llm_dfc_ef_qwen, but evaluates/trains over one fixed
example for each of the eight target labels (the first domain).  The resulting
metric is not used as a learning-quality benchmark; the primary gate is exact
external-EF vs DFC-EF state/trajectory equality plus the storage ledger.
"""
from __future__ import annotations
import hashlib, json
import torch
import transformers
import llm_dfc_ef_qwen as core
from chunked_fp32_adamw import FullFP32AdamWChunked
from llm_continual_qwen import SEQ_LEN, make_stream, target_id

_original_build_optimizer = core.build_optimizer

def _safe_build_optimizer(method, params, args):
    if method == "fp32_dense":
        return FullFP32AdamWChunked(params, lr=args.lr, betas=(0.9,0.999), eps=1e-8,
                                   weight_decay=args.weight_decay, chunk_coordinates=args.optimizer_chunk)
    return _original_build_optimizer(method, params, args)
core.build_optimizer = _safe_build_optimizer

# NumPy cannot directly materialize torch.bfloat16.  Hash the physical BF16
# words as uint16 bytes so placement comparisons remain bitwise, not numeric.
def _bf16_safe_hash_tensor(h, tensor: torch.Tensor) -> None:
    x=tensor.detach().contiguous().cpu()
    if x.dtype == torch.bfloat16:
        h.update(x.view(torch.uint16).numpy().tobytes())
    else:
        h.update(x.numpy().tobytes())
core._update_hash_tensor = _bf16_safe_hash_tensor

_orig_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

def _bf16_from_pretrained(*args, **kwargs):
    kwargs["torch_dtype"] = torch.bfloat16
    return _orig_from_pretrained(*args, **kwargs)
transformers.AutoModelForCausalLM.from_pretrained = _bf16_from_pretrained


def _micro_pretokenize(tokenizer, device: torch.device):
    # First domain has exactly one example for each of the fixed eight labels.
    examples = list(make_stream()[0])
    ids=[]; masks=[]; targets=[]
    for ex in examples:
        enc=tokenizer(ex.text,return_tensors="pt",add_special_tokens=False,
                      truncation=True,max_length=SEQ_LEN,padding="max_length")
        ids.append(enc["input_ids"][0]); masks.append(enc["attention_mask"][0]); targets.append(target_id(tokenizer,ex.target_text))
    if len(examples)!=8 or len(set(targets))!=8:
        raise RuntimeError(f"micro label coverage failure: {targets}")
    return torch.stack(ids).to(device), torch.stack(masks).to(device), torch.tensor(targets,device=device,dtype=torch.long)
core.pretokenize = _micro_pretokenize

_orig_run=core.run

def _run(args):
    if str(args.device)!="cpu": raise RuntimeError("BF16 micro wrapper is CPU-only")
    r=_orig_run(args)
    r["protocol"]="qwen-dfc-ef-blocktopk-cpu-bf16-micro-v1"
    r["parameter_dtype"]="torch.bfloat16"
    r["loading_dtype_override"]="cpu-bfloat16"
    r["evaluation_examples"]=8
    r["evaluation_scope"]="first-domain-eight-label-covering-examples"
    r.pop("result_sha256",None)
    raw=json.dumps(r,sort_keys=True,separators=(",",":"),allow_nan=True).encode(); r["result_sha256"]=hashlib.sha256(raw).hexdigest()
    return r
core.run=_run

if __name__=="__main__": core.main()
