"""CPU BF16 loading wrapper for large free-runner DFC-EF placement tests.

The scientific comparison remains external-EF vs DFC-EF under identical
LOW16 moment semantics.  Only model parameter storage/compute dtype is changed
to BF16 to make larger pretrained models feasible on free CPU runners.
"""
from __future__ import annotations
import hashlib, json
import torch
import transformers
import llm_dfc_ef_qwen as core
from chunked_fp32_adamw import FullFP32AdamWChunked

_original_build_optimizer = core.build_optimizer

def _safe_build_optimizer(method, params, args):
    if method == "fp32_dense":
        return FullFP32AdamWChunked(params, lr=args.lr, betas=(0.9,0.999), eps=1e-8,
                                   weight_decay=args.weight_decay, chunk_coordinates=args.optimizer_chunk)
    return _original_build_optimizer(method, params, args)
core.build_optimizer = _safe_build_optimizer

_orig_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

def _bf16_from_pretrained(*args, **kwargs):
    kwargs["torch_dtype"] = torch.bfloat16
    return _orig_from_pretrained(*args, **kwargs)
transformers.AutoModelForCausalLM.from_pretrained = _bf16_from_pretrained

_orig_run = core.run

def _run(args):
    if str(args.device) != "cpu":
        raise RuntimeError("cpu-bf16 wrapper is CPU-only")
    r = _orig_run(args)
    r["protocol"] = "qwen-dfc-ef-blocktopk-cpu-bf16-v1"
    r["parameter_dtype"] = "torch.bfloat16"
    r["loading_dtype_override"] = "cpu-bfloat16"
    r.pop("result_sha256",None)
    raw=json.dumps(r,sort_keys=True,separators=(",",":"),allow_nan=True).encode()
    r["result_sha256"]=hashlib.sha256(raw).hexdigest()
    return r
core.run = _run

if __name__ == "__main__":
    core.main()
