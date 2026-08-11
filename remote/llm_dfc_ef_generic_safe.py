"""Tokenizer-agnostic actual-model DFC-EF entry point.

The core Qwen validation uses eight target strings chosen to be single tokens in
Qwen2.5.  For cross-architecture validation this wrapper keeps the exact same
model/optimizer/compressor/update path but defines each target as the *last
subtoken* of the target string.  This is a deterministic tokenizer adapter, not
a scientific retuning: the purpose of this entry point is placement equivalence
across model families, not task accuracy.
"""
from __future__ import annotations
import hashlib, json
import torch
import llm_dfc_ef_qwen as core
from chunked_fp32_adamw import FullFP32AdamWChunked
from llm_continual_qwen import SEQ_LEN, make_stream

_original_build_optimizer = core.build_optimizer

def _safe_build_optimizer(method, params, args):
    if method == "fp32_dense":
        return FullFP32AdamWChunked(
            params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=args.weight_decay, chunk_coordinates=args.optimizer_chunk,
        )
    return _original_build_optimizer(method, params, args)


def _generic_pretokenize(tokenizer, device: torch.device):
    examples = [ex for task in make_stream() for ex in task]
    ids, masks, targets = [], [], []
    for ex in examples:
        enc = tokenizer(
            ex.text, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=SEQ_LEN, padding="max_length",
        )
        t = tokenizer(ex.target_text, add_special_tokens=False)["input_ids"]
        if not t:
            raise RuntimeError(f"empty target tokenization: {ex.target_text!r}")
        ids.append(enc["input_ids"][0]); masks.append(enc["attention_mask"][0]); targets.append(int(t[-1]))
    if len(set(targets)) != 8:
        raise RuntimeError(f"last-subtoken target collision: {targets}")
    return torch.stack(ids).to(device), torch.stack(masks).to(device), torch.tensor(targets, device=device, dtype=torch.long)

core.build_optimizer = _safe_build_optimizer
core.pretokenize = _generic_pretokenize
_original_run = core.run

def _run(args):
    r = _original_run(args)
    r["protocol"] = "generic-last-subtoken-dfc-ef-blocktopk-v1"
    r["target_adapter"] = "last-subtoken-of-fixed-eight-target-strings"
    # Replace the result digest after protocol metadata changes.
    r.pop("result_sha256", None)
    raw = json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
    r["result_sha256"] = hashlib.sha256(raw).hexdigest()
    return r

core.run = _run

if __name__ == "__main__":
    core.main()
