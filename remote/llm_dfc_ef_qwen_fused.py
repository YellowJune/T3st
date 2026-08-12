"""Actual Qwen training with one-pass fused EF + BF16-high AdamW.

This experiment isolates physical placement while removing the implementation
artifact that dominated the earlier unfused DFC path. `external_fused` and
`dfc_fused` use the same structured stride compressor, FP16 transport
quantization, semantic high-16 Adam moments, parameter update ordering, model,
data, and seeds. The DFC variant differs only by placing the logical FP32 EF
residual in the two low-16 optimizer fibers and therefore allocates no separate
model-sized residual tensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fused_stride_ef_adamw import FusedStrideEFAdamW
from llm_continual_qwen import SEQ_LEN, make_stream, target_id


def cuda_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0, "max_reserved": 0}
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
        "max_reserved": int(torch.cuda.max_memory_reserved()),
    }


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def trainable_named(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def pretokenize(tokenizer, device):
    examples = [ex for task in make_stream() for ex in task]
    ids, masks, targets = [], [], []
    for ex in examples:
        enc = tokenizer(ex.text, return_tensors="pt", add_special_tokens=False,
                        truncation=True, max_length=SEQ_LEN, padding="max_length")
        ids.append(enc["input_ids"][0]); masks.append(enc["attention_mask"][0])
        targets.append(target_id(tokenizer, ex.target_text))
    if len(set(targets)) != 8:
        raise RuntimeError(f"target-token collision: {targets}")
    return (torch.stack(ids).to(device), torch.stack(masks).to(device),
            torch.tensor(targets, device=device, dtype=torch.long))


@torch.inference_mode()
def evaluate(model, ids, masks, targets, batch_size=8):
    model.eval(); correct = 0; total_loss = 0.0; total = int(ids.shape[0])
    for s in range(0, total, batch_size):
        e = min(total, s + batch_size)
        logits = model(input_ids=ids[s:e], attention_mask=masks[s:e], use_cache=False).logits[:, -1, :].float()
        total_loss += float(F.cross_entropy(logits, targets[s:e], reduction="sum"))
        correct += int((logits.argmax(-1) == targets[s:e]).sum())
    model.train()
    return correct / total, total_loss / total


def update_hash_tensor(h, tensor):
    x = tensor.detach().contiguous().cpu()
    h.update(x.numpy().tobytes())


def parameter_digest(named):
    h = hashlib.sha256()
    for name, p in named:
        h.update(name.encode() + b"\0"); update_hash_tensor(h, p)
    return h.hexdigest()


@torch.no_grad()
def semantic_optimizer_digest(opt):
    h = hashlib.sha256()
    for mb, vb in opt.semantic_moment_bits():
        update_hash_tensor(h, mb); update_hash_tensor(h, vb)
    return h.hexdigest()


@torch.no_grad()
def logical_residual_digest(opt):
    h = hashlib.sha256()
    for r in opt.logical_residuals():
        update_hash_tensor(h, r.view(torch.int32))
    return h.hexdigest()


def checkpoint_bytes(path: Path, named, opt) -> int:
    payload = {
        "schema_version": 1,
        "trainable": {n: p.detach() for n, p in named},
        "optimizer": opt.checkpoint_dict(),
    }
    torch.save(payload, path)
    return int(path.stat().st_size)


def run(args):
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.method not in ("external_fused", "dfc_fused"):
        raise ValueError(args.method)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    resolved = HfApi().model_info(args.model, revision=args.revision).sha
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    for p in model.parameters(): p.requires_grad_(False)
    layers = model.model.layers
    if args.train_last_layers < 0 or args.train_last_layers > len(layers):
        raise ValueError("invalid train_last_layers")
    selected = layers if args.train_last_layers == 0 else layers[-args.train_last_layers:]
    for layer in selected:
        for p in layer.parameters(): p.requires_grad_(True)
    model = model.to(device); model.train(); sync()
    named = trainable_named(model); params = [p for _, p in named]
    if any(not p.is_contiguous() for p in params):
        raise RuntimeError("non-contiguous trainable parameter")
    trainable_n = int(sum(p.numel() for p in params)); total_n = int(sum(p.numel() for p in model.parameters()))
    mem_after_model = cuda_memory()

    compile_started = time.perf_counter()
    opt = FusedStrideEFAdamW(params, method=args.method, lr=args.lr,
                             weight_decay=args.weight_decay, stride=args.stride)
    sync(); compile_seconds = time.perf_counter() - compile_started
    mem_after_state = cuda_memory()

    ids, masks, targets = pretokenize(tok, device)
    order = np.random.default_rng(args.seed + 9901).permutation(ids.shape[0]).tolist()
    acc0, ev0 = evaluate(model, ids, masks, targets, args.eval_batch_size)
    history = [{"update": 0, "accuracy": acc0, "eval_loss": ev0}]

    torch.cuda.reset_peak_memory_stats(); sync()
    start_wall = time.perf_counter(); losses = []; transmitted = 0; dense_values = 0
    event_pairs = []
    for update in range(args.updates):
        idx = [order[(update * args.batch_size + j) % len(order)] for j in range(args.batch_size)]
        index = torch.tensor(idx, device=device, dtype=torch.long)
        batch_ids = ids.index_select(0, index); batch_masks = masks.index_select(0, index); batch_targets = targets.index_select(0, index)
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=batch_ids, attention_mask=batch_masks, use_cache=False).logits[:, -1, :]
        loss = F.cross_entropy(logits.float(), batch_targets); loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        for p in params:
            if p.grad is not None and not p.grad.is_contiguous():
                p.grad = p.grad.contiguous()
        dense_values += sum(int(p.grad.numel()) for p in params if p.grad is not None)
        evs = torch.cuda.Event(enable_timing=True); eve = torch.cuda.Event(enable_timing=True)
        evs.record(); transmitted += opt.step(phase=update % args.stride); eve.record(); event_pairs.append((evs, eve))
        losses.append(float(loss.detach()))
        u = update + 1
        if args.eval_every > 0 and (u % args.eval_every == 0 or u == args.updates):
            acc, evl = evaluate(model, ids, masks, targets, args.eval_batch_size)
            history.append({"update": u, "accuracy": acc, "eval_loss": evl,
                            "mean_recent_train_loss": float(np.mean(losses[-args.eval_every:]))})
    sync(); wall_seconds = time.perf_counter() - start_wall
    training_peak = cuda_memory()
    fused_update_ms = float(sum(a.elapsed_time(b) for a, b in event_pairs))

    accf, evf = evaluate(model, ids, masks, targets, args.eval_batch_size); sync()
    param_sha = parameter_digest(named)
    sem_sha = semantic_optimizer_digest(opt)
    res_sha = logical_residual_digest(opt)

    cp = Path(args.checkpoint) if args.checkpoint else Path(tempfile.gettempdir()) / f"dfc_fused_{args.method}_{args.seed}.pt"
    cp.parent.mkdir(parents=True, exist_ok=True)
    cbytes = checkpoint_bytes(cp, named, opt)
    if not args.keep_checkpoint:
        cp.unlink(missing_ok=True)

    result = {
        "schema_version": 1, "protocol": "dfc-fused-stride-ef-adamw-qwen-v1",
        "method": args.method, "model": args.model, "requested_revision": args.revision,
        "resolved_hub_revision": resolved, "seed": args.seed,
        "train_last_layers": args.train_last_layers, "trainable_parameters": trainable_n,
        "total_parameters": total_n, "updates": args.updates, "batch_size": args.batch_size,
        "seq_len": SEQ_LEN, "lr": args.lr, "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip, "stride": args.stride,
        "transport_dtype": "float16", "semantic_optimizer": "BF16-high in FP32 containers",
        "actual_external_residual_bytes": int(opt.external_residual_bytes),
        "dfc_fiber_capacity_bytes": 4 * trainable_n if args.method == "dfc_fused" else 0,
        "model_scale_external_residual_removed_bytes": 4 * trainable_n if args.method == "dfc_fused" else 0,
        "persistent_optimizer_plus_ef_bytes": int(opt.physical_state_bytes),
        "memory_after_model": mem_after_model, "memory_after_state_setup": mem_after_state,
        "training_peak_memory": training_peak, "backend_compile_seconds": compile_seconds,
        "initial_accuracy": acc0, "initial_eval_loss": ev0, "final_accuracy": accf,
        "final_eval_loss": evf, "history": history, "wall_seconds": wall_seconds,
        "fused_update_total_ms": fused_update_ms,
        "fused_update_mean_ms": fused_update_ms / args.updates,
        "transmitted_values": int(transmitted), "dense_gradient_values": int(dense_values),
        "communication_ratio": float(transmitted / dense_values),
        "checkpoint_bytes": cbytes, "parameter_sha256": param_sha,
        "semantic_optimizer_sha256": sem_sha, "logical_residual_sha256": res_sha,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0), "gpu_capability": list(torch.cuda.get_device_capability(0)),
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["result_sha256"] = hashlib.sha256(canonical).hexdigest()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=("external_fused", "dfc_fused"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--revision", default="060db6499f32faf8b98477b0a26969ef7d8b9987")
    ap.add_argument("--seed", type=int, default=3101); ap.add_argument("--train-last-layers", type=int, default=0)
    ap.add_argument("--updates", type=int, default=64); ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=8); ap.add_argument("--eval-every", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0); ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--output", required=True); ap.add_argument("--checkpoint"); ap.add_argument("--keep-checkpoint", action="store_true")
    args = ap.parse_args(); result = run(args)
    print(json.dumps({k: result[k] for k in ("method", "model", "trainable_parameters", "final_accuracy",
          "final_eval_loss", "wall_seconds", "fused_update_mean_ms", "checkpoint_bytes", "result_sha256")}, indent=2))


if __name__ == "__main__": main()
