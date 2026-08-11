"""Actual Qwen learning-path validation for DFC-EF on free CUDA GPUs.

The experiment separates three questions that must not be conflated:

1. numerical contract: ordinary full-FP32 AdamW moments vs BF16-high moment
   semantics (``fp32_dense`` vs ``low16_dense``);
2. compression utility: dense vs blockwise magnitude sparsification with and
   without error feedback;
3. physical placement: matched external FP32 EF state vs exactly the same EF
   state packed into DFC low-word fibers (``external_ef`` vs ``dfc_ef``).

Only the third comparison supports the DFC memory-substitution claim. DFC does
not receive credit for the communication reduction: blockwise top-k is the
compressor and is identical in external-EF and DFC-EF.

The default model is pinned Qwen2.5-0.5B. By default the last 8 transformer
blocks are trainable; ``--train-last-layers 0`` makes all transformer blocks
trainable while embeddings/final norm/lm_head stay frozen.  FP16 parameters are
used on CUDA so the protocol runs on free P100/T4-class Kaggle GPUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from block_topk_ef import block_topk_dfc_inplace_, block_topk_external_inplace_, block_topk_noef_inplace_
from chunked_low16_adamw import DFCLow16AdamWChunked
from dfc_ef import PackedFP32Residual, allocate_external_residuals
from kaggle_checkpoint import atomic_json_dump, atomic_torch_save, capture_rng_state, restore_rng_state
from llm_continual_qwen import SEQ_LEN, make_stream, target_id
from torch_fiber import DFCAdamW, HIGH16_MASK_I32, LOW16_MASK_I32


METHODS = ("fp32_dense", "low16_dense", "low16_noef", "external_ef", "dfc_ef")


def cuda_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0, "max_reserved": 0}
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
        "max_reserved": int(torch.cuda.max_memory_reserved()),
    }


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def trainable_named(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def build_optimizer(method: str, params: list[torch.nn.Parameter], args):
    if method == "fp32_dense":
        return DFCAdamW(
            params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=args.weight_decay, enable_fiber=False,
        )
    return DFCLow16AdamWChunked(
        params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=args.weight_decay, enable_fiber=(method == "dfc_ef"),
        chunk_coordinates=args.optimizer_chunk,
    )


def pretokenize(tokenizer, device: torch.device):
    examples = [ex for task in make_stream() for ex in task]
    ids, masks, targets = [], [], []
    for ex in examples:
        enc = tokenizer(
            ex.text, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=SEQ_LEN, padding="max_length",
        )
        ids.append(enc["input_ids"][0])
        masks.append(enc["attention_mask"][0])
        targets.append(target_id(tokenizer, ex.target_text))
    if len(set(targets)) != 8:
        raise RuntimeError(f"target-token collision: {targets}")
    return (
        torch.stack(ids).to(device),
        torch.stack(masks).to(device),
        torch.tensor(targets, device=device, dtype=torch.long),
    )


@torch.inference_mode()
def evaluate(model, ids, masks, targets, batch_size: int = 8) -> tuple[float, float]:
    model.eval()
    correct = 0
    total_loss = 0.0
    total = int(ids.shape[0])
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        logits = model(
            input_ids=ids[start:end], attention_mask=masks[start:end], use_cache=False
        ).logits[:, -1, :].float()
        total_loss += float(F.cross_entropy(logits, targets[start:end], reduction="sum"))
        correct += int((logits.argmax(-1) == targets[start:end]).sum())
    model.train()
    return correct / total, total_loss / total


def prepare_gradient(p: torch.nn.Parameter) -> torch.Tensor | None:
    g = p.grad
    if g is None:
        return None
    if not g.is_contiguous():
        p.grad = g.contiguous()
        g = p.grad
    return g


@torch.no_grad()
def compress_gradients(
    method: str,
    params: list[torch.nn.Parameter],
    *,
    external_residuals: list[torch.Tensor] | None,
    channel: PackedFP32Residual | None,
    keep_ratio: float,
    chunk_coordinates: int,
) -> int:
    sent = 0
    if method in ("fp32_dense", "low16_dense"):
        return sum(int(p.grad.numel()) for p in params if p.grad is not None)
    for i, p in enumerate(params):
        g = prepare_gradient(p)
        if g is None:
            continue
        if method == "low16_noef":
            sent += block_topk_noef_inplace_(
                g, keep_ratio=keep_ratio, chunk_coordinates=chunk_coordinates
            )
        elif method == "external_ef":
            assert external_residuals is not None
            sent += block_topk_external_inplace_(
                g, external_residuals[i], keep_ratio=keep_ratio,
                chunk_coordinates=chunk_coordinates,
            )
        elif method == "dfc_ef":
            assert channel is not None
            sent += block_topk_dfc_inplace_(
                p, g, channel, keep_ratio=keep_ratio,
                chunk_coordinates=chunk_coordinates,
            )
        else:
            raise ValueError(method)
    return sent


def _update_hash_tensor(h: "hashlib._Hash", tensor: torch.Tensor) -> None:
    x = tensor.detach().contiguous().cpu()
    h.update(x.numpy().tobytes())


def parameter_digest(named: list[tuple[str, torch.nn.Parameter]]) -> str:
    h = hashlib.sha256()
    for name, p in named:
        h.update(name.encode() + b"\0")
        _update_hash_tensor(h, p)
    return h.hexdigest()


def external_residual_digest(residuals: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for r in residuals:
        _update_hash_tensor(h, r.view(torch.int32))
    return h.hexdigest()


@torch.no_grad()
def dfc_residual_digest(channel: PackedFP32Residual, chunk: int) -> str:
    """Hash logical FP32 residual bytes without a model-sized decoded tensor."""
    h = hashlib.sha256()
    for p in channel.params:
        first, second = channel._state_pair(channel.optimizer, p)
        fb = first.view(-1).view(torch.int32)
        sb = second.view(-1).view(torch.int32)
        for start in range(0, p.numel(), chunk):
            end = min(p.numel(), start + chunk)
            residual = channel._decode_words(fb[start:end], sb[start:end])
            _update_hash_tensor(h, residual.view(torch.int32))
    return h.hexdigest()


@torch.no_grad()
def semantic_optimizer_digest(optimizer, params: list[torch.nn.Parameter], low16: bool) -> str:
    h = hashlib.sha256()
    for p in params:
        st = optimizer.state[p]
        for key in ("exp_avg", "exp_avg_sq"):
            x = st[key]
            if low16:
                bits = torch.bitwise_and(x.view(torch.int32), HIGH16_MASK_I32)
                _update_hash_tensor(h, bits)
            else:
                _update_hash_tensor(h, x.view(torch.int32))
    return h.hexdigest()


def save_checkpoint(
    path: Path,
    *,
    update: int,
    named_params: list[tuple[str, torch.nn.Parameter]],
    optimizer,
    external_residuals: list[torch.Tensor] | None,
    method: str,
    result_state: dict,
) -> int:
    payload = {
        "schema_version": 1,
        "method": method,
        "update": int(update),
        "trainable": {name: p.detach() for name, p in named_params},
        "optimizer": optimizer.state_dict(),
        "external_residuals": external_residuals,
        "rng": capture_rng_state(),
        "result_state": result_state,
    }
    atomic_torch_save(payload, path)
    return int(path.stat().st_size)


def load_checkpoint(
    path: Path,
    *,
    named_params: list[tuple[str, torch.nn.Parameter]],
    optimizer,
    external_residuals: list[torch.Tensor] | None,
    device: torch.device,
    expected_method: str,
):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("method") != expected_method:
        raise RuntimeError("checkpoint method mismatch")
    saved = ckpt["trainable"]
    if set(saved) != {n for n, _ in named_params}:
        raise RuntimeError("trainable parameter set mismatch")
    for name, p in named_params:
        p.data.copy_(saved[name].to(device=device, dtype=p.dtype))
    optimizer.load_state_dict(ckpt["optimizer"])
    if external_residuals is not None:
        src = ckpt.get("external_residuals")
        if src is None or len(src) != len(external_residuals):
            raise RuntimeError("missing external EF residual in checkpoint")
        for dst, value in zip(external_residuals, src):
            dst.copy_(value.to(device=device, dtype=torch.float32))
    restore_rng_state(ckpt["rng"])
    return int(ckpt["update"]), dict(ckpt.get("result_state", {}))


def run(args) -> dict:
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.method not in METHODS:
        raise ValueError(args.method)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    requested_dtype = torch.float16 if device.type == "cuda" else torch.float32
    resolved = HfApi().model_info(args.model, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=requested_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    if args.train_last_layers < 0 or args.train_last_layers > len(layers):
        raise ValueError(f"train_last_layers must be 0..{len(layers)}")
    selected = layers if args.train_last_layers == 0 else layers[-args.train_last_layers:]
    for layer in selected:
        for p in layer.parameters():
            p.requires_grad_(True)
    model = model.to(device)
    model.train()
    named = trainable_named(model)
    params = [p for _, p in named]
    trainable_n = int(sum(p.numel() for p in params))
    total_n = int(sum(p.numel() for p in model.parameters()))
    if trainable_n == 0:
        raise RuntimeError("no trainable parameters")
    synchronize()
    mem_after_model = cuda_memory()

    optimizer = build_optimizer(args.method, params, args)
    synchronize()
    mem_after_optimizer = cuda_memory()

    external_residuals = None
    channel = None
    if args.method == "external_ef":
        external_residuals = allocate_external_residuals(params)
    elif args.method == "dfc_ef":
        channel = PackedFP32Residual(optimizer)
        channel.zero_()
    synchronize()
    mem_after_residual_setup = cuda_memory()

    ids, masks, targets = pretokenize(tokenizer, device)
    order = np.random.default_rng(args.seed + 9901).permutation(ids.shape[0]).tolist()
    update = 0
    history: list[dict] = []
    checkpoint_size = None
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path and args.resume and checkpoint_path.exists():
        update, state = load_checkpoint(
            checkpoint_path, named_params=named, optimizer=optimizer,
            external_residuals=external_residuals, device=device,
            expected_method=args.method,
        )
        history = list(state.get("history", []))
        # Rebind channel after state_dict load; optimizer state tensors may have
        # been replaced by load_state_dict.
        if args.method == "dfc_ef":
            channel = PackedFP32Residual(optimizer)

    eval_acc0, eval_loss0 = evaluate(model, ids, masks, targets, args.eval_batch_size)
    if not history:
        history.append({"update": update, "accuracy": eval_acc0, "eval_loss": eval_loss0})
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses: list[float] = []
    transmitted = 0
    total_gradient_values = 0

    while update < args.updates:
        idx = [order[(update * args.batch_size + j) % len(order)] for j in range(args.batch_size)]
        index = torch.tensor(idx, device=device, dtype=torch.long)
        batch_ids = ids.index_select(0, index)
        batch_masks = masks.index_select(0, index)
        batch_targets = targets.index_select(0, index)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=batch_ids, attention_mask=batch_masks, use_cache=False).logits[:, -1, :]
        loss = F.cross_entropy(logits.float(), batch_targets)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        # Make gradient layout common across all methods before timing/packing.
        for p in params:
            prepare_gradient(p)
        total_gradient_values += sum(int(p.grad.numel()) for p in params if p.grad is not None)
        transmitted += compress_gradients(
            args.method, params, external_residuals=external_residuals,
            channel=channel, keep_ratio=args.keep_ratio,
            chunk_coordinates=args.ef_chunk,
        )
        optimizer.step()
        update += 1
        losses.append(float(loss.detach()))

        if args.eval_every > 0 and (update % args.eval_every == 0 or update == args.updates):
            acc, evl = evaluate(model, ids, masks, targets, args.eval_batch_size)
            history.append({"update": update, "accuracy": acc, "eval_loss": evl,
                            "mean_recent_train_loss": float(np.mean(losses[-args.eval_every:]))})
            if args.progress_output:
                atomic_json_dump({
                    "schema_version": 1, "method": args.method, "seed": args.seed,
                    "update": update, "history": history, "memory": cuda_memory(),
                }, args.progress_output)

        if checkpoint_path and args.checkpoint_every > 0 and update % args.checkpoint_every == 0:
            checkpoint_size = save_checkpoint(
                checkpoint_path, update=update, named_params=named, optimizer=optimizer,
                external_residuals=external_residuals, method=args.method,
                result_state={"history": history},
            )

    synchronize()
    wall = time.perf_counter() - started
    train_peak = cuda_memory()
    final_acc, final_eval_loss = evaluate(model, ids, masks, targets, args.eval_batch_size)

    # Save a final restart point if checkpointing was requested.
    if checkpoint_path:
        checkpoint_size = save_checkpoint(
            checkpoint_path, update=update, named_params=named, optimizer=optimizer,
            external_residuals=external_residuals, method=args.method,
            result_state={"history": history},
        )

    param_sha = parameter_digest(named)
    low16 = args.method != "fp32_dense"
    optimizer_sha = semantic_optimizer_digest(optimizer, params, low16=low16)
    if external_residuals is not None:
        residual_sha = external_residual_digest(external_residuals)
    elif channel is not None:
        residual_sha = dfc_residual_digest(channel, args.ef_chunk)
    else:
        residual_sha = None

    actual_external_residual_bytes = 0 if external_residuals is None else int(sum(r.numel() * r.element_size() for r in external_residuals))
    fiber_capacity_bytes = int(4 * trainable_n) if channel is not None else 0
    theoretical_dense_values = int(total_gradient_values)
    result = {
        "schema_version": 1,
        "protocol": "qwen-dfc-ef-blocktopk-v1",
        "method": args.method,
        "seed": args.seed,
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_hub_revision": resolved,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "parameter_dtype": str(requested_dtype),
        "total_model_parameters": total_n,
        "trainable_parameters": trainable_n,
        "train_last_layers": args.train_last_layers,
        "optimizer_semantics": "full-fp32-moments" if args.method == "fp32_dense" else "bf16-high-moments-in-fp32-containers",
        "compressor": "none" if args.method in ("fp32_dense", "low16_dense") else "blockwise-topk",
        "keep_ratio": 1.0 if args.method in ("fp32_dense", "low16_dense") else args.keep_ratio,
        "nominal_value_compression_ratio": 1.0 if args.method in ("fp32_dense", "low16_dense") else 1.0 / args.keep_ratio,
        "updates": update,
        "batch_size": args.batch_size,
        "seq_len": SEQ_LEN,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "ef_chunk_coordinates": args.ef_chunk,
        "optimizer_chunk_coordinates": args.optimizer_chunk,
        "actual_external_residual_bytes": actual_external_residual_bytes,
        "dfc_fiber_capacity_bytes": fiber_capacity_bytes,
        "model_scale_external_residual_removed_bytes": int(4 * trainable_n) if args.method == "dfc_ef" else 0,
        "transmitted_values": int(transmitted),
        "dense_gradient_values": theoretical_dense_values,
        "observed_value_fraction": float(transmitted / theoretical_dense_values) if theoretical_dense_values else None,
        "initial_accuracy": eval_acc0,
        "initial_eval_loss": eval_loss0,
        "final_accuracy": final_acc,
        "final_eval_loss": final_eval_loss,
        "mean_train_loss": float(np.mean(losses)) if losses else None,
        "history": history,
        "wall_seconds": wall,
        "updates_per_second": float(args.updates / wall) if wall > 0 else None,
        "memory_after_model": mem_after_model,
        "memory_after_optimizer": mem_after_optimizer,
        "memory_after_residual_setup": mem_after_residual_setup,
        "training_peak_memory": train_peak,
        "checkpoint_bytes": checkpoint_size,
        "parameter_sha256": param_sha,
        "semantic_optimizer_sha256": optimizer_sha,
        "logical_residual_sha256": residual_sha,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
    result["result_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--seed", type=int, default=260811)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--revision", default="060db6499f32faf8b98477b0a26969ef7d8b9987")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-last-layers", type=int, default=8,
                   help="0 = all transformer blocks; embeddings/head remain frozen")
    p.add_argument("--updates", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--keep-ratio", type=float, default=0.125)
    p.add_argument("--ef-chunk", type=int, default=262_144)
    p.add_argument("--optimizer-chunk", type=int, default=262_144)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--checkpoint-every", type=int, default=64)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--progress-output", default="")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = run(args)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(result, out)
    summary_keys = [
        "protocol", "method", "seed", "device_name", "trainable_parameters",
        "optimizer_semantics", "actual_external_residual_bytes",
        "dfc_fiber_capacity_bytes", "model_scale_external_residual_removed_bytes",
        "initial_accuracy", "final_accuracy", "final_eval_loss", "wall_seconds",
        "updates_per_second", "result_sha256",
    ]
    print(json.dumps({k: result[k] for k in summary_keys}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
