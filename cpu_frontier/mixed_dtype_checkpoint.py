#!/usr/bin/env python3
"""Validate physical FP32 fiber state across FP16/BF16 optimizer restore."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REMOTE = Path(__file__).resolve().parents[1] / "remote"
sys.path.insert(0, str(REMOTE))

from torch_fiber import (  # noqa: E402
    DFCAdamW,
    DFCLow16AdamW,
    TorchLow16FiberChannel,
    TorchSignFiberChannel,
)


class Tiny(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        gen = torch.Generator().manual_seed(991)
        self.a = nn.Parameter(torch.randn(1024, generator=gen).to(dtype) * 0.03)
        self.b = nn.Parameter(torch.randn(257, generator=gen).to(dtype) * 0.03)


def clone_model(model: Tiny, dtype) -> Tiny:
    out = Tiny(dtype)
    out.load_state_dict(model.state_dict())
    return out


def set_pair_grads(a: Tiny, b: Tiny, gen: torch.Generator):
    for pa, pb in zip(a.parameters(), b.parameters()):
        grad = (torch.randn(pa.shape, generator=gen) * 0.02).to(pa.dtype)
        pa.grad = grad.clone()
        pb.grad = grad.clone()


def exact_tensor(a: torch.Tensor, b: torch.Tensor, label: str):
    if a.dtype != b.dtype:
        raise AssertionError(f"{label}: dtype mismatch {a.dtype} != {b.dtype}")
    if not torch.equal(a, b):
        raise AssertionError(f"{label}: physical tensor mismatch")


def exact_optimizer(a, b, keys=("exp_avg", "exp_avg_sq")):
    for ga, gb in zip(a.param_groups, b.param_groups):
        for pa, pb in zip(ga["params"], gb["params"]):
            exact_tensor(pa, pb, "parameter")
            sa, sb = a.state[pa], b.state[pb]
            for key in keys:
                exact_tensor(sa[key], sb[key], key)
                if sa[key].dtype != torch.float32:
                    raise AssertionError(f"{key}: fiber physical state is not FP32 after restore")


def run_case(dtype: torch.dtype, method: str, seed: int) -> dict:
    base = Tiny(dtype)
    uninterrupted = clone_model(base, dtype)
    checkpointed = clone_model(base, dtype)
    kwargs = dict(lr=7e-4, betas=(0.87, 0.997), eps=1e-7, weight_decay=0.013)

    if method == "sign":
        Opt, Channel = DFCAdamW, TorchSignFiberChannel
    elif method == "low16":
        Opt, Channel = DFCLow16AdamW, TorchLow16FiberChannel
    else:
        raise ValueError(method)

    opt_a = Opt(uninterrupted.parameters(), enable_fiber=True, **kwargs)
    opt_b = Opt(checkpointed.parameters(), enable_fiber=True, **kwargs)
    ch_a, ch_b = Channel(opt_a), Channel(opt_b)
    payload = np.random.default_rng(seed).bytes(ch_a.byte_capacity)
    ch_a.write_bytes(0, payload)
    ch_b.write_bytes(0, payload)

    gen = torch.Generator().manual_seed(seed + 1)
    pre_steps, post_steps = 7, 11
    for _ in range(pre_steps):
        set_pair_grads(uninterrupted, checkpointed, gen)
        opt_a.step(); opt_b.step()
        opt_a.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
        exact_optimizer(opt_a, opt_b)

    model_buf, opt_buf = io.BytesIO(), io.BytesIO()
    torch.save(checkpointed.state_dict(), model_buf)
    torch.save(opt_b.state_dict(), opt_buf)
    checkpoint_bytes = len(model_buf.getvalue()) + len(opt_buf.getvalue())
    model_buf.seek(0); opt_buf.seek(0)

    restored = Tiny(dtype)
    restored.load_state_dict(torch.load(model_buf, weights_only=True))
    opt_r = Opt(restored.parameters(), enable_fiber=True, **kwargs)
    opt_r.load_state_dict(torch.load(opt_buf, weights_only=True))
    ch_r = Channel(opt_r)

    exact_optimizer(opt_a, opt_r)
    if ch_r.read_bytes(0, len(payload)) != payload:
        raise AssertionError(f"{method}/{dtype}: payload changed on restore")

    for _ in range(post_steps):
        set_pair_grads(uninterrupted, restored, gen)
        opt_a.step(); opt_r.step()
        opt_a.zero_grad(set_to_none=True); opt_r.zero_grad(set_to_none=True)
        exact_optimizer(opt_a, opt_r)

    recovered = ch_r.read_bytes(0, len(payload))
    if recovered != payload:
        raise AssertionError(f"{method}/{dtype}: payload changed after resumed training")

    return {
        "status": "PASS",
        "method": method,
        "parameter_dtype": str(dtype).replace("torch.", ""),
        "physical_state_dtype": "float32",
        "pre_steps": pre_steps,
        "post_steps": post_steps,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "checkpoint_bytes": checkpoint_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for i, dtype in enumerate((torch.float16, torch.bfloat16)):
        for j, method in enumerate(("sign", "low16")):
            cases.append(run_case(dtype, method, 88000 + 100 * i + j))
    report = {
        "schema": "dfc-mixed-dtype-physical-checkpoint-v1",
        "status": "PASS",
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "cases": cases,
        "claim": "FP16/BF16 parameter checkpoints preserve physical FP32 DFC-SIGN/DFC-LOW16 payload state and resume bit-identically to uninterrupted execution.",
        "excluded": ["GPU performance", "accelerator behavior"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
