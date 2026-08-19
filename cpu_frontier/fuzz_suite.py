#!/usr/bin/env python3
"""Randomized CPU property tests for DFC-SIGN and DFC-LOW16.

The goal is breadth rather than benchmark performance: tensor partitions,
parameter dtypes, AdamW hyperparameters, gradients, and payload bytes vary per
trial. Every accepted trial requires exact equality to its decoder-defined
reference trajectory and exact payload persistence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
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

DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def make_params(sizes: list[int], dtype: torch.dtype, seed: int) -> nn.ParameterList:
    gen = torch.Generator().manual_seed(seed)
    return nn.ParameterList([
        nn.Parameter((0.05 * torch.randn(size, generator=gen)).to(dtype))
        for size in sizes
    ])


def clone_params(params: nn.ParameterList) -> nn.ParameterList:
    return nn.ParameterList([nn.Parameter(p.detach().clone()) for p in params])


def exact(a: torch.Tensor, b: torch.Tensor, label: str):
    if not torch.equal(a, b):
        neq = int(torch.count_nonzero(a != b).item())
        raise AssertionError(f"{label}: {neq} unequal elements")


def compare_sign(ref_opt: DFCAdamW, dfc_opt: DFCAdamW):
    for rg, dg in zip(ref_opt.param_groups, dfc_opt.param_groups):
        for rp, dp in zip(rg["params"], dg["params"]):
            exact(rp, dp, "sign parameter")
            rs, ds = ref_opt.state[rp], dfc_opt.state[dp]
            exact(rs["exp_avg"], ds["exp_avg"], "sign first moment")
            exact(rs["exp_avg_sq"], ds["exp_avg_sq"].abs(), "sign decoded second moment")


def compare_low16(ref_opt: DFCLow16AdamW, dfc_opt: DFCLow16AdamW):
    for rg, dg in zip(ref_opt.param_groups, dfc_opt.param_groups):
        for rp, dp in zip(rg["params"], dg["params"]):
            exact(rp, dp, "low16 parameter")
            rs, ds = ref_opt.state[rp], dfc_opt.state[dp]
            exact(DFCLow16AdamW._decode(rs["exp_avg"]), DFCLow16AdamW._decode(ds["exp_avg"]), "low16 decoded first")
            exact(DFCLow16AdamW._decode(rs["exp_avg_sq"]), DFCLow16AdamW._decode(ds["exp_avg_sq"]), "low16 decoded second")


def random_config(rng: np.random.Generator, trial: int) -> dict:
    n_tensors = int(rng.integers(1, 6))
    sizes = [int(rng.integers(8, 4097)) for _ in range(n_tensors)]
    dtype_index = int(rng.integers(0, len(DTYPES)))
    beta1 = float(rng.choice(np.array([0.0, 0.5, 0.9, 0.95, 0.99])))
    beta2 = float(rng.choice(np.array([0.5, 0.9, 0.99, 0.999, 0.9999])))
    lr = float(10 ** rng.uniform(-5.0, -2.3))
    eps = float(rng.choice(np.array([1e-8, 1e-7, 1e-6, 1e-4])))
    weight_decay = float(rng.choice(np.array([0.0, 1e-4, 1e-2, 0.1])))
    steps = int(rng.integers(1, 8))
    grad_scale = float(10 ** rng.uniform(-3.0, -0.3))
    return {
        "trial": trial,
        "sizes": sizes,
        "dtype_index": dtype_index,
        "dtype": str(DTYPES[dtype_index]).replace("torch.", ""),
        "beta1": beta1,
        "beta2": beta2,
        "lr": lr,
        "eps": eps,
        "weight_decay": weight_decay,
        "steps": steps,
        "grad_scale": grad_scale,
    }


def assign_grads(
    a: nn.ParameterList,
    b: nn.ParameterList,
    gen: torch.Generator,
    rng: np.random.Generator,
    scale: float,
):
    active = 0
    for pa, pb in zip(a, b):
        mode = int(rng.integers(0, 10))
        if mode == 0:
            pa.grad = None
            pb.grad = None
            continue
        if mode == 1:
            g = torch.zeros(pa.shape, dtype=torch.float32)
        else:
            g = torch.randn(pa.shape, generator=gen, dtype=torch.float32) * scale
        cast = g.to(pa.dtype)
        pa.grad = cast.clone()
        pb.grad = cast.clone()
        active += pa.numel()
    return active


def sign_trial(config: dict, seed: int) -> int:
    dtype = DTYPES[config["dtype_index"]]
    base = make_params(config["sizes"], dtype, seed)
    ref_params, dfc_params = clone_params(base), clone_params(base)
    kwargs = dict(
        lr=config["lr"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["eps"],
        weight_decay=config["weight_decay"],
    )
    ref = DFCAdamW(ref_params, enable_fiber=False, **kwargs)
    dfc = DFCAdamW(dfc_params, enable_fiber=True, **kwargs)
    channel = TorchSignFiberChannel(dfc)
    payload_rng = np.random.default_rng(seed + 1)
    payload = payload_rng.bytes(channel.byte_capacity)
    channel.write_bytes(0, payload)
    grad_gen = torch.Generator().manual_seed(seed + 2)
    decision_rng = np.random.default_rng(seed + 3)
    updates = 0
    for _ in range(config["steps"]):
        updates += assign_grads(ref_params, dfc_params, grad_gen, decision_rng, config["grad_scale"])
        ref.step()
        dfc.step()
        compare_sign(ref, dfc)
        ref.zero_grad(set_to_none=True)
        dfc.zero_grad(set_to_none=True)
    if channel.read_bytes(0, len(payload)) != payload:
        raise AssertionError("sign payload persistence failed")
    return updates


def low16_trial(config: dict, seed: int) -> int:
    dtype = DTYPES[config["dtype_index"]]
    base = make_params(config["sizes"], dtype, seed)
    ref_params, dfc_params = clone_params(base), clone_params(base)
    kwargs = dict(
        lr=config["lr"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["eps"],
        weight_decay=config["weight_decay"],
    )
    ref = DFCLow16AdamW(ref_params, enable_fiber=False, **kwargs)
    dfc = DFCLow16AdamW(dfc_params, enable_fiber=True, **kwargs)
    channel = TorchLow16FiberChannel(dfc)
    payload_rng = np.random.default_rng(seed + 11)
    payload = payload_rng.bytes(channel.byte_capacity)
    channel.write_bytes(0, payload)
    grad_gen = torch.Generator().manual_seed(seed + 12)
    decision_rng = np.random.default_rng(seed + 13)
    updates = 0
    for _ in range(config["steps"]):
        updates += assign_grads(ref_params, dfc_params, grad_gen, decision_rng, config["grad_scale"])
        ref.step()
        dfc.step()
        compare_low16(ref, dfc)
        ref.zero_grad(set_to_none=True)
        dfc.zero_grad(set_to_none=True)
    if channel.read_bytes(0, len(payload)) != payload:
        raise AssertionError("low16 payload persistence failed")
    return updates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    total_updates = 0
    started = time.time()
    for trial in range(args.trials):
        config = random_config(rng, trial)
        method = "sign" if trial % 2 == 0 else "low16"
        trial_seed = args.seed + 1000 * trial
        t0 = time.perf_counter()
        if method == "sign":
            updates = sign_trial(config, trial_seed)
        else:
            updates = low16_trial(config, trial_seed)
        total_updates += updates
        rows.append({
            "method": method,
            "config": config,
            "active_coordinate_updates": updates,
            "status": "PASS",
            "wall_seconds": time.perf_counter() - t0,
        })

    canonical_rows = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    report = {
        "schema": "dfc-cpu-randomized-fuzz-v1",
        "status": "PASS",
        "seed": args.seed,
        "trials": args.trials,
        "sign_trials": sum(r["method"] == "sign" for r in rows),
        "low16_trials": sum(r["method"] == "low16" for r in rows),
        "active_coordinate_updates": total_updates,
        "configuration_digest_sha256": hashlib.sha256(canonical_rows).hexdigest(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "coverage": {
            "dtypes": sorted(set(r["config"]["dtype"] for r in rows)),
            "beta1": sorted(set(r["config"]["beta1"] for r in rows)),
            "beta2": sorted(set(r["config"]["beta2"] for r in rows)),
            "weight_decay": sorted(set(r["config"]["weight_decay"] for r in rows)),
            "tensor_count_min": min(len(r["config"]["sizes"]) for r in rows),
            "tensor_count_max": max(len(r["config"]["sizes"]) for r in rows),
        },
        "started_unix": started,
        "finished_unix": time.time(),
        "rows": rows,
        "excluded_claims": ["GPU performance", "accelerator counters", "H100-class behavior"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
