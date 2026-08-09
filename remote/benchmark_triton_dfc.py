#!/usr/bin/env python3
"""Audited CUDA benchmark for fused full-FP32 DFC-Sign AdamW.

The primary comparison uses two shape-identical Triton kernels with the same
floating-point operation order.  Their only difference is sign decode and
re-embedding.  PyTorch fused AdamW is reported as a secondary library anchor.
Every timed size first passes bitwise matched-kernel trajectory equality and
arbitrary-payload preservation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from triton_dfc_adamw import dfc_adamw_step, reference_adamw_step, triton


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def timed_ms(fn, warmup: int, repetitions: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def reset_state(size: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    parameter = torch.randn(size, device="cuda", dtype=torch.float32, generator=generator)
    gradient = torch.randn(size, device="cuda", dtype=torch.float32, generator=generator)
    first = torch.randn(size, device="cuda", dtype=torch.float32, generator=generator) * 0.01
    second = torch.rand(size, device="cuda", dtype=torch.float32, generator=generator) * 0.01
    return parameter, gradient, first, second


def exactness_gate(size: int, seed: int) -> dict[str, object]:
    p_ref, gradient, m_ref, v_ref = reset_state(size, seed)
    p_dfc, m_dfc = p_ref.clone(), m_ref.clone()
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    payload = torch.randint(0, 2, (size,), device="cuda", dtype=torch.int32,
                            generator=generator) << 31
    v_dfc = ((v_ref.view(torch.int32) & 0x7FFFFFFF) | payload).view(torch.float32).clone()
    for step in range(1, 9):
        reference_adamw_step(p_ref, gradient, m_ref, v_ref, step, lr=3e-4,
                             weight_decay=0.01)
        dfc_adamw_step(p_dfc, gradient, m_dfc, v_dfc, step, lr=3e-4,
                       weight_decay=0.01)
    torch.cuda.synchronize()
    magnitude = v_dfc.view(torch.int32) & 0x7FFFFFFF
    checks = {
        "parameter_bitwise_equal": bool(torch.equal(p_ref, p_dfc)),
        "first_moment_bitwise_equal": bool(torch.equal(m_ref, m_dfc)),
        "second_magnitude_bitwise_equal": bool(torch.equal(v_ref.view(torch.int32), magnitude)),
        "payload_bitwise_preserved": bool(torch.equal(v_dfc.view(torch.int32) & -2147483648,
                                                       payload)),
    }
    if not all(checks.values()):
        raise AssertionError(f"fused exactness gate failed: {checks}")
    digest = hashlib.sha256()
    for tensor in (p_dfc, m_dfc, magnitude, payload):
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return {**checks, "trajectory_sha256": digest.hexdigest(), "updates": 8}


def benchmark_size(size: int, warmup: int, repetitions: int, rounds: int,
                   seed: int) -> dict[str, object]:
    gate = exactness_gate(size, seed)
    p0, gradient, m0, v0 = reset_state(size, seed + 9)
    payload_generator = torch.Generator(device="cuda").manual_seed(seed + 10)
    payload = torch.randint(0, 2, (size,), device="cuda", dtype=torch.int32,
                            generator=payload_generator) << 31

    reference_rounds: list[float] = []
    dfc_rounds: list[float] = []
    pytorch_rounds: list[float] = []
    ratios: list[float] = []
    orders = []
    for round_index in range(rounds):
        order = ["reference", "dfc"] if round_index % 2 == 0 else ["dfc", "reference"]
        orders.append(order)
        measured: dict[str, list[float]] = {}
        for method in order:
            p, m, v = p0.clone(), m0.clone(), v0.clone()
            if method == "dfc":
                v = ((v.view(torch.int32) & 0x7FFFFFFF) | payload).view(torch.float32).clone()
                measured[method] = timed_ms(
                    lambda: dfc_adamw_step(p, gradient, m, v, 17, lr=3e-4,
                                           weight_decay=0.01), warmup, repetitions)
            else:
                measured[method] = timed_ms(
                    lambda: reference_adamw_step(p, gradient, m, v, 17, lr=3e-4,
                                                 weight_decay=0.01), warmup, repetitions)
        ref_median = statistics.median(measured["reference"])
        dfc_median = statistics.median(measured["dfc"])
        reference_rounds.append(ref_median)
        dfc_rounds.append(dfc_median)
        ratios.append(dfc_median / ref_median)

        parameter = torch.nn.Parameter(p0.clone())
        parameter.grad = gradient
        optimizer = torch.optim.AdamW([parameter], lr=3e-4, weight_decay=0.01, fused=True)
        optimizer.step()  # allocate and compile state before timing
        pytorch_samples = timed_ms(optimizer.step, warmup, repetitions)
        pytorch_rounds.append(statistics.median(pytorch_samples))

    overhead = (statistics.median(ratios) - 1.0) * 100.0
    return {
        "elements": size,
        "state_bytes_touched_per_update": size * 4 * 4,
        "warmup_per_round": warmup,
        "repetitions_per_round": repetitions,
        "rounds": rounds,
        "interleaving_orders": orders,
        "matched_reference_ms": reference_rounds,
        "dfc_sign_ms": dfc_rounds,
        "pytorch_fused_ms": pytorch_rounds,
        "paired_ratio": ratios,
        "median_matched_reference_ms": statistics.median(reference_rounds),
        "median_dfc_sign_ms": statistics.median(dfc_rounds),
        "median_pytorch_fused_ms": statistics.median(pytorch_rounds),
        "median_overhead_percent": overhead,
        "ratio_iqr": [percentile(ratios, 25), percentile(ratios, 75)],
        "exactness": gate,
    }


def device_metadata() -> dict[str, object]:
    props = torch.cuda.get_device_properties(0)
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,power.limit",
             "--format=csv,noheader"], text=True, timeout=20,
        ).strip()
    except Exception as exc:  # pragma: no cover - depends on runner image
        smi = f"unavailable: {exc}"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": getattr(triton, "__version__", "unknown"),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": props.name,
        "gpu_total_memory": props.total_memory,
        "compute_capability": [props.major, props.minor],
        "nvidia_smi": smi,
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_sha": os.environ.get("GITHUB_SHA"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1048576,8388608,33554432")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--maximum-primary-overhead-percent", type=float, default=5.0)
    parser.add_argument("--primary-size", type=int, default=8388608)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or triton is None:
        raise RuntimeError("an actual CUDA GPU and Triton are mandatory; emulation is rejected")
    sizes = [int(value) for value in args.sizes.split(",")]
    if args.primary_size not in sizes:
        raise ValueError("primary size must be present in --sizes")
    torch.manual_seed(args.seed)
    results = [benchmark_size(size, args.warmup, args.repetitions, args.rounds,
                              args.seed + index * 101) for index, size in enumerate(sizes)]
    primary = next(row for row in results if row["elements"] == args.primary_size)
    gate = {
        "primary_size": args.primary_size,
        "maximum_overhead_percent": args.maximum_primary_overhead_percent,
        "observed_overhead_percent": primary["median_overhead_percent"],
        "passed": primary["median_overhead_percent"] <= args.maximum_primary_overhead_percent,
    }
    payload = {
        "schema": "dfc-triton-gpu-v1",
        "device": device_metadata(),
        "results": results,
        "acceptance_gate": gate,
        "source_sha256": {
            "benchmark": sha256_file(Path(__file__)),
            "kernel": sha256_file(Path(__file__).with_name("triton_dfc_adamw.py")),
        },
        "timestamp_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit("predeclared <=5% primary overhead gate failed")


if __name__ == "__main__":
    main()
