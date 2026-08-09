"""GPU benchmark for incremental DFC-SIGN overhead over matched fused AdamW.

The reference and DFC kernels are identical in arithmetic order and memory
traffic; only the DFC kernel clears/restores the FP32 second-moment sign bit.
Each reported point first passes bitwise state equality and payload persistence,
then measures CUDA-event time over repeated in-place steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from triton_dfc_adamw import dfc_adamw_step, reference_adamw_step, triton


def _event_time_ms(fn, iterations: int, repeats: int) -> list[float]:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)) / iterations)
    return samples


def benchmark_size(n: int, seed: int, repeats: int) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    gradient = torch.randn(n, device=device, dtype=torch.float32) * 0.01
    p0 = torch.randn(n, device=device, dtype=torch.float32)
    m0 = torch.randn(n, device=device, dtype=torch.float32) * 0.01
    v0 = torch.rand(n, device=device, dtype=torch.float32) * 0.1 + 1e-5
    payload = torch.randint(0, 2, (n,), device=device, dtype=torch.int32) << 31

    p_ref, m_ref, v_ref = p0.clone(), m0.clone(), v0.clone()
    p_dfc, m_dfc = p0.clone(), m0.clone()
    v_dfc = ((v0.view(torch.int32) & 0x7FFFFFFF) | payload).view(torch.float32).clone()
    kwargs = dict(step=101, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    reference_adamw_step(p_ref, gradient, m_ref, v_ref, **kwargs)
    dfc_adamw_step(p_dfc, gradient, m_dfc, v_dfc, **kwargs)
    torch.cuda.synchronize()
    exact = bool(
        torch.equal(p_ref, p_dfc)
        and torch.equal(m_ref, m_dfc)
        and torch.equal(v_ref.view(torch.int32), v_dfc.view(torch.int32) & 0x7FFFFFFF)
        and torch.equal(v_dfc.view(torch.int32) & -2147483648, payload)
    )
    if not exact:
        raise RuntimeError(f"bitwise equality gate failed for n={n}")

    p_ref, m_ref, v_ref = p0.clone(), m0.clone(), v0.clone()
    p_dfc, m_dfc = p0.clone(), m0.clone()
    v_dfc = ((v0.view(torch.int32) & 0x7FFFFFFF) | payload).view(torch.float32).clone()
    iterations = max(8, min(64, 32_000_000 // max(1, n)))
    ref_samples = _event_time_ms(
        lambda: reference_adamw_step(p_ref, gradient, m_ref, v_ref, **kwargs), iterations, repeats
    )
    dfc_samples = _event_time_ms(
        lambda: dfc_adamw_step(p_dfc, gradient, m_dfc, v_dfc, **kwargs), iterations, repeats
    )
    ref_ms = statistics.median(ref_samples)
    dfc_ms = statistics.median(dfc_samples)
    overhead = 100.0 * (dfc_ms / ref_ms - 1.0)
    bytes_per_coordinate = 28
    return {
        "elements": n,
        "iterations_per_sample": iterations,
        "repeats": repeats,
        "reference_ms": ref_ms,
        "dfc_ms": dfc_ms,
        "overhead_percent": overhead,
        "reference_samples_ms": ref_samples,
        "dfc_samples_ms": dfc_samples,
        "reference_effective_gbps": bytes_per_coordinate * n / (ref_ms * 1e6),
        "dfc_effective_gbps": bytes_per_coordinate * n / (dfc_ms * 1e6),
        "bitwise_exact": exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1048576,4194304,16777216,33554432")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or triton is None:
        raise RuntimeError("CUDA and Triton are required for this benchmark")
    properties = torch.cuda.get_device_properties(0)
    rows = []
    started = time.time()
    for index, text in enumerate(args.sizes.split(",")):
        rows.append(benchmark_size(int(text), args.seed + index, args.repeats))
    result = {
        "schema_version": 1,
        "torch": torch.__version__,
        "triton": getattr(triton, "__version__", "unknown"),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": int(properties.total_memory),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "rows": rows,
        "median_overhead_percent": float(statistics.median(row["overhead_percent"] for row in rows)),
        "max_overhead_percent": float(max(row["overhead_percent"] for row in rows)),
        "started_unix": started,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["result_sha256"] = hashlib.sha256(canonical).hexdigest()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "gpu": result["gpu_name"],
        "median_overhead_percent": result["median_overhead_percent"],
        "max_overhead_percent": result["max_overhead_percent"],
        "result_sha256": result["result_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
