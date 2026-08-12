"""Synthetic systems benchmark for fused external-EF vs fused DFC-EF.

Each row uses identical FP16 parameter/gradient tensors and BF16-high AdamW
semantics. The external kernel additionally reads/writes one FP32 residual. The
DFC kernel stores exactly the same logical residual in the two low-word fibers
of the moment containers. The benchmark first proves bitwise equality of
parameter state, semantic moment state, and logical residual state, then times
repeated fused updates with CUDA events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from fused_stride_ef_adamw import FusedStrideEFAdamW, load_backend


def logical_dfc(m, v):
    mb = m.view(torch.int32); vb = v.view(torch.int32)
    return ((mb & 65535) | ((vb & 65535) << 16)).view(torch.float32)


def semantic_bits(x):
    return x.view(torch.int32) & -65536


def time_steps(opt, p, g, iterations, repeats, seed_phase=0):
    # Warmup plus phase cycling to avoid a degenerate fixed residual pattern.
    for j in range(8):
        p.grad = g
        opt.step(phase=(seed_phase + j) % opt.stride)
    torch.cuda.synchronize()
    samples = []
    for rep in range(repeats):
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        for j in range(iterations):
            p.grad = g
            opt.step(phase=(seed_phase + j) % opt.stride)
        end.record(); torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)) / iterations)
    return samples


def bench(n, seed, repeats, stride):
    device = "cuda"; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    p0 = (torch.randn(n, device=device, dtype=torch.float16) * 0.02).contiguous()
    g = (torch.randn(n, device=device, dtype=torch.float16) * 0.01).contiguous()

    # Exactness gate after several changing phases.
    pe = torch.nn.Parameter(p0.clone()); pd = torch.nn.Parameter(p0.clone())
    oe = FusedStrideEFAdamW([pe], method="external_fused", stride=stride)
    od = FusedStrideEFAdamW([pd], method="dfc_fused", stride=stride)
    for j in range(min(2 * stride, 16)):
        pe.grad = g; pd.grad = g
        oe.step(phase=j % stride); od.step(phase=j % stride)
    torch.cuda.synchronize()
    param_equal = torch.equal(pe, pd)
    m_equal = torch.equal(semantic_bits(oe.m[0]), semantic_bits(od.m[0]))
    v_equal = torch.equal(semantic_bits(oe.v[0]), semantic_bits(od.v[0]))
    residual_equal = torch.equal(oe.residual[0].view(torch.int32), logical_dfc(od.m[0], od.v[0]).view(torch.int32))
    exact = bool(param_equal and m_equal and v_equal and residual_equal)
    if not exact:
        raise RuntimeError(f"fused exactness gate failed n={n}: p={param_equal} m={m_equal} v={v_equal} r={residual_equal}")
    del pe, pd, oe, od
    torch.cuda.empty_cache(); torch.cuda.synchronize()

    # Persistent allocation measured independently for each method.
    torch.cuda.reset_peak_memory_stats(); base = int(torch.cuda.memory_allocated())
    pe = torch.nn.Parameter(p0.clone()); pe.grad = g
    oe = FusedStrideEFAdamW([pe], method="external_fused", stride=stride)
    torch.cuda.synchronize(); ext_alloc = int(torch.cuda.memory_allocated()) - base
    iterations = max(8, min(128, 64_000_000 // max(1, n)))
    ext_samples = time_steps(oe, pe, g, iterations, repeats)
    del pe, oe
    torch.cuda.empty_cache(); torch.cuda.synchronize()

    base2 = int(torch.cuda.memory_allocated())
    pd = torch.nn.Parameter(p0.clone()); pd.grad = g
    od = FusedStrideEFAdamW([pd], method="dfc_fused", stride=stride)
    torch.cuda.synchronize(); dfc_alloc = int(torch.cuda.memory_allocated()) - base2
    dfc_samples = time_steps(od, pd, g, iterations, repeats)
    del pd, od
    torch.cuda.empty_cache(); torch.cuda.synchronize()

    ext_ms = statistics.median(ext_samples); dfc_ms = statistics.median(dfc_samples)
    overhead = 100.0 * (dfc_ms / ext_ms - 1.0)
    speedup = 100.0 * (ext_ms / dfc_ms - 1.0)
    # Explicit global-memory transactions in the fused kernels, excluding
    # cache effects and metadata: p2+g2+m4+v4(+r4) reads and p2+m4+v4(+r4) writes.
    ext_bytes_per_coord = 30; dfc_bytes_per_coord = 22
    return {
        "elements": n, "stride": stride, "iterations_per_sample": iterations, "repeats": repeats,
        "bitwise_exact": exact, "parameter_equal": param_equal, "semantic_m_equal": m_equal,
        "semantic_v_equal": v_equal, "logical_residual_equal": residual_equal,
        "external_ms": ext_ms, "dfc_ms": dfc_ms, "runtime_overhead_percent": overhead,
        "runtime_speedup_percent": speedup, "external_samples_ms": ext_samples, "dfc_samples_ms": dfc_samples,
        "external_persistent_allocated_bytes": ext_alloc, "dfc_persistent_allocated_bytes": dfc_alloc,
        "persistent_allocated_saved_bytes": ext_alloc - dfc_alloc,
        "explicit_external_bytes_per_coordinate": ext_bytes_per_coord,
        "explicit_dfc_bytes_per_coordinate": dfc_bytes_per_coord,
        "explicit_bytes_saved_per_coordinate": ext_bytes_per_coord - dfc_bytes_per_coord,
        "explicit_traffic_reduction_percent": 100.0 * (ext_bytes_per_coord - dfc_bytes_per_coord) / ext_bytes_per_coord,
        "external_effective_gbps": ext_bytes_per_coord * n / (ext_ms * 1e6),
        "dfc_effective_gbps": dfc_bytes_per_coord * n / (dfc_ms * 1e6),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", default="1048576,4194304,16777216,33554432")
    ap.add_argument("--repeats", type=int, default=9); ap.add_argument("--seed", type=int, default=2701)
    ap.add_argument("--stride", type=int, default=8); ap.add_argument("--output", required=True)
    args = ap.parse_args(); load_backend(verbose=True)
    rows = []; started = time.time()
    for i, s in enumerate(args.sizes.split(",")):
        rows.append(bench(int(s), args.seed + i, args.repeats, args.stride))
    result = {
        "schema_version": 1, "protocol": "dfc-fused-stride-systems-v1",
        "gpu_name": torch.cuda.get_device_name(0), "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "rows": rows, "all_exact": all(r["bitwise_exact"] for r in rows),
        "median_runtime_overhead_percent": float(statistics.median(r["runtime_overhead_percent"] for r in rows)),
        "median_runtime_speedup_percent": float(statistics.median(r["runtime_speedup_percent"] for r in rows)),
        "max_runtime_overhead_percent": float(max(r["runtime_overhead_percent"] for r in rows)),
        "min_runtime_speedup_percent": float(min(r["runtime_speedup_percent"] for r in rows)),
        "started_unix": started,
    }
    result["promotion_gate"] = bool(result["all_exact"] and result["median_runtime_overhead_percent"] <= 2.0)
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["result_sha256"] = hashlib.sha256(canonical).hexdigest()
    p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("gpu_name", "all_exact", "median_runtime_overhead_percent",
          "median_runtime_speedup_percent", "max_runtime_overhead_percent", "promotion_gate", "result_sha256")}, indent=2))


if __name__ == "__main__": main()
