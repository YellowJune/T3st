"""GPU state-memory frontier benchmark for DFC-EF.

This is a state-only systems benchmark. It intentionally excludes model weights,
gradients, and activations because those allocations are common to both methods.
The external baseline owns two FP32 Adam moment tensors plus one FP32 EF
residual (12 bytes/coordinate). DFC-EF owns only the two pre-existing FP32
moment tensors (8 bytes/coordinate); its 32-bit residual payload is stored in
their low-word fibers.

Run without --single to binary-search both frontiers in fresh subprocesses so a
failed allocation cannot poison the CUDA allocator state of later probes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _single(method: str, coordinates: int) -> int:
    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"ok": False, "error": "CUDA unavailable"}))
        return 3
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free0, total = torch.cuda.mem_get_info(device)
    n = int(coordinates)
    tensors = []
    try:
        tensors.append(torch.empty(n, dtype=torch.float32, device=device))  # exp_avg
        tensors.append(torch.empty(n, dtype=torch.float32, device=device))  # exp_avg_sq
        if method == "external":
            tensors.append(torch.empty(n, dtype=torch.float32, device=device))  # EF residual
        elif method != "dfc":
            raise ValueError(method)
        # One tiny write per allocation forces normal stream visibility without
        # sweeping the whole tensor and turning an allocation test into a BW test.
        for t in tensors:
            if t.numel():
                t[0] = 0.0
                t[-1] = 0.0
        torch.cuda.synchronize(device)
        free1, _ = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        result = {
            "ok": True,
            "method": method,
            "coordinates": n,
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "total_hbm_bytes": int(total),
            "free_before_bytes": int(free0),
            "free_after_bytes": int(free1),
            "allocated_bytes": int(allocated),
            "reserved_bytes": int(reserved),
            "theoretical_state_bytes": int(n * (12 if method == "external" else 8)),
            "external_residual_bytes": int(4 * n if method == "external" else 0),
            "dfc_fiber_capacity_bytes": int(4 * n if method == "dfc" else 0),
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and "out of memory" not in str(exc).lower():
            raise
        print(json.dumps({
            "ok": False,
            "method": method,
            "coordinates": n,
            "error": "cuda_oom",
            "message": str(exc).splitlines()[0][:500],
            "device": torch.cuda.get_device_name(device),
            "total_hbm_bytes": int(total),
            "free_before_bytes": int(free0),
        }, sort_keys=True))
        return 42


def _probe(method: str, n: int) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--single", "--method", method, "--coordinates", str(int(n))]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = [x for x in proc.stdout.splitlines() if x.strip().startswith("{")]
    if lines:
        row = json.loads(lines[-1])
    else:
        row = {"ok": False, "method": method, "coordinates": int(n), "error": "probe_failed", "returncode": proc.returncode, "stderr": proc.stderr[-2000:]}
    row["returncode"] = proc.returncode
    return row


def _search(method: str, total_bytes: int, resolution: int, safety: float) -> tuple[int, list[dict]]:
    bpp = 12 if method == "external" else 8
    # Search a little beyond the nominal free-memory estimate so the failing
    # side of the frontier is actually observed.
    hi = max(resolution, int(total_bytes * safety / bpp))
    hi = ((hi + resolution - 1) // resolution) * resolution
    lo = 0
    rows: list[dict] = []
    # First ensure hi fails; if not, expand until it does or exceeds 1.25x HBM.
    cap = int(total_bytes * 1.25 / bpp)
    while hi <= cap:
        row = _probe(method, hi); rows.append(row)
        if not row.get("ok"):
            break
        lo = hi
        hi += max(resolution, hi // 8)
        hi = ((hi + resolution - 1) // resolution) * resolution
    if rows and rows[-1].get("ok"):
        # Conservative fallback: hi is one resolution above last success.
        hi = lo + resolution
    while hi - lo > resolution:
        mid = ((lo + hi) // (2 * resolution)) * resolution
        mid = max(lo + resolution, mid)
        row = _probe(method, mid); rows.append(row)
        if row.get("ok"):
            lo = mid
        else:
            hi = mid
    return lo, rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--single", action="store_true")
    p.add_argument("--method", choices=["external", "dfc"])
    p.add_argument("--coordinates", type=int)
    p.add_argument("--resolution", type=int, default=8_388_608, help="binary-search resolution in coordinates")
    p.add_argument("--safety", type=float, default=1.05, help="initial high bound as fraction of total-HBM theoretical capacity")
    p.add_argument("--output", default="results/kaggle_free/gpu_memory_frontier.json")
    a = p.parse_args()
    if a.single:
        if a.method is None or a.coordinates is None:
            p.error("--single requires --method and --coordinates")
        raise SystemExit(_single(a.method, a.coordinates))

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    _, total = torch.cuda.mem_get_info(0)
    ext_max, ext_rows = _search("external", int(total), a.resolution, a.safety)
    dfc_max, dfc_rows = _search("dfc", int(total), a.resolution, a.safety)

    crossover = None
    if dfc_max >= ext_max + a.resolution:
        candidate = min(dfc_max, ext_max + a.resolution)
        ext_check = _probe("external", candidate)
        dfc_check = _probe("dfc", candidate)
        crossover = {"coordinates": candidate, "external": ext_check, "dfc": dfc_check,
                     "verified": (not ext_check.get("ok", False)) and bool(dfc_check.get("ok", False))}

    result = {
        "schema_version": 1,
        "protocol": "dfc-ef-state-memory-frontier-v1",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "total_hbm_bytes": int(total),
        "resolution_coordinates": int(a.resolution),
        "external_bytes_per_coordinate": 12,
        "dfc_bytes_per_coordinate": 8,
        "external_max_success_coordinates": int(ext_max),
        "dfc_max_success_coordinates": int(dfc_max),
        "frontier_ratio_dfc_over_external": (float(dfc_max) / ext_max) if ext_max else None,
        "external_residual_bytes_removed_at_dfc_frontier": int(4 * dfc_max),
        "crossover": crossover,
        "external_probes": ext_rows,
        "dfc_probes": dfc_rows,
    }
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ["protocol", "device", "total_hbm_bytes", "external_max_success_coordinates", "dfc_max_success_coordinates", "frontier_ratio_dfc_over_external", "external_residual_bytes_removed_at_dfc_frontier", "crossover"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
