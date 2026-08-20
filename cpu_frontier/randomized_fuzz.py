#!/usr/bin/env python3
"""Randomized CPU-only falsification suite for DFC-SIGN and DFC-LOW16.

The purpose of this file is not performance benchmarking.  It repeatedly
randomizes tensor partitions, parameter dtypes, AdamW hyperparameters, missing
gradients, payload rewrites, unaligned byte ranges, and checkpoint boundaries,
then requires the decoded DFC trajectory to remain bit-identical to the
corresponding decoder-defined reference.  It is intentionally suitable for an
ordinary GitHub-hosted CPU runner and emits a sealed JSON record.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import random
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


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def clone_params(params: list[nn.Parameter]) -> list[nn.Parameter]:
    return [nn.Parameter(p.detach().clone()) for p in params]


def equal_sign(ref: DFCAdamW, dfc: DFCAdamW) -> None:
    for gr, gd in zip(ref.param_groups, dfc.param_groups):
        for pr, pd in zip(gr["params"], gd["params"]):
            if not torch.equal(pr, pd):
                raise AssertionError("DFC-SIGN parameter trajectory mismatch")
            sr, sd = ref.state[pr], dfc.state[pd]
            if not torch.equal(sr["exp_avg"], sd["exp_avg"]):
                raise AssertionError("DFC-SIGN first-moment mismatch")
            if not torch.equal(sr["exp_avg_sq"], sd["exp_avg_sq"].abs()):
                raise AssertionError("DFC-SIGN decoded second-moment mismatch")


def equal_low16(ref: DFCLow16AdamW, dfc: DFCLow16AdamW) -> None:
    for gr, gd in zip(ref.param_groups, dfc.param_groups):
        for pr, pd in zip(gr["params"], gd["params"]):
            if not torch.equal(pr, pd):
                raise AssertionError("DFC-LOW16 parameter trajectory mismatch")
            sr, sd = ref.state[pr], dfc.state[pd]
            for key in ("exp_avg", "exp_avg_sq"):
                if not torch.equal(
                    DFCLow16AdamW._decode(sr[key]),
                    DFCLow16AdamW._decode(sd[key]),
                ):
                    raise AssertionError(f"DFC-LOW16 decoded {key} mismatch")


def random_payload_write(channel, rng: np.random.Generator) -> tuple[int, bytes]:
    if channel.byte_capacity == 0:
        return 0, b""
    length = int(rng.integers(1, min(channel.byte_capacity, 97) + 1))
    start = int(rng.integers(0, channel.byte_capacity - length + 1))
    raw = rng.bytes(length)
    channel.write_bytes(start, raw)
    if channel.read_bytes(start, length) != raw:
        raise AssertionError("immediate payload round-trip mismatch")
    return start, raw


def track_latest_write(
    tracked: list[tuple[int, bytes]], start: int, raw: bytes
) -> list[tuple[int, bytes]]:
    """Track only ranges whose asserted bytes have not been overwritten.

    A later write invalidates an older assertion whenever their byte ranges
    overlap, even partially.  This prevents the fuzz oracle itself from
    reporting a false payload-mutation failure after a legitimate later write.
    """
    end = start + len(raw)
    kept = [
        (old_start, old_raw)
        for old_start, old_raw in tracked
        if old_start + len(old_raw) <= start or old_start >= end
    ]
    kept.append((start, raw))
    return kept


def checkpoint_roundtrip(kind: str, ref_params, dfc_params, ref_opt, dfc_opt):
    rb, db = io.BytesIO(), io.BytesIO()
    torch.save({"p": [p.detach() for p in ref_params], "o": ref_opt.state_dict()}, rb)
    torch.save({"p": [p.detach() for p in dfc_params], "o": dfc_opt.state_dict()}, db)
    rb.seek(0)
    db.seek(0)
    rckpt = torch.load(rb, weights_only=True)
    dckpt = torch.load(db, weights_only=True)

    r2 = [nn.Parameter(x.clone()) for x in rckpt["p"]]
    d2 = [nn.Parameter(x.clone()) for x in dckpt["p"]]
    if kind == "sign":
        ro2 = DFCAdamW(r2, enable_fiber=False)
        do2 = DFCAdamW(d2, enable_fiber=True)
    else:
        ro2 = DFCLow16AdamW(r2, enable_fiber=False)
        do2 = DFCLow16AdamW(d2, enable_fiber=True)
    ro2.load_state_dict(rckpt["o"])
    do2.load_state_dict(dckpt["o"])
    return r2, d2, ro2, do2


def run_case(case_id: int, base_seed: int, kind: str) -> dict:
    seed = base_seed + 10007 * case_id + (0 if kind == "sign" else 500_003)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    torch_gen = torch.Generator().manual_seed(seed)

    dtype = py_rng.choice([torch.float32, torch.bfloat16, torch.float16])
    ntensors = py_rng.randint(1, 5)
    sizes = [py_rng.randint(8, 1024) for _ in range(ntensors)]
    ref_params = [nn.Parameter(torch.randn(n, generator=torch_gen).to(dtype)) for n in sizes]
    dfc_params = clone_params(ref_params)

    lr = 10 ** py_rng.uniform(-4.2, -2.8)
    beta1 = py_rng.uniform(0.75, 0.97)
    beta2 = py_rng.uniform(0.95, 0.9999)
    eps = py_rng.choice([1e-8, 1e-7, 1e-6])
    weight_decay = py_rng.choice([0.0, 0.001, 0.01, 0.05])

    cls = DFCAdamW if kind == "sign" else DFCLow16AdamW
    ref_opt = cls(ref_params, lr=lr, betas=(beta1, beta2), eps=eps,
                  weight_decay=weight_decay, enable_fiber=False)
    dfc_opt = cls(dfc_params, lr=lr, betas=(beta1, beta2), eps=eps,
                  weight_decay=weight_decay, enable_fiber=True)
    channel_cls = TorchSignFiberChannel if kind == "sign" else TorchLow16FiberChannel
    channel = channel_cls(dfc_opt)
    checker = equal_sign if kind == "sign" else equal_low16

    payload_digest_stream = hashlib.sha256()
    tracked: list[tuple[int, bytes]] = []
    # Initial adversarial/random payload.  Unaligned offsets are selected naturally.
    for _ in range(py_rng.randint(1, 4)):
        start, raw = random_payload_write(channel, rng)
        tracked = track_latest_write(tracked, start, raw)
        payload_digest_stream.update(start.to_bytes(8, "little"))
        payload_digest_stream.update(raw)

    steps = py_rng.randint(8, 28)
    checkpoint_at = py_rng.randint(2, steps - 2)
    checkpointed = False
    updates = 0

    for step in range(steps):
        # Payload may change at any transition boundary.
        if step % py_rng.randint(2, 5) == 0:
            start, raw = random_payload_write(channel, rng)
            tracked = track_latest_write(tracked, start, raw)
            payload_digest_stream.update(start.to_bytes(8, "little"))
            payload_digest_stream.update(raw)

        active = 0
        for i, (pr, pd) in enumerate(zip(ref_params, dfc_params)):
            # Deliberately omit some gradients, but identically in both paths.
            if ((step + i + case_id) % 7) == 0:
                pr.grad = None
                pd.grad = None
                continue
            grad = torch.randn(pr.shape, generator=torch_gen, dtype=torch.float32)
            grad.mul_(10 ** py_rng.uniform(-2.0, 0.2))
            cast = grad.to(dtype)
            pr.grad = cast.clone()
            pd.grad = cast.clone()
            active += pr.numel()

        try:
            ref_opt.step()
            dfc_opt.step()
            ref_opt.zero_grad(set_to_none=True)
            dfc_opt.zero_grad(set_to_none=True)
            checker(ref_opt, dfc_opt)
        except Exception as exc:
            raise AssertionError(
                f"case={case_id} kind={kind} dtype={dtype} step={step}: semantic trajectory failure: {exc}"
            ) from exc
        updates += active

        for start, raw in tracked:
            if channel.read_bytes(start, len(raw)) != raw:
                raise AssertionError(
                    f"case={case_id} kind={kind} dtype={dtype} step={step}: payload mutated across optimizer transition at byte range [{start},{start+len(raw)})"
                )

        if step == checkpoint_at:
            # Save/restore physical optimizer state, then continue from restored objects.
            snapshots = [(s, bytes(b)) for s, b in tracked]
            ref_params, dfc_params, ref_opt, dfc_opt = checkpoint_roundtrip(
                kind, ref_params, dfc_params, ref_opt, dfc_opt
            )
            channel = channel_cls(dfc_opt)
            checker(ref_opt, dfc_opt)
            for start, raw in snapshots:
                if channel.read_bytes(start, len(raw)) != raw:
                    raise AssertionError(
                        f"case={case_id} kind={kind} dtype={dtype} checkpoint_step={step}: payload lost across randomized checkpoint at byte range [{start},{start+len(raw)})"
                    )
            checkpointed = True

    return {
        "case": case_id,
        "kind": kind,
        "seed": seed,
        "dtype": str(dtype).replace("torch.", ""),
        "tensor_sizes": sizes,
        "steps": steps,
        "checkpoint_step": checkpoint_at,
        "checkpointed": checkpointed,
        "coordinate_updates": updates,
        "fiber_bytes": channel.byte_capacity,
        "payload_write_digest": payload_digest_stream.hexdigest(),
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=910_241)
    args = parser.parse_args()
    if args.cases < 2:
        raise SystemExit("--cases must be >= 2")
    if torch.cuda.is_available():
        raise SystemExit("randomized CPU fuzz unexpectedly sees CUDA")

    rows = []
    for i in range(args.cases):
        rows.append(run_case(i, args.seed, "sign" if i % 2 == 0 else "low16"))

    total_updates = sum(r["coordinate_updates"] for r in rows)
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    report = {
        "schema": "dfc-github-cpu-randomized-fuzz-v1",
        "status": "PASS",
        "cases": args.cases,
        "base_seed": args.seed,
        "sign_cases": sum(r["kind"] == "sign" for r in rows),
        "low16_cases": sum(r["kind"] == "low16" for r in rows),
        "total_coordinate_updates": total_updates,
        "case_manifest_sha256": sha256(canonical),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "randomized_dimensions": [
            "tensor partition/count",
            "FP32/BF16/FP16 parameter dtype",
            "AdamW lr/betas/eps/weight decay",
            "gradient magnitude and missing gradients",
            "payload content/length/unaligned offset/rewrite schedule",
            "checkpoint position",
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
