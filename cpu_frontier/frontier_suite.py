#!/usr/bin/env python3
"""CPU-only frontier validation for Decoder-Fiber Computing.

This suite intentionally requires no CUDA device. It validates properties that
can be established on GitHub-hosted CPU runners and emits an auditable JSON
record. Hardware latency, Nsight traffic counters, and accelerator-specific
claims are out of scope by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
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


class ParamModule(nn.Module):
    def __init__(self, shapes, dtype=torch.float32, noncontiguous=False):
        super().__init__()
        params = []
        for i, shape in enumerate(shapes):
            g = torch.Generator().manual_seed(100 + i)
            value = torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype)
            if noncontiguous and value.ndim == 2:
                value = value.t()
            params.append(nn.Parameter(value))
        self.params = nn.ParameterList(params)


def clone_module(module: nn.Module) -> nn.Module:
    shapes = [tuple(p.shape) for p in module.parameters()]
    dtype = next(module.parameters()).dtype
    out = ParamModule(shapes, dtype=dtype)
    out.load_state_dict(module.state_dict())
    return out


def assign_identical_grads(a: nn.Module, b: nn.Module, generator: torch.Generator, scale=1.0):
    for pa, pb in zip(a.parameters(), b.parameters()):
        g = torch.randn(pa.shape, generator=generator, dtype=torch.float32) * scale
        ga = g.to(pa.dtype)
        pa.grad = ga.clone()
        pb.grad = ga.clone()


def assert_equal_tensor(a: torch.Tensor, b: torch.Tensor, label: str):
    if not torch.equal(a, b):
        mismatch = int(torch.count_nonzero(a != b).item())
        raise AssertionError(f"{label}: {mismatch} unequal elements")


def assert_sign_optimizer_equal(ref: DFCAdamW, dfc: DFCAdamW):
    for group_ref, group_dfc in zip(ref.param_groups, dfc.param_groups):
        for p_ref, p_dfc in zip(group_ref["params"], group_dfc["params"]):
            assert_equal_tensor(p_ref, p_dfc, "parameter")
            s_ref, s_dfc = ref.state[p_ref], dfc.state[p_dfc]
            assert_equal_tensor(s_ref["exp_avg"], s_dfc["exp_avg"], "first moment")
            assert_equal_tensor(
                s_ref["exp_avg_sq"],
                s_dfc["exp_avg_sq"].abs(),
                "decoded second moment",
            )


def assert_low16_optimizer_equal(ref: DFCLow16AdamW, dfc: DFCLow16AdamW):
    for group_ref, group_dfc in zip(ref.param_groups, dfc.param_groups):
        for p_ref, p_dfc in zip(group_ref["params"], group_dfc["params"]):
            assert_equal_tensor(p_ref, p_dfc, "parameter")
            s_ref, s_dfc = ref.state[p_ref], dfc.state[p_dfc]
            assert_equal_tensor(
                DFCLow16AdamW._decode(s_ref["exp_avg"]),
                DFCLow16AdamW._decode(s_dfc["exp_avg"]),
                "decoded first moment",
            )
            assert_equal_tensor(
                DFCLow16AdamW._decode(s_ref["exp_avg_sq"]),
                DFCLow16AdamW._decode(s_dfc["exp_avg_sq"]),
                "decoded second moment",
            )


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def experiment_sign_payload_patterns() -> dict:
    base = ParamModule([(4096,), (1025,), (17,)])
    ref, dfc_model = clone_module(base), clone_module(base)
    ref_opt = DFCAdamW(ref.parameters(), lr=7e-4, weight_decay=0.013, enable_fiber=False)
    dfc_opt = DFCAdamW(dfc_model.parameters(), lr=7e-4, weight_decay=0.013, enable_fiber=True)
    channel = TorchSignFiberChannel(dfc_opt)
    rng = np.random.default_rng(7001)
    patterns = [
        bytes(channel.byte_capacity),
        bytes([0xFF]) * channel.byte_capacity,
        bytes([0xAA]) * channel.byte_capacity,
        bytes([0x55]) * channel.byte_capacity,
        rng.bytes(channel.byte_capacity),
    ]
    gen = torch.Generator().manual_seed(7002)
    steps_per_pattern = 8
    for payload in patterns:
        channel.write_bytes(0, payload)
        for _ in range(steps_per_pattern):
            assign_identical_grads(ref, dfc_model, gen)
            ref_opt.step()
            dfc_opt.step()
            ref_opt.zero_grad(set_to_none=True)
            dfc_opt.zero_grad(set_to_none=True)
            assert_sign_optimizer_equal(ref_opt, dfc_opt)
        if channel.read_bytes(0, len(payload)) != payload:
            raise AssertionError("sign-fiber payload changed")
    coords = sum(p.numel() for p in ref.parameters())
    return {
        "status": "PASS",
        "patterns": len(patterns),
        "steps": len(patterns) * steps_per_pattern,
        "coordinates": coords,
        "coordinate_updates": coords * len(patterns) * steps_per_pattern,
        "payload_bytes": channel.byte_capacity,
    }


def experiment_sign_dtype_matrix() -> dict:
    rows = {}
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        name = str(dtype).replace("torch.", "")
        base = ParamModule([(513,), (129,)], dtype=dtype)
        ref, dfc_model = clone_module(base), clone_module(base)
        ref_opt = DFCAdamW(ref.parameters(), lr=3e-4, enable_fiber=False)
        dfc_opt = DFCAdamW(dfc_model.parameters(), lr=3e-4, enable_fiber=True)
        channel = TorchSignFiberChannel(dfc_opt)
        payload = np.random.default_rng(7100).bytes(channel.byte_capacity)
        channel.write_bytes(0, payload)
        gen = torch.Generator().manual_seed(7101)
        for _ in range(6):
            assign_identical_grads(ref, dfc_model, gen, scale=0.1)
            ref_opt.step()
            dfc_opt.step()
            ref_opt.zero_grad(set_to_none=True)
            dfc_opt.zero_grad(set_to_none=True)
            assert_sign_optimizer_equal(ref_opt, dfc_opt)
        if channel.read_bytes(0, len(payload)) != payload:
            raise AssertionError(f"dtype {name} payload changed")
        rows[name] = {"status": "PASS", "payload_bytes": channel.byte_capacity}
    return {"status": "PASS", "dtypes": rows}


def experiment_cross_tensor_boundaries() -> dict:
    model = ParamModule([(7,), (9,), (15,), (33,)])
    opt = DFCAdamW(model.parameters(), enable_fiber=True)
    channel = TorchSignFiberChannel(opt)
    bits = np.array([(i * 5 + 1) & 1 for i in range(channel.bit_capacity - 3)], dtype=np.uint8)
    channel.write_bits(3, bits)
    if not np.array_equal(channel.read_bits(3, bits.size), bits):
        raise AssertionError("cross-tensor sign-bit range mismatch")

    low_model = ParamModule([(3,), (5,), (11,)])
    low_opt = DFCLow16AdamW(low_model.parameters(), enable_fiber=True)
    low_channel = TorchLow16FiberChannel(low_opt)
    payload = bytes((i * 29 + 7) & 0xFF for i in range(low_channel.byte_capacity - 1))
    low_channel.write_bytes(1, payload)
    if low_channel.read_bytes(1, len(payload)) != payload:
        raise AssertionError("cross-tensor low16 byte range mismatch")
    return {
        "status": "PASS",
        "sign_bits_checked": int(bits.size),
        "low16_bytes_checked": len(payload),
    }


def _step_manual_pair(ref, dfc_model, ref_opt, dfc_opt, generator, steps, checker):
    for _ in range(steps):
        assign_identical_grads(ref, dfc_model, generator, scale=0.2)
        ref_opt.step()
        dfc_opt.step()
        ref_opt.zero_grad(set_to_none=True)
        dfc_opt.zero_grad(set_to_none=True)
        checker(ref_opt, dfc_opt)


def experiment_checkpoint_resume_sign() -> dict:
    base = ParamModule([(2048,), (257,)])
    ref, dfc_model = clone_module(base), clone_module(base)
    ref_opt = DFCAdamW(ref.parameters(), lr=5e-4, weight_decay=0.01, enable_fiber=False)
    dfc_opt = DFCAdamW(dfc_model.parameters(), lr=5e-4, weight_decay=0.01, enable_fiber=True)
    channel = TorchSignFiberChannel(dfc_opt)
    rng = np.random.default_rng(7200)
    payload = rng.bytes(channel.byte_capacity)
    channel.write_bytes(0, payload)
    gen = torch.Generator().manual_seed(7201)
    _step_manual_pair(ref, dfc_model, ref_opt, dfc_opt, gen, 9, assert_sign_optimizer_equal)

    ref_blob, dfc_blob = io.BytesIO(), io.BytesIO()
    torch.save({"m": ref.state_dict(), "o": ref_opt.state_dict()}, ref_blob)
    torch.save({"m": dfc_model.state_dict(), "o": dfc_opt.state_dict()}, dfc_blob)

    ref2, dfc2 = clone_module(base), clone_module(base)
    ro2 = DFCAdamW(ref2.parameters(), lr=5e-4, weight_decay=0.01, enable_fiber=False)
    do2 = DFCAdamW(dfc2.parameters(), lr=5e-4, weight_decay=0.01, enable_fiber=True)
    ref_blob.seek(0)
    dfc_blob.seek(0)
    rckpt = torch.load(ref_blob, weights_only=True)
    dckpt = torch.load(dfc_blob, weights_only=True)
    ref2.load_state_dict(rckpt["m"])
    ro2.load_state_dict(rckpt["o"])
    dfc2.load_state_dict(dckpt["m"])
    do2.load_state_dict(dckpt["o"])
    channel2 = TorchSignFiberChannel(do2)
    if channel2.read_bytes(0, len(payload)) != payload:
        raise AssertionError("sign payload lost on checkpoint restore")
    assert_sign_optimizer_equal(ro2, do2)
    _step_manual_pair(ref2, dfc2, ro2, do2, gen, 11, assert_sign_optimizer_equal)
    if channel2.read_bytes(0, len(payload)) != payload:
        raise AssertionError("sign payload changed after resumed training")
    return {
        "status": "PASS",
        "pre_resume_steps": 9,
        "post_resume_steps": 11,
        "payload_sha256": digest_bytes(payload),
    }


def experiment_checkpoint_resume_low16() -> dict:
    base = ParamModule([(1024,), (129,)])
    ref, dfc_model = clone_module(base), clone_module(base)
    ref_opt = DFCLow16AdamW(ref.parameters(), lr=4e-4, enable_fiber=False)
    dfc_opt = DFCLow16AdamW(dfc_model.parameters(), lr=4e-4, enable_fiber=True)
    channel = TorchLow16FiberChannel(dfc_opt)
    payload = np.random.default_rng(7300).bytes(channel.byte_capacity)
    channel.write_bytes(0, payload)
    gen = torch.Generator().manual_seed(7301)
    _step_manual_pair(ref, dfc_model, ref_opt, dfc_opt, gen, 7, assert_low16_optimizer_equal)

    rb, db = io.BytesIO(), io.BytesIO()
    torch.save({"m": ref.state_dict(), "o": ref_opt.state_dict()}, rb)
    torch.save({"m": dfc_model.state_dict(), "o": dfc_opt.state_dict()}, db)
    ref2, dfc2 = clone_module(base), clone_module(base)
    ro2 = DFCLow16AdamW(ref2.parameters(), lr=4e-4, enable_fiber=False)
    do2 = DFCLow16AdamW(dfc2.parameters(), lr=4e-4, enable_fiber=True)
    rb.seek(0)
    db.seek(0)
    rckpt, dckpt = torch.load(rb, weights_only=True), torch.load(db, weights_only=True)
    ref2.load_state_dict(rckpt["m"])
    ro2.load_state_dict(rckpt["o"])
    dfc2.load_state_dict(dckpt["m"])
    do2.load_state_dict(dckpt["o"])
    channel2 = TorchLow16FiberChannel(do2)
    if channel2.read_bytes(0, len(payload)) != payload:
        raise AssertionError("low16 payload lost on checkpoint restore")
    _step_manual_pair(ref2, dfc2, ro2, do2, gen, 9, assert_low16_optimizer_equal)
    if channel2.read_bytes(0, len(payload)) != payload:
        raise AssertionError("low16 payload changed after resumed training")
    return {
        "status": "PASS",
        "pre_resume_steps": 7,
        "post_resume_steps": 9,
        "payload_sha256": digest_bytes(payload),
    }


def experiment_parameter_groups_and_accumulation() -> dict:
    a1 = nn.Parameter(torch.randn(777))
    a2 = nn.Parameter(torch.randn(333))
    b1 = nn.Parameter(a1.detach().clone())
    b2 = nn.Parameter(a2.detach().clone())
    ref = DFCAdamW(
        [{"params": [a1], "lr": 2e-4, "weight_decay": 0.0},
         {"params": [a2], "lr": 9e-4, "weight_decay": 0.07}],
        enable_fiber=False,
    )
    dfc = DFCAdamW(
        [{"params": [b1], "lr": 2e-4, "weight_decay": 0.0},
         {"params": [b2], "lr": 9e-4, "weight_decay": 0.07}],
        enable_fiber=True,
    )
    ch = TorchSignFiberChannel(dfc)
    payload = np.random.default_rng(7400).bytes(ch.byte_capacity)
    ch.write_bytes(0, payload)
    gen = torch.Generator().manual_seed(7401)
    for step in range(12):
        for pa, pb in ((a1, b1), (a2, b2)):
            total = torch.zeros_like(pa)
            for _ in range(3):
                total.add_(torch.randn(pa.shape, generator=gen) * 0.05)
            pa.grad = total.clone()
            pb.grad = total.clone()
        if step % 4 == 3:
            a2.grad = None
            b2.grad = None
        ref.step()
        dfc.step()
        ref.zero_grad(set_to_none=True)
        dfc.zero_grad(set_to_none=True)
        assert_sign_optimizer_equal(ref, dfc)
    if ch.read_bytes(0, len(payload)) != payload:
        raise AssertionError("parameter-group payload changed")
    return {
        "status": "PASS",
        "steps": 12,
        "accumulations_per_step": 3,
        "includes_missing_gradient_steps": True,
    }


def experiment_capacity_law() -> dict:
    rows = []
    for p in (1, 7, 8, 9, 127, 128, 129, 1024, 4097):
        m1 = ParamModule([(p,)])
        sopt = DFCAdamW(m1.parameters(), enable_fiber=True)
        sch = TorchSignFiberChannel(sopt)
        expected_sign = p // 8
        if sch.byte_capacity != expected_sign:
            raise AssertionError(f"sign capacity {p}: {sch.byte_capacity} != {expected_sign}")

        m2 = ParamModule([(p,)])
        lopt = DFCLow16AdamW(m2.parameters(), enable_fiber=True)
        lch = TorchLow16FiberChannel(lopt)
        expected_low16 = 4 * p
        if lch.byte_capacity != expected_low16:
            raise AssertionError(f"low16 capacity {p}: {lch.byte_capacity} != {expected_low16}")
        rows.append({"coordinates": p, "sign_bytes": sch.byte_capacity, "low16_bytes": lch.byte_capacity})
    return {"status": "PASS", "rows": rows}


def experiment_subnormal_and_signed_zero() -> dict:
    ref_model = ParamModule([(64,)])
    dfc_model = clone_module(ref_model)
    ref = DFCAdamW(ref_model.parameters(), lr=1e-4, enable_fiber=False)
    dfc = DFCAdamW(dfc_model.parameters(), lr=1e-4, enable_fiber=True)
    pr = next(ref_model.parameters())
    pd = next(dfc_model.parameters())
    sr = ref.state[pr]["exp_avg_sq"]
    sd = dfc.state[pd]["exp_avg_sq"]
    sr.view(torch.int32).fill_(1)
    bits = sd.view(torch.int32)
    pattern = torch.arange(bits.numel(), dtype=torch.int32) & 1
    bits.copy_(torch.where(pattern.bool(), torch.full_like(bits, -2147483647), torch.ones_like(bits)))
    pr.grad = torch.zeros_like(pr)
    pd.grad = torch.zeros_like(pd)
    ref.step()
    dfc.step()
    assert_sign_optimizer_equal(ref, dfc)
    observed = TorchSignFiberChannel(dfc).read_bits(0, 64)
    expected = pattern.to(torch.uint8).numpy()
    if not np.array_equal(observed, expected):
        raise AssertionError("subnormal sign payload changed")
    return {"status": "PASS", "coordinates": 64, "covers_signed_zero_or_subnormal_boundary": True}


def experiment_sign_stress(coordinates: int, steps: int) -> dict:
    base = ParamModule([(coordinates,)])
    ref, dfc_model = clone_module(base), clone_module(base)
    ref_opt = DFCAdamW(ref.parameters(), lr=2e-4, weight_decay=0.01, enable_fiber=False)
    dfc_opt = DFCAdamW(dfc_model.parameters(), lr=2e-4, weight_decay=0.01, enable_fiber=True)
    ch = TorchSignFiberChannel(dfc_opt)
    payload = np.random.default_rng(7500).bytes(ch.byte_capacity)
    ch.write_bytes(0, payload)
    gen = torch.Generator().manual_seed(7501)
    _step_manual_pair(ref, dfc_model, ref_opt, dfc_opt, gen, steps, assert_sign_optimizer_equal)
    if ch.read_bytes(0, len(payload)) != payload:
        raise AssertionError("sign stress payload changed")
    return {
        "status": "PASS",
        "coordinates": coordinates,
        "steps": steps,
        "coordinate_updates": coordinates * steps,
        "payload_bytes": ch.byte_capacity,
        "payload_sha256": digest_bytes(payload),
    }


def experiment_low16_stress(coordinates: int, steps: int) -> dict:
    base = ParamModule([(coordinates,)])
    ref, dfc_model = clone_module(base), clone_module(base)
    ref_opt = DFCLow16AdamW(ref.parameters(), lr=2e-4, enable_fiber=False)
    dfc_opt = DFCLow16AdamW(dfc_model.parameters(), lr=2e-4, enable_fiber=True)
    ch = TorchLow16FiberChannel(dfc_opt)
    payload = np.random.default_rng(7600).bytes(ch.byte_capacity)
    ch.write_bytes(0, payload)
    gen = torch.Generator().manual_seed(7601)
    _step_manual_pair(ref, dfc_model, ref_opt, dfc_opt, gen, steps, assert_low16_optimizer_equal)
    if ch.read_bytes(0, len(payload)) != payload:
        raise AssertionError("low16 stress payload changed")
    return {
        "status": "PASS",
        "coordinates": coordinates,
        "steps": steps,
        "coordinate_updates": coordinates * steps,
        "payload_bytes": ch.byte_capacity,
        "payload_sha256": digest_bytes(payload),
    }


def diagnostic_noncontiguous() -> dict:
    try:
        value = torch.randn(31, 17).t()
        p = nn.Parameter(value)
        opt = DFCAdamW([p], enable_fiber=True)
        ch = TorchSignFiberChannel(opt)
        payload = bytes([0xA5]) * ch.byte_capacity
        ch.write_bytes(0, payload)
        p.grad = torch.randn_like(p)
        opt.step()
        survived = ch.read_bytes(0, len(payload)) == payload
        return {"status": "SUPPORTED" if survived else "UNSUPPORTED", "payload_survived": survived}
    except Exception as exc:
        return {"status": "UNSUPPORTED", "exception": f"{type(exc).__name__}: {exc}"}


def run_all(sign_coordinates: int, sign_steps: int, low_coordinates: int, low_steps: int) -> dict:
    experiments = {}
    funcs = [
        ("sign_payload_patterns", experiment_sign_payload_patterns),
        ("sign_dtype_matrix", experiment_sign_dtype_matrix),
        ("cross_tensor_boundaries", experiment_cross_tensor_boundaries),
        ("checkpoint_resume_sign", experiment_checkpoint_resume_sign),
        ("checkpoint_resume_low16", experiment_checkpoint_resume_low16),
        ("parameter_groups_and_accumulation", experiment_parameter_groups_and_accumulation),
        ("capacity_law", experiment_capacity_law),
        ("subnormal_and_signed_zero", experiment_subnormal_and_signed_zero),
    ]
    started = time.time()
    for name, fn in funcs:
        t0 = time.perf_counter()
        result = fn()
        result["wall_seconds"] = time.perf_counter() - t0
        experiments[name] = result

    t0 = time.perf_counter()
    experiments["sign_stress"] = experiment_sign_stress(sign_coordinates, sign_steps)
    experiments["sign_stress"]["wall_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    experiments["low16_stress"] = experiment_low16_stress(low_coordinates, low_steps)
    experiments["low16_stress"]["wall_seconds"] = time.perf_counter() - t0

    experiments["noncontiguous_storage_diagnostic"] = diagnostic_noncontiguous()

    gated = [v for k, v in experiments.items() if k != "noncontiguous_storage_diagnostic"]
    if not all(v.get("status") == "PASS" for v in gated):
        raise AssertionError("one or more CPU frontier publication gates failed")

    coord_updates = sum(int(v.get("coordinate_updates", 0)) for v in experiments.values())
    return {
        "schema": "dfc-github-cpu-frontier-v1",
        "status": "PASS",
        "started_unix": started,
        "finished_unix": time.time(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_threads": torch.get_num_threads(),
        },
        "hardware_claims_excluded": [
            "GPU latency or throughput",
            "Nsight DRAM/L2 hardware counters",
            "H100/H200/B200/B300 architecture generalization",
            "CUDA/Triton performance overhead",
        ],
        "total_stress_coordinate_updates": coord_updates,
        "experiments": experiments,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sign-coordinates", type=int, default=262_144)
    parser.add_argument("--sign-steps", type=int, default=64)
    parser.add_argument("--low-coordinates", type=int, default=131_072)
    parser.add_argument("--low-steps", type=int, default=32)
    args = parser.parse_args()
    report = run_all(args.sign_coordinates, args.sign_steps, args.low_coordinates, args.low_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
