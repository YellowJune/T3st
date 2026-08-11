"""Actual-CUDA exactness/checkpoint gate for chunked DFC-EF."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from chunked_low16_adamw import DFCLow16AdamWChunked
from dfc_ef import PackedFP32Residual, stride_error_feedback_dfc_inplace_, stride_error_feedback_external_inplace_
from torch_fiber import HIGH16_MASK_I32


def digest_tensor(t: torch.Tensor) -> str:
    h = hashlib.sha256()
    x = t.detach().contiguous().cpu()
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def decoded_bits(t: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_and(t.view(torch.int32), HIGH16_MASK_I32)


def run(dtype: torch.dtype, n: int, steps: int, chunk: int, seed: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    gen = torch.Generator(device=device).manual_seed(seed)
    init = torch.randn(n, generator=gen, device=device, dtype=dtype)
    p_ext = torch.nn.Parameter(init.clone())
    p_dfc = torch.nn.Parameter(init.clone())
    opt_ext = DFCLow16AdamWChunked([p_ext], lr=3e-4, enable_fiber=False, chunk_coordinates=chunk)
    opt_dfc = DFCLow16AdamWChunked([p_dfc], lr=3e-4, enable_fiber=True, chunk_coordinates=chunk)
    residual_ext = torch.zeros(n, dtype=torch.float32, device=device)
    ch = PackedFP32Residual(opt_dfc); ch.zero_()
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(steps):
        g = torch.randn(n, generator=gen, device=device, dtype=dtype)
        ge = g.clone(); gd = g.clone()
        se = stride_error_feedback_external_inplace_(
            ge, residual_ext, stride=8, offset=step % 8, global_start=11,
            chunk_coordinates=chunk,
        )
        sd = stride_error_feedback_dfc_inplace_(
            p_dfc, gd, ch, stride=8, offset=step % 8, global_start=11,
            chunk_coordinates=chunk,
        )
        if se != sd or not torch.equal(ge, gd):
            raise AssertionError(f"compressed gradient mismatch at step {step}")
        hidden = ch.read_for_parameter(p_dfc)
        if not torch.equal(residual_ext.view(torch.int32), hidden.view(torch.int32)):
            raise AssertionError(f"residual mismatch at step {step}")
        p_ext.grad = ge; p_dfc.grad = gd
        opt_ext.step(); opt_dfc.step()
        if not torch.equal(p_ext, p_dfc):
            raise AssertionError(f"parameter mismatch at step {step}")
        a = opt_ext.state[p_ext]; b = opt_dfc.state[p_dfc]
        if not torch.equal(decoded_bits(a['exp_avg']), decoded_bits(b['exp_avg'])):
            raise AssertionError(f"first moment semantic mismatch at step {step}")
        if not torch.equal(decoded_bits(a['exp_avg_sq']), decoded_bits(b['exp_avg_sq'])):
            raise AssertionError(f"second moment semantic mismatch at step {step}")

    # DFC checkpoint is self-contained: hidden residual rides inside optimizer state.
    payload_before = ch.read_for_parameter(p_dfc).clone()
    buf = io.BytesIO()
    torch.save({'parameter': p_dfc.detach(), 'optimizer': opt_dfc.state_dict()}, buf)
    checkpoint_bytes = buf.tell(); buf.seek(0)
    p2 = torch.nn.Parameter(torch.empty_like(p_dfc))
    opt2 = DFCLow16AdamWChunked([p2], lr=3e-4, enable_fiber=True, chunk_coordinates=chunk)
    saved = torch.load(buf, map_location=device, weights_only=False)
    p2.data.copy_(saved['parameter']); opt2.load_state_dict(saved['optimizer'])
    ch2 = PackedFP32Residual(opt2)
    if not torch.equal(payload_before.view(torch.int32), ch2.read_for_parameter(p2).view(torch.int32)):
        raise AssertionError("checkpoint did not preserve hidden residual")

    torch.cuda.synchronize(device)
    return {
        'dtype': str(dtype),
        'coordinates': n,
        'steps': steps,
        'chunk_coordinates': chunk,
        'parameter_sha256': digest_tensor(p_dfc),
        'residual_sha256': digest_tensor(ch.read_for_parameter(p_dfc)),
        'dfc_checkpoint_bytes': int(checkpoint_bytes),
        'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coordinates', type=int, default=1_048_579)
    ap.add_argument('--steps', type=int, default=32)
    ap.add_argument('--chunk', type=int, default=131_071)
    ap.add_argument('--seed', type=int, default=260811)
    ap.add_argument('--output', default='results/kaggle_free/gpu_dfc_ef_exactness.json')
    a = ap.parse_args()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    rows = [run(torch.float32, a.coordinates, a.steps, a.chunk, a.seed),
            run(torch.float16, a.coordinates, a.steps, a.chunk, a.seed + 1)]
    result = {
        'schema_version': 1,
        'protocol': 'dfc-ef-cuda-exactness-v1',
        'device': device_name,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'rows': rows,
        'pass': True,
    }
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
