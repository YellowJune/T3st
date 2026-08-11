"""CUDA end-to-end compressor+optimizer throughput benchmark for DFC-EF."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _hash_tensor(t):
    h = hashlib.sha256(); h.update(t.detach().contiguous().cpu().numpy().tobytes()); return h.hexdigest()


def _single(method: str, n: int, repeats: int, warmup: int, chunk: int, stride: int, seed: int) -> int:
    import torch
    from chunked_low16_adamw import DFCLow16AdamWChunked
    from dfc_ef import PackedFP32Residual, stride_error_feedback_dfc_inplace_, stride_error_feedback_external_inplace_

    if not torch.cuda.is_available():
        print(json.dumps({'ok': False, 'error': 'CUDA unavailable'})); return 3
    dev = torch.device('cuda:0')
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    p = torch.nn.Parameter(torch.randn(n, device=dev, dtype=torch.float16) * 1e-2)
    base_grad = torch.randn(n, device=dev, dtype=torch.float16) * 1e-3
    grad = torch.empty_like(base_grad)
    enable = method == 'dfc'
    opt = DFCLow16AdamWChunked([p], lr=1e-4, enable_fiber=enable, chunk_coordinates=chunk)
    if method == 'external':
        residual = torch.zeros(n, device=dev, dtype=torch.float32)
        channel = None
    elif method == 'dfc':
        residual = None
        channel = PackedFP32Residual(opt); channel.zero_()
    else:
        raise ValueError(method)

    def one(step: int):
        grad.copy_(base_grad)
        if method == 'external':
            sent = stride_error_feedback_external_inplace_(
                grad, residual, stride=stride, offset=step % stride,
                global_start=13, chunk_coordinates=chunk,
            )
        else:
            sent = stride_error_feedback_dfc_inplace_(
                p, grad, channel, stride=stride, offset=step % stride,
                global_start=13, chunk_coordinates=chunk,
            )
        p.grad = grad
        opt.step()
        return sent

    for i in range(warmup): one(i)
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    times = []
    sent = None
    for i in range(repeats):
        grad.copy_(base_grad)
        start.record()
        if method == 'external':
            sent = stride_error_feedback_external_inplace_(
                grad, residual, stride=stride, offset=(warmup+i) % stride,
                global_start=13, chunk_coordinates=chunk,
            )
        else:
            sent = stride_error_feedback_dfc_inplace_(
                p, grad, channel, stride=stride, offset=(warmup+i) % stride,
                global_start=13, chunk_coordinates=chunk,
            )
        p.grad = grad; opt.step(); end.record(); end.synchronize()
        times.append(float(start.elapsed_time(end)))
    peak = int(torch.cuda.max_memory_allocated(dev))
    alloc = int(torch.cuda.memory_allocated(dev))
    result = {
        'ok': True,
        'method': method,
        'coordinates': n,
        'stride': stride,
        'compression_ratio_values': stride,
        'chunk_coordinates': chunk,
        'repeats': repeats,
        'median_ms': float(__import__('statistics').median(times)),
        'mean_ms': float(sum(times)/len(times)),
        'min_ms': float(min(times)),
        'max_ms': float(max(times)),
        'transmitted_values_last_step': int(sent),
        'peak_allocated_bytes': peak,
        'steady_allocated_bytes': alloc,
        'theoretical_external_residual_bytes': int(4*n if method == 'external' else 0),
        'fiber_capacity_bytes': int(4*n if method == 'dfc' else 0),
        'parameter_sha256': _hash_tensor(p),
        'residual_sha256': _hash_tensor(residual if method == 'external' else channel.read_for_parameter(p)),
        'device': torch.cuda.get_device_name(dev),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
    }
    print(json.dumps(result, sort_keys=True)); return 0


def _run_single(method, n, repeats, warmup, chunk, stride, seed):
    cmd = [sys.executable, str(Path(__file__).resolve()), '--single', '--method', method,
           '--coordinates', str(n), '--repeats', str(repeats), '--warmup', str(warmup),
           '--chunk', str(chunk), '--stride', str(stride), '--seed', str(seed)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = [x for x in proc.stdout.splitlines() if x.strip().startswith('{')]
    if not lines:
        return {'ok': False, 'method': method, 'coordinates': n, 'returncode': proc.returncode, 'stderr': proc.stderr[-4000:]}
    row = json.loads(lines[-1]); row['returncode'] = proc.returncode; return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--single', action='store_true')
    ap.add_argument('--method', choices=['external','dfc'])
    ap.add_argument('--coordinates', type=int, default=1_048_576)
    ap.add_argument('--sizes', default='1048576,4194304,16777216,33554432')
    ap.add_argument('--repeats', type=int, default=9)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--chunk', type=int, default=1_048_576)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--seed', type=int, default=260811)
    ap.add_argument('--output', default='results/kaggle_free/gpu_dfc_ef_throughput.json')
    a = ap.parse_args()
    if a.single:
        if a.method is None: ap.error('--single requires --method')
        raise SystemExit(_single(a.method, a.coordinates, a.repeats, a.warmup, a.chunk, a.stride, a.seed))
    sizes = [int(x) for x in a.sizes.split(',') if x.strip()]
    rows = []
    for n in sizes:
        ext = _run_single('external', n, a.repeats, a.warmup, a.chunk, a.stride, a.seed)
        dfc = _run_single('dfc', n, a.repeats, a.warmup, a.chunk, a.stride, a.seed)
        paired = {'coordinates': n, 'external': ext, 'dfc': dfc}
        if ext.get('ok') and dfc.get('ok'):
            paired['dfc_overhead_fraction'] = dfc['median_ms']/ext['median_ms'] - 1.0
            paired['peak_memory_bytes_saved'] = ext['peak_allocated_bytes'] - dfc['peak_allocated_bytes']
            paired['trajectory_digest_equal'] = (ext['parameter_sha256'] == dfc['parameter_sha256'] and ext['residual_sha256'] == dfc['residual_sha256'])
        rows.append(paired)
    result = {'schema_version': 1, 'protocol': 'dfc-ef-cuda-throughput-v1', 'rows': rows}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__': main()
