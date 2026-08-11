"""Budget-aware free-GPU campaign for the DFC-EF paper validation.

This orchestrator intentionally runs the cheapest/falsifying gates first. If a
free Kaggle GPU session dies late, the highest-value systems evidence should
already exist in ``results/kaggle_free``.

Phases:
  S0 environment ledger
  S1 actual-CUDA exactness + checkpoint roundtrip
  S2 state-only HBM frontier/OOM crossover
  S3 compressor+optimizer throughput/memory scaling
  S4 Qwen smoke: external-EF vs DFC-EF
  S5 Qwen primary 3-seed paired external-EF/DFC-EF
  S6 precision/compression ablations
  S7 scale-up transformer-block stress if time remains

The campaign never interprets communication reduction as a DFC contribution.
Blockwise top-k supplies compression; DFC only changes EF residual placement.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + '\n')
    os.replace(tmp, path)


def run_cmd(name: str, cmd: list[str], *, cwd: Path, timeout_s: int | None = None) -> dict:
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=timeout_s)
    row = {
        'name': name,
        'command': cmd,
        'returncode': proc.returncode,
        'wall_seconds': time.time() - started,
        'stdout_tail': proc.stdout[-12000:],
        'stderr_tail': proc.stderr[-12000:],
        'success': proc.returncode == 0,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget-hours', type=float, default=10.5)
    ap.add_argument('--reserve-minutes', type=float, default=25.0)
    ap.add_argument('--results', default='../results/kaggle_free')
    ap.add_argument('--seeds', default='1901,1931,1951')
    ap.add_argument('--primary-updates', type=int, default=128)
    ap.add_argument('--primary-layers', type=int, default=8)
    ap.add_argument('--stress-layers', type=int, default=0, help='0 = all transformer blocks')
    ap.add_argument('--stress-updates', type=int, default=64)
    ap.add_argument('--skip-frontier', action='store_true')
    a = ap.parse_args()

    root = Path(__file__).resolve().parent
    results = (root / a.results).resolve()
    results.mkdir(parents=True, exist_ok=True)
    journal_path = results / 'campaign_manifest.json'
    t0 = time.time()
    deadline = t0 + a.budget_hours * 3600
    reserve = a.reserve_minutes * 60
    phases: list[dict] = []

    def remaining():
        return deadline - time.time()

    def save():
        atomic_json({
            'schema_version': 1,
            'protocol': 'dfc-ef-kaggle-free-campaign-v1',
            'started_unix': t0,
            'updated_unix': time.time(),
            'budget_hours': a.budget_hours,
            'remaining_seconds': remaining(),
            'python': sys.version,
            'platform': platform.platform(),
            'phases': phases,
        }, journal_path)

    def execute(name, cmd, min_remaining_s=0, timeout_s=None, required=False):
        if remaining() < reserve + min_remaining_s:
            row = {'name': name, 'skipped': True, 'reason': 'time_budget',
                   'remaining_seconds': remaining(), 'success': False}
            phases.append(row); save(); return row
        try:
            row = run_cmd(name, cmd, cwd=root, timeout_s=timeout_s)
        except subprocess.TimeoutExpired as exc:
            row = {'name': name, 'success': False, 'timeout': True,
                   'wall_seconds': exc.timeout, 'command': cmd}
        phases.append(row); save()
        if required and not row.get('success'):
            raise SystemExit(f'required phase failed: {name}')
        return row

    # S0: provenance and accelerator ledger.
    env_out = results / 'environment.txt'
    env_cmd = [sys.executable, '-c',
        "import torch,platform; print(platform.platform()); print(torch.__version__, torch.version.cuda); "
        "print(torch.cuda.is_available()); "
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA'); "
        "print(torch.cuda.get_device_properties(0) if torch.cuda.is_available() else '')"]
    row = execute('S0_environment', env_cmd, required=True)
    env_out.write_text(row.get('stdout_tail','') + '\n' + row.get('stderr_tail',''))

    # S1: correctness first. Cheap and scientifically mandatory.
    execute('S1_cuda_exactness', [sys.executable, 'gpu_dfc_ef_exactness.py',
            '--output', str(results / 'gpu_dfc_ef_exactness.json')],
            min_remaining_s=300, timeout_s=1800, required=True)

    # S2: actual HBM frontier. This runs each OOM probe in a fresh subprocess.
    if not a.skip_frontier:
        execute('S2_hbm_frontier', [sys.executable, 'gpu_memory_frontier.py',
                '--resolution', '4194304', '--output', str(results / 'gpu_memory_frontier.json')],
                min_remaining_s=600, timeout_s=3600, required=True)

    # S3: end-to-end compressor + optimizer timing over several state scales.
    execute('S3_throughput', [sys.executable, 'gpu_dfc_ef_throughput.py',
            '--sizes', '1048576,4194304,16777216,33554432,67108864',
            '--repeats', '11', '--warmup', '3',
            '--output', str(results / 'gpu_dfc_ef_throughput.json')],
            min_remaining_s=600, timeout_s=3600, required=True)

    # S4: small actual-model smoke; fail before spending hours on a broken path.
    for method in ('external_ef', 'dfc_ef'):
        execute(f'S4_smoke_{method}', [sys.executable, 'llm_dfc_ef_qwen.py',
                '--method', method, '--seed', '1801', '--train-last-layers', '2',
                '--updates', '8', '--batch-size', '2', '--eval-every', '4',
                '--keep-ratio', '0.125', '--deterministic',
                '--progress-output', str(results / f'smoke_{method}_progress.json'),
                '--output', str(results / f'smoke_{method}.json')],
                min_remaining_s=900, timeout_s=2700, required=True)

    # S5: primary paired 3-seed evidence.
    seeds = [int(x) for x in a.seeds.split(',') if x.strip()]
    for seed in seeds:
        for method in ('external_ef', 'dfc_ef'):
            execute(f'S5_primary_{method}_{seed}', [sys.executable, 'llm_dfc_ef_qwen.py',
                    '--method', method, '--seed', str(seed),
                    '--train-last-layers', str(a.primary_layers),
                    '--updates', str(a.primary_updates), '--batch-size', '4',
                    '--eval-every', '32', '--keep-ratio', '0.125', '--deterministic',
                    '--progress-output', str(results / f'primary_{method}_{seed}_progress.json'),
                    '--output', str(results / f'primary_{method}_{seed}.json')],
                    min_remaining_s=1800, timeout_s=max(1800, int(remaining()-reserve)), required=False)

    # S6: one-seed causal attribution. These prevent reviewers from attributing
    # a learning change to low16 semantics or to compression rather than DFC.
    for method in ('fp32_dense', 'low16_dense', 'low16_noef'):
        execute(f'S6_ablation_{method}', [sys.executable, 'llm_dfc_ef_qwen.py',
                '--method', method, '--seed', '1901',
                '--train-last-layers', str(a.primary_layers),
                '--updates', str(a.primary_updates), '--batch-size', '4',
                '--eval-every', '32', '--keep-ratio', '0.125', '--deterministic',
                '--output', str(results / f'ablation_{method}_1901.json')],
                min_remaining_s=1800, timeout_s=max(1800, int(remaining()-reserve)), required=False)

    # S7: maximize free HBM use. All transformer blocks are trainable by default
    # here; if it OOMs the failure is retained rather than hidden.
    for method in ('external_ef', 'dfc_ef'):
        execute(f'S7_stress_{method}', [sys.executable, 'llm_dfc_ef_qwen.py',
                '--method', method, '--seed', '1999',
                '--train-last-layers', str(a.stress_layers),
                '--updates', str(a.stress_updates), '--batch-size', '2',
                '--eval-every', '16', '--keep-ratio', '0.125', '--deterministic',
                '--output', str(results / f'stress_{method}.json')],
                min_remaining_s=900, timeout_s=max(900, int(remaining()-reserve)), required=False)

    save()
    print(json.dumps({'manifest': str(journal_path), 'phases': len(phases),
                      'elapsed_hours': (time.time()-t0)/3600,
                      'remaining_minutes': remaining()/60}, indent=2))


if __name__ == '__main__':
    main()
