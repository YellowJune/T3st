"""Aggregate sealed Kaggle DFC-EF evidence without retuning thresholds.

Thresholds are copied from docs/DFC_EF_FREE_VALIDATION_PROTOCOL.md.  The script
can be run over one merged result directory or a GitHub-collected tree. Missing
required primary evidence yields INCOMPLETE, never an inferred PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def candidates(root: Path, name: str) -> list[Path]:
    return sorted(root.rglob(name))


def first(root: Path, name: str) -> tuple[Path | None, dict | None]:
    xs = candidates(root, name)
    return (xs[0], load_json(xs[0])) if xs else (None, None)


def qwen_pairs(root: Path) -> list[tuple[Path, dict]]:
    rows = []
    for p in sorted(root.rglob('pair_*.json')):
        try:
            row = load_json(p)
        except Exception:
            continue
        if 'exact_trajectory_gate' in row:
            rows.append((p, row))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--output', default='dfc_ef_kaggle_aggregate.json')
    a = ap.parse_args()
    root = Path(a.root).resolve()

    exact_path, exact = first(root, 'gpu_dfc_ef_exactness.json')
    frontier_path, frontier = first(root, 'gpu_memory_frontier.json')
    throughput_path, throughput = first(root, 'gpu_dfc_ef_throughput.json')
    tpu_path, tpu = first(root, 'tpu_dfc_ef_jax.json')
    pairs = qwen_pairs(root)

    gates: dict[str, Any] = {}
    gates['H1_cuda_exactness'] = {
        'status': 'PASS' if exact and exact.get('pass') is True else ('FAIL' if exact else 'INCOMPLETE'),
        'source': str(exact_path) if exact_path else None,
    }

    crossover_ok = bool(frontier and frontier.get('crossover', {}).get('verified'))
    ratio = frontier.get('frontier_ratio_dfc_over_external') if frontier else None
    h4_pass = crossover_ok and ratio is not None and float(ratio) >= 1.25
    gates['H4_cuda_frontier'] = {
        'status': 'PASS' if h4_pass else ('FAIL' if frontier else 'INCOMPLETE'),
        'ratio': ratio,
        'minimum_ratio': 1.25,
        'crossover_verified': crossover_ok if frontier else None,
        'source': str(frontier_path) if frontier_path else None,
    }

    primary_pairs = [(p, r) for p, r in pairs if str(r.get('tag', '')).startswith('primary_')]
    smoke_pairs = [(p, r) for p, r in pairs if r.get('tag') == 'smoke']
    # Frozen primary campaign targets three paired seeds; do not silently lower n.
    pair_exact = len(primary_pairs) >= 3 and all(bool(r.get('exact_trajectory_gate')) for _, r in primary_pairs[:3])
    resource_equal = len(primary_pairs) >= 3 and all(bool(r.get('resource_contract_equal')) for _, r in primary_pairs[:3])
    removed_matches_capacity = len(primary_pairs) >= 3 and all(
        int(r.get('dfc_model_scale_external_removed_bytes') or -1) == int(r.get('dfc_fiber_capacity_bytes') or -2)
        and int(r.get('dfc_model_scale_external_removed_bytes') or 0) > 0
        for _, r in primary_pairs[:3]
    )
    gates['H2_qwen_placement_trajectory'] = {
        'status': 'PASS' if pair_exact else ('FAIL' if len(primary_pairs) >= 3 else 'INCOMPLETE'),
        'primary_pairs_found': len(primary_pairs),
        'required_pairs': 3,
        'sources': [str(p) for p, _ in primary_pairs],
    }
    gates['H3_model_scale_allocation_contract'] = {
        'status': 'PASS' if removed_matches_capacity else ('FAIL' if len(primary_pairs) >= 3 else 'INCOMPLETE'),
        'primary_pairs_found': len(primary_pairs),
        'sources': [str(p) for p, _ in primary_pairs],
    }
    gates['H5_matched_communication_contract'] = {
        'status': 'PASS' if resource_equal else ('FAIL' if len(primary_pairs) >= 3 else 'INCOMPLETE'),
        'primary_pairs_found': len(primary_pairs),
    }

    tpu_exact = None
    if tpu:
        ex = tpu.get('exactness', {})
        tpu_exact = bool(ex.get('roundtrip_bitwise') and ex.get('semantic_invariance_bitwise') and ex.get('trajectory_bitwise'))
    secondary = {
        'throughput_available': throughput is not None,
        'throughput_source': str(throughput_path) if throughput_path else None,
        'smoke_pairs_found': len(smoke_pairs),
        'tpu_cross_substrate': {
            'status': 'PASS' if tpu_exact else ('FAIL' if tpu else 'INCOMPLETE'),
            'source': str(tpu_path) if tpu_path else None,
            'backend': tpu.get('backend') if tpu else None,
            'device_count': tpu.get('device_count') if tpu else None,
        },
    }

    statuses = [g['status'] for g in gates.values()]
    if 'FAIL' in statuses:
        overall = 'FAIL'
    elif all(s == 'PASS' for s in statuses):
        overall = 'PASS'
    else:
        overall = 'INCOMPLETE'

    result = {
        'schema_version': 1,
        'protocol': 'dfc-ef-free-validation-aggregate-v1',
        'root': str(root),
        'primary_status': overall,
        'gates': gates,
        'secondary': secondary,
        'projection_only_storage_law_decimal_bytes': {
            '7B': 28_000_000_000,
            '30B': 120_000_000_000,
            '70B': 280_000_000_000,
        },
        'projection_warning': 'These 7B/30B/70B values are algebraic 4P projections, not measured HBM savings.',
    }
    canonical = json.dumps(result, sort_keys=True, separators=(',', ':')).encode()
    result['aggregate_sha256'] = hashlib.sha256(canonical).hexdigest()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
