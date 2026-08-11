from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/YellowJune/T3st.git"
BRANCH = "dfc-h200-final"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
CHECKOUT = WORK / "T3st_dfc_ef"
OUT = WORK / "dfc_ef_gpu_gate_results"
OUT.mkdir(parents=True, exist_ok=True)


def run(name, cmd, cwd=None, required=True):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    row = {"name": name, "command": cmd, "returncode": p.returncode,
           "wall_seconds": time.time()-t0, "stdout": p.stdout, "stderr": p.stderr}
    (OUT / f"{name}.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(f"\n===== {name} rc={p.returncode} {row['wall_seconds']:.1f}s =====")
    print(p.stdout[-6000:]); print(p.stderr[-6000:], file=sys.stderr)
    if required and p.returncode != 0:
        raise SystemExit(f"required gate failed: {name}")
    return row


# Provenance first.
run("nvidia_smi", ["nvidia-smi"], required=True)
if CHECKOUT.exists():
    shutil.rmtree(CHECKOUT)
run("clone", ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, str(CHECKOUT)], required=True)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True).strip()
(OUT / "source_commit.txt").write_text(commit + "\n")
remote = CHECKOUT / "remote"

# No model download in this kernel: use the first free-GPU minutes only for
# hard correctness, HBM frontier, and timing evidence.
run("cpu_exactness", [sys.executable, "-m", "pytest", "-q",
    "test_dfc_ef.py", "test_block_topk_ef.py", "test_chunked_low16_adamw.py",
    "test_kaggle_checkpoint.py"], cwd=remote)
run("cuda_exactness", [sys.executable, "gpu_dfc_ef_exactness.py",
    "--coordinates", "1048579", "--steps", "32", "--chunk", "131071",
    "--output", str(OUT / "gpu_dfc_ef_exactness.json")], cwd=remote)
run("memory_frontier", [sys.executable, "gpu_memory_frontier.py",
    "--resolution", "4194304", "--output", str(OUT / "gpu_memory_frontier.json")], cwd=remote)
run("throughput", [sys.executable, "gpu_dfc_ef_throughput.py",
    "--sizes", "1048576,4194304,16777216,33554432,67108864",
    "--repeats", "11", "--warmup", "3", "--chunk", "1048576", "--stride", "8",
    "--output", str(OUT / "gpu_dfc_ef_throughput.json")], cwd=remote)

manifest = {
    "schema_version": 1,
    "protocol": "dfc-ef-kaggle-gpu-gate-v1",
    "source_commit": commit,
    "branch": BRANCH,
    "python": sys.version,
    "files": sorted(p.name for p in OUT.iterdir()),
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
shutil.make_archive(str(WORK / "dfc_ef_gpu_gate_results"), "zip", OUT)
print(json.dumps(manifest, indent=2))
