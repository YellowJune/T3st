from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
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
    print(p.stdout[-12000:]); print(p.stderr[-12000:], file=sys.stderr)
    if required and p.returncode != 0:
        raise RuntimeError(f"required gate failed: {name}")
    return row


def archive(status: str, source_commit: str | None = None, error: str | None = None):
    manifest = {
        "schema_version": 2,
        "protocol": "dfc-ef-kaggle-gpu-gate-v2-p100-compatible",
        "status": status,
        "source_commit": source_commit,
        "branch": BRANCH,
        "python": sys.version,
        "error": error,
        "files": sorted(p.name for p in OUT.iterdir()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shutil.make_archive(str(WORK / "dfc_ef_gpu_gate_results"), "zip", OUT)
    print(json.dumps(manifest, indent=2, sort_keys=True))


commit = None
try:
    smi = run("nvidia_smi", ["nvidia-smi"], required=True)
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()[0]
    (OUT / "gpu_name.txt").write_text(gpu_name + "\n")

    # Kaggle may assign a P100 even when a T4-class accelerator is requested.
    # Current Kaggle torch wheels can omit sm_60.  PyTorch's official 2.7.1
    # CUDA-12.6 wheel retains Pascal support, so install it only when needed.
    torch_probe = run("torch_preflight_before", [sys.executable, "-c",
        "import torch,json; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'arch_list':torch.cuda.get_arch_list()}))"],
        required=False)
    if "P100" in gpu_name or "sm_60" not in torch_probe.get("stdout", ""):
        run("install_p100_torch", [sys.executable, "-m", "pip", "install",
            "--no-cache-dir", "--upgrade", "--force-reinstall",
            "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu126"], required=True)

    compat = run("torch_preflight_after", [sys.executable, "-c",
        "import torch,json; d={'torch':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'capability':torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,'arch_list':torch.cuda.get_arch_list()}; print(json.dumps(d)); assert torch.cuda.is_available(); cc=torch.cuda.get_device_capability(0); assert cc[0] != 6 or 'sm_60' in torch.cuda.get_arch_list()"], required=True)

    if CHECKOUT.exists():
        shutil.rmtree(CHECKOUT)
    run("clone", ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, str(CHECKOUT)], required=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True).strip()
    (OUT / "source_commit.txt").write_text(commit + "\n")
    remote = CHECKOUT / "remote"

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
    archive("PASS", commit)
except BaseException as exc:
    tb = traceback.format_exc()
    (OUT / "fatal_traceback.txt").write_text(tb)
    print(tb, file=sys.stderr)
    archive("FAIL", commit, f"{type(exc).__name__}: {exc}")
    raise
