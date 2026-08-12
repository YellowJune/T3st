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
MODEL05 = "Qwen/Qwen2.5-0.5B"
REV05 = "060db6499f32faf8b98477b0a26969ef7d8b9987"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
CHECKOUT = WORK / "T3st_dfc_fused_qwen"
OUT = WORK / "dfc_fused_qwen_results"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time(); BUDGET_HOURS = float(os.environ.get("DFC_FUSED_BUDGET_HOURS", "10.0"))
DEADLINE = T0 + BUDGET_HOURS * 3600; RESERVE = 20 * 60
MANIFEST = {"schema_version": 1, "protocol": "dfc-fused-traffic-elimination-campaign-v1", "jobs": [], "pairs": []}


def save_manifest():
    MANIFEST["elapsed_hours"] = (time.time() - T0) / 3600
    MANIFEST["remaining_minutes"] = (DEADLINE - time.time()) / 60
    tmp = OUT / "manifest.json.tmp"; tmp.write_text(json.dumps(MANIFEST, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, OUT / "manifest.json")


def shell(name, cmd, *, cwd=None, required=True, timeout=None, env=None):
    st = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout, env=env)
        row = {"name": name, "returncode": p.returncode, "success": p.returncode == 0,
               "wall_seconds": time.time() - st, "command": cmd,
               "stdout_tail": p.stdout[-20000:], "stderr_tail": p.stderr[-20000:]}
    except subprocess.TimeoutExpired as e:
        row = {"name": name, "returncode": -999, "success": False, "timeout": True,
               "wall_seconds": time.time() - st, "command": cmd,
               "stdout_tail": (e.stdout or "")[-20000:] if isinstance(e.stdout, str) else "",
               "stderr_tail": (e.stderr or "")[-20000:] if isinstance(e.stderr, str) else ""}
    MANIFEST["jobs"].append(row); save_manifest()
    print(f"\n===== {name} success={row['success']} {row['wall_seconds']:.1f}s =====")
    print(row.get("stdout_tail", "")); print(row.get("stderr_tail", ""), file=sys.stderr)
    if required and not row["success"]: raise RuntimeError(f"required job failed: {name}")
    return row


def enough(seconds): return DEADLINE - time.time() > RESERVE + seconds


def qwen_job(name, method, seed, *, model=MODEL05, revision=REV05, layers=0, updates=64, batch=2, required=True):
    result = OUT / f"{name}.json"
    cmd = [sys.executable, "llm_dfc_ef_qwen_fused.py", "--method", method, "--seed", str(seed),
           "--model", model, "--revision", revision, "--train-last-layers", str(layers),
           "--updates", str(updates), "--batch-size", str(batch), "--eval-every", str(max(8, updates // 4)),
           "--stride", "8", "--output", str(result)]
    timeout = max(900, int(DEADLINE - time.time() - RESERVE))
    row = shell(name, cmd, cwd=CHECKOUT / "remote", required=required, timeout=timeout)
    if row["success"] and result.exists():
        data = json.loads(result.read_text())
        row["result_summary"] = {k: data.get(k) for k in (
            "model", "resolved_hub_revision", "method", "seed", "trainable_parameters",
            "actual_external_residual_bytes", "model_scale_external_residual_removed_bytes",
            "final_accuracy", "final_eval_loss", "wall_seconds", "fused_update_mean_ms",
            "checkpoint_bytes", "parameter_sha256", "semantic_optimizer_sha256",
            "logical_residual_sha256", "result_sha256")}
        save_manifest()
    return row


def compare_pair(tag, ext_name, dfc_name):
    e = json.loads((OUT / f"{ext_name}.json").read_text()); d = json.loads((OUT / f"{dfc_name}.json").read_text())
    contract_keys = ("model", "resolved_hub_revision", "seed", "train_last_layers", "trainable_parameters",
                     "updates", "batch_size", "seq_len", "lr", "weight_decay", "grad_clip", "stride",
                     "transmitted_values", "dense_gradient_values", "transport_dtype", "semantic_optimizer")
    c = {
        "tag": tag,
        "resource_contract_equal": all(e.get(k) == d.get(k) for k in contract_keys),
        "parameter_digest_equal": e["parameter_sha256"] == d["parameter_sha256"],
        "semantic_optimizer_digest_equal": e["semantic_optimizer_sha256"] == d["semantic_optimizer_sha256"],
        "logical_residual_digest_equal": e["logical_residual_sha256"] == d["logical_residual_sha256"],
        "final_accuracy_equal": e["final_accuracy"] == d["final_accuracy"],
        "final_eval_loss_equal": e["final_eval_loss"] == d["final_eval_loss"],
        "peak_allocated_external": e["training_peak_memory"]["max_allocated"],
        "peak_allocated_dfc": d["training_peak_memory"]["max_allocated"],
        "setup_allocated_external": e["memory_after_state_setup"]["allocated"],
        "setup_allocated_dfc": d["memory_after_state_setup"]["allocated"],
        "checkpoint_bytes_external": e["checkpoint_bytes"], "checkpoint_bytes_dfc": d["checkpoint_bytes"],
        "wall_seconds_external": e["wall_seconds"], "wall_seconds_dfc": d["wall_seconds"],
        "fused_update_mean_ms_external": e["fused_update_mean_ms"], "fused_update_mean_ms_dfc": d["fused_update_mean_ms"],
        "external_residual_bytes": e["actual_external_residual_bytes"],
        "dfc_external_residual_bytes": d["actual_external_residual_bytes"],
        "dfc_removed_bytes": d["model_scale_external_residual_removed_bytes"],
    }
    c["exact_trajectory_gate"] = all(c[k] for k in ("resource_contract_equal", "parameter_digest_equal",
        "semantic_optimizer_digest_equal", "logical_residual_digest_equal", "final_accuracy_equal", "final_eval_loss_equal"))
    c["peak_allocated_saved_bytes"] = c["peak_allocated_external"] - c["peak_allocated_dfc"]
    c["peak_allocated_saved_percent"] = 100.0 * c["peak_allocated_saved_bytes"] / c["peak_allocated_external"]
    c["setup_allocated_saved_bytes"] = c["setup_allocated_external"] - c["setup_allocated_dfc"]
    c["checkpoint_saved_bytes"] = c["checkpoint_bytes_external"] - c["checkpoint_bytes_dfc"]
    c["checkpoint_saved_percent"] = 100.0 * c["checkpoint_saved_bytes"] / c["checkpoint_bytes_external"]
    c["end_to_end_overhead_percent"] = 100.0 * (c["wall_seconds_dfc"] / c["wall_seconds_external"] - 1.0)
    c["end_to_end_speedup_percent"] = 100.0 * (c["wall_seconds_external"] / c["wall_seconds_dfc"] - 1.0)
    c["optimizer_stage_overhead_percent"] = 100.0 * (c["fused_update_mean_ms_dfc"] / c["fused_update_mean_ms_external"] - 1.0)
    c["optimizer_stage_speedup_percent"] = 100.0 * (c["fused_update_mean_ms_external"] / c["fused_update_mean_ms_dfc"] - 1.0)
    c["traffic_elimination_gate"] = bool(c["exact_trajectory_gate"] and c["peak_allocated_saved_bytes"] > 0 and
                                         c["checkpoint_saved_bytes"] > 0 and c["optimizer_stage_overhead_percent"] <= 2.0)
    (OUT / f"pair_{tag}.json").write_text(json.dumps(c, indent=2, sort_keys=True) + "\n")
    MANIFEST["pairs"].append(c); save_manifest(); return c


def archive(status, error=None):
    MANIFEST["status"] = status; MANIFEST["error"] = error; save_manifest()
    shutil.make_archive(str(WORK / "dfc_fused_qwen_results"), "zip", OUT)


try:
    shell("nvidia_smi", ["nvidia-smi"], required=True)
    gpu_name = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    MANIFEST["gpu_name"] = gpu_name; save_manifest()

    probe = shell("torch_preflight_before", [sys.executable, "-c",
        "import torch,json; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'capability':torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,'arch_list':torch.cuda.get_arch_list()}))"], required=False)
    if "P100" in gpu_name or "sm_60" not in probe.get("stdout_tail", "") and "P100" in gpu_name:
        shell("install_pascal_torch", [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", "--force-reinstall",
              "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu126"], required=True)
        shell("remove_optional_binary_stacks", [sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"], required=False)

    shell("torch_preflight_after", [sys.executable, "-c",
        "import torch,json; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'name':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'arch_list':torch.cuda.get_arch_list()})); assert torch.cuda.is_available()"], required=True)
    shell("compiler_preflight", ["bash", "-lc", "set -e; command -v g++; g++ --version | head -1; command -v nvcc; nvcc --version | tail -1"], required=True)

    if CHECKOUT.exists(): shutil.rmtree(CHECKOUT)
    shell("clone", ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, str(CHECKOUT)], required=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True).strip()
    MANIFEST["source_commit"] = commit; save_manifest()
    shell("install_stack", [sys.executable, "-m", "pip", "install", "-q", "ninja", "transformers>=4.51,<5",
          "huggingface_hub>=0.30,<1", "sentencepiece", "accelerate", "numpy"], required=True)

    systems_path = OUT / "fused_systems.json"
    shell("fused_systems", [sys.executable, "benchmark_fused_stride_ef.py", "--output", str(systems_path)],
          cwd=CHECKOUT / "remote", required=True, timeout=45 * 60)
    systems = json.loads(systems_path.read_text()); MANIFEST["systems"] = systems; save_manifest()
    if not systems.get("promotion_gate"):
        archive("NO_PROMOTION", "systems promotion gate failed")
        print(json.dumps({"status": "NO_PROMOTION", "systems": {k: systems.get(k) for k in (
            "median_runtime_overhead_percent", "median_runtime_speedup_percent", "all_exact")}}, indent=2)); raise SystemExit(0)

    # First actual Qwen pair is the go/no-go learning-scale test.
    qwen_job("qwen05_external_3101", "external_fused", 3101, updates=64, batch=2)
    qwen_job("qwen05_dfc_3101", "dfc_fused", 3101, updates=64, batch=2)
    p05 = compare_pair("qwen05_3101", "qwen05_external_3101", "qwen05_dfc_3101")
    MANIFEST["qwen05_promotion_gate"] = p05["traffic_elimination_gate"]; save_manifest()
    if not p05["traffic_elimination_gate"]:
        archive("NO_PROMOTION", "0.5B traffic-elimination gate failed")
        print(json.dumps({"status": "NO_PROMOTION", "pair": p05}, indent=2)); raise SystemExit(0)

    # Scale is prioritized immediately once the small real-model pair validates.
    if enough(60 * 60):
        qwen_job("qwen15_external_3201", "external_fused", 3201, model="Qwen/Qwen2.5-1.5B", revision="main", layers=4, updates=48, batch=1)
        qwen_job("qwen15_dfc_3201", "dfc_fused", 3201, model="Qwen/Qwen2.5-1.5B", revision="main", layers=4, updates=48, batch=1)
        compare_pair("qwen15_3201", "qwen15_external_3201", "qwen15_dfc_3201")

    if enough(75 * 60):
        qwen_job("qwen3b_external_3301", "external_fused", 3301, model="Qwen/Qwen2.5-3B", revision="main", layers=2, updates=32, batch=1)
        qwen_job("qwen3b_dfc_3301", "dfc_fused", 3301, model="Qwen/Qwen2.5-3B", revision="main", layers=2, updates=32, batch=1)
        compare_pair("qwen3b_3301", "qwen3b_external_3301", "qwen3b_dfc_3301")

    # Additional 0.5B seeds only after scale evidence is secured.
    for seed in (3131, 3161):
        if not enough(35 * 60): break
        qwen_job(f"qwen05_external_{seed}", "external_fused", seed, updates=64, batch=2)
        qwen_job(f"qwen05_dfc_{seed}", "dfc_fused", seed, updates=64, batch=2)
        compare_pair(f"qwen05_{seed}", f"qwen05_external_{seed}", f"qwen05_dfc_{seed}")

    all_exact = all(p.get("exact_trajectory_gate") for p in MANIFEST["pairs"])
    all_traffic = all(p.get("traffic_elimination_gate") for p in MANIFEST["pairs"])
    MANIFEST["all_exact"] = all_exact; MANIFEST["all_traffic_elimination_gates"] = all_traffic
    archive("PASS" if all_exact else "FAIL")
    print(json.dumps({"status": MANIFEST["status"], "source_commit": commit,
                      "pairs": len(MANIFEST["pairs"]), "all_exact": all_exact,
                      "all_traffic_elimination_gates": all_traffic}, indent=2))
except SystemExit:
    raise
except BaseException as exc:
    tb = traceback.format_exc(); (OUT / "fatal_traceback.txt").write_text(tb); print(tb, file=sys.stderr)
    archive("FAIL", f"{type(exc).__name__}: {exc}"); raise
