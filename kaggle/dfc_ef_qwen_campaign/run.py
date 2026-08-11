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
CHECKOUT = WORK / "T3st_dfc_ef_qwen"
OUT = WORK / "dfc_ef_qwen_results"
CKPT = WORK / "dfc_ef_qwen_checkpoints"
OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
BUDGET_HOURS = float(os.environ.get("DFC_KAGGLE_BUDGET_HOURS", "10.25"))
RESERVE_SECONDS = 20 * 60
T0 = time.time(); DEADLINE = T0 + BUDGET_HOURS * 3600
MANIFEST = {"schema_version": 3, "protocol": "dfc-ef-kaggle-qwen-campaign-v3-p100-textonly",
            "budget_hours": BUDGET_HOURS, "jobs": []}


def save_manifest():
    MANIFEST["elapsed_hours"] = (time.time()-T0)/3600
    MANIFEST["remaining_minutes"] = (DEADLINE-time.time())/60
    tmp = OUT / "manifest.json.tmp"
    tmp.write_text(json.dumps(MANIFEST, indent=2, sort_keys=True, allow_nan=True)+"\n")
    os.replace(tmp, OUT / "manifest.json")


def shell(name, cmd, cwd=None, required=True, timeout=None):
    start = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        row = {"name": name, "returncode": p.returncode, "success": p.returncode == 0,
               "wall_seconds": time.time()-start, "command": cmd,
               "stdout_tail": p.stdout[-16000:], "stderr_tail": p.stderr[-16000:]}
    except subprocess.TimeoutExpired as e:
        row = {"name": name, "returncode": -999, "success": False, "timeout": True,
               "wall_seconds": time.time()-start, "command": cmd,
               "stdout_tail": (e.stdout or "")[-16000:] if isinstance(e.stdout, str) else "",
               "stderr_tail": (e.stderr or "")[-16000:] if isinstance(e.stderr, str) else ""}
    MANIFEST["jobs"].append(row); save_manifest()
    print(f"\n===== {name}: success={row['success']} {row['wall_seconds']:.1f}s =====")
    print(row.get("stdout_tail", "")); print(row.get("stderr_tail", ""), file=sys.stderr)
    if required and not row["success"]:
        raise RuntimeError(f"required job failed: {name}")
    return row


def enough(seconds):
    return DEADLINE-time.time() > RESERVE_SECONDS + seconds


def qwen_job(name, method, seed, *, model=MODEL05, revision=REV05,
             layers=0, updates=128, batch=2, checkpoint=False, required=False):
    if not enough(25*60):
        row = {"name": name, "skipped": True, "success": False, "reason": "time_budget"}
        MANIFEST["jobs"].append(row); save_manifest(); return row
    result = OUT / f"{name}.json"
    progress = OUT / f"{name}_progress.json"
    ckpt = CKPT / f"{name}.pt"
    cmd = [sys.executable, "llm_dfc_ef_qwen_safe.py",
           "--method", method, "--seed", str(seed), "--model", model,
           "--revision", revision, "--train-last-layers", str(layers),
           "--updates", str(updates), "--batch-size", str(batch),
           "--eval-every", str(max(8, updates//4)), "--keep-ratio", "0.125",
           "--ef-chunk", "262144", "--optimizer-chunk", "262144",
           "--deterministic", "--progress-output", str(progress),
           "--output", str(result)]
    if checkpoint:
        cmd += ["--checkpoint", str(ckpt), "--checkpoint-every", str(max(8, updates//2)), "--resume"]
    timeout = max(900, int(DEADLINE-time.time()-RESERVE_SECONDS))
    row = shell(name, cmd, cwd=CHECKOUT/"remote", required=required, timeout=timeout)
    if row["success"] and result.exists():
        data = json.loads(result.read_text())
        row["result_summary"] = {k: data.get(k) for k in (
            "model", "resolved_hub_revision", "method", "seed", "trainable_parameters",
            "actual_external_residual_bytes", "dfc_fiber_capacity_bytes",
            "model_scale_external_residual_removed_bytes", "initial_accuracy", "final_accuracy",
            "final_eval_loss", "wall_seconds", "checkpoint_bytes", "parameter_sha256",
            "semantic_optimizer_sha256", "logical_residual_sha256", "result_sha256")}
        save_manifest()
    if row["success"] and ckpt.exists():
        ckpt.unlink()
    return row


def compare_pair(tag, ext_name, dfc_name):
    ep, dp = OUT/f"{ext_name}.json", OUT/f"{dfc_name}.json"
    if not ep.exists() or not dp.exists(): return
    e, d = json.loads(ep.read_text()), json.loads(dp.read_text())
    comparison = {
        "tag": tag,
        "resource_contract_equal": all(e.get(k) == d.get(k) for k in (
            "model", "resolved_hub_revision", "trainable_parameters", "train_last_layers",
            "updates", "batch_size", "seq_len", "lr", "weight_decay", "grad_clip",
            "keep_ratio", "ef_chunk_coordinates", "optimizer_chunk_coordinates",
            "transmitted_values", "dense_gradient_values")),
        "parameter_digest_equal": e.get("parameter_sha256") == d.get("parameter_sha256"),
        "semantic_optimizer_digest_equal": e.get("semantic_optimizer_sha256") == d.get("semantic_optimizer_sha256"),
        "logical_residual_digest_equal": e.get("logical_residual_sha256") == d.get("logical_residual_sha256"),
        "final_accuracy_equal": e.get("final_accuracy") == d.get("final_accuracy"),
        "final_eval_loss_equal": e.get("final_eval_loss") == d.get("final_eval_loss"),
        "external_residual_bytes": e.get("actual_external_residual_bytes"),
        "dfc_fiber_capacity_bytes": d.get("dfc_fiber_capacity_bytes"),
        "dfc_model_scale_external_removed_bytes": d.get("model_scale_external_residual_removed_bytes"),
        "peak_train_allocated_external": e.get("training_peak_memory",{}).get("max_allocated"),
        "peak_train_allocated_dfc": d.get("training_peak_memory",{}).get("max_allocated"),
        "peak_train_reserved_external": e.get("training_peak_memory",{}).get("max_reserved"),
        "peak_train_reserved_dfc": d.get("training_peak_memory",{}).get("max_reserved"),
        "setup_allocated_external": e.get("memory_after_residual_setup",{}).get("allocated"),
        "setup_allocated_dfc": d.get("memory_after_residual_setup",{}).get("allocated"),
        "final_accuracy_external": e.get("final_accuracy"),
        "final_accuracy_dfc": d.get("final_accuracy"),
        "checkpoint_bytes_external": e.get("checkpoint_bytes"),
        "checkpoint_bytes_dfc": d.get("checkpoint_bytes"),
        "wall_seconds_external": e.get("wall_seconds"),
        "wall_seconds_dfc": d.get("wall_seconds"),
    }
    comparison["setup_allocated_saved_bytes"] = (
        comparison["setup_allocated_external"] - comparison["setup_allocated_dfc"]
        if comparison["setup_allocated_external"] is not None and comparison["setup_allocated_dfc"] is not None else None)
    comparison["peak_allocated_saved_bytes"] = (
        comparison["peak_train_allocated_external"] - comparison["peak_train_allocated_dfc"]
        if comparison["peak_train_allocated_external"] is not None and comparison["peak_train_allocated_dfc"] is not None else None)
    comparison["exact_trajectory_gate"] = all(comparison[k] for k in (
        "resource_contract_equal", "parameter_digest_equal",
        "semantic_optimizer_digest_equal", "logical_residual_digest_equal",
        "final_accuracy_equal", "final_eval_loss_equal"))
    (OUT/f"pair_{tag}.json").write_text(json.dumps(comparison,indent=2,sort_keys=True)+"\n")
    MANIFEST.setdefault("pairs", []).append(comparison); save_manifest()
    if not comparison["exact_trajectory_gate"]:
        raise RuntimeError(f"paired placement-only exactness failed: {tag}")


def archive(status, error=None):
    MANIFEST["status"] = status
    MANIFEST["error"] = error
    save_manifest()
    shutil.make_archive(str(WORK/"dfc_ef_qwen_results"), "zip", OUT)


try:
    shell("nvidia_smi", ["nvidia-smi"], required=True)
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()[0]
    MANIFEST["gpu_name"] = gpu_name; save_manifest()
    probe = shell("torch_preflight_before", [sys.executable, "-c",
        "import torch,json; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'arch_list':torch.cuda.get_arch_list()}))"], required=False)
    if "P100" in gpu_name or "sm_60" not in probe.get("stdout_tail", ""):
        shell("install_p100_torch", [sys.executable,"-m","pip","install","--no-cache-dir",
              "--upgrade","--force-reinstall","torch==2.7.1","--index-url",
              "https://download.pytorch.org/whl/cu126"], required=True)
    shell("torch_preflight_after", [sys.executable, "-c",
          "import torch,json; d={'torch':torch.__version__,'cuda':torch.version.cuda,'name':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'arch_list':torch.cuda.get_arch_list()}; print(json.dumps(d)); assert torch.cuda.is_available(); assert torch.cuda.get_device_capability(0)[0]!=6 or 'sm_60' in torch.cuda.get_arch_list()"], required=True)

    # Kaggle's preinstalled torchvision can be ABI-incompatible after replacing
    # torch for Pascal support. Qwen is text-only, so remove optional vision/audio
    # packages rather than letting transformers import a stale torchvision::nms.
    shell("remove_optional_binary_stacks", [sys.executable,"-m","pip","uninstall","-y",
          "torchvision","torchaudio"], required=False)

    if CHECKOUT.exists(): shutil.rmtree(CHECKOUT)
    shell("clone", ["git","clone","--depth","1","--branch",BRANCH,REPO,str(CHECKOUT)], required=True)
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=CHECKOUT,text=True).strip()
    MANIFEST["source_commit"] = commit; save_manifest()

    shell("install_model_stack", [sys.executable,"-m","pip","install","-q",
          "transformers>=4.51,<5","huggingface_hub>=0.30,<1","sentencepiece","accelerate","numpy"], required=True)
    shell("qwen_import_preflight", [sys.executable,"-c",
          "import torch,transformers,json; from transformers import Qwen2ForCausalLM,AutoTokenizer; print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'qwen_class':Qwen2ForCausalLM.__name__}))"], required=True)

    # A: checkpointed smoke: real-model restart + persistent-state delta.
    qwen_job("smoke_external", "external_ef", 1801, layers=2, updates=16, batch=2, checkpoint=True, required=True)
    qwen_job("smoke_dfc", "dfc_ef", 1801, layers=2, updates=16, batch=2, checkpoint=True, required=True)
    compare_pair("smoke", "smoke_external", "smoke_dfc")

    # B: strongest primary free-GPU result: all transformer blocks trainable,
    # three independent sealed seeds.
    for seed in (1901, 1931, 1951):
        e = f"primary_external_{seed}"; d = f"primary_dfc_{seed}"
        qwen_job(e, "external_ef", seed, layers=0, updates=128, batch=2, required=(seed==1901))
        qwen_job(d, "dfc_ef", seed, layers=0, updates=128, batch=2, required=(seed==1901))
        compare_pair(f"primary_{seed}", e, d)

    # C: causal ablations at the same envelope.
    for method in ("fp32_dense", "low16_dense", "low16_noef"):
        qwen_job(f"ablation_{method}", method, 1901, layers=0, updates=128, batch=2)

    # D: scale extensions only if free time remains.
    if enough(75*60):
        for method in ("external_ef", "dfc_ef"):
            qwen_job(f"qwen15b_{method}", method, 2111, model="Qwen/Qwen2.5-1.5B",
                     revision="main", layers=4, updates=64, batch=1)
        compare_pair("qwen15b", "qwen15b_external_ef", "qwen15b_dfc_ef")

    if enough(100*60):
        for method in ("external_ef", "dfc_ef"):
            qwen_job(f"qwen3b_{method}", method, 2203, model="Qwen/Qwen2.5-3B",
                     revision="main", layers=2, updates=32, batch=1)
        compare_pair("qwen3b", "qwen3b_external_ef", "qwen3b_dfc_ef")

    archive("PASS")
    print(json.dumps({"source_commit": commit, "elapsed_hours": (time.time()-T0)/3600,
                      "jobs": len(MANIFEST["jobs"]), "pairs": len(MANIFEST.get("pairs",[]))}, indent=2))
except BaseException as exc:
    tb = traceback.format_exc()
    (OUT/"fatal_traceback.txt").write_text(tb)
    print(tb, file=sys.stderr)
    archive("FAIL", f"{type(exc).__name__}: {exc}")
    raise
