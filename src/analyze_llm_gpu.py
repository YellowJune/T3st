#!/usr/bin/env python3
"""Validate accepted Qwen/GPU evidence and make publication artifacts.

This script is deliberately fail-closed: it produces no figure or TeX table
unless both predeclared dominance gates passed and all expected raw Qwen cells
are present.  It does not rerun or reinterpret either acceptance test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("sequential", "external_derpp", "dfc_sign_derpp")
LABELS = {
    "sequential": "Sequential",
    "external_derpp": "External DER++",
    "dfc_sign_derpp": "DFC-Sign + DER++",
}
COLORS = {
    "sequential": "#777777",
    "external_derpp": "#4477AA",
    "dfc_sign_derpp": "#009988",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_qwen(raw_dir: Path, aggregate_path: Path) -> tuple[list[dict], dict]:
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    require(aggregate["schema"] == "dfc-qwen-continual-aggregate-v4", "Qwen aggregate schema")
    require(aggregate["dominance_gate"]["passed"] is True, "Qwen gate did not pass")
    require(aggregate["cells"] == 9, "Qwen aggregate is not a 3x3 matrix")
    expected_hashes = aggregate["source_sha256"]
    candidates: dict[str, Path] = {}
    for path in raw_dir.rglob("qwen_*.json"):
        if path.name == "qwen_aggregate.json":
            continue
        if path.name in candidates:
            require(sha256(path) == sha256(candidates[path.name]), f"conflicting duplicate {path.name}")
        else:
            candidates[path.name] = path
    require(set(candidates) == set(expected_hashes), "Qwen raw-file set differs from accepted aggregate")
    rows = []
    for name, expected in expected_hashes.items():
        path = candidates[name]
        require(sha256(path) == expected, f"Qwen raw hash {name}")
        row = json.loads(path.read_text(encoding="utf-8"))
        require(row["environment"]["github_sha"] == aggregate["accepted_commit_sha"],
                f"Qwen commit binding {name}")
        rows.append(row)
    require({row["method"] for row in rows} == set(METHODS), "Qwen method set")
    require({int(row["seed"]) for row in rows} == set(aggregate["seeds"]), "Qwen seed set")
    return rows, aggregate


def load_gpu(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    require(result["schema"] == "dfc-triton-gpu-v1", "GPU schema")
    require(result["acceptance_gate"]["passed"] is True, "GPU <=5% gate did not pass")
    require(result["device"]["gpu_name"], "GPU identity absent")
    require(result["device"]["github_run_id"], "GPU workflow binding absent")
    for row in result["results"]:
        require(all(row["exactness"].values()), f"GPU exactness at {row['elements']}")
        require(len(row["paired_ratio"]) == 9, f"GPU paired rounds at {row['elements']}")
    return result


def mean_matrix(rows: list[dict], method: str) -> np.ndarray:
    selected = [row for row in rows if row["method"] == method]
    matrices = []
    for row in selected:
        matrices.append(np.asarray([
            [np.nan if value is None else float(value) for value in line]
            for line in row["accuracy_matrix"]
        ]))
    return np.nanmean(np.stack(matrices), axis=0)


def save(fig, figures: Path, stem: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_qwen(rows: list[dict], aggregate: dict, figures: Path) -> None:
    fig = plt.figure(figsize=(11.6, 3.8))
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.15), wspace=0.36)
    for index, method in enumerate(("external_derpp", "dfc_sign_derpp")):
        ax = fig.add_subplot(grid[0, index])
        matrix = 100 * mean_matrix(rows, method)
        image = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                if not np.isnan(matrix[row, col]):
                    ax.text(col, row, f"{matrix[row, col]:.0f}", ha="center", va="center",
                            fontsize=8, color="white" if matrix[row, col] < 58 else "black")
        ax.set_xticks(range(4), ["O", "C", "M", "S"])
        ax.set_yticks(range(4), ["O", "C", "M", "S"])
        ax.set_xlabel("Evaluated task")
        ax.set_ylabel("After learning task")
        ax.set_title(LABELS[method])
    colorbar = fig.colorbar(image, ax=fig.axes[:2], fraction=0.035, pad=0.025)
    colorbar.set_label("Accuracy (%)")

    ax = fig.add_subplot(grid[0, 2])
    keys = ("final_average_accuracy", "current_task_accuracy")
    x = np.arange(len(keys))
    width = 0.36
    for offset, method in zip((-width / 2, width / 2), ("external_derpp", "dfc_sign_derpp")):
        means = [100 * aggregate["aggregate"][method][key]["mean"] for key in keys]
        sems = [100 * aggregate["aggregate"][method][key]["sem"] for key in keys]
        ax.bar(x + offset, means, width, yerr=sems, capsize=3,
               label=LABELS[method], color=COLORS[method])
    ax.set_xticks(x, ["Final average", "Current task"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    gate = aggregate["dominance_gate"]
    ax.set_title(
        f"Gain {gate['final_accuracy_gain_pp']:+.1f} pp\n"
        f"Forgetting reduction {gate['forgetting_reduction_pp']:+.1f} pp"
    )
    save(fig, figures, "fig17_qwen_fullfp32")


def plot_gpu(gpu: dict, figures: Path) -> None:
    rows = sorted(gpu["results"], key=lambda row: row["elements"])
    x = np.asarray([row["elements"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.8))
    for key, label, color in (
        ("matched_reference_ms", "Matched Triton AdamW", "#4477AA"),
        ("dfc_sign_ms", "DFC-Sign Triton AdamW", "#009988"),
        ("pytorch_fused_ms", "PyTorch fused AdamW", "#AA3377"),
    ):
        medians = np.asarray([np.median(row[key]) for row in rows])
        lower = np.asarray([np.percentile(row[key], 25) for row in rows])
        upper = np.asarray([np.percentile(row[key], 75) for row in rows])
        axes[0].errorbar(x, medians, yerr=np.vstack((medians - lower, upper - medians)),
                         marker="o", capsize=3, lw=2, label=label, color=color)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("FP32 coordinates")
    axes[0].set_ylabel("Kernel time (ms)")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8)
    axes[0].set_title(gpu["device"]["gpu_name"])

    overhead = 100 * (np.asarray([np.median(row["paired_ratio"]) for row in rows]) - 1)
    axes[1].plot(x, overhead, marker="o", color="#009988", lw=2)
    axes[1].axhline(5.0, color="#CC3311", linestyle="--", label="Sealed 5% ceiling")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("FP32 coordinates")
    axes[1].set_ylabel("Paired median overhead (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    primary = gpu["acceptance_gate"]
    axes[1].set_title(f"Primary overhead {primary['observed_overhead_percent']:.2f}%")
    save(fig, figures, "fig18_triton_actual_gpu")


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_tex(aggregate: dict, gpu: dict, output: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Full-FP32 Qwen2.5-0.5B continual adaptation under one physical envelope. Values are mean$\pm$SEM over three sealed seeds.}",
        r"\label{tab:qwen_fullfp32}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Final acc. (\%) & Forgetting (pp) & Current (\%) \\",
        r"\midrule",
    ]
    for method in METHODS:
        block = aggregate["aggregate"][method]
        lines.append(
            f"{LABELS[method]} & "
            f"{100*block['final_average_accuracy']['mean']:.2f}$\\pm${100*block['final_average_accuracy']['sem']:.2f} & "
            f"{100*block['average_forgetting']['mean']:.2f}$\\pm${100*block['average_forgetting']['sem']:.2f} & "
            f"{100*block['current_task_accuracy']['mean']:.2f}$\\pm${100*block['current_task_accuracy']['sem']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "", r"\begin{table}[t]",
              r"\centering", r"\caption{Actual-GPU fused AdamW benchmark.}",
              r"\label{tab:triton_gpu}", r"\begin{tabular}{rrrr}", r"\toprule",
              r"Coordinates & Reference (ms) & DFC-Sign (ms) & Overhead (\%) \\", r"\midrule"]
    for row in sorted(gpu["results"], key=lambda item: item["elements"]):
        lines.append(
            f"{row['elements']:,} & {row['median_matched_reference_ms']:.4f} & "
            f"{row['median_dfc_sign_ms']:.4f} & {row['median_overhead_percent']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}",
              f"\\par\\smallskip\\footnotesize GPU: {tex_escape(gpu['device']['gpu_name'])}; "
              f"workflow {tex_escape(str(gpu['device']['github_run_id']))}.", r"\end{table}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-raw", type=Path, required=True)
    parser.add_argument("--qwen-aggregate", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--tex", type=Path, default=Path("paper/generated/llm_gpu_tables.tex"))
    parser.add_argument("--summary", type=Path, default=Path("results/summary/llm_gpu_v3.json"))
    args = parser.parse_args()

    rows, aggregate = load_qwen(args.qwen_raw, args.qwen_aggregate)
    gpu = load_gpu(args.gpu)
    plot_qwen(rows, aggregate, args.figures)
    plot_gpu(gpu, args.figures)
    write_tex(aggregate, gpu, args.tex)
    summary = {
        "schema": "dfc-llm-gpu-publication-v1",
        "qwen": {
            "run_id": aggregate["accepted_workflow_run"],
            "commit_sha": aggregate["accepted_commit_sha"],
            "resolved_revision": aggregate["resolved_revision"],
            "dominance_gate": aggregate["dominance_gate"],
            "aggregate": aggregate["aggregate"],
        },
        "gpu": {
            "run_id": gpu["device"]["github_run_id"],
            "commit_sha": gpu["device"]["github_sha"],
            "device": gpu["device"],
            "acceptance_gate": gpu["acceptance_gate"],
            "results": gpu["results"],
        },
        "evidence_sha256": {
            "qwen_aggregate": sha256(args.qwen_aggregate),
            "gpu": sha256(args.gpu),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
