#!/usr/bin/env python3
"""Fail-closed aggregation for the sealed Qwen continual-adaptation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


METHODS = ["sequential", "external_derpp", "dfc_sign_derpp"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def recompute(row: dict) -> dict[str, float]:
    accuracy = row["accuracy_matrix"]
    nll = row["nll_matrix"]
    final_accuracy = [float(value) for value in accuracy[-1]]
    final_nll = [float(value) for value in nll[-1]]
    forgetting, nll_forgetting = [], []
    for task in range(len(final_accuracy) - 1):
        a = [float(accuracy[index][task]) for index in range(task, len(accuracy))]
        n = [float(nll[index][task]) for index in range(task, len(nll))]
        forgetting.append(max(a) - final_accuracy[task])
        nll_forgetting.append(final_nll[task] - min(n))
    return {
        "final_average_accuracy": float(np.mean(final_accuracy)),
        "final_average_nll": float(np.mean(final_nll)),
        "average_forgetting": float(np.mean(forgetting)),
        "average_nll_forgetting": float(np.mean(nll_forgetting)),
        "current_task_accuracy": final_accuracy[-1],
        "mean_learning_accuracy": float(np.mean([accuracy[i][i] for i in range(len(accuracy))])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--seeds", default="1319,1321,1327")
    parser.add_argument("--minimum-learning-accuracy", type=float, default=0.92)
    parser.add_argument("--minimum-accuracy-gain-pp", type=float, default=10.0)
    parser.add_argument("--minimum-forgetting-reduction-pp", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    paths = sorted(args.input.glob("qwen_*.json"))
    require(len(paths) == len(METHODS) * len(seeds), "incomplete LLM matrix")
    rows, hashes = [], {}
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        require(row["schema"] == "dfc-qwen-choice-v1", f"schema {path.name}")
        require(row["model"] == "Qwen/Qwen2.5-0.5B-Instruct",
                f"model {path.name}")
        claimed_result_sha = row.get("result_sha256")
        unhashed = dict(row)
        unhashed.pop("result_sha256", None)
        canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
        require(hashlib.sha256(canonical).hexdigest() == claimed_result_sha,
                f"result digest {path.name}")
        require(row["requested_revision"] == row["resolved_revision"],
                f"model revision {path.name}")
        require(row["dataset"]["suite"] == "hf_public_choice_v1",
                f"dataset suite {path.name}")
        require(row["dataset"]["train_examples_per_task"] == 128,
                f"train rows {path.name}")
        require(row["dataset"]["evaluation_examples_per_task"] == 128,
                f"test rows {path.name}")
        require(len(row["dataset"]["repositories"]) == 4,
                f"dataset repositories {path.name}")
        for task in row["dataset"]["repositories"]:
            require(task["verbalizers"] == ["A", "B", "C", "D"],
                    f"choice words {path.name}")
            require(len(set(task["verbalizer_token_ids"])) == 4,
                    f"choice token ids {path.name}")
        require(row["lora"]["rank"] == 16 and row["lora"]["alpha"] == 32.0,
                f"LoRA geometry {path.name}")
        require(len(row["lora"]["target_modules"]) == 96,
                f"LoRA targets {path.name}")
        protocol = row["protocol"]
        require(protocol["tasks"] == 4 and protocol["updates_per_task"] == 1024,
                f"task/update protocol {path.name}")
        require(protocol["total_updates"] == 4096 and protocol["batch_size"] == 2,
                f"total update protocol {path.name}")
        require(protocol["maximum_length"] == 48 and protocol["dense_tokens"] == 393216,
                f"token protocol {path.name}")
        require(protocol["learning_rate"] == 5e-4,
                f"learning rate {path.name}")
        require(protocol["distill_weight"] == 0.1,
                f"distillation coefficient {path.name}")
        require(row["resources"]["external_allocated_bytes"] == 2048,
                f"external bytes {path.name}")
        metrics = recompute(row)
        for key, value in metrics.items():
            require(math.isclose(value, row["metrics"][key], rel_tol=0.0, abs_tol=1e-10),
                    f"metric {key} {path.name}")
        require(row["checkpoint_gate"]["passed"] is True, f"checkpoint {path.name}")
        require(row["environment"]["github_sha"] == args.commit_sha, f"commit {path.name}")
        require(row["environment"]["github_run_id"] == args.run_id, f"run id {path.name}")
        require(row["environment"]["device"] == "cpu", f"device {path.name}")
        rows.append(row)
        hashes[path.name] = sha256(path)
    require({row["method"] for row in rows} == set(METHODS), "method set")
    require({int(row["seed"]) for row in rows} == set(seeds), "seed set")
    require(len({(row["method"], int(row["seed"])) for row in rows}) == len(rows), "duplicates")
    require(len({row["resolved_revision"] for row in rows}) == 1, "model revision mismatch")
    require(len({row["dataset"]["mapping_sha256"] for row in rows}) == 1, "dataset mismatch")
    require(len({row["dataset"]["manifest_sha256"] for row in rows}) == 1,
            "dataset manifest mismatch")
    require(len({json.dumps(row["dataset"]["repositories"], sort_keys=True) for row in rows}) == 1,
            "dataset subset mismatch")
    require(len({row["digests"]["source_sha256"] for row in rows}) == 1,
            "core source mismatch")
    require(len({row["digests"]["optimizer_source_sha256"] for row in rows}) == 1,
            "optimizer source mismatch")
    for seed in seeds:
        paired = [row for row in rows if int(row["seed"]) == seed]
        resource_keys = (
            "model_physical_bytes", "optimizer_state_physical_bytes",
            "external_allocated_bytes", "common_persistent_bytes",
        )
        for key in resource_keys:
            require(len({row["resources"][key] for row in paired}) == 1,
                    f"resource {key} seed {seed}")
        protocol_keys = ("total_updates", "batch_size", "maximum_length", "dense_tokens",
                         "counted_dense_neural_flops", "learning_rate",
                         "distill_weight")
        for key in protocol_keys:
            require(len({row["protocol"][key] for row in paired}) == 1,
                    f"protocol {key} seed {seed}")
        dfc = next(row for row in paired if row["method"] == "dfc_sign_derpp")
        require(dfc["lora"]["rank"] == 16 and dfc["lora"]["alpha"] == 32.0,
                f"LoRA geometry seed {seed}")
        p = int(dfc["lora"]["trainable_parameters"])
        require(dfc["resources"]["internal_sign_fiber_bytes"] == p // 8,
                f"sign capacity seed {seed}")
        require(dfc["resources"]["logical_replay_bytes"] == 2048 + p // 8,
                f"logical capacity seed {seed}")
        external = next(row for row in paired if row["method"] == "external_derpp")
        sequential = next(row for row in paired if row["method"] == "sequential")
        require(external["resources"]["logical_replay_bytes"] == 2048,
                f"external logical bytes seed {seed}")
        require(sequential["resources"]["logical_replay_bytes"] == 0,
                f"sequential logical bytes seed {seed}")
        require(all(boundary["sha256"] for boundary in dfc["payload_boundaries"]),
                f"payload boundary seed {seed}")

    aggregate = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        aggregate[method] = {
            key: {
                "mean": float(np.mean([row["metrics"][key] for row in selected])),
                "sem": float(np.std([row["metrics"][key] for row in selected], ddof=1) /
                             math.sqrt(len(selected))),
            }
            for key in selected[0]["metrics"]
        }
        aggregate[method]["record_capacity"] = selected[0]["resources"]["record_capacity"]
        aggregate[method]["mean_wall_seconds"] = float(np.mean(
            [row["runtime"]["wall_seconds"] for row in selected]
        ))

    external = aggregate["external_derpp"]
    dfc = aggregate["dfc_sign_derpp"]
    gate = {
        "final_accuracy_gain_pp": 100 * (
            dfc["final_average_accuracy"]["mean"] - external["final_average_accuracy"]["mean"]
        ),
        "forgetting_reduction_pp": 100 * (
            external["average_forgetting"]["mean"] - dfc["average_forgetting"]["mean"]
        ),
        "current_task_delta_pp": 100 * (
            dfc["current_task_accuracy"]["mean"] - external["current_task_accuracy"]["mean"]
        ),
        "final_nll_reduction": (
            external["final_average_nll"]["mean"] - dfc["final_average_nll"]["mean"]
        ),
        "dfc_mean_learning_accuracy": dfc["mean_learning_accuracy"]["mean"],
        "minimum_learning_accuracy": args.minimum_learning_accuracy,
        "minimum_accuracy_gain_pp": args.minimum_accuracy_gain_pp,
        "minimum_forgetting_reduction_pp": args.minimum_forgetting_reduction_pp,
    }
    gate["passed"] = bool(
        gate["dfc_mean_learning_accuracy"] >= args.minimum_learning_accuracy
        and gate["final_accuracy_gain_pp"] >= args.minimum_accuracy_gain_pp
        and gate["forgetting_reduction_pp"] >= args.minimum_forgetting_reduction_pp
        and gate["current_task_delta_pp"] >= -1.0
        and gate["final_nll_reduction"] >= 0
    )
    report = {
        "schema": "dfc-qwen-choice-aggregate-v1",
        "accepted_workflow_run": args.run_id,
        "accepted_commit_sha": args.commit_sha,
        "methods": METHODS,
        "seeds": seeds,
        "cells": len(rows),
        "resolved_revision": rows[0]["resolved_revision"],
        "dataset_manifest_sha256": rows[0]["dataset"]["manifest_sha256"],
        "dataset_mapping_sha256": rows[0]["dataset"]["mapping_sha256"],
        "all_resource_pairs_equal": True,
        "aggregate": aggregate,
        "dominance_gate": gate,
        "source_sha256": hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit("predeclared LLM dominance gate failed")


if __name__ == "__main__":
    main()
