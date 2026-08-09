#!/usr/bin/env python3
"""Pinned real-dataset Qwen continual adaptation using full-FP32 DFC-Sign.

This benchmark reuses the audited optimizer, record codec, checkpoint gate,
resource ledger, and training loop from llm_continual_benchmark.py.  It replaces
only the controlled semantic suite with deterministic, disjoint subsets of four
public text-classification datasets whose immutable Hugging Face revisions are
recorded by the workflow before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import llm_continual_benchmark as core


TASK_SPECS = (
    {
        "name": "ag_news",
        "repo": "fancyzhx/ag_news",
        "source_ref": "main",
        "config": None,
        "text_column": "text",
        "label_column": "label",
        "train_split": "train",
        "eval_split": "test",
        "label_map": {0: 0, 1: 1, 2: 2, 3: 3},
        "class_prompt": "0 world, 1 sports, 2 business, 3 science and technology",
    },
    {
        "name": "emotion",
        "repo": "dair-ai/emotion",
        "source_ref": "main",
        "config": "split",
        "text_column": "text",
        "label_column": "label",
        "train_split": "train",
        "eval_split": "test",
        "label_map": {0: 0, 1: 1, 3: 2, 4: 3},
        "class_prompt": "0 sadness, 1 joy, 2 anger, 3 fear",
    },
    {
        "name": "banking77",
        "repo": "mteb/banking77",
        "source_ref": "main",
        "config": None,
        "text_column": "text",
        "label_column": "label",
        "train_split": "train",
        "eval_split": "test",
        "label_map": {0: 0, 8: 1, 24: 2, 44: 3},
        "class_prompt": "0 activate card, 1 cancel transfer, 2 supported countries, 3 forgotten passcode",
    },
    {
        "name": "trec",
        "repo": "CogComp/trec",
        "source_ref": "refs/convert/parquet",
        "config": None,
        "text_column": "text",
        "label_column": "coarse_label",
        "train_split": "train",
        "eval_split": "test",
        "label_map": {1: 0, 2: 1, 3: 2, 5: 3},
        "class_prompt": "0 entity, 1 description, 2 human, 3 numeric answer",
    },
)
TRAIN_PER_CLASS = 32
EVAL_PER_CLASS = 32
MANIFEST: dict = {}


def _canonical_sha(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _select_rows(dataset, spec: dict, per_class: int, split_name: str):
    buckets = {target: [] for target in range(4)}
    reverse = {int(source): int(target) for source, target in spec["label_map"].items()}
    for row in dataset:
        source_label = int(row[spec["label_column"]])
        if source_label not in reverse:
            continue
        text = " ".join(str(row[spec["text_column"]]).split())
        if not text:
            continue
        target = reverse[source_label]
        key = hashlib.sha256(
            f'{spec["repo"]}\0{split_name}\0{source_label}\0{text}'.encode()
        ).hexdigest()
        buckets[target].append((key, text))
    selected = []
    for target in range(4):
        ordered = sorted(buckets[target])
        if len(ordered) < per_class:
            raise RuntimeError(
                f'{spec["name"]} {split_name} class {target} has '
                f'{len(ordered)} rows, need {per_class}'
            )
        selected.extend((key, text, target) for key, text in ordered[:per_class])
    selected.sort()
    return selected


def _encode_rows(tokenizer, spec: dict, task_index: int, rows):
    examples = []
    for _, text, target in rows:
        prompt = (
            f'Dataset {spec["name"]}. Classes are {spec["class_prompt"]}. '
            f'Classify: {text}'
        )
        ids = tuple(
            int(value)
            for value in tokenizer.encode(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=core.LLMRecordCodec.MAX_LENGTH,
            )
        )
        if not ids:
            raise RuntimeError("tokenizer emitted an empty prompt")
        examples.append(core.Example(task_index, ids, len(ids) - 1, int(target)))
    return examples


def build_real_tasks(tokenizer, keys_per_task: int, num_labels: int):
    if keys_per_task != 4 or num_labels != 4:
        raise ValueError("sealed public-dataset protocol uses four classes")
    from datasets import load_dataset

    expected_names = {spec["name"] for spec in TASK_SPECS}
    if set(MANIFEST["datasets"]) != expected_names:
        raise RuntimeError("dataset manifest task set mismatch")
    train_tasks, eval_tasks, declarations = [], [], []
    for task_index, spec in enumerate(TASK_SPECS):
        sealed = MANIFEST["datasets"][spec["name"]]
        if sealed["repo"] != spec["repo"] or sealed["source_ref"] != spec["source_ref"]:
            raise RuntimeError(f'dataset identity mismatch for {spec["name"]}')
        revision = sealed["revision"]
        if not isinstance(revision, str) or len(revision) < 40:
            raise RuntimeError(f'unpinned dataset revision for {spec["name"]}')
        kwargs = {"revision": revision}
        if spec["config"] is None:
            dataset = load_dataset(spec["repo"], **kwargs)
        else:
            dataset = load_dataset(spec["repo"], spec["config"], **kwargs)
        train_rows = _select_rows(
            dataset[spec["train_split"]], spec, TRAIN_PER_CLASS, spec["train_split"]
        )
        eval_rows = _select_rows(
            dataset[spec["eval_split"]], spec, EVAL_PER_CLASS, spec["eval_split"]
        )
        train_tasks.append(_encode_rows(tokenizer, spec, task_index, train_rows))
        eval_tasks.append(_encode_rows(tokenizer, spec, task_index, eval_rows))
        declarations.append(
            {
                "name": spec["name"],
                "repo": spec["repo"],
                "revision": revision,
                "source_ref": spec["source_ref"],
                "config": spec["config"],
                "label_map": spec["label_map"],
                "class_prompt": spec["class_prompt"],
                "train_subset_sha256": _canonical_sha(
                    {"rows": [(key, target) for key, _, target in train_rows]}
                ),
                "eval_subset_sha256": _canonical_sha(
                    {"rows": [(key, target) for key, _, target in eval_rows]}
                ),
            }
        )
    declaration = {
        "protocol": "hf_public_four_domain_v1",
        "train_per_class": TRAIN_PER_CLASS,
        "eval_per_class": EVAL_PER_CLASS,
        "tasks": declarations,
    }
    metadata = {
        "suite": "hf_public_four_domain_v1",
        "num_labels": 4,
        "label_ids": [0, 1, 2, 3],
        "mapping_sha256": _canonical_sha(declaration),
        "train_examples_per_task": 4 * TRAIN_PER_CLASS,
        "evaluation_examples_per_task": 4 * EVAL_PER_CLASS,
        "repositories": declarations,
        "manifest_sha256": MANIFEST["manifest_sha256"],
    }
    return train_tasks, eval_tasks, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=core.METHODS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--partial-layers", type=int, default=1)
    parser.add_argument("--keys-per-task", type=int, default=4)
    parser.add_argument("--num-labels", type=int, default=4, choices=[4])
    parser.add_argument("--head-width", type=int, default=256)
    parser.add_argument("--updates-per-task", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2, choices=[2])
    parser.add_argument("--external-bytes", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    global MANIFEST
    MANIFEST = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    claimed_manifest_sha = MANIFEST.get("manifest_sha256")
    unhashed_manifest = dict(MANIFEST)
    unhashed_manifest.pop("manifest_sha256", None)
    if claimed_manifest_sha != _canonical_sha(unhashed_manifest):
        raise RuntimeError("dataset manifest SHA256 mismatch")
    if MANIFEST["model"]["repo"] != args.model:
        raise RuntimeError("model manifest identity mismatch")
    if MANIFEST["model"]["revision"] != args.revision:
        raise RuntimeError("model manifest revision mismatch")

    core.TASK_NAMES = tuple(spec["name"] for spec in TASK_SPECS)
    core.build_tasks = build_real_tasks
    result = core.run(args)
    result["schema"] = "dfc-qwen-public-continual-v1"
    result["digests"]["public_dataset_wrapper_sha256"] = core.sha256_file(Path(__file__))
    result.pop("result_sha256", None)
    result["result_sha256"] = _canonical_sha(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "method": result["method"],
                "seed": result["seed"],
                "metrics": result["metrics"],
                "resources": result["resources"],
                "dataset": result["dataset"],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
