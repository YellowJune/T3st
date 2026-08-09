#!/usr/bin/env python3
"""Full-FP32 DFC-Sign verbalizer adaptation on Qwen2.5-0.5B-Instruct.

Qwen's own next-token likelihood over four natural-language verbalizers is
optimized with FP32 LoRA and ordinary FP32 Adam moments. A fixed-size
token-and-dark-logit record is stored either in an actual external byte
envelope or in that envelope composed with the exact Adam second-moment sign
fiber. Batch shape, updates, dense tokens, model, trainable parameters,
optimizer tensors, and external physical bytes are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import random
import struct
import time
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F

from torch_fiber import DFCAdamW, TorchSignFiberChannel


METHODS = ("sequential", "external_derpp", "dfc_sign_derpp")
TASK_SPECS = (
    {
        "name": "ag_news", "repo": "fancyzhx/ag_news", "source_ref": "main",
        "config": None, "text_column": "text", "label_column": "label",
        "train_split": "train", "eval_split": "test",
        "label_map": {0: 0, 1: 1, 2: 2, 3: 3},
        "verbalizer_candidates": (
            ("world", "global"), ("sports", "sport"),
            ("business", "finance"), ("science", "technology"),
        ),
    },
    {
        "name": "emotion", "repo": "dair-ai/emotion", "source_ref": "main",
        "config": "split", "text_column": "text", "label_column": "label",
        "train_split": "train", "eval_split": "test",
        "label_map": {0: 0, 1: 1, 3: 2, 4: 3},
        "verbalizer_candidates": (
            ("sad", "sadness"), ("joy", "happy"),
            ("anger", "angry"), ("fear", "afraid"),
        ),
    },
    {
        "name": "banking77", "repo": "mteb/banking77", "source_ref": "main",
        "config": None, "text_column": "text", "label_column": "label",
        "train_split": "train", "eval_split": "test",
        "label_map": {0: 0, 8: 1, 24: 2, 44: 3},
        "verbalizer_candidates": (
            ("activate", "activation"), ("cancel", "cancellation"),
            ("country", "countries"), ("passcode", "password"),
        ),
    },
    {
        "name": "trec", "repo": "CogComp/trec",
        "source_ref": "refs/convert/parquet", "config": None,
        "text_column": "text", "label_column": "coarse_label",
        "train_split": "train", "eval_split": "test",
        "label_map": {1: 0, 2: 1, 3: 2, 5: 3},
        "verbalizer_candidates": (
            ("entity", "thing"), ("description", "definition"),
            ("human", "person"), ("number", "numeric"),
        ),
    },
)
TASK_NAMES = tuple(spec["name"] for spec in TASK_SPECS)
TRAIN_PER_CLASS = 32
EVAL_PER_CLASS = 32
MANIFEST: dict = {}
TASK_LABEL_TOKEN_IDS: list[tuple[int, ...]] = []


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Example:
    task: int
    input_ids: tuple[int, ...]
    answer_index: int
    target_id: int
    top_ids: tuple[int, ...] = (0, 0, 0, 0)
    top_logits: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)


class LLMRecordCodec:
    MAGIC = b"QLM1"
    VERSION = 1
    MAX_LENGTH = 48
    TOP_K = 4
    RECORD_BYTES = 256
    HEADER = struct.Struct("<4sBBHHI")

    @classmethod
    def encode(cls, example: Example) -> bytes:
        if len(example.input_ids) > cls.MAX_LENGTH:
            raise ValueError("token record exceeds fixed maximum length")
        if not 0 < example.answer_index < len(example.input_ids):
            raise ValueError("invalid answer index")
        body = bytearray(cls.RECORD_BYTES)
        cls.HEADER.pack_into(body, 0, cls.MAGIC, cls.VERSION, example.task,
                             len(example.input_ids), example.answer_index, example.target_id)
        struct.pack_into("<4I", body, 16, *example.top_ids)
        struct.pack_into("<4e", body, 32, *example.top_logits)
        padded = list(example.input_ids) + [0] * (cls.MAX_LENGTH - len(example.input_ids))
        struct.pack_into(f"<{cls.MAX_LENGTH}I", body, 40, *padded)
        crc = zlib.crc32(body[:-4]) & 0xFFFFFFFF
        struct.pack_into("<I", body, cls.RECORD_BYTES - 4, crc)
        return bytes(body)

    @classmethod
    def decode(cls, raw: bytes) -> Example:
        if len(raw) != cls.RECORD_BYTES:
            raise RuntimeError("record length mismatch")
        expected = struct.unpack_from("<I", raw, cls.RECORD_BYTES - 4)[0]
        if (zlib.crc32(raw[:-4]) & 0xFFFFFFFF) != expected:
            raise RuntimeError("record CRC mismatch")
        magic, version, task, length, answer_index, target_id = cls.HEADER.unpack_from(raw, 0)
        if magic != cls.MAGIC or version != cls.VERSION or length > cls.MAX_LENGTH:
            raise RuntimeError("record header mismatch")
        top_ids = struct.unpack_from("<4I", raw, 16)
        top_logits = struct.unpack_from("<4e", raw, 32)
        all_ids = struct.unpack_from(f"<{cls.MAX_LENGTH}I", raw, 40)
        return Example(task, tuple(all_ids[:length]), answer_index, target_id,
                       tuple(top_ids), tuple(float(value) for value in top_logits))


class ByteStore(Protocol):
    @property
    def capacity(self) -> int: ...
    def read(self, start: int, count: int) -> bytes: ...
    def write(self, start: int, payload: bytes) -> None: ...


class ExternalByteStore:
    def __init__(self, storage: bytearray):
        self.storage = storage

    @property
    def capacity(self) -> int:
        return len(self.storage)

    def read(self, start: int, count: int) -> bytes:
        if start < 0 or count < 0 or start + count > self.capacity:
            raise IndexError("external read out of bounds")
        return bytes(self.storage[start:start + count])

    def write(self, start: int, payload: bytes) -> None:
        if start < 0 or start + len(payload) > self.capacity:
            raise IndexError("external write out of bounds")
        self.storage[start:start + len(payload)] = payload


class CompositeByteStore:
    """Concatenate the common external allocation and an optimizer sign fiber."""
    def __init__(self, external: bytearray, channel: TorchSignFiberChannel):
        self.external = ExternalByteStore(external)
        self.channel = channel

    @property
    def capacity(self) -> int:
        return self.external.capacity + self.channel.byte_capacity

    def read(self, start: int, count: int) -> bytes:
        if start < 0 or count < 0 or start + count > self.capacity:
            raise IndexError("composite read out of bounds")
        if count == 0:
            return b""
        split = self.external.capacity
        if start >= split:
            return self.channel.read_bytes(start - split, count)
        first = min(count, split - start)
        return self.external.read(start, first) + self.channel.read_bytes(0, count - first)

    def write(self, start: int, payload: bytes) -> None:
        if start < 0 or start + len(payload) > self.capacity:
            raise IndexError("composite write out of bounds")
        if not payload:
            return
        split = self.external.capacity
        if start >= split:
            self.channel.write_bytes(start - split, payload)
            return
        first = min(len(payload), split - start)
        self.external.write(start, payload[:first])
        if first < len(payload):
            self.channel.write_bytes(0, payload[first:])


class PersistentReservoir:
    MAGIC = b"DFCLLM1\0"
    VERSION = 1
    HEADER_BYTES = 64
    HEADER = struct.Struct("<8sIIIIQI")

    def __init__(self, store: ByteStore, initialize: bool):
        self.store = store
        self.record_bytes = LLMRecordCodec.RECORD_BYTES
        self.record_capacity = max(0, (store.capacity - self.HEADER_BYTES) // self.record_bytes)
        if self.record_capacity < 1:
            raise ValueError("byte envelope cannot hold one record")
        self.count = 0
        self.seen = 0
        if initialize:
            self._write_header()
        else:
            self._read_header()

    def _header_payload(self, crc: int = 0) -> bytes:
        raw = bytearray(self.HEADER_BYTES)
        self.HEADER.pack_into(raw, 0, self.MAGIC, self.VERSION, self.record_bytes,
                              self.record_capacity, self.count, self.seen, crc)
        return bytes(raw)

    def _write_header(self) -> None:
        raw = bytearray(self._header_payload(0))
        crc = zlib.crc32(raw[:self.HEADER.size - 4]) & 0xFFFFFFFF
        struct.pack_into("<I", raw, self.HEADER.size - 4, crc)
        self.store.write(0, bytes(raw))

    def _read_header(self) -> None:
        raw = self.store.read(0, self.HEADER_BYTES)
        magic, version, record_bytes, capacity, count, seen, crc = self.HEADER.unpack_from(raw, 0)
        actual = zlib.crc32(raw[:self.HEADER.size - 4]) & 0xFFFFFFFF
        if (magic, version, record_bytes, capacity, crc) != (
            self.MAGIC, self.VERSION, self.record_bytes, self.record_capacity, actual
        ):
            raise RuntimeError("reservoir header mismatch")
        if count > capacity:
            raise RuntimeError("reservoir count exceeds capacity")
        self.count, self.seen = int(count), int(seen)

    def _offset(self, index: int) -> int:
        if not 0 <= index < self.record_capacity:
            raise IndexError("record index out of range")
        return self.HEADER_BYTES + index * self.record_bytes

    def insert(self, example: Example, rng: np.random.Generator) -> None:
        self.seen += 1
        if self.count < self.record_capacity:
            index = self.count
            self.count += 1
        else:
            candidate = int(rng.integers(0, self.seen))
            if candidate >= self.record_capacity:
                self._write_header()
                return
            index = candidate
        self.store.write(self._offset(index), LLMRecordCodec.encode(example))
        self._write_header()

    def sample(self, rng: np.random.Generator) -> Example:
        if self.count == 0:
            raise IndexError("cannot sample empty reservoir")
        index = int(rng.integers(0, self.count))
        return LLMRecordCodec.decode(self.store.read(self._offset(index), self.record_bytes))

    def digest(self) -> str:
        used = self.HEADER_BYTES + self.count * self.record_bytes
        return sha256_bytes(self.store.read(0, used))


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = float(alpha) / rank
        self.lora_a = torch.nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base(inputs)
        update = F.linear(F.linear(inputs.float(), self.lora_a), self.lora_b) * self.scaling
        return base + update.to(base.dtype)


def inject_lora(model: torch.nn.Module, rank: int, alpha: float) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if not (
            name.endswith("self_attn.q_proj")
            or name.endswith("self_attn.v_proj")
            or name.endswith("self_attn.o_proj")
            or name.endswith("mlp.down_proj")
        ):
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module, rank, alpha))
        targets.append(name)
    if not targets:
        raise RuntimeError("no Qwen projection modules were replaced")
    return targets


def adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def adapter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter_state(model).items()):
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def canonical_sha(payload: dict) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def select_rows(dataset, spec: dict, per_class: int, split_name: str):
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


def choose_verbalizers(tokenizer, spec: dict) -> tuple[tuple[str, ...], tuple[int, ...]]:
    words, token_ids = [], []
    for candidates in spec["verbalizer_candidates"]:
        chosen = None
        for word in candidates:
            encoded = tokenizer.encode(" " + word, add_special_tokens=False)
            if len(encoded) == 1 and int(encoded[0]) not in token_ids:
                chosen = (word, int(encoded[0]))
                break
        if chosen is None:
            raise RuntimeError(f'no one-token verbalizer for {spec["name"]}: {candidates}')
        words.append(chosen[0])
        token_ids.append(chosen[1])
    return tuple(words), tuple(token_ids)


def encode_rows(tokenizer, spec: dict, task_index: int, rows,
                words: tuple[str, ...], token_ids: tuple[int, ...]):
    examples = []
    labels = ", ".join(words)
    for _, text, target in rows:
        prompt = (
            f'Classify {spec["name"]}. Reply with exactly one word from '
            f'[{labels}]. Text: {text}\nAnswer:'
        )
        prompt_ids = tuple(
            int(value)
            for value in tokenizer.encode(
                prompt, add_special_tokens=False, truncation=True,
                max_length=LLMRecordCodec.MAX_LENGTH - 1,
            )
        )
        if not prompt_ids:
            raise RuntimeError("tokenizer emitted an empty prompt")
        ids = prompt_ids + (int(token_ids[target]),)
        examples.append(Example(task_index, ids, len(prompt_ids), int(target)))
    return examples


def build_tasks(tokenizer, keys_per_task: int):
    if keys_per_task != 4:
        raise ValueError("sealed verbalizer protocol uses four classes")
    from datasets import load_dataset

    TASK_LABEL_TOKEN_IDS.clear()
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
        train_rows = select_rows(
            dataset[spec["train_split"]], spec, TRAIN_PER_CLASS, spec["train_split"]
        )
        eval_rows = select_rows(
            dataset[spec["eval_split"]], spec, EVAL_PER_CLASS, spec["eval_split"]
        )
        words, token_ids = choose_verbalizers(tokenizer, spec)
        TASK_LABEL_TOKEN_IDS.append(token_ids)
        train_tasks.append(
            encode_rows(tokenizer, spec, task_index, train_rows, words, token_ids)
        )
        eval_tasks.append(
            encode_rows(tokenizer, spec, task_index, eval_rows, words, token_ids)
        )
        declarations.append(
            {
                "name": spec["name"], "repo": spec["repo"],
                "revision": revision, "source_ref": spec["source_ref"],
                "config": spec["config"], "label_map": spec["label_map"],
                "verbalizers": words, "verbalizer_token_ids": token_ids,
                "train_subset_sha256": canonical_sha(
                    {"rows": [(key, target) for key, _, target in train_rows]}
                ),
                "eval_subset_sha256": canonical_sha(
                    {"rows": [(key, target) for key, _, target in eval_rows]}
                ),
            }
        )
    declaration = {
        "protocol": "hf_public_verbalizer_v1",
        "train_per_class": TRAIN_PER_CLASS,
        "eval_per_class": EVAL_PER_CLASS,
        "tasks": declarations,
    }
    metadata = {
        "suite": "hf_public_verbalizer_v1", "num_labels": 4,
        "label_ids": [0, 1, 2, 3],
        "mapping_sha256": canonical_sha(declaration),
        "manifest_sha256": MANIFEST["manifest_sha256"],
        "train_examples_per_task": 4 * TRAIN_PER_CLASS,
        "evaluation_examples_per_task": 4 * EVAL_PER_CLASS,
        "repositories": declarations,
    }
    return train_tasks, eval_tasks, metadata


def pack_batch(examples: list[Example], pad_token_id: int, device: torch.device):
    batch = len(examples)
    inputs = torch.full((batch, LLMRecordCodec.MAX_LENGTH), pad_token_id,
                        dtype=torch.long, device=device)
    attention = torch.zeros_like(inputs)
    answer_positions, targets, candidates = [], [], []
    for index, example in enumerate(examples):
        length = len(example.input_ids)
        inputs[index, :length] = torch.tensor(example.input_ids, dtype=torch.long, device=device)
        attention[index, :length] = 1
        answer_positions.append(example.answer_index - 1)
        targets.append(example.target_id)
        candidates.append(TASK_LABEL_TOKEN_IDS[example.task])
    return (
        inputs,
        attention,
        torch.tensor(answer_positions, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
        torch.tensor(candidates, dtype=torch.long, device=device),
    )


@torch.no_grad()
def evaluate(model, tasks: list[list[Example]], learned: int, pad_token_id: int,
             device: torch.device, batch_size: int = 8):
    model.eval()
    accuracies, nlls = [], []
    for task_index in range(learned + 1):
        correct, count, nll_sum = 0, 0, 0.0
        examples = tasks[task_index]
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            inputs, attention, positions, targets, candidates = pack_batch(
                batch, pad_token_id, device
            )
            logits = model(input_ids=inputs, attention_mask=attention, use_cache=False).logits
            vocabulary_logits = logits[
                torch.arange(len(batch), device=device), positions
            ].float()
            selected = vocabulary_logits.gather(1, candidates)
            nll_sum += float(F.cross_entropy(selected, targets, reduction="sum"))
            correct += int((selected.argmax(dim=-1) == targets).sum())
            count += len(batch)
        accuracies.append(correct / count)
        nlls.append(nll_sum / count)
    return accuracies, nlls


def rebuild_reservoir(method: str, external: bytearray, optimizer: DFCAdamW,
                      initialize: bool) -> PersistentReservoir | None:
    if method == "sequential":
        return None
    if method == "dfc_sign_derpp":
        store: ByteStore = CompositeByteStore(external, TorchSignFiberChannel(optimizer))
    else:
        store = ExternalByteStore(external)
    return PersistentReservoir(store, initialize=initialize)


def checkpoint_roundtrip(model, optimizer, external: bytearray, method: str,
                         reservoir: PersistentReservoir | None):
    before = reservoir.digest() if reservoir is not None else sha256_bytes(bytes(external))
    buffer = io.BytesIO()
    torch.save({
        "adapter": adapter_state(model),
        "optimizer": optimizer.state_dict(),
        "external": bytes(external),
    }, buffer)
    checkpoint_bytes = buffer.tell()
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in restored["adapter"].items():
            named[name].copy_(value)
    optimizer.load_state_dict(restored["optimizer"])
    external[:] = restored["external"]
    rebuilt = rebuild_reservoir(method, external, optimizer, initialize=False) if reservoir else None
    after = rebuilt.digest() if rebuilt is not None else sha256_bytes(bytes(external))
    if before != after:
        raise AssertionError("checkpoint changed replay payload")
    return rebuilt, {"payload_sha256": before, "bytes": checkpoint_bytes, "passed": True}


def metric_summary(accuracy_matrix: list[list[float | None]],
                   nll_matrix: list[list[float | None]]) -> dict[str, float]:
    final_accuracy = [float(value) for value in accuracy_matrix[-1] if value is not None]
    final_nll = [float(value) for value in nll_matrix[-1] if value is not None]
    forgetting = []
    nll_forgetting = []
    for task in range(len(final_accuracy) - 1):
        learned_accuracy = [float(accuracy_matrix[row][task]) for row in range(task, len(accuracy_matrix))]
        learned_nll = [float(nll_matrix[row][task]) for row in range(task, len(nll_matrix))]
        forgetting.append(max(learned_accuracy) - final_accuracy[task])
        nll_forgetting.append(final_nll[task] - min(learned_nll))
    diagonal = [float(accuracy_matrix[index][index]) for index in range(len(accuracy_matrix))]
    return {
        "final_average_accuracy": float(np.mean(final_accuracy)),
        "final_average_nll": float(np.mean(final_nll)),
        "average_forgetting": float(np.mean(forgetting)),
        "average_nll_forgetting": float(np.mean(nll_forgetting)),
        "current_task_accuracy": final_accuracy[-1],
        "mean_learning_accuracy": float(np.mean(diagonal)),
    }


def run(args) -> dict:
    if args.method not in METHODS:
        raise ValueError(args.method)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                              trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float32,
        trust_remote_code=False, attn_implementation="eager",
    ).to(device)
    model.config.use_cache = False
    targets = inject_lora(model, args.rank, args.alpha)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    optimizer = DFCAdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.999),
                         eps=1e-8, weight_decay=args.weight_decay,
                         enable_fiber=args.method == "dfc_sign_derpp")
    external = bytearray(args.external_bytes)
    reservoir = rebuild_reservoir(args.method, external, optimizer, initialize=True)
    train_tasks, eval_tasks, dataset_metadata = build_tasks(tokenizer, args.keys_per_task)
    current_rng = np.random.default_rng(args.seed + 10_001)
    replay_rng = np.random.default_rng(args.seed + 20_003)
    reservoir_rng = np.random.default_rng(args.seed + 30_007)

    accuracy_matrix: list[list[float | None]] = []
    nll_matrix: list[list[float | None]] = []
    payload_boundaries = []
    checkpoint_gate = None
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    for task_index in range(len(TASK_NAMES)):
        model.train()
        for update in range(args.updates_per_task):
            current = train_tasks[task_index][int(current_rng.integers(0, len(train_tasks[task_index])))]
            batch = [current]
            replay_example = None
            if reservoir is not None and reservoir.count:
                replay_example = reservoir.sample(replay_rng)
                batch.append(replay_example)
            else:
                batch.append(train_tasks[task_index][int(current_rng.integers(
                    0, len(train_tasks[task_index])))])
            inputs, attention, positions, target_ids, candidate_ids = pack_batch(
                batch, tokenizer.pad_token_id, device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=inputs, attention_mask=attention, use_cache=False).logits
            vocabulary_logits = logits[
                torch.arange(args.batch_size, device=device), positions
            ].float()
            selected = vocabulary_logits.gather(1, candidate_ids)
            loss = F.cross_entropy(selected, target_ids)
            if replay_example is not None:
                top_ids = torch.tensor(replay_example.top_ids, dtype=torch.long, device=device)
                teacher = torch.tensor(replay_example.top_logits, dtype=torch.float32, device=device)
                replay_logits = selected[1].gather(0, top_ids)
                loss = loss + args.distill_weight * F.mse_loss(replay_logits, teacher)
            teacher_values, teacher_ids = torch.topk(selected[0].detach(), LLMRecordCodec.TOP_K)
            inserted = replace(
                current,
                top_ids=tuple(int(value) for value in teacher_ids.cpu()),
                top_logits=tuple(float(value) for value in teacher_values.to(torch.float16).cpu()),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            if reservoir is not None:
                reservoir.insert(inserted, reservoir_rng)
                if update % 8 == 0:
                    reservoir._read_header()

        accuracies, nlls = evaluate(model, eval_tasks, task_index,
                                    tokenizer.pad_token_id, device)
        accuracy_matrix.append(accuracies + [None] * (len(TASK_NAMES) - len(accuracies)))
        nll_matrix.append(nlls + [None] * (len(TASK_NAMES) - len(nlls)))
        payload_boundaries.append({
            "task": task_index,
            "count": reservoir.count if reservoir else 0,
            "seen": reservoir.seen if reservoir else 0,
            "sha256": reservoir.digest() if reservoir else sha256_bytes(bytes(external)),
        })
        if task_index == 1:
            reservoir, checkpoint_gate = checkpoint_roundtrip(
                model, optimizer, external, args.method, reservoir
            )

    wall_seconds = time.perf_counter() - start_wall
    cpu_seconds = time.process_time() - start_cpu
    metrics = metric_summary(accuracy_matrix, nll_matrix)
    model_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    state_bytes = trainable_count * 8
    fiber_bytes = trainable_count // 8 if args.method == "dfc_sign_derpp" else 0
    logical_bytes = args.external_bytes + fiber_bytes if reservoir is not None else 0
    record_capacity = max(0, (logical_bytes - PersistentReservoir.HEADER_BYTES) //
                          LLMRecordCodec.RECORD_BYTES) if reservoir is not None else 0
    dense_tokens = len(TASK_NAMES) * args.updates_per_task * args.batch_size * LLMRecordCodec.MAX_LENGTH
    total_model_parameters = sum(parameter.numel() for parameter in model.parameters())
    counted_flops = 6 * total_model_parameters * dense_tokens
    payload = {
        "schema": "dfc-qwen-verbalizer-v1",
        "method": args.method,
        "seed": args.seed,
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None) or args.revision,
        "tokenizer_class": tokenizer.__class__.__name__,
        "task_names": list(TASK_NAMES),
        "dataset": dataset_metadata,
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "target_modules": targets,
            "trainable_parameters": trainable_count,
        },
        "protocol": {
            "tasks": len(TASK_NAMES),
            "keys_per_task": args.keys_per_task,
            "updates_per_task": args.updates_per_task,
            "total_updates": len(TASK_NAMES) * args.updates_per_task,
            "batch_size": args.batch_size,
            "current_per_update": args.batch_size if args.method == "sequential" else 1,
            "replay_per_update": 0 if args.method == "sequential" else 1,
            "maximum_length": LLMRecordCodec.MAX_LENGTH,
            "dense_tokens": dense_tokens,
            "counted_dense_neural_flops": counted_flops,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "distill_weight": args.distill_weight,
        },
        "resources": {
            "model_physical_bytes": model_bytes,
            "optimizer_state_physical_bytes": state_bytes,
            "external_allocated_bytes": len(external),
            "common_persistent_bytes": model_bytes + state_bytes + len(external),
            "internal_sign_fiber_bytes": fiber_bytes,
            "logical_replay_bytes": logical_bytes,
            "record_bytes": LLMRecordCodec.RECORD_BYTES,
            "record_capacity": record_capacity,
            "final_record_count": reservoir.count if reservoir else 0,
        },
        "accuracy_matrix": accuracy_matrix,
        "nll_matrix": nll_matrix,
        "metrics": metrics,
        "payload_boundaries": payload_boundaries,
        "checkpoint_gate": checkpoint_gate,
        "digests": {
            "adapter_sha256": adapter_digest(model),
            "final_payload_sha256": reservoir.digest() if reservoir else sha256_bytes(bytes(external)),
            "source_sha256": sha256_file(Path(__file__)),
            "optimizer_source_sha256": sha256_file(Path(__file__).with_name("torch_fiber.py")),
        },
        "runtime": {"wall_seconds": wall_seconds, "cpu_seconds_energy_proxy": cpu_seconds},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": str(device),
            "threads": args.threads,
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_sha": os.environ.get("GITHUB_SHA"),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["result_sha256"] = sha256_bytes(canonical)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--keys-per-task", type=int, default=4)
    parser.add_argument("--updates-per-task", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2, choices=[2])
    parser.add_argument("--external-bytes", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
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
    if claimed_manifest_sha != canonical_sha(unhashed_manifest):
        raise RuntimeError("dataset manifest SHA256 mismatch")
    if MANIFEST["model"]["repo"] != args.model:
        raise RuntimeError("model manifest identity mismatch")
    if MANIFEST["model"]["revision"] != args.revision:
        raise RuntimeError("model manifest revision mismatch")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "method": result["method"],
        "seed": result["seed"],
        "metrics": result["metrics"],
        "resources": result["resources"],
        "result_sha256": result["result_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
