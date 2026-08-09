"""Executed Qwen-0.5B continual adaptation on AG News for full-FP32 DFC-SIGN.

Four class-incremental tasks are learned sequentially with LoRA plus the sequence
classifier. External DER++ and DFC-SIGN+DER++ use the same 512-byte external
allocation, batch shape, update count, model, and ordinary FP32 AdamW state.
DFC-SIGN only addresses the sign fiber already resident in FP32 second moments.
"""
from __future__ import annotations

import argparse, binascii, hashlib, json, random, struct, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch_fiber import DFCAdamW, TorchSignFiberChannel

SEQ_LEN = 24
N_LABELS = 4
HEADER_BYTES = 32
RECORD_BYTES = 120
RECORD_MAGIC = b"QAG1"
STORE_MAGIC = b"DFCAGN1\0"


class ReplayCodec:
    @staticmethod
    def encode(input_ids, attention_mask, label, task, logits):
        ids = input_ids.detach().cpu().to(torch.int64).reshape(-1)
        mask = attention_mask.detach().cpu().to(torch.uint8).reshape(-1)
        vals = logits.detach().cpu().to(torch.float16).reshape(-1)
        if ids.numel() != SEQ_LEN or mask.numel() != SEQ_LEN or vals.numel() != N_LABELS:
            raise ValueError("fixed codec shape required")
        mask_bits = sum((int(b) & 1) << i for i, b in enumerate(mask.tolist()))
        body = bytearray(RECORD_MAGIC)
        body.extend(struct.pack("<BBHI", int(task), int(label), 0, mask_bits))
        body.extend(np.asarray(ids, dtype="<u4").tobytes())
        body.extend(np.asarray(vals, dtype="<f2").tobytes())
        if len(body) != RECORD_BYTES - 4:
            raise AssertionError(len(body))
        body.extend(struct.pack("<I", binascii.crc32(body) & 0xFFFFFFFF))
        return bytes(body)

    @staticmethod
    def decode(raw):
        if len(raw) != RECORD_BYTES:
            raise ValueError("bad record length")
        body = raw[:-4]
        if (binascii.crc32(body) & 0xFFFFFFFF) != struct.unpack("<I", raw[-4:])[0]:
            raise RuntimeError("record CRC mismatch")
        if body[:4] != RECORD_MAGIC:
            raise RuntimeError("record magic mismatch")
        task, label, _, mask_bits = struct.unpack("<BBHI", body[4:12])
        cursor = 12
        ids = np.frombuffer(body[cursor:cursor + 4 * SEQ_LEN], dtype="<u4").astype(np.int64)
        cursor += 4 * SEQ_LEN
        vals = np.frombuffer(body[cursor:cursor + 2 * N_LABELS], dtype="<f2").astype(np.float32)
        mask = np.asarray([(mask_bits >> i) & 1 for i in range(SEQ_LEN)], dtype=np.int64)
        return {
            "task": int(task),
            "label": int(label),
            "input_ids": torch.from_numpy(ids.copy()),
            "attention_mask": torch.from_numpy(mask.copy()),
            "logits": torch.from_numpy(vals.copy()),
        }


class CombinedByteChannel:
    def __init__(self, external_bytes, fiber):
        if external_bytes < HEADER_BYTES:
            raise ValueError("external envelope too small")
        self.external = bytearray(external_bytes)
        self.fiber = fiber
        self.fiber_bytes = 0 if fiber is None else fiber.byte_capacity

    @property
    def byte_capacity(self):
        return len(self.external) + self.fiber_bytes

    def read(self, start, count):
        if start < 0 or count < 0 or start + count > self.byte_capacity:
            raise IndexError
        out = bytearray(); pos = start; remaining = count
        if pos < len(self.external) and remaining:
            take = min(remaining, len(self.external) - pos)
            out.extend(self.external[pos:pos + take]); pos += take; remaining -= take
        if remaining:
            out.extend(self.fiber.read_bytes(pos - len(self.external), remaining))
        return bytes(out)

    def write(self, start, payload):
        raw = bytes(payload)
        if start < 0 or start + len(raw) > self.byte_capacity:
            raise IndexError
        pos = start; cursor = 0; remaining = len(raw)
        if pos < len(self.external) and remaining:
            take = min(remaining, len(self.external) - pos)
            self.external[pos:pos + take] = raw[cursor:cursor + take]
            pos += take; cursor += take; remaining -= take
        if remaining:
            self.fiber.write_bytes(pos - len(self.external), raw[cursor:])


class ReservoirStore:
    def __init__(self, channel, rng):
        self.channel = channel
        self.rng = rng
        self.capacity_records = max(0, (channel.byte_capacity - HEADER_BYTES) // RECORD_BYTES)
        self._write_header(0, 0)

    def _header(self, count, seen):
        body = bytearray(STORE_MAGIC)
        body.extend(struct.pack("<IIIII", 1, RECORD_BYTES, self.capacity_records, count, seen))
        body.extend(struct.pack("<I", binascii.crc32(body) & 0xFFFFFFFF))
        return bytes(body)

    def _write_header(self, count, seen):
        self.channel.write(0, self._header(count, seen))

    def _read_header(self):
        raw = self.channel.read(0, HEADER_BYTES); body = raw[:-4]
        if (binascii.crc32(body) & 0xFFFFFFFF) != struct.unpack("<I", raw[-4:])[0]:
            raise RuntimeError("header CRC mismatch")
        if body[:8] != STORE_MAGIC:
            raise RuntimeError("header magic mismatch")
        version, record_bytes, capacity, count, seen = struct.unpack("<IIIII", body[8:28])
        if (version, record_bytes, capacity) != (1, RECORD_BYTES, self.capacity_records):
            raise RuntimeError("header schema mismatch")
        return int(count), int(seen)

    @property
    def count(self): return self._read_header()[0]
    @property
    def seen(self): return self._read_header()[1]
    def _offset(self, index): return HEADER_BYTES + index * RECORD_BYTES

    def insert(self, record):
        count, seen = self._read_header(); seen += 1
        if self.capacity_records == 0:
            self._write_header(count, seen); return
        if count < self.capacity_records:
            slot = count; count += 1
        else:
            slot = int(self.rng.integers(0, seen))
            if slot >= self.capacity_records:
                self._write_header(count, seen); return
        self.channel.write(self._offset(slot), record)
        self._write_header(count, seen)

    def sample(self):
        count, _ = self._read_header()
        if count == 0: return None
        index = int(self.rng.integers(0, count))
        return ReplayCodec.decode(self.channel.read(self._offset(index), RECORD_BYTES))

    def digest(self):
        count, _ = self._read_header()
        return hashlib.sha256(self.channel.read(0, HEADER_BYTES + count * RECORD_BYTES)).hexdigest()


@dataclass(frozen=True)
class Example:
    task: int
    text: str
    label: int


def load_stream(seed: int, train_per_class: int, eval_per_class: int):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    dataset_id = "fancyzhx/ag_news"
    revision = HfApi().dataset_info(dataset_id).sha
    ds = load_dataset(dataset_id, revision=revision)
    rng = np.random.default_rng(seed + 31007)
    tasks = []
    canonical = []
    train_labels = np.asarray(ds["train"]["label"])
    test_labels = np.asarray(ds["test"]["label"])
    for label in range(N_LABELS):
        train_indices = np.flatnonzero(train_labels == label)
        eval_indices = np.flatnonzero(test_labels == label)
        rng.shuffle(train_indices); rng.shuffle(eval_indices)
        train = [Example(label, str(ds["train"][int(i)]["text"]), label) for i in train_indices[:train_per_class]]
        evaluation = [Example(label, str(ds["test"][int(i)]["text"]), label) for i in eval_indices[:eval_per_class]]
        tasks.append((train, evaluation))
        canonical.append({"label": label, "train": [x.text for x in train], "eval": [x.text for x in evaluation]})
    subset_sha = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return tasks, dataset_id, revision, subset_sha


def tokenize(tokenizer, text, device):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=SEQ_LEN,
        padding="max_length",
    )
    return encoded["input_ids"][0].to(device), encoded["attention_mask"][0].to(device)


def logits_of(model, ids, mask):
    return model(input_ids=ids, attention_mask=mask, use_cache=False).logits


@torch.inference_mode()
def evaluate(model, tokenizer, tasks, device, upto):
    model.eval(); result = []
    for task_index, (_, evaluation) in enumerate(tasks):
        if task_index > upto:
            result.append(float("nan")); continue
        correct = 0; total = 0
        for start in range(0, len(evaluation), 16):
            chunk = evaluation[start:start + 16]
            ids, masks = zip(*(tokenize(tokenizer, ex.text, device) for ex in chunk))
            pred = logits_of(model, torch.stack(ids), torch.stack(masks)).argmax(-1).cpu().numpy()
            labels = np.asarray([ex.label for ex in chunk])
            correct += int(np.sum(pred == labels)); total += len(chunk)
        result.append(correct / total)
    model.train()
    return result


def metrics(matrix):
    final = float(np.nanmean(matrix[-1]))
    current = float(np.nanmean(np.diag(matrix)))
    forgetting = []
    for task in range(matrix.shape[0] - 1):
        history = matrix[task:, task]
        forgetting.append(float(np.nanmax(history) - history[-1]))
    return {
        "final_average_accuracy": final,
        "current_task_accuracy": current,
        "average_forgetting": float(np.mean(forgetting)),
    }


def run(args):
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model_revision = HfApi().model_info(args.model).sha
    tasks, dataset_id, dataset_revision, subset_sha = load_stream(args.seed, args.train_per_class, args.eval_per_class)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
        num_labels=N_LABELS,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    n_layers = int(model.config.num_hidden_layers)
    first = max(0, n_layers - args.lora_last_layers)
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.0,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["score"],
        layers_to_transform=list(range(first, n_layers)),
        layers_pattern="layers",
    )
    model = get_peft_model(model, config).to(device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_n = sum(p.numel() for p in trainable)
    optimizer = DFCAdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
        enable_fiber=args.method == "dfc_sign_derpp",
    )
    fiber = TorchSignFiberChannel(optimizer) if args.method == "dfc_sign_derpp" else None
    store = ReservoirStore(
        CombinedByteChannel(args.external_bytes, fiber),
        np.random.default_rng(args.seed + 10003),
    )

    T = len(tasks)
    matrix = np.full((T, T), np.nan, dtype=np.float64)
    rng = np.random.default_rng(args.seed + 20003)
    updates = 0; losses = []; started = time.perf_counter()
    for task_index, (train_examples, _) in enumerate(tasks):
        for _ in range(args.steps_per_task):
            current = train_examples[int(rng.integers(0, len(train_examples)))]
            current_ids, current_mask = tokenize(tokenizer, current.text, device)
            replay = store.sample() if args.method != "naive" else None
            if replay is None:
                replay_ids, replay_mask, replay_label = current_ids.clone(), current_mask.clone(), current.label
                replay_logits = None
            else:
                replay_ids = replay["input_ids"].to(device)
                replay_mask = replay["attention_mask"].to(device)
                replay_label = int(replay["label"])
                replay_logits = replay["logits"].to(device)

            logits = logits_of(model, torch.stack([current_ids, replay_ids]), torch.stack([current_mask, replay_mask]))
            loss = F.cross_entropy(logits[0:1], torch.tensor([current.label], device=device))
            if replay is not None and args.method != "naive":
                loss = (
                    loss
                    + args.replay_ce_weight * F.cross_entropy(logits[1:2], torch.tensor([replay_label], device=device))
                    + args.distill_weight * F.mse_loss(logits[1].float(), replay_logits.float())
                )
            else:
                loss = loss + 0.0 * logits[1].sum()

            record = ReplayCodec.encode(current_ids, current_mask, current.label, current.task, logits[0].detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            updates += 1; losses.append(float(loss.detach()))
            if args.method != "naive": store.insert(record)
        matrix[task_index] = np.asarray(evaluate(model, tokenizer, tasks, device, task_index))

    result = {
        "schema_version": 1,
        "protocol": "qwen-agnews-classinc-lora-v1",
        "method": args.method,
        "seed": args.seed,
        "model": args.model,
        "requested_model_revision": args.revision,
        "resolved_model_revision": model_revision,
        "dataset": dataset_id,
        "resolved_dataset_revision": dataset_revision,
        "dataset_subset_sha256": subset_sha,
        "torch": torch.__version__,
        "trainable_parameters": int(trainable_n),
        "total_model_parameters": int(sum(p.numel() for p in model.parameters())),
        "lora_rank": args.lora_rank,
        "lora_last_layers": args.lora_last_layers,
        "external_bytes": args.external_bytes,
        "sign_fiber_bytes": 0 if fiber is None else fiber.byte_capacity,
        "record_bytes": RECORD_BYTES,
        "record_capacity": 0 if args.method == "naive" else store.capacity_records,
        "records_final": 0 if args.method == "naive" else store.count,
        "records_seen": 0 if args.method == "naive" else store.seen,
        "store_sha256": None if args.method == "naive" else store.digest(),
        "batch_size": 2,
        "steps_per_task": args.steps_per_task,
        "tasks": T,
        "updates": updates,
        "processed_text_slots": 2 * updates,
        "seq_len": SEQ_LEN,
        "num_labels": N_LABELS,
        "train_per_class": args.train_per_class,
        "eval_per_class": args.eval_per_class,
        "distill_weight": args.distill_weight,
        "replay_ce_weight": args.replay_ce_weight,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "accuracy_matrix": matrix.tolist(),
        **metrics(matrix),
        "mean_training_loss": float(np.mean(losses)),
        "wall_seconds": time.perf_counter() - started,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
    result["result_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["naive", "derpp", "dfc_sign_derpp"])
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps-per-task", type=int, default=64)
    parser.add_argument("--external-bytes", type=int, default=512)
    parser.add_argument("--train-per-class", type=int, default=32)
    parser.add_argument("--eval-per-class", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-last-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--replay-ce-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in [
        "protocol", "method", "seed", "final_average_accuracy", "average_forgetting",
        "current_task_accuracy", "record_capacity", "sign_fiber_bytes", "wall_seconds", "result_sha256"
    ]}, indent=2))


if __name__ == "__main__":
    main()
