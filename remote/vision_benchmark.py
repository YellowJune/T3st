#!/usr/bin/env python3
"""Resource-matched Split CIFAR-100 benchmark for DFC sign-fiber replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import resource
import struct
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor

from torch_fiber import DFCAdamW, TorchSignFiberChannel


class ArrayChannel:
    def __init__(self, capacity: int):
        self.storage = bytearray(max(0, int(capacity)))

    @property
    def byte_capacity(self) -> int:
        return len(self.storage)

    def read_bytes(self, start: int, count: int) -> bytes:
        return bytes(self.storage[start : start + count])

    def write_bytes(self, start: int, payload) -> None:
        raw = bytes(payload)
        self.storage[start : start + len(raw)] = raw


class CompositeChannel:
    def __init__(self, channels):
        self.channels = tuple(channels)
        self.ends = np.cumsum([channel.byte_capacity for channel in channels], dtype=np.int64)

    @property
    def byte_capacity(self) -> int:
        return int(self.ends[-1])

    def _segments(self, start: int, count: int):
        if start < 0 or count < 0 or start + count > self.byte_capacity:
            raise IndexError("channel range out of bounds")
        position, remaining = int(start), int(count)
        while remaining:
            index = int(np.searchsorted(self.ends, position, side="right"))
            base = 0 if index == 0 else int(self.ends[index - 1])
            local = position - base
            take = min(remaining, self.channels[index].byte_capacity - local)
            yield self.channels[index], local, take
            position += take
            remaining -= take

    def read_bytes(self, start: int, count: int) -> bytes:
        return b"".join(c.read_bytes(local, take) for c, local, take in self._segments(start, count))

    def write_bytes(self, start: int, payload) -> None:
        raw, cursor = bytes(payload), 0
        for channel, local, take in self._segments(start, len(raw)):
            channel.write_bytes(local, raw[cursor : cursor + take])
            cursor += take


class CIFAR4BitCodec:
    MAGIC = b"CF4B"
    SIDE = 16
    RECORD_BYTES = 400

    @classmethod
    def encode(cls, x: torch.Tensor, y: int, task: int) -> bytes:
        small = F.interpolate(
            x.detach().cpu().unsqueeze(0), size=(cls.SIDE, cls.SIDE),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        q = torch.clamp(torch.round(small.reshape(-1) * 15.0), 0, 15).to(torch.uint8).numpy()
        packed = q[0::2] | (q[1::2] << np.uint8(4))
        raw = bytearray(cls.RECORD_BYTES)
        raw[:4] = cls.MAGIC
        raw[4:6] = int(y).to_bytes(2, "little")
        raw[6:8] = int(task).to_bytes(2, "little")
        payload_end = 8 + packed.size
        raw[8:payload_end] = packed.tobytes()
        raw[payload_end : payload_end + 4] = (
            zlib.crc32(raw[:payload_end]) & 0xFFFFFFFF
        ).to_bytes(4, "little")
        return bytes(raw)

    @classmethod
    def decode(cls, raw: bytes) -> tuple[torch.Tensor, int, int]:
        if len(raw) != cls.RECORD_BYTES or raw[:4] != cls.MAGIC:
            raise RuntimeError("invalid CIFAR replay record")
        packed_bytes = 3 * cls.SIDE * cls.SIDE // 2
        payload_end = 8 + packed_bytes
        if zlib.crc32(raw[:payload_end]) & 0xFFFFFFFF != int.from_bytes(
            raw[payload_end : payload_end + 4], "little"
        ):
            raise RuntimeError("CIFAR replay checksum mismatch")
        packed = np.frombuffer(raw[8:payload_end], dtype=np.uint8)
        q = np.empty(3 * cls.SIDE * cls.SIDE, dtype=np.uint8)
        q[0::2], q[1::2] = packed & 15, packed >> 4
        small = torch.from_numpy(q.astype(np.float32).reshape(3, cls.SIDE, cls.SIDE) / 15.0)
        x = F.interpolate(
            small.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False
        ).squeeze(0)
        return x, int.from_bytes(raw[4:6], "little"), int.from_bytes(raw[6:8], "little")


class FiberReservoir:
    HEADER_BYTES = 16
    MAGIC = b"CVR2"

    def __init__(self, channel):
        self.channel = channel
        self.capacity = max(0, (channel.byte_capacity - self.HEADER_BYTES) // CIFAR4BitCodec.RECORD_BYTES)
        raw = channel.read_bytes(0, min(self.HEADER_BYTES, channel.byte_capacity))
        if len(raw) == self.HEADER_BYTES and raw[:4] == self.MAGIC:
            self.size = int.from_bytes(raw[4:8], "little")
            self.seen = int.from_bytes(raw[8:16], "little")
        else:
            self.size = self.seen = 0
            if channel.byte_capacity >= self.HEADER_BYTES:
                self._header()

    def _header(self) -> None:
        self.channel.write_bytes(
            0, self.MAGIC + self.size.to_bytes(4, "little") + self.seen.to_bytes(8, "little")
        )

    def _offset(self, index: int) -> int:
        return self.HEADER_BYTES + index * CIFAR4BitCodec.RECORD_BYTES

    def add(self, x: torch.Tensor, y: int, task: int, rng: np.random.Generator) -> None:
        if self.capacity == 0:
            return
        self.seen += 1
        if self.size < self.capacity:
            index = self.size
            self.size += 1
        else:
            index = int(rng.integers(0, self.seen))
            if index >= self.capacity:
                self._header()
                return
        self.channel.write_bytes(self._offset(index), CIFAR4BitCodec.encode(x, y, task))
        self._header()

    def sample(self, count: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        indices = rng.integers(0, self.size, size=count)
        items = [
            CIFAR4BitCodec.decode(
                self.channel.read_bytes(self._offset(int(index)), CIFAR4BitCodec.RECORD_BYTES)
            )
            for index in indices
        ]
        return torch.stack([item[0] for item in items]), torch.tensor([item[1] for item in items])


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class CifarResNet(nn.Module):
    def __init__(self, width: int = 32, blocks: int = 2, classes: int = 100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU()
        )
        channels = width
        stages = []
        for output, stride in ((width, 1), (2 * width, 2), (4 * width, 2)):
            stages.append(BasicBlock(channels, output, stride))
            channels = output
            for _ in range(blocks - 1):
                stages.append(BasicBlock(channels, output))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(channels, classes)
        self.register_buffer("input_mean", torch.tensor((0.5071, 0.4867, 0.4408)).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor((0.2675, 0.2565, 0.2761)).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.input_mean) / self.input_std
        x = self.stages(self.stem(x))
        return self.head(torch.mean(x, dim=(2, 3)))


@torch.no_grad()
def dense_macs_per_example(model: nn.Module) -> int:
    """Count convolution/linear MACs for the fixed 32x32 model shape."""
    total = 0
    hooks = []

    def hook(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            per_output = module.in_channels // module.groups
            per_output *= module.kernel_size[0] * module.kernel_size[1]
            total += int(output[0].numel()) * per_output
        elif isinstance(module, nn.Linear):
            total += module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    training = model.training
    model.eval()
    model(torch.zeros(1, 3, 32, 32, device=next(model.parameters()).device))
    model.train(training)
    for handle in hooks:
        handle.remove()
    return total


class RaplMeter:
    def __init__(self):
        self.files = sorted(Path("/sys/class/powercap").glob("**/energy_uj"))
        self.before = []

    def start(self):
        self.before = [int(path.read_text()) for path in self.files]

    def stop(self):
        if not self.files:
            return None
        after = [int(path.read_text()) for path in self.files]
        return sum(max(0, b - a) for a, b in zip(after, self.before)) / 1e6


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        prediction = model(x.to(device)).argmax(1).cpu()
        correct += int((prediction == y).sum())
        total += y.numel()
    return correct / total


def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    train = CIFAR100(args.data, train=True, download=True, transform=ToTensor())
    test = CIFAR100(args.data, train=False, download=True, transform=ToTensor())
    train_targets, test_targets = np.asarray(train.targets), np.asarray(test.targets)
    train_tasks = [np.flatnonzero((train_targets // 10) == task).tolist() for task in range(10)]
    test_tasks = [
        DataLoader(Subset(test, np.flatnonzero((test_targets // 10) == task).tolist()),
                   batch_size=256, shuffle=False, num_workers=2)
        for task in range(10)
    ]

    model = CifarResNet(width=args.width).to(device)
    macs_per_example = dense_macs_per_example(model)
    optimizer = DFCAdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        enable_fiber=args.method == "dfc_sign_er"
    )
    external = ArrayChannel(args.memory_bytes)
    if args.method == "dfc_sign_er":
        fiber = TorchSignFiberChannel(optimizer)
        channel = CompositeChannel([fiber, external])
        internal_bytes = fiber.byte_capacity
    elif args.method == "er":
        channel, internal_bytes = external, 0
    else:
        channel, internal_bytes = ArrayChannel(0), 0
    reservoir = FiberReservoir(channel)
    loss_fn = nn.CrossEntropyLoss()
    replay_rng = np.random.default_rng(args.seed + 900_001)
    reservoir_rng = np.random.default_rng(args.seed + 1_900_001)
    matrix = np.full((10, 10), np.nan, dtype=np.float64)
    meter = RaplMeter()
    meter.start()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    updates = examples = 0

    for task_index, indices in enumerate(train_tasks):
        generator = torch.Generator().manual_seed(args.seed * 100 + task_index)
        loader = DataLoader(
            Subset(train, indices), batch_size=args.batch_size, shuffle=True,
            generator=generator, num_workers=2, drop_last=True
        )
        iterator = iter(loader)
        model.train()
        for _ in range(args.updates_per_task):
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                x, y = next(iterator)
            n_replay = min(args.replay_count, args.batch_size - 1) if reservoir.size else 0
            if n_replay:
                x_old, y_old = reservoir.sample(n_replay, replay_rng)
                x = torch.cat([x[: args.batch_size - n_replay], x_old], 0)
                y = torch.cat([y[: args.batch_size - n_replay], y_old], 0)
            else:
                x, y = x[: args.batch_size], y[: args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            # Class-incremental single-head training; unseen classes are masked,
            # while all classes observed so far compete in the same softmax.
            seen_classes = 10 * (task_index + 1)
            loss = loss_fn(model(x.to(device))[:, :seen_classes], y.to(device))
            loss.backward()
            optimizer.step()
            updates += 1
            examples += args.batch_size

        insertion_order = reservoir_rng.permutation(indices)
        for index in insertion_order:
            x, y = train[int(index)]
            reservoir.add(x, int(y), task_index, reservoir_rng)
        for old_task in range(task_index + 1):
            matrix[task_index, old_task] = evaluate(model, test_tasks[old_task], device)

    wall, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
    joules = meter.stop()
    final = matrix[-1]
    forgetting = np.mean([np.nanmax(matrix[i:, i]) - final[i] for i in range(9)])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "dataset": "CIFAR-100 class-incremental 10x10",
        "method": args.method,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "model": f"CifarResNet-width{args.width}-blocks2",
        "final_average_accuracy": float(np.mean(final)),
        "average_forgetting": float(forgetting),
        "current_task_accuracy": float(matrix[-1, -1]),
        "mean_learning_accuracy": float(np.mean(np.diag(matrix))),
        "accuracy_matrix": matrix.tolist(),
        "accuracy_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "external_bytes": args.memory_bytes,
        "internal_fiber_bytes": internal_bytes,
        "base_persistent_bytes": 12 * parameter_count,
        "common_envelope_bytes": 12 * parameter_count + args.memory_bytes,
        "replay_capacity": reservoir.capacity,
        "replay_size": reservoir.size,
        "record_bytes": CIFAR4BitCodec.RECORD_BYTES,
        "batch_size": args.batch_size,
        "updates": updates,
        "examples_processed": examples,
        "dense_macs_per_example": macs_per_example,
        "neural_flops": 6 * macs_per_example * examples,
        "replay_per_update": args.replay_count,
        "input_codec": "16x16 RGB, 4-bit, bilinear reconstruction, CRC32",
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "rapl_joules": joules,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "torch_version": torch.__version__,
        "device": str(device),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("naive", "er", "dfc_sign_er"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-task", type=int, default=120)
    parser.add_argument("--replay-count", type=int, default=32)
    parser.add_argument("--memory-bytes", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
