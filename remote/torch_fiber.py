"""PyTorch implementation of exact FP32 sign-fiber AdamW.

This file has no dependency on the NumPy reference implementation.  Optimizer
states remain ordinary FP32 tensors, so state_dict checkpoints, AMP master
states, gradient accumulation, DDP, and FSDP local shards retain the payload in
the same tensors they already serialize or communicate.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch


SIGN_MASK_I32 = -2147483648
MAGNITUDE_MASK_I32 = 2147483647
LOW16_MASK_I32 = 65535
HIGH16_MASK_I32 = -65536


class DFCAdamW(torch.optim.Optimizer):
    """AdamW with one exact payload bit per FP32 second-moment coordinate."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        enable_fiber: bool = True,
    ):
        if lr < 0 or eps < 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("invalid AdamW hyperparameters")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.enable_fiber = bool(enable_fiber)
        self.initialize_state()

    @torch.no_grad()
    def initialize_state(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise TypeError("DFCAdamW expects floating parameters")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter, dtype=torch.float32, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, dtype=torch.float32, memory_format=torch.preserve_format
                    )

    def second_moment_tensors(self) -> list[torch.Tensor]:
        return [
            self.state[parameter]["exp_avg_sq"]
            for group in self.param_groups
            for parameter in group["params"]
        ]

    @property
    def payload_bit_capacity(self) -> int:
        return sum(tensor.numel() for tensor in self.second_moment_tensors()) if self.enable_fiber else 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("DFCAdamW does not support sparse gradients")
                gradient = gradient.float()
                state = self.state[parameter]
                state["step"] += 1
                step = state["step"]
                first = state["exp_avg"]
                physical_second = state["exp_avg_sq"]

                if self.enable_fiber:
                    physical_bits = physical_second.view(torch.int32)
                    payload = torch.bitwise_and(physical_bits, SIGN_MASK_I32)
                    second = physical_second.abs()
                else:
                    payload = None
                    second = physical_second

                first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                if self.enable_fiber:
                    second_bits = second.view(torch.int32)
                    packed_bits = torch.bitwise_or(
                        torch.bitwise_and(second_bits, MAGNITUDE_MASK_I32), payload
                    )
                    physical_second.copy_(packed_bits.view(torch.float32))
                    second_for_update = physical_second.abs()
                else:
                    second_for_update = physical_second

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = second_for_update.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.addcdiv_(first.to(parameter.dtype), denominator.to(parameter.dtype), value=-step_size)
        return loss


class TorchSignFiberChannel:
    """Byte-addressable sign fiber over a DFCAdamW state shard."""

    def __init__(self, optimizer: DFCAdamW):
        if not optimizer.enable_fiber:
            raise ValueError("optimizer fiber is disabled")
        self.optimizer = optimizer
        self.tensors = optimizer.second_moment_tensors()
        self.ends = np.cumsum([tensor.numel() for tensor in self.tensors], dtype=np.int64)

    @property
    def bit_capacity(self) -> int:
        return int(self.ends[-1]) if self.ends.size else 0

    @property
    def byte_capacity(self) -> int:
        return self.bit_capacity // 8

    def _segments(self, start: int, count: int):
        if start < 0 or count < 0 or start + count > self.bit_capacity:
            raise IndexError("fiber range out of bounds")
        position, remaining = int(start), int(count)
        while remaining:
            index = int(np.searchsorted(self.ends, position, side="right"))
            base = 0 if index == 0 else int(self.ends[index - 1])
            local = position - base
            take = min(remaining, self.tensors[index].numel() - local)
            yield self.tensors[index], local, take
            position += take
            remaining -= take

    @torch.no_grad()
    def read_bits(self, start: int, count: int) -> np.ndarray:
        pieces = []
        for tensor, local, take in self._segments(start, count):
            words = tensor.reshape(-1).view(torch.int32)[local : local + take]
            piece = torch.bitwise_and(torch.bitwise_right_shift(words, 31), 1)
            pieces.append(piece.to(torch.uint8).cpu().numpy())
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.uint8)

    @torch.no_grad()
    def write_bits(self, start: int, payload: np.ndarray) -> None:
        bits = np.ascontiguousarray(payload, dtype=np.uint8).reshape(-1)
        if np.any(bits > 1):
            raise ValueError("payload must be binary")
        cursor = 0
        for tensor, local, take in self._segments(start, bits.size):
            words = tensor.reshape(-1).view(torch.int32)
            incoming = torch.from_numpy(bits[cursor : cursor + take].astype(np.int32)).to(words.device)
            words[local : local + take] = torch.bitwise_or(
                torch.bitwise_and(words[local : local + take], MAGNITUDE_MASK_I32),
                torch.bitwise_left_shift(incoming, 31),
            )
            cursor += take

    def read_bytes(self, start: int, count: int) -> bytes:
        if start < 0 or count < 0 or start + count > self.byte_capacity:
            raise IndexError("fiber byte range out of bounds")
        return np.packbits(self.read_bits(start * 8, count * 8), bitorder="little").tobytes()

    def write_bytes(self, start: int, payload: bytes | bytearray | memoryview) -> None:
        raw = bytes(payload)
        if start < 0 or start + len(raw) > self.byte_capacity:
            raise IndexError("fiber byte range out of bounds")
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
        self.write_bits(start * 8, bits)


def local_fiber_capacity(optimizer: DFCAdamW) -> dict[str, int]:
    """Report capacity for a single optimizer/DDP/FSDP shard."""
    bits = optimizer.payload_bit_capacity
    return {"payload_bits": bits, "payload_bytes": bits // 8, "state_coordinates": bits}


class DFCLow16AdamW(torch.optim.Optimizer):
    """BF16-high semantic AdamW in FP32 containers with 16 payload bits/word.

    Both first and second moments are decoded by clearing their low 16 physical
    bits.  ``enable_fiber=False`` is the canonical zero-payload reference for
    the same numerical transition; it is deliberately not full-FP32 AdamW.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        enable_fiber: bool = True,
    ):
        if lr < 0 or eps < 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("invalid AdamW hyperparameters")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.enable_fiber = bool(enable_fiber)
        self.initialize_state()

    @torch.no_grad()
    def initialize_state(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise TypeError("DFCLow16AdamW expects floating parameters")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter, dtype=torch.float32, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, dtype=torch.float32, memory_format=torch.preserve_format
                    )

    def low_word_tensors(self) -> list[torch.Tensor]:
        first = [
            self.state[parameter]["exp_avg"]
            for group in self.param_groups for parameter in group["params"]
        ]
        second = [
            self.state[parameter]["exp_avg_sq"]
            for group in self.param_groups for parameter in group["params"]
        ]
        return first + second

    @property
    def payload_bit_capacity(self) -> int:
        if not self.enable_fiber:
            return 0
        return 16 * sum(tensor.numel() for tensor in self.low_word_tensors())

    @staticmethod
    def _decode(tensor: torch.Tensor) -> torch.Tensor:
        bits = tensor.view(torch.int32)
        return torch.bitwise_and(bits, HIGH16_MASK_I32).view(torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("DFCLow16AdamW does not support sparse gradients")
                gradient = gradient.float()
                state = self.state[parameter]
                state["step"] += 1
                step = state["step"]
                first_physical = state["exp_avg"]
                second_physical = state["exp_avg_sq"]
                first_bits = first_physical.view(torch.int32)
                second_bits = second_physical.view(torch.int32)
                if self.enable_fiber:
                    first_payload = torch.bitwise_and(first_bits, LOW16_MASK_I32)
                    second_payload = torch.bitwise_and(second_bits, LOW16_MASK_I32)
                else:
                    first_payload = second_payload = None
                first = self._decode(first_physical)
                second = self._decode(second_physical)
                first_new = beta1 * first + (1.0 - beta1) * gradient
                second_new = beta2 * second + (1.0 - beta2) * gradient * gradient
                first_new_bits = torch.bitwise_and(first_new.view(torch.int32), HIGH16_MASK_I32)
                second_new_bits = torch.bitwise_and(second_new.view(torch.int32), HIGH16_MASK_I32)
                if self.enable_fiber:
                    first_new_bits = torch.bitwise_or(first_new_bits, first_payload)
                    second_new_bits = torch.bitwise_or(second_new_bits, second_payload)
                first_physical.copy_(first_new_bits.view(torch.float32))
                second_physical.copy_(second_new_bits.view(torch.float32))
                first_use = self._decode(first_physical)
                second_use = self._decode(second_physical)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = second_use.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.addcdiv_(
                    first_use.to(parameter.dtype), denominator.to(parameter.dtype),
                    value=-(lr / bias_correction1),
                )
        return loss


class TorchLow16FiberChannel:
    """Byte-addressable low-word fiber over both AdamW moment tensors."""

    def __init__(self, optimizer: DFCLow16AdamW):
        if not optimizer.enable_fiber:
            raise ValueError("optimizer fiber is disabled")
        self.optimizer = optimizer
        self.tensors = optimizer.low_word_tensors()
        self.ends = np.cumsum([tensor.numel() for tensor in self.tensors], dtype=np.int64)

    @property
    def word_capacity(self) -> int:
        return int(self.ends[-1]) if self.ends.size else 0

    @property
    def byte_capacity(self) -> int:
        return 2 * self.word_capacity

    def _segments(self, start: int, count: int):
        if start < 0 or count < 0 or start + count > self.word_capacity:
            raise IndexError("low-word range out of bounds")
        position, remaining = int(start), int(count)
        while remaining:
            index = int(np.searchsorted(self.ends, position, side="right"))
            base = 0 if index == 0 else int(self.ends[index - 1])
            local = position - base
            take = min(remaining, self.tensors[index].numel() - local)
            yield self.tensors[index], local, take
            position += take
            remaining -= take

    @torch.no_grad()
    def read_words(self, start: int, count: int) -> np.ndarray:
        pieces = []
        for tensor, local, take in self._segments(start, count):
            bits = tensor.reshape(-1).view(torch.int32)[local : local + take]
            pieces.append(
                torch.bitwise_and(bits, LOW16_MASK_I32).to(torch.int32).cpu().numpy().astype(np.uint16)
            )
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.uint16)

    @torch.no_grad()
    def write_words(self, start: int, payload: np.ndarray) -> None:
        words_in = np.ascontiguousarray(payload, dtype=np.uint16).reshape(-1)
        cursor = 0
        for tensor, local, take in self._segments(start, words_in.size):
            bits = tensor.reshape(-1).view(torch.int32)
            incoming = torch.from_numpy(words_in[cursor : cursor + take].astype(np.int32)).to(bits.device)
            bits[local : local + take] = torch.bitwise_or(
                torch.bitwise_and(bits[local : local + take], HIGH16_MASK_I32), incoming
            )
            cursor += take

    def read_bytes(self, start: int, count: int) -> bytes:
        if start < 0 or count < 0 or start + count > self.byte_capacity:
            raise IndexError("low-word byte range out of bounds")
        first_word = start // 2
        end = start + count
        word_count = (end + 1) // 2 - first_word
        raw = self.read_words(first_word, word_count).astype("<u2", copy=False).tobytes()
        offset = start - 2 * first_word
        return raw[offset : offset + count]

    def write_bytes(self, start: int, payload: bytes | bytearray | memoryview) -> None:
        raw = bytes(payload)
        if start < 0 or start + len(raw) > self.byte_capacity:
            raise IndexError("low-word byte range out of bounds")
        if not raw:
            return
        first_word = start // 2
        end = start + len(raw)
        word_count = (end + 1) // 2 - first_word
        existing = bytearray(
            self.read_words(first_word, word_count).astype("<u2", copy=False).tobytes()
        )
        offset = start - 2 * first_word
        existing[offset : offset + len(raw)] = raw
        words = np.frombuffer(bytes(existing), dtype="<u2").copy()
        self.write_words(first_word, words)
