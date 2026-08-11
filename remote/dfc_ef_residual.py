"""DFC-LOW16 error-feedback state adapter.

A full FP32 error-feedback residual requires 32 bits per trainable coordinate.
DFC-LOW16 exposes exactly 16 payload bits in each of AdamW's two FP32 moment
words, so the two low words can store one arbitrary FP32 residual bit-for-bit
without an additional model-sized tensor.

The numerical optimizer semantics are the HIGH16 (BF16-high) portions of the
moments. Packing/unpacking below never changes those semantic bits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import torch

LOW16_MASK_I32 = 0xFFFF
HIGH16_MASK_I32 = -65536


def _require_fp32_pair(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.dtype != torch.float32 or second.dtype != torch.float32:
        raise TypeError("DFC-EF moment tensors must be float32")
    if first.shape != second.shape:
        raise ValueError("moment shape mismatch")
    if first.device != second.device:
        raise ValueError("moment device mismatch")


@torch.no_grad()
def pack_fp32_residual_(first: torch.Tensor, second: torch.Tensor, residual: torch.Tensor) -> None:
    """Embed one FP32 residual word per coordinate into two LOW16 fibers."""
    _require_fp32_pair(first, second)
    if residual.dtype != torch.float32 or residual.shape != first.shape or residual.device != first.device:
        raise TypeError("residual must be same-shape/device float32")
    rb = residual.contiguous().view(torch.int32)
    low = torch.bitwise_and(rb, LOW16_MASK_I32)
    high = torch.bitwise_and(torch.bitwise_right_shift(rb, 16), LOW16_MASK_I32)
    fb = first.view(torch.int32)
    sb = second.view(torch.int32)
    fb.copy_(torch.bitwise_or(torch.bitwise_and(fb, HIGH16_MASK_I32), low))
    sb.copy_(torch.bitwise_or(torch.bitwise_and(sb, HIGH16_MASK_I32), high))


@torch.no_grad()
def unpack_fp32_residual(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Decode the exact FP32 residual stored across two LOW16 fibers."""
    _require_fp32_pair(first, second)
    fb = torch.bitwise_and(first.view(torch.int32), LOW16_MASK_I32)
    sb = torch.bitwise_and(second.view(torch.int32), LOW16_MASK_I32)
    bits = torch.bitwise_or(fb, torch.bitwise_left_shift(sb, 16))
    return bits.contiguous().view(torch.float32)


def semantic_high16_bits(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.float32:
        raise TypeError("expected float32")
    return torch.bitwise_and(tensor.view(torch.int32), HIGH16_MASK_I32)


@torch.no_grad()
def ef_int8_tensor(gradient: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic per-tensor symmetric INT8 quantization with error feedback."""
    g = gradient.float()
    r = residual.float()
    corrected = g + r
    max_abs = corrected.abs().amax()
    scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
    q = torch.clamp(torch.round(corrected / scale), -127, 127).to(torch.int8)
    deq = q.float() * scale
    new_residual = corrected - deq
    return deq, new_residual, scale.float()


@dataclass
class CompressionStats:
    encoded_bytes: int = 0
    tensors: int = 0

    def add(self, n: int) -> None:
        self.encoded_bytes += int(n) + 4
        self.tensors += 1


class ExternalErrorFeedback:
    """Conventional model-sized FP32 error-feedback allocation."""
    def __init__(self, params: Iterable[torch.nn.Parameter]):
        self.params = list(params)
        self.residuals = [torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format) for p in self.params]
        self.stats = CompressionStats()

    @torch.no_grad()
    def compress_grads_(self) -> None:
        for p, residual in zip(self.params, self.residuals):
            if p.grad is None:
                continue
            deq, new_residual, _ = ef_int8_tensor(p.grad, residual)
            residual.copy_(new_residual)
            p.grad.copy_(deq.to(p.grad.dtype))
            self.stats.add(p.numel())

    @property
    def allocated_bytes(self) -> int:
        return sum(r.numel() * r.element_size() for r in self.residuals)

    def state_dict(self) -> dict:
        return {"residuals": [r.detach().cpu().clone() for r in self.residuals], "stats": self.stats.__dict__.copy()}

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        if len(state["residuals"]) != len(self.residuals):
            raise ValueError("residual count mismatch")
        for dst, src in zip(self.residuals, state["residuals"]):
            dst.copy_(src.to(dst.device, dtype=torch.float32))
        self.stats = CompressionStats(**state.get("stats", {}))


class DFCLow16ErrorFeedback:
    """Error feedback stored entirely in the two LOW16 Adam moment fibers."""
    def __init__(self, optimizer, params: Iterable[torch.nn.Parameter]):
        self.optimizer = optimizer
        self.params = list(params)
        self.stats = CompressionStats()
        if not getattr(optimizer, "enable_fiber", False):
            raise ValueError("DFC LOW16 fiber must be enabled")
        for p in self.params:
            state = optimizer.state[p]
            _require_fp32_pair(state["exp_avg"], state["exp_avg_sq"])

    @torch.no_grad()
    def compress_grads_(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            state = self.optimizer.state[p]
            first, second = state["exp_avg"], state["exp_avg_sq"]
            residual = unpack_fp32_residual(first, second)
            deq, new_residual, _ = ef_int8_tensor(p.grad, residual)
            pack_fp32_residual_(first, second, new_residual)
            p.grad.copy_(deq.to(p.grad.dtype))
            self.stats.add(p.numel())

    @property
    def allocated_bytes(self) -> int:
        return 0

    def state_dict(self) -> dict:
        return {"stats": self.stats.__dict__.copy(), "payload_location": "optimizer_low16_fibers"}

    def load_state_dict(self, state: dict) -> None:
        self.stats = CompressionStats(**state.get("stats", {}))

    @torch.no_grad()
    def decoded_residuals(self) -> list[torch.Tensor]:
        out = []
        for p in self.params:
            s = self.optimizer.state[p]
            out.append(unpack_fp32_residual(s["exp_avg"], s["exp_avg_sq"]))
        return out
