"""DFC-EF: model-scale error-feedback state inside optimizer-state decoder fibers.

DFC-EF stores one logical FP32 error-feedback (EF) residual per trainable
coordinate inside the two 16-bit low-word fibers exposed by ``DFCLow16AdamW``.
The numerical contract is therefore *BF16-high AdamW semantics in FP32
containers*, not ordinary full-FP32 AdamW. The matched external baseline uses
the exact same optimizer semantics and compressor; only residual placement
differs.

Two low16 fibers provide 32 physical payload bits per coordinate. The low and
high 16-bit halves of each logical FP32 residual word are packed into exp_avg
and exp_avg_sq respectively. Decoding reconstructs the residual bit-for-bit.

For the model-scale path, use the chunked stride-k routines below. They never
materialize a model-sized decoded residual tensor: only a bounded chunk-sized
workspace exists transiently. That distinction is essential for a truthful
"zero model-scale external residual allocation" memory claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from torch_fiber import DFCLow16AdamW, HIGH16_MASK_I32, LOW16_MASK_I32


@dataclass(frozen=True)
class EFMemoryLedger:
    coordinates: int
    optimizer_state_bytes: int
    external_residual_bytes: int
    dfc_external_residual_bytes: int
    fiber_capacity_bytes: int

    @property
    def bytes_removed(self) -> int:
        return self.external_residual_bytes - self.dfc_external_residual_bytes


def memory_ledger(coordinates: int) -> EFMemoryLedger:
    p = int(coordinates)
    if p < 0:
        raise ValueError("coordinates must be nonnegative")
    return EFMemoryLedger(
        coordinates=p,
        optimizer_state_bytes=8 * p,       # two FP32 moment tensors
        external_residual_bytes=4 * p,     # one FP32 EF residual tensor
        dfc_external_residual_bytes=0,
        fiber_capacity_bytes=4 * p,        # 16+16 payload bits
    )


class PackedFP32Residual:
    """Tensor-native FP32 residual channel over both Adam moment low words."""

    def __init__(self, optimizer: DFCLow16AdamW):
        if not optimizer.enable_fiber:
            raise ValueError("DFC fiber must be enabled")
        self.optimizer = optimizer
        self.params = [p for g in optimizer.param_groups for p in g["params"]]

    @staticmethod
    def _state_pair(optimizer: DFCLow16AdamW, parameter: torch.nn.Parameter):
        state = optimizer.state[parameter]
        return state["exp_avg"], state["exp_avg_sq"]

    @property
    def coordinates(self) -> int:
        return sum(p.numel() for p in self.params)

    @property
    def byte_capacity(self) -> int:
        return 4 * self.coordinates

    @staticmethod
    def _decode_words(first_bits: torch.Tensor, second_bits: torch.Tensor) -> torch.Tensor:
        lo = torch.bitwise_and(first_bits, LOW16_MASK_I32)
        hi = torch.bitwise_and(second_bits, LOW16_MASK_I32)
        bits = torch.bitwise_or(lo, torch.bitwise_left_shift(hi, 16))
        return bits.view(torch.float32)

    @staticmethod
    def _pack_words_(first_bits: torch.Tensor, second_bits: torch.Tensor, residual: torch.Tensor) -> None:
        if residual.dtype != torch.float32:
            raise TypeError("residual must be float32")
        source = residual.contiguous().view(torch.int32)
        lo = torch.bitwise_and(source, LOW16_MASK_I32)
        hi = torch.bitwise_and(torch.bitwise_right_shift(source, 16), LOW16_MASK_I32)
        first_bits.copy_(torch.bitwise_or(torch.bitwise_and(first_bits, HIGH16_MASK_I32), lo))
        second_bits.copy_(torch.bitwise_or(torch.bitwise_and(second_bits, HIGH16_MASK_I32), hi))

    @torch.no_grad()
    def write_for_parameter(self, parameter: torch.nn.Parameter, residual: torch.Tensor) -> None:
        if residual.dtype != torch.float32:
            raise TypeError("residual must be float32")
        if residual.shape != parameter.shape:
            raise ValueError("residual shape must match parameter")
        first, second = self._state_pair(self.optimizer, parameter)
        self._pack_words_(first.view(torch.int32), second.view(torch.int32), residual)

    @torch.no_grad()
    def read_for_parameter(self, parameter: torch.nn.Parameter) -> torch.Tensor:
        first, second = self._state_pair(self.optimizer, parameter)
        return self._decode_words(first.view(torch.int32), second.view(torch.int32))

    @torch.no_grad()
    def zero_(self) -> None:
        for p in self.params:
            first, second = self._state_pair(self.optimizer, p)
            first_bits = first.view(torch.int32)
            second_bits = second.view(torch.int32)
            first_bits.copy_(torch.bitwise_and(first_bits, HIGH16_MASK_I32))
            second_bits.copy_(torch.bitwise_and(second_bits, HIGH16_MASK_I32))

    @torch.no_grad()
    def write(self, residuals: Iterable[torch.Tensor]) -> None:
        values = list(residuals)
        if len(values) != len(self.params):
            raise ValueError("residual list length mismatch")
        for p, r in zip(self.params, values):
            self.write_for_parameter(p, r)

    @torch.no_grad()
    def read(self) -> list[torch.Tensor]:
        return [self.read_for_parameter(p) for p in self.params]


@torch.no_grad()
def topk_error_feedback_external(
    gradient: torch.Tensor,
    residual: torch.Tensor,
    keep_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small-scale deterministic top-k reference with external FP32 EF state.

    This helper materializes full temporary tensors and is intended for
    correctness tests, not the model-scale memory claim.
    """
    if gradient.dtype != torch.float32 or residual.dtype != torch.float32:
        raise TypeError("gradient and residual must be float32")
    if gradient.shape != residual.shape:
        raise ValueError("gradient/residual shape mismatch")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    compensated = gradient + residual
    n = compensated.numel()
    k = max(1, min(n, int(round(n * keep_ratio))))
    flat = compensated.reshape(-1)
    if k == n:
        compressed = compensated.clone()
    else:
        idx = torch.topk(flat.abs(), k, sorted=False).indices
        compressed_flat = torch.zeros_like(flat)
        compressed_flat[idx] = flat[idx]
        compressed = compressed_flat.view_as(compensated)
    residual_new = compensated - compressed
    return compressed, residual_new


@torch.no_grad()
def topk_error_feedback_dfc(
    parameter: torch.nn.Parameter,
    gradient: torch.Tensor,
    channel: PackedFP32Residual,
    keep_ratio: float,
) -> torch.Tensor:
    """Small-scale top-k reference with residual stored in DFC fibers."""
    residual = channel.read_for_parameter(parameter)
    compressed, residual_new = topk_error_feedback_external(gradient, residual, keep_ratio)
    channel.write_for_parameter(parameter, residual_new)
    return compressed


def _keep_start(global_coordinate: int, stride: int, offset: int) -> int:
    return (int(offset) - int(global_coordinate)) % int(stride)


@torch.no_grad()
def stride_error_feedback_external_inplace_(
    gradient: torch.Tensor,
    residual: torch.Tensor,
    *,
    stride: int,
    offset: int,
    global_start: int = 0,
    chunk_coordinates: int = 1_048_576,
) -> int:
    """Chunked structured stride-k compression with conventional FP32 EF.

    ``gradient`` is replaced in-place by the communicated/decompressed sparse
    gradient. ``residual`` remains an external FP32 model-sized tensor. Only a
    bounded FP32 compensated chunk is transiently allocated.

    Positions satisfying ``global_index % stride == offset`` are transmitted.
    The offset can be cycled across steps so every coordinate is selected once
    per ``stride`` steps. Returned value is the number of transmitted values.
    """
    if residual.dtype != torch.float32:
        raise TypeError("external residual must be float32")
    if gradient.shape != residual.shape:
        raise ValueError("gradient/residual shape mismatch")
    if not gradient.is_floating_point():
        raise TypeError("gradient must be floating point")
    if not gradient.is_contiguous() or not residual.is_contiguous():
        raise ValueError("chunked EF expects contiguous tensors")
    if stride <= 0 or not 0 <= offset < stride or chunk_coordinates <= 0:
        raise ValueError("invalid stride/offset/chunk size")

    g = gradient.view(-1)
    r = residual.view(-1)
    sent = 0
    for start in range(0, g.numel(), chunk_coordinates):
        end = min(g.numel(), start + chunk_coordinates)
        gs = g[start:end]
        rs = r[start:end]
        compensated = gs.float() + rs
        rs.copy_(compensated)
        gs.zero_()
        first = _keep_start(global_start + start, stride, offset)
        if first < end - start:
            chosen = compensated[first::stride]
            communicated = chosen.to(dtype=gradient.dtype)
            gs[first::stride].copy_(communicated)
            # Keep quantization/transport rounding error in EF state.
            rs[first::stride].copy_(chosen - communicated.float())
            sent += int(chosen.numel())
    return sent


@torch.no_grad()
def stride_error_feedback_dfc_inplace_(
    parameter: torch.nn.Parameter,
    gradient: torch.Tensor,
    channel: PackedFP32Residual,
    *,
    stride: int,
    offset: int,
    global_start: int = 0,
    chunk_coordinates: int = 1_048_576,
) -> int:
    """Same chunked stride-k EF transition with no model-sized residual tensor.

    Residual chunks are decoded from optimizer low words, combined with the
    gradient, then immediately repacked. Peak temporary residual workspace is
    O(``chunk_coordinates``), independent of model size.
    """
    if gradient.shape != parameter.shape:
        raise ValueError("gradient shape must match parameter")
    if not gradient.is_floating_point():
        raise TypeError("gradient must be floating point")
    if not gradient.is_contiguous():
        raise ValueError("chunked EF expects contiguous gradients")
    if stride <= 0 or not 0 <= offset < stride or chunk_coordinates <= 0:
        raise ValueError("invalid stride/offset/chunk size")

    first, second = channel._state_pair(channel.optimizer, parameter)
    fb = first.view(-1).view(torch.int32)
    sb = second.view(-1).view(torch.int32)
    g = gradient.view(-1)
    sent = 0
    for start in range(0, g.numel(), chunk_coordinates):
        end = min(g.numel(), start + chunk_coordinates)
        fbs = fb[start:end]
        sbs = sb[start:end]
        residual = channel._decode_words(fbs, sbs)
        gs = g[start:end]
        compensated = gs.float() + residual
        gs.zero_()
        residual_new = compensated
        first_keep = _keep_start(global_start + start, stride, offset)
        if first_keep < end - start:
            chosen = compensated[first_keep::stride]
            communicated = chosen.to(dtype=gradient.dtype)
            gs[first_keep::stride].copy_(communicated)
            # clone only the chunk when a selected slice must be altered;
            # this remains bounded by chunk_coordinates.
            residual_new = compensated.clone()
            residual_new[first_keep::stride] = chosen - communicated.float()
            sent += int(chosen.numel())
        channel._pack_words_(fbs, sbs, residual_new)
    return sent


def allocate_external_residuals(parameters: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
    """Allocate the matched conventional FP32 model-sized EF state."""
    return [torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format) for p in parameters]
