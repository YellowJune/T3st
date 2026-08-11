"""DFC-EF: zero-extra-allocation error-feedback state for compressed training.

This module stores one logical FP32 error-feedback residual per trainable
coordinate inside the two 16-bit low-word fibers exposed by ``DFCLow16AdamW``.
The numerical contract is therefore *BF16-high AdamW semantics in FP32
containers*, not ordinary full-FP32 AdamW.  The external baseline implemented
here uses the exact same optimizer semantics and the exact same residual update;
the only difference is whether the residual lives in a separate FP32 tensor or
inside the optimizer-state fibers.

The packed residual representation is exact at the bit level: the high and low
16-bit halves of each residual FP32 word are stored in the low 16 physical bits
of exp_avg and exp_avg_sq respectively.  Decoding reconstructs the original
FP32 residual bit-for-bit.
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
    # two ordinary FP32 Adam moment tensors
    optimizer_state = 2 * 4 * p
    # one external FP32 EF residual tensor
    external_residual = 4 * p
    # two low16 fibers -> 16+16 bits = 4 bytes/coordinate
    fiber_capacity = 4 * p
    return EFMemoryLedger(
        coordinates=p,
        optimizer_state_bytes=optimizer_state,
        external_residual_bytes=external_residual,
        dfc_external_residual_bytes=0,
        fiber_capacity_bytes=fiber_capacity,
    )


class PackedFP32Residual:
    """Tensor-native FP32 residual channel over both Adam moment low words.

    The channel is intentionally tensor-native: no CPU/NumPy round-trip is
    needed during training. For each optimizer parameter tensor, residual word
    i is split into two uint16 halves. The low half is packed into exp_avg's low
    word and the high half into exp_avg_sq's low word.
    """

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

    @torch.no_grad()
    def write_for_parameter(self, parameter: torch.nn.Parameter, residual: torch.Tensor) -> None:
        if residual.dtype != torch.float32:
            raise TypeError("residual must be float32")
        if residual.shape != parameter.shape:
            raise ValueError("residual shape must match parameter")
        first, second = self._state_pair(self.optimizer, parameter)
        source = residual.contiguous().view(torch.int32)
        lo = torch.bitwise_and(source, LOW16_MASK_I32)
        hi = torch.bitwise_and(torch.bitwise_right_shift(source, 16), LOW16_MASK_I32)
        first_bits = first.view(torch.int32)
        second_bits = second.view(torch.int32)
        first_bits.copy_(torch.bitwise_or(torch.bitwise_and(first_bits, HIGH16_MASK_I32), lo))
        second_bits.copy_(torch.bitwise_or(torch.bitwise_and(second_bits, HIGH16_MASK_I32), hi))

    @torch.no_grad()
    def read_for_parameter(self, parameter: torch.nn.Parameter) -> torch.Tensor:
        first, second = self._state_pair(self.optimizer, parameter)
        lo = torch.bitwise_and(first.view(torch.int32), LOW16_MASK_I32)
        hi = torch.bitwise_and(second.view(torch.int32), LOW16_MASK_I32)
        bits = torch.bitwise_or(lo, torch.bitwise_left_shift(hi, 16))
        return bits.view(torch.float32)

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
    """Deterministic top-k compressor with conventional external EF residual."""
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
    """Same EF transition as external baseline, with residual stored in DFC."""
    residual = channel.read_for_parameter(parameter)
    compressed, residual_new = topk_error_feedback_external(gradient, residual, keep_ratio)
    channel.write_for_parameter(parameter, residual_new)
    return compressed
