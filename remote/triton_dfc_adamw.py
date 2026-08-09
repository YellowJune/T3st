"""Matched fused Triton AdamW kernels with and without DFC-SIGN.

Both kernels use the same ordered FP32 arithmetic and the same tensor load/store
footprint. The DFC variant adds only sign decode/re-embedding integer operations
inside the fused pass. This makes ``reference_adamw_step`` the appropriate
systems control for measuring the incremental DFC-SIGN overhead while
``dfc_adamw_step`` remains the deployable payload-preserving primitive.

The module is import-safe on CPU-only systems. GPU calls raise clear errors
unless CUDA and Triton are present.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by CPU CI
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _reference_adamw_kernel(
        parameter,
        gradient,
        first_moment,
        second_moment,
        n_elements: tl.constexpr,
        lr: tl.constexpr,
        beta1: tl.constexpr,
        beta2: tl.constexpr,
        eps: tl.constexpr,
        weight_decay: tl.constexpr,
        bias_correction1: tl.constexpr,
        bias_correction2: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        p = tl.load(parameter + offsets, mask=mask).to(tl.float32)
        g = tl.load(gradient + offsets, mask=mask).to(tl.float32)
        m = tl.load(first_moment + offsets, mask=mask).to(tl.float32)
        v = tl.load(second_moment + offsets, mask=mask).to(tl.float32)

        m_new = beta1 * m + (1.0 - beta1) * g
        v_new = beta2 * v + (1.0 - beta2) * g * g
        denominator = tl.sqrt(v_new / bias_correction2) + eps
        p_new = p * (1.0 - lr * weight_decay) - (lr / bias_correction1) * m_new / denominator

        tl.store(parameter + offsets, p_new, mask=mask)
        tl.store(first_moment + offsets, m_new, mask=mask)
        tl.store(second_moment + offsets, v_new, mask=mask)


    @triton.jit
    def _dfc_adamw_kernel(
        parameter,
        gradient,
        first_moment,
        physical_second_moment,
        n_elements: tl.constexpr,
        lr: tl.constexpr,
        beta1: tl.constexpr,
        beta2: tl.constexpr,
        eps: tl.constexpr,
        weight_decay: tl.constexpr,
        bias_correction1: tl.constexpr,
        bias_correction2: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        p = tl.load(parameter + offsets, mask=mask).to(tl.float32)
        g = tl.load(gradient + offsets, mask=mask).to(tl.float32)
        m = tl.load(first_moment + offsets, mask=mask).to(tl.float32)
        v_physical = tl.load(physical_second_moment + offsets, mask=mask).to(tl.float32)
        v_bits = v_physical.to(tl.int32, bitcast=True)
        payload = v_bits & -2147483648
        magnitude_bits = v_bits & 2147483647
        v = magnitude_bits.to(tl.float32, bitcast=True)

        m_new = beta1 * m + (1.0 - beta1) * g
        v_new = beta2 * v + (1.0 - beta2) * g * g
        denominator = tl.sqrt(v_new / bias_correction2) + eps
        p_new = p * (1.0 - lr * weight_decay) - (lr / bias_correction1) * m_new / denominator
        v_new_bits = v_new.to(tl.int32, bitcast=True) & 2147483647
        v_packed = (v_new_bits | payload).to(tl.float32, bitcast=True)

        tl.store(parameter + offsets, p_new, mask=mask)
        tl.store(first_moment + offsets, m_new, mask=mask)
        tl.store(physical_second_moment + offsets, v_packed, mask=mask)


def _validate_tensors(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
) -> None:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not parameter.is_cuda:
        raise RuntimeError("DFC Triton kernels require CUDA tensors")
    tensors = (parameter, gradient, first_moment, second_moment)
    if any(tensor.dtype != torch.float32 or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("kernel requires contiguous FP32 tensors")
    if any(tensor.numel() != parameter.numel() for tensor in tensors):
        raise ValueError("kernel tensors must have equal sizes")


def _launch(kernel, parameter, gradient, first_moment, second_moment, step, lr, betas, eps, weight_decay):
    _validate_tensors(parameter, gradient, first_moment, second_moment)
    beta1, beta2 = betas
    block = 256
    grid = (triton.cdiv(parameter.numel(), block),)
    kernel[grid](
        parameter,
        gradient,
        first_moment,
        second_moment,
        parameter.numel(),
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        1.0 - beta1**step,
        1.0 - beta2**step,
        block_size=block,
    )


def reference_adamw_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    step: int,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> None:
    """Launch the matched fused AdamW control kernel in-place."""
    if triton is None:
        raise RuntimeError("Triton is not installed")
    _launch(
        _reference_adamw_kernel,
        parameter,
        gradient,
        first_moment,
        second_moment,
        step,
        lr,
        betas,
        eps,
        weight_decay,
    )


def dfc_adamw_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    first_moment: torch.Tensor,
    physical_second_moment: torch.Tensor,
    step: int,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> None:
    """Launch one fused in-place DFC-SIGN AdamW update."""
    if triton is None:
        raise RuntimeError("Triton is not installed")
    _launch(
        _dfc_adamw_kernel,
        parameter,
        gradient,
        first_moment,
        physical_second_moment,
        step,
        lr,
        betas,
        eps,
        weight_decay,
    )
