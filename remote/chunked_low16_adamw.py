"""Bounded-workspace BF16-high AdamW for model-scale DFC-EF experiments."""

from __future__ import annotations

import math

import torch

from torch_fiber import DFCLow16AdamW, HIGH16_MASK_I32, LOW16_MASK_I32


class DFCLow16AdamWChunked(DFCLow16AdamW):
    """Numerically matched DFCLow16AdamW with O(chunk) transient workspace.

    The parent optimizer is the semantic reference. This implementation performs
    the same per-coordinate transition in flattened chunks so a very large
    parameter tensor does not create several parameter-sized FP32 temporaries.
    ``enable_fiber`` has the same meaning as in the parent class.
    """

    def __init__(self, *args, chunk_coordinates: int = 1_048_576, **kwargs):
        if int(chunk_coordinates) <= 0:
            raise ValueError("chunk_coordinates must be positive")
        self.chunk_coordinates = int(chunk_coordinates)
        super().__init__(*args, **kwargs)

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
                    raise RuntimeError("DFCLow16AdamWChunked does not support sparse gradients")
                if not parameter.is_contiguous() or not gradient.is_contiguous():
                    raise RuntimeError("chunked optimizer requires contiguous parameter and gradient tensors")
                state = self.state[parameter]
                state["step"] += 1
                step = state["step"]
                first_physical = state["exp_avg"]
                second_physical = state["exp_avg_sq"]
                pflat = parameter.view(-1)
                gflat = gradient.view(-1)
                fbits = first_physical.view(-1).view(torch.int32)
                sbits = second_physical.view(-1).view(torch.int32)
                step_size = lr / (1.0 - beta1**step)
                bias2_sqrt = math.sqrt(1.0 - beta2**step)

                for start in range(0, pflat.numel(), self.chunk_coordinates):
                    end = min(pflat.numel(), start + self.chunk_coordinates)
                    pf = pflat[start:end]
                    gf = gflat[start:end].float()
                    fb = fbits[start:end]
                    sb = sbits[start:end]

                    if self.enable_fiber:
                        fpayload = torch.bitwise_and(fb, LOW16_MASK_I32)
                        spayload = torch.bitwise_and(sb, LOW16_MASK_I32)
                    else:
                        fpayload = spayload = None

                    first = torch.bitwise_and(fb, HIGH16_MASK_I32).view(torch.float32)
                    second = torch.bitwise_and(sb, HIGH16_MASK_I32).view(torch.float32)
                    first_new = beta1 * first + (1.0 - beta1) * gf
                    second_new = beta2 * second + (1.0 - beta2) * gf * gf
                    fnew = torch.bitwise_and(first_new.view(torch.int32), HIGH16_MASK_I32)
                    snew = torch.bitwise_and(second_new.view(torch.int32), HIGH16_MASK_I32)
                    if self.enable_fiber:
                        fnew = torch.bitwise_or(fnew, fpayload)
                        snew = torch.bitwise_or(snew, spayload)
                    fb.copy_(fnew); sb.copy_(snew)

                    first_use = torch.bitwise_and(fb, HIGH16_MASK_I32).view(torch.float32)
                    second_use = torch.bitwise_and(sb, HIGH16_MASK_I32).view(torch.float32)
                    denominator = second_use.sqrt().div_(bias2_sqrt).add_(eps)
                    if weight_decay:
                        pf.mul_(1.0 - lr * weight_decay)
                    pf.addcdiv_(
                        first_use.to(dtype=parameter.dtype),
                        denominator.to(dtype=parameter.dtype),
                        value=-step_size,
                    )
        return loss


@torch.no_grad()
def stride_no_error_feedback_inplace_(
    gradient: torch.Tensor,
    *,
    stride: int,
    offset: int,
    global_start: int = 0,
) -> int:
    """Matched structured compressor with no EF state for utility ablations."""
    if stride <= 0 or not 0 <= offset < stride:
        raise ValueError("invalid stride/offset")
    if not gradient.is_contiguous():
        raise ValueError("gradient must be contiguous")
    flat = gradient.view(-1)
    # Save selected values only; this is ~1/stride of a tensor and is not an EF
    # state. It is analogous to the communicated sparse values themselves.
    first = (offset - global_start) % stride
    selected = flat[first::stride].clone()
    flat.zero_()
    flat[first::stride].copy_(selected)
    return int(selected.numel())
