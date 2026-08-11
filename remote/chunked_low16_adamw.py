"""Bounded-workspace BF16-high AdamW for model-scale DFC-EF experiments."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from itertools import chain

import torch

from torch_fiber import DFCLow16AdamW, HIGH16_MASK_I32, LOW16_MASK_I32


class DFCLow16AdamWChunked(DFCLow16AdamW):
    """BF16-high AdamW with O(chunk) transient workspace."""

    def __init__(self, *args, chunk_coordinates: int = 1_048_576, **kwargs):
        if int(chunk_coordinates) <= 0:
            raise ValueError("chunk_coordinates must be positive")
        self.chunk_coordinates = int(chunk_coordinates)
        super().__init__(*args, **kwargs)

    def load_state_dict(self, state_dict):
        """Reload DFC state without PyTorch's parameter-dtype state casting.

        PyTorch's generic optimizer loader intentionally converts every tensor
        state associated with a floating parameter to that parameter's dtype.
        DFC cannot use that policy: ``exp_avg`` and ``exp_avg_sq`` are physical
        FP32 containers even when the model parameter is FP16/BF16, and their
        low words contain payload.  This loader performs the same positional
        parameter-ID mapping while preserving FP32 moment bytes exactly.
        """
        if not isinstance(state_dict, dict) or "state" not in state_dict or "param_groups" not in state_dict:
            raise ValueError("invalid optimizer state_dict")

        groups = self.param_groups
        saved_groups = copy.deepcopy(state_dict["param_groups"])
        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        if any(len(g["params"]) != len(sg["params"]) for g, sg in zip(groups, saved_groups)):
            raise ValueError("loaded state dict parameter-group size mismatch")

        id_map = dict(zip(
            chain.from_iterable(g["params"] for g in saved_groups),
            chain.from_iterable(g["params"] for g in groups),
        ))
        new_state = defaultdict(dict)
        for saved_id, saved_state in state_dict["state"].items():
            if saved_id not in id_map:
                new_state[saved_id] = copy.deepcopy(saved_state)
                continue
            parameter = id_map[saved_id]
            restored = {}
            for key, value in saved_state.items():
                if isinstance(value, torch.Tensor):
                    if key in ("exp_avg", "exp_avg_sq"):
                        if value.dtype != torch.float32:
                            raise TypeError(f"{key} must be a physical float32 container")
                        # float32 -> float32 device copy is bit-preserving.
                        restored[key] = value.detach().to(
                            device=parameter.device, dtype=torch.float32
                        ).contiguous().clone()
                    else:
                        # Preserve non-moment tensor dtype; only relocate device.
                        restored[key] = value.detach().to(device=parameter.device).clone()
                else:
                    restored[key] = copy.deepcopy(value)
            new_state[parameter] = restored

        new_groups = []
        for current, saved in zip(groups, saved_groups):
            saved["params"] = current["params"]
            if "param_names" in current and "param_names" not in saved:
                saved["param_names"] = current["param_names"]
            new_groups.append(saved)
        self.__setstate__({"state": new_state, "param_groups": new_groups})

        # Fail closed: the physical contract must survive every reload.
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state and state[key].dtype != torch.float32:
                        raise RuntimeError(f"{key} reload violated physical FP32 contract")
        return None

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
                    fb.copy_(fnew)
                    sb.copy_(snew)

                    first_use = torch.bitwise_and(fb, HIGH16_MASK_I32).view(torch.float32)
                    second_use = torch.bitwise_and(sb, HIGH16_MASK_I32).view(torch.float32)
                    denominator = second_use.sqrt().div_(bias2_sqrt).add_(eps)
                    if weight_decay:
                        pf.mul_(1.0 - lr * weight_decay)
                    if parameter.dtype == torch.float32:
                        pf.addcdiv_(first_use, denominator, value=-step_size)
                    else:
                        update = first_use.div(denominator).mul_(-step_size)
                        pf.add_(update.to(dtype=parameter.dtype))
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
    first = (offset - global_start) % stride
    selected = flat[first::stride].clone()
    flat.zero_()
    flat[first::stride].copy_(selected)
    return int(selected.numel())
