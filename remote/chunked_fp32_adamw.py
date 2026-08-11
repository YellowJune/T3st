"""Bounded-workspace AdamW with ordinary FP32 moment semantics.

Used only as the precision-control baseline for DFC-LOW16 experiments. Parameters
may be FP16/BF16/FP32, but first and second moments and the Adam ratio are
computed in FP32. The final update is cast to parameter dtype.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


class FullFP32AdamWChunked(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0,
                 chunk_coordinates: int = 1_048_576):
        if lr < 0 or eps < 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("invalid AdamW hyperparameters")
        if int(chunk_coordinates) <= 0:
            raise ValueError("chunk_coordinates must be positive")
        self.chunk_coordinates = int(chunk_coordinates)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.initialize_state()

    @torch.no_grad()
    def initialize_state(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise TypeError("floating parameters required")
                st = self.state[p]
                if not st:
                    st['step'] = 0
                    st['exp_avg'] = torch.zeros_like(p, dtype=torch.float32,
                                                     memory_format=torch.preserve_format)
                    st['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32,
                                                        memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError("sparse gradients unsupported")
                if not p.is_contiguous() or not g.is_contiguous():
                    raise RuntimeError("chunked optimizer requires contiguous tensors")
                st = self.state[p]
                st['step'] += 1
                step = st['step']
                pf = p.view(-1)
                gf = g.view(-1)
                m = st['exp_avg'].view(-1)
                v = st['exp_avg_sq'].view(-1)
                step_size = lr / (1.0 - beta1 ** step)
                bias2_sqrt = math.sqrt(1.0 - beta2 ** step)
                for start in range(0, pf.numel(), self.chunk_coordinates):
                    end = min(pf.numel(), start + self.chunk_coordinates)
                    pc, gc = pf[start:end], gf[start:end].float()
                    mc, vc = m[start:end], v[start:end]
                    mc.mul_(beta1).add_(gc, alpha=1.0-beta1)
                    vc.mul_(beta2).addcmul_(gc, gc, value=1.0-beta2)
                    denom = vc.sqrt().div_(bias2_sqrt).add_(eps)
                    if wd:
                        pc.mul_(1.0 - lr * wd)
                    update = mc.div(denom).mul_(-step_size)
                    pc.add_(update.to(dtype=p.dtype))
        return loss
