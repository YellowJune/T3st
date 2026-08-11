"""Bounded-workspace blockwise top-k compressors for DFC-EF experiments.

Blockwise top-k is used for the learning experiments because it is closer to a
standard magnitude sparsifier than the stride-k systems microbenchmark while
remaining bounded-workspace.  External-EF and DFC-EF execute the same logical
transition; only the physical location of the FP32 error-feedback residual
differs.
"""

from __future__ import annotations

import torch

from dfc_ef import PackedFP32Residual


def _k_for_block(n: int, keep_ratio: float) -> int:
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0,1]")
    return max(1, min(int(n), int(round(int(n) * float(keep_ratio)))))


@torch.no_grad()
def block_topk_external_inplace_(
    gradient: torch.Tensor,
    residual: torch.Tensor,
    *,
    keep_ratio: float = 0.125,
    chunk_coordinates: int = 1_048_576,
) -> int:
    """Blockwise top-k with a conventional model-sized FP32 EF residual.

    ``gradient`` is overwritten by the sparse communicated/decompressed values.
    Residual state remains FP32. Only O(chunk_coordinates) transient workspace
    is required in addition to the model-sized external residual.
    """
    if residual.dtype != torch.float32:
        raise TypeError("external residual must be float32")
    if gradient.shape != residual.shape:
        raise ValueError("gradient/residual shape mismatch")
    if not gradient.is_floating_point() or not gradient.is_contiguous() or not residual.is_contiguous():
        raise ValueError("floating contiguous gradient/residual required")
    if chunk_coordinates <= 0:
        raise ValueError("chunk_coordinates must be positive")

    g = gradient.view(-1)
    r = residual.view(-1)
    sent = 0
    for start in range(0, g.numel(), chunk_coordinates):
        end = min(g.numel(), start + chunk_coordinates)
        gs = g[start:end]
        rs = r[start:end]
        compensated = gs.float() + rs
        k = _k_for_block(compensated.numel(), keep_ratio)
        if k == compensated.numel():
            communicated = compensated.to(dtype=gradient.dtype)
            gs.copy_(communicated)
            rs.copy_(compensated - communicated.float())
            sent += int(k)
            continue
        idx = torch.topk(compensated.abs(), k, sorted=False).indices
        chosen = compensated[idx]
        communicated = chosen.to(dtype=gradient.dtype)
        rs.copy_(compensated)
        rs[idx] = chosen - communicated.float()
        gs.zero_()
        gs[idx] = communicated
        sent += int(k)
    return sent


@torch.no_grad()
def block_topk_dfc_inplace_(
    parameter: torch.nn.Parameter,
    gradient: torch.Tensor,
    channel: PackedFP32Residual,
    *,
    keep_ratio: float = 0.125,
    chunk_coordinates: int = 1_048_576,
) -> int:
    """Same blockwise top-k EF transition with residual state in DFC fibers.

    No model-sized decoded residual is materialized. Each residual chunk is
    reconstructed from low-word fibers, updated, and repacked immediately.
    """
    if gradient.shape != parameter.shape:
        raise ValueError("gradient shape must match parameter")
    if not gradient.is_floating_point() or not gradient.is_contiguous():
        raise ValueError("floating contiguous gradient required")
    if chunk_coordinates <= 0:
        raise ValueError("chunk_coordinates must be positive")

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
        k = _k_for_block(compensated.numel(), keep_ratio)
        if k == compensated.numel():
            communicated = compensated.to(dtype=gradient.dtype)
            gs.copy_(communicated)
            residual_new = compensated - communicated.float()
            channel._pack_words_(fbs, sbs, residual_new)
            sent += int(k)
            continue
        idx = torch.topk(compensated.abs(), k, sorted=False).indices
        chosen = compensated[idx]
        communicated = chosen.to(dtype=gradient.dtype)
        # compensated itself becomes the residual chunk, avoiding a second
        # chunk-sized FP32 copy.
        compensated[idx] = chosen - communicated.float()
        gs.zero_()
        gs[idx] = communicated
        channel._pack_words_(fbs, sbs, compensated)
        sent += int(k)
    return sent


@torch.no_grad()
def block_topk_noef_inplace_(
    gradient: torch.Tensor,
    *,
    keep_ratio: float = 0.125,
    chunk_coordinates: int = 1_048_576,
) -> int:
    """Matched blockwise top-k sparsifier without error feedback."""
    if not gradient.is_floating_point() or not gradient.is_contiguous():
        raise ValueError("floating contiguous gradient required")
    if chunk_coordinates <= 0:
        raise ValueError("chunk_coordinates must be positive")
    g = gradient.view(-1)
    sent = 0
    for start in range(0, g.numel(), chunk_coordinates):
        end = min(g.numel(), start + chunk_coordinates)
        gs = g[start:end]
        k = _k_for_block(gs.numel(), keep_ratio)
        if k == gs.numel():
            sent += int(k)
            continue
        idx = torch.topk(gs.abs(), k, sorted=False).indices
        chosen = gs[idx].clone()
        gs.zero_(); gs[idx] = chosen
        sent += int(k)
    return sent
