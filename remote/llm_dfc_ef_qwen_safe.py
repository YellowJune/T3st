"""Canonical free-GPU entry point for llm_dfc_ef_qwen.

This thin wrapper replaces only the full-FP32 precision-control optimizer with
``FullFP32AdamWChunked`` so FP16 model parameters never suffer epsilon underflow.
All DFC-EF protocol logic remains in ``llm_dfc_ef_qwen``.
"""

from __future__ import annotations

import llm_dfc_ef_qwen as core
from chunked_fp32_adamw import FullFP32AdamWChunked

_original_build_optimizer = core.build_optimizer


def _safe_build_optimizer(method, params, args):
    if method == "fp32_dense":
        return FullFP32AdamWChunked(
            params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=args.weight_decay, chunk_coordinates=args.optimizer_chunk,
        )
    return _original_build_optimizer(method, params, args)


core.build_optimizer = _safe_build_optimizer

if __name__ == "__main__":
    core.main()
