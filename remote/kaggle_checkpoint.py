"""Crash-tolerant checkpoint helpers for short Kaggle/Vast validation jobs."""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Write a checkpoint by rename so interrupted writes never replace the last good file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".tmp.", dir=str(target.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(payload, tmp)
        # Flush file contents before atomic rename.
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".tmp.", dir=str(target.parent), text=True)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def append_jsonl(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = dict(payload)
    row.setdefault("unix_time", time.time())
    with target.open("a", buffering=1) as fh:
        fh.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")
        fh.flush(); os.fsync(fh.fileno())


def checkpoint_payload(*, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(),
        "extra": {} if extra is None else extra,
    }


def load_checkpoint(path: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    map_location: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    restore_rng_state(ckpt["rng"])
    return ckpt
