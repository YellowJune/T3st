from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from kaggle_checkpoint import atomic_json_dump, atomic_torch_save, capture_rng_state, load_checkpoint, restore_rng_state


def test_rng_capture_restore_roundtrip():
    random.seed(10); np.random.seed(10); torch.manual_seed(10)
    state = capture_rng_state()
    a = (random.random(), float(np.random.rand()), torch.rand(5))
    restore_rng_state(state)
    b = (random.random(), float(np.random.rand()), torch.rand(5))
    assert a[0] == b[0]
    assert a[1] == b[1]
    assert torch.equal(a[2], b[2])


def test_atomic_save_and_load_model_optimizer(tmp_path: Path):
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 4); loss = model(x).square().mean(); loss.backward(); opt.step()
    ref = {k: v.detach().clone() for k, v in model.state_dict().items()}
    payload = {"schema_version": 1, "step": 1, "model": model.state_dict(), "optimizer": opt.state_dict(), "rng": capture_rng_state(), "extra": {"x": 7}}
    path = tmp_path / "ckpt.pt"
    atomic_torch_save(payload, path)
    model2 = torch.nn.Linear(4, 3); opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    ckpt = load_checkpoint(path, model=model2, optimizer=opt2)
    assert ckpt["step"] == 1 and ckpt["extra"]["x"] == 7
    for k, v in model2.state_dict().items():
        assert torch.equal(v, ref[k])


def test_atomic_json_dump(tmp_path: Path):
    path = tmp_path / "r.json"
    atomic_json_dump({"b": 2, "a": 1}, path)
    assert '"a": 1' in path.read_text()
