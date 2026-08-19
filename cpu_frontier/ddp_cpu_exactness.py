#!/usr/bin/env python3
"""Two-rank CPU/Gloo DDP exactness gate for DFC-SIGN.

Each rank carries a different hidden payload while DDP exposes the same semantic
optimizer trajectory. No CUDA device is required or used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

REMOTE = Path(__file__).resolve().parents[1] / "remote"
sys.path.insert(0, str(REMOTE))

from torch_fiber import DFCAdamW, TorchSignFiberChannel  # noqa: E402


def tensor_digest(parameters) -> str:
    raw = b"".join(p.detach().cpu().contiguous().numpy().tobytes() for p in parameters)
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"expected two ranks, got {world}")

    torch.manual_seed(8100)
    reference = nn.Sequential(nn.Linear(64, 96), nn.GELU(), nn.Linear(96, 32))
    lifted = nn.Sequential(nn.Linear(64, 96), nn.GELU(), nn.Linear(96, 32))
    lifted.load_state_dict(reference.state_dict())

    reference = DDP(reference)
    lifted = DDP(lifted)
    ref_opt = DFCAdamW(reference.parameters(), lr=5e-4, weight_decay=0.01, enable_fiber=False)
    dfc_opt = DFCAdamW(lifted.parameters(), lr=5e-4, weight_decay=0.01, enable_fiber=True)
    channel = TorchSignFiberChannel(dfc_opt)

    rng = np.random.default_rng(8200 + rank)
    payload = rng.bytes(channel.byte_capacity)
    channel.write_bytes(0, payload)
    gen = torch.Generator().manual_seed(8300 + rank)

    loss_fn = nn.MSELoss()
    for _ in range(args.steps):
        x = torch.randn(23, 64, generator=gen)
        y = torch.randn(23, 32, generator=gen)
        for model, optimizer in ((reference, ref_opt), (lifted, dfc_opt)):
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            optimizer.step()

        for a, b in zip(reference.parameters(), lifted.parameters()):
            if not torch.equal(a, b):
                raise AssertionError(f"rank {rank}: DFC changed DDP parameter trajectory")

    if channel.read_bytes(0, len(payload)) != payload:
        raise AssertionError(f"rank {rank}: hidden payload changed")

    local_digest = tensor_digest(lifted.parameters())
    digests = [None for _ in range(world)]
    payload_digests = [None for _ in range(world)]
    dist.all_gather_object(digests, local_digest)
    dist.all_gather_object(payload_digests, hashlib.sha256(payload).hexdigest())

    if len(set(digests)) != 1:
        raise AssertionError(f"DDP semantic parameters diverged across ranks: {digests}")
    if len(set(payload_digests)) != world:
        raise AssertionError("rank-specific hidden payloads unexpectedly identical")

    if rank == 0:
        coordinates = sum(p.numel() for p in lifted.parameters())
        report = {
            "schema": "dfc-github-cpu-ddp-v1",
            "status": "PASS",
            "backend": "gloo",
            "world_size": world,
            "steps": args.steps,
            "coordinates_per_rank": coordinates,
            "coordinate_updates": coordinates * args.steps * world,
            "semantic_parameter_sha256": digests[0],
            "rank_payload_sha256": payload_digests,
            "payloads_distinct": True,
            "cuda_available": torch.cuda.is_available(),
            "claim": "rank-distinct DFC-SIGN payloads preserve the bit-identical DDP semantic parameter trajectory",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
