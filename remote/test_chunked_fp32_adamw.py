from __future__ import annotations

import torch

from chunked_fp32_adamw import FullFP32AdamWChunked


def _assert_state_close(actual: torch.Tensor, reference: torch.Tensor) -> None:
    """Audit arithmetic-order drift against torch.optim.AdamW.

    The chunked implementation evaluates the same FP32 recurrence but executes
    it in independently sliced kernels, so bitwise identity with PyTorch's
    monolithic foreach=False path is not the contract.  We require a tight FP32
    numerical match and separately cap the maximum absolute deviation.
    """
    max_abs = float((actual - reference).abs().max())
    assert max_abs < 3e-6, max_abs
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


def test_fp32_chunked_matches_torch_adamw_close():
    gen = torch.Generator().manual_seed(1001)
    x = torch.randn(4099, generator=gen)
    p1 = torch.nn.Parameter(x.clone())
    p2 = torch.nn.Parameter(x.clone())
    a = FullFP32AdamWChunked([p1], lr=3e-4, weight_decay=0.01, chunk_coordinates=127)
    b = torch.optim.AdamW([p2], lr=3e-4, weight_decay=0.01, foreach=False)
    for _ in range(20):
        g = torch.randn(x.shape, generator=gen)
        p1.grad = g.clone(); p2.grad = g.clone()
        a.step(); b.step()
    torch.testing.assert_close(p1, p2, rtol=3e-6, atol=3e-6)
    _assert_state_close(a.state[p1]['exp_avg'], b.state[p2]['exp_avg'])
    _assert_state_close(a.state[p1]['exp_avg_sq'], b.state[p2]['exp_avg_sq'])


def test_fp16_sparse_gradients_remain_finite_with_eps_1e8():
    gen = torch.Generator().manual_seed(1002)
    p = torch.nn.Parameter(torch.randn(5003, generator=gen).half())
    opt = FullFP32AdamWChunked([p], lr=3e-4, eps=1e-8, chunk_coordinates=113)
    for step in range(16):
        g = torch.zeros_like(p)
        d = torch.randn(p.shape, generator=gen).half()
        g[step % 8::8] = d[step % 8::8]
        p.grad = g
        opt.step()
        assert torch.isfinite(p).all()
