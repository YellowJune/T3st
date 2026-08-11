from __future__ import annotations

import torch

from chunked_fp32_adamw import FullFP32AdamWChunked


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
    assert torch.allclose(p1, p2, rtol=2e-6, atol=2e-6)
    assert torch.allclose(a.state[p1]['exp_avg'], b.state[p2]['exp_avg'], rtol=1e-6, atol=1e-7)
    assert torch.allclose(a.state[p1]['exp_avg_sq'], b.state[p2]['exp_avg_sq'], rtol=1e-6, atol=1e-7)


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
