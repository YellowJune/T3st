from __future__ import annotations

import io

import torch

from dfc_ef import PackedFP32Residual, memory_ledger, topk_error_feedback_dfc, topk_error_feedback_external
from torch_fiber import DFCLow16AdamW, HIGH16_MASK_I32


def _decoded(t: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_and(t.view(torch.int32), HIGH16_MASK_I32).view(torch.float32)


def test_memory_ledger_exact_capacity():
    x = memory_ledger(7_000_000_000)
    assert x.optimizer_state_bytes == 56_000_000_000
    assert x.external_residual_bytes == 28_000_000_000
    assert x.fiber_capacity_bytes == 28_000_000_000
    assert x.bytes_removed == 28_000_000_000


def test_packed_residual_bitwise_roundtrip():
    p = torch.nn.Parameter(torch.zeros(4096, dtype=torch.float32))
    opt = DFCLow16AdamW([p], enable_fiber=True)
    channel = PackedFP32Residual(opt)
    gen = torch.Generator().manual_seed(7)
    residual = torch.randn(p.shape, generator=gen, dtype=torch.float32)
    # Include signed zero and extreme finite values.
    residual[:6] = torch.tensor([0.0, -0.0, 1.0, -1.0, 1e-30, 1e30], dtype=torch.float32)
    channel.write_for_parameter(p, residual)
    recovered = channel.read_for_parameter(p)
    assert torch.equal(recovered.view(torch.int32), residual.view(torch.int32))


def test_external_and_dfc_ef_match_for_many_steps():
    gen = torch.Generator().manual_seed(1234)
    init = torch.randn(2048, generator=gen, dtype=torch.float32)
    p_ext = torch.nn.Parameter(init.clone())
    p_dfc = torch.nn.Parameter(init.clone())
    opt_ext = DFCLow16AdamW([p_ext], lr=3e-4, enable_fiber=False)
    opt_dfc = DFCLow16AdamW([p_dfc], lr=3e-4, enable_fiber=True)
    channel = PackedFP32Residual(opt_dfc)
    channel.zero_()
    residual_ext = torch.zeros_like(p_ext)

    for _ in range(100):
        g = torch.randn(p_ext.shape, generator=gen, dtype=torch.float32)
        c_ext, residual_ext = topk_error_feedback_external(g, residual_ext, keep_ratio=0.1)
        c_dfc = topk_error_feedback_dfc(p_dfc, g, channel, keep_ratio=0.1)
        assert torch.equal(c_ext.view(torch.int32), c_dfc.view(torch.int32))
        assert torch.equal(residual_ext.view(torch.int32), channel.read_for_parameter(p_dfc).view(torch.int32))
        p_ext.grad = c_ext.clone()
        p_dfc.grad = c_dfc.clone()
        opt_ext.step(); opt_dfc.step()
        assert torch.equal(p_ext.view(torch.int32), p_dfc.view(torch.int32))
        se = opt_ext.state[p_ext]; sd = opt_dfc.state[p_dfc]
        assert torch.equal(_decoded(se['exp_avg']).view(torch.int32), _decoded(sd['exp_avg']).view(torch.int32))
        assert torch.equal(_decoded(se['exp_avg_sq']).view(torch.int32), _decoded(sd['exp_avg_sq']).view(torch.int32))
        assert torch.equal(residual_ext.view(torch.int32), channel.read_for_parameter(p_dfc).view(torch.int32))


def test_checkpoint_resume_preserves_hidden_residual_and_future_trajectory():
    gen = torch.Generator().manual_seed(55)
    init = torch.randn(512, generator=gen)
    p = torch.nn.Parameter(init.clone())
    opt = DFCLow16AdamW([p], lr=1e-3, enable_fiber=True)
    ch = PackedFP32Residual(opt); ch.zero_()

    gradients = [torch.randn(p.shape, generator=gen) for _ in range(20)]
    for g in gradients[:9]:
        c = topk_error_feedback_dfc(p, g.float(), ch, 0.2)
        p.grad = c; opt.step()

    payload_before = ch.read_for_parameter(p).clone()
    buf = io.BytesIO()
    torch.save({'parameter': p.detach().clone(), 'optimizer': opt.state_dict()}, buf)
    buf.seek(0)

    p2 = torch.nn.Parameter(torch.empty_like(p))
    opt2 = DFCLow16AdamW([p2], lr=1e-3, enable_fiber=True)
    ckpt = torch.load(buf, weights_only=False)
    p2.data.copy_(ckpt['parameter'])
    opt2.load_state_dict(ckpt['optimizer'])
    ch2 = PackedFP32Residual(opt2)
    assert torch.equal(payload_before.view(torch.int32), ch2.read_for_parameter(p2).view(torch.int32))

    for g in gradients[9:]:
        c1 = topk_error_feedback_dfc(p, g.float(), ch, 0.2)
        c2 = topk_error_feedback_dfc(p2, g.float(), ch2, 0.2)
        assert torch.equal(c1.view(torch.int32), c2.view(torch.int32))
        p.grad = c1; p2.grad = c2
        opt.step(); opt2.step()
        assert torch.equal(p.view(torch.int32), p2.view(torch.int32))
        assert torch.equal(ch.read_for_parameter(p).view(torch.int32), ch2.read_for_parameter(p2).view(torch.int32))
