from __future__ import annotations

import torch

from chunked_low16_adamw import DFCLow16AdamWChunked, stride_no_error_feedback_inplace_
from torch_fiber import DFCLow16AdamW, HIGH16_MASK_I32


def _decoded(t):
    return torch.bitwise_and(t.view(torch.int32), HIGH16_MASK_I32).view(torch.float32)


def _install_payload(opt, p, salt1=17, salt2=23):
    s = opt.state[p]
    for key, salt in [('exp_avg', salt1), ('exp_avg_sq', salt2)]:
        bits = s[key].view(torch.int32)
        payload = (torch.arange(bits.numel(), dtype=torch.int32) * salt) & 0xFFFF
        bits.copy_((bits & HIGH16_MASK_I32) | payload)


def test_chunked_fp32_matches_reference_with_and_without_payload():
    for enable_fiber in (False, True):
        gen = torch.Generator().manual_seed(123)
        init = torch.randn(5003, generator=gen, dtype=torch.float32)
        p_ref = torch.nn.Parameter(init.clone())
        p_chk = torch.nn.Parameter(init.clone())
        ref = DFCLow16AdamW([p_ref], lr=2e-4, weight_decay=0.01, enable_fiber=enable_fiber)
        chk = DFCLow16AdamWChunked([p_chk], lr=2e-4, weight_decay=0.01,
                                  enable_fiber=enable_fiber, chunk_coordinates=257)
        if enable_fiber:
            _install_payload(ref, p_ref); _install_payload(chk, p_chk)
        for _ in range(15):
            g = torch.randn(p_ref.shape, generator=gen, dtype=torch.float32)
            p_ref.grad = g.clone(); p_chk.grad = g.clone()
            ref.step(); chk.step()
            assert torch.equal(p_ref.view(torch.int32), p_chk.view(torch.int32))
            sr, sc = ref.state[p_ref], chk.state[p_chk]
            assert torch.equal(_decoded(sr['exp_avg']).view(torch.int32), _decoded(sc['exp_avg']).view(torch.int32))
            assert torch.equal(_decoded(sr['exp_avg_sq']).view(torch.int32), _decoded(sc['exp_avg_sq']).view(torch.int32))
            if enable_fiber:
                assert torch.equal(sr['exp_avg'].view(torch.int32) & 0xFFFF,
                                   sc['exp_avg'].view(torch.int32) & 0xFFFF)
                assert torch.equal(sr['exp_avg_sq'].view(torch.int32) & 0xFFFF,
                                   sc['exp_avg_sq'].view(torch.int32) & 0xFFFF)


def test_chunked_fp16_fiber_is_semantically_invisible_and_sparse_safe():
    """The FP16 path uses an FP32 final ratio to avoid eps underflow on zeros."""
    gen = torch.Generator().manual_seed(456)
    init = torch.randn(5003, generator=gen, dtype=torch.float32).to(torch.float16)
    p_zero = torch.nn.Parameter(init.clone())
    p_fiber = torch.nn.Parameter(init.clone())
    zero = DFCLow16AdamWChunked([p_zero], lr=2e-4, enable_fiber=False, chunk_coordinates=257)
    fiber = DFCLow16AdamWChunked([p_fiber], lr=2e-4, enable_fiber=True, chunk_coordinates=257)
    _install_payload(fiber, p_fiber)
    for step in range(20):
        # Deliberately sparse: most second-moment coordinates stay exactly zero
        # on early steps, which previously exposed FP16 epsilon underflow.
        g = torch.zeros_like(p_zero)
        dense = torch.randn(p_zero.shape, generator=gen, dtype=torch.float32).to(torch.float16)
        g[(step % 7)::7] = dense[(step % 7)::7]
        p_zero.grad = g.clone(); p_fiber.grad = g.clone()
        zero.step(); fiber.step()
        assert torch.isfinite(p_zero).all() and torch.isfinite(p_fiber).all()
        assert torch.equal(p_zero.view(torch.int16), p_fiber.view(torch.int16))
        sz, sf = zero.state[p_zero], fiber.state[p_fiber]
        assert torch.equal(_decoded(sz['exp_avg']).view(torch.int32), _decoded(sf['exp_avg']).view(torch.int32))
        assert torch.equal(_decoded(sz['exp_avg_sq']).view(torch.int32), _decoded(sf['exp_avg_sq']).view(torch.int32))


def test_noef_stride_keeps_expected_positions():
    g = torch.arange(17, dtype=torch.float32)
    sent = stride_no_error_feedback_inplace_(g, stride=4, offset=2, global_start=3)
    expected = torch.zeros(17)
    expected[3::4] = torch.arange(17, dtype=torch.float32)[3::4]
    assert torch.equal(g, expected)
    assert sent == len(expected[3::4])
