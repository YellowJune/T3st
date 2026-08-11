from __future__ import annotations

import pytest
import torch

from chunked_low16_adamw import DFCLow16AdamWChunked, stride_no_error_feedback_inplace_
from torch_fiber import DFCLow16AdamW, HIGH16_MASK_I32


def _decoded(t):
    return torch.bitwise_and(t.view(torch.int32), HIGH16_MASK_I32).view(torch.float32)


@pytest.mark.parametrize('enable_fiber', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_chunked_optimizer_matches_reference(enable_fiber, dtype):
    gen = torch.Generator().manual_seed(123)
    init = torch.randn(5003, generator=gen, dtype=torch.float32).to(dtype)
    p_ref = torch.nn.Parameter(init.clone())
    p_chk = torch.nn.Parameter(init.clone())
    ref = DFCLow16AdamW([p_ref], lr=2e-4, weight_decay=0.01, enable_fiber=enable_fiber)
    chk = DFCLow16AdamWChunked([p_chk], lr=2e-4, weight_decay=0.01, enable_fiber=enable_fiber, chunk_coordinates=257)

    if enable_fiber:
        # Install deterministic low-word payload in both optimizers so payload
        # preservation is tested in addition to semantic trajectory equality.
        for opt, p in [(ref, p_ref), (chk, p_chk)]:
            s = opt.state[p]
            for key, salt in [('exp_avg', 17), ('exp_avg_sq', 23)]:
                bits = s[key].view(torch.int32)
                payload = (torch.arange(bits.numel(), dtype=torch.int32) * salt) & 0xFFFF
                bits.copy_((bits & HIGH16_MASK_I32) | payload)

    for _ in range(15):
        g = torch.randn(p_ref.shape, generator=gen, dtype=torch.float32).to(dtype)
        p_ref.grad = g.clone(); p_chk.grad = g.clone()
        ref.step(); chk.step()
        view_dtype = torch.int32 if dtype == torch.float32 else torch.int16
        assert torch.equal(p_ref.view(view_dtype), p_chk.view(view_dtype))
        sr, sc = ref.state[p_ref], chk.state[p_chk]
        assert torch.equal(_decoded(sr['exp_avg']).view(torch.int32), _decoded(sc['exp_avg']).view(torch.int32))
        assert torch.equal(_decoded(sr['exp_avg_sq']).view(torch.int32), _decoded(sc['exp_avg_sq']).view(torch.int32))
        if enable_fiber:
            assert torch.equal(sr['exp_avg'].view(torch.int32) & 0xFFFF, sc['exp_avg'].view(torch.int32) & 0xFFFF)
            assert torch.equal(sr['exp_avg_sq'].view(torch.int32) & 0xFFFF, sc['exp_avg_sq'].view(torch.int32) & 0xFFFF)


def test_noef_stride_keeps_expected_positions():
    g = torch.arange(17, dtype=torch.float32)
    sent = stride_no_error_feedback_inplace_(g, stride=4, offset=2, global_start=3)
    expected = torch.zeros(17)
    # (global_start + i) % 4 == 2 -> i % 4 == 3
    expected[3::4] = torch.arange(17, dtype=torch.float32)[3::4]
    assert torch.equal(g, expected)
    assert sent == len(expected[3::4])
