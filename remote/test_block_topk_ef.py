from __future__ import annotations

import pytest
import torch

from block_topk_ef import block_topk_dfc_inplace_, block_topk_external_inplace_, block_topk_noef_inplace_
from chunked_low16_adamw import DFCLow16AdamWChunked
from dfc_ef import PackedFP32Residual
from torch_fiber import HIGH16_MASK_I32


def _decoded_bits(t):
    return torch.bitwise_and(t.view(torch.int32), HIGH16_MASK_I32)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('chunk', [31, 257, 2048])
@pytest.mark.parametrize('ratio', [0.125, 0.25, 1.0])
def test_block_topk_external_dfc_exact(dtype, chunk, ratio):
    gen = torch.Generator().manual_seed(8181)
    n = 5003
    init = torch.randn(n, generator=gen, dtype=torch.float32).to(dtype)
    p_e = torch.nn.Parameter(init.clone())
    p_d = torch.nn.Parameter(init.clone())
    oe = DFCLow16AdamWChunked([p_e], lr=4e-4, enable_fiber=False, chunk_coordinates=113)
    od = DFCLow16AdamWChunked([p_d], lr=4e-4, enable_fiber=True, chunk_coordinates=113)
    residual = torch.zeros(n, dtype=torch.float32)
    channel = PackedFP32Residual(od); channel.zero_()
    for _ in range(12):
        g = torch.randn(n, generator=gen, dtype=torch.float32).to(dtype)
        ge, gd = g.clone(), g.clone()
        se = block_topk_external_inplace_(ge, residual, keep_ratio=ratio, chunk_coordinates=chunk)
        sd = block_topk_dfc_inplace_(p_d, gd, channel, keep_ratio=ratio, chunk_coordinates=chunk)
        assert se == sd
        assert torch.equal(ge, gd)
        assert torch.equal(residual.view(torch.int32), channel.read_for_parameter(p_d).view(torch.int32))
        p_e.grad = ge; p_d.grad = gd
        oe.step(); od.step()
        assert torch.equal(p_e, p_d)
        aes, ads = oe.state[p_e], od.state[p_d]
        assert torch.equal(_decoded_bits(aes['exp_avg']), _decoded_bits(ads['exp_avg']))
        assert torch.equal(_decoded_bits(aes['exp_avg_sq']), _decoded_bits(ads['exp_avg_sq']))


def test_noef_block_topk_count_and_sparsity():
    x = torch.tensor([1., -8., 2., 7., 5., -3., 4., 6., 10., 9.])
    sent = block_topk_noef_inplace_(x, keep_ratio=0.25, chunk_coordinates=4)
    # blocks 4,4,2 -> k 1,1,1
    assert sent == 3
    assert int((x != 0).sum()) == 3
    assert x[1] == -8 and x[7] == 6 and x[8] == 10
