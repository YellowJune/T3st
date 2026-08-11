from __future__ import annotations
import io
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_fiber import DFCLow16AdamW
from dfc_ef_residual import (
    pack_fp32_residual_, unpack_fp32_residual, semantic_high16_bits,
    ExternalErrorFeedback, DFCLow16ErrorFeedback,
)


def assert_bits_equal(a: torch.Tensor, b: torch.Tensor):
    assert a.dtype == b.dtype == torch.float32
    assert torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def test_pack_roundtrip_and_semantics():
    gen = torch.Generator().manual_seed(123)
    first = torch.randn(4099, generator=gen, dtype=torch.float32)
    second = torch.randn(4099, generator=gen, dtype=torch.float32)
    residual = torch.randn(4099, generator=gen, dtype=torch.float32)
    f_sem = semantic_high16_bits(first).clone(); s_sem = semantic_high16_bits(second).clone()
    pack_fp32_residual_(first, second, residual)
    assert torch.equal(f_sem, semantic_high16_bits(first))
    assert torch.equal(s_sem, semantic_high16_bits(second))
    assert_bits_equal(residual, unpack_fp32_residual(first, second))


def _make_pair(n=257):
    p_ext = torch.nn.Parameter(torch.linspace(-.5,.5,n,dtype=torch.float32))
    p_dfc = torch.nn.Parameter(p_ext.detach().clone())
    opt_ext = DFCLow16AdamW([p_ext], lr=3e-3, enable_fiber=False)
    opt_dfc = DFCLow16AdamW([p_dfc], lr=3e-3, enable_fiber=True)
    ef_ext = ExternalErrorFeedback([p_ext])
    ef_dfc = DFCLow16ErrorFeedback(opt_dfc, [p_dfc])
    return p_ext,p_dfc,opt_ext,opt_dfc,ef_ext,ef_dfc


def _one_step(pair, grad):
    p_ext,p_dfc,opt_ext,opt_dfc,ef_ext,ef_dfc = pair
    p_ext.grad=grad.clone(); p_dfc.grad=grad.clone()
    ef_ext.compress_grads_(); ef_dfc.compress_grads_()
    assert_bits_equal(ef_ext.residuals[0], ef_dfc.decoded_residuals()[0])
    assert torch.equal(p_ext.grad.view(torch.int32), p_dfc.grad.view(torch.int32))
    opt_ext.step(); opt_dfc.step(); opt_ext.zero_grad(set_to_none=True); opt_dfc.zero_grad(set_to_none=True)
    assert torch.equal(p_ext.detach().view(torch.int32), p_dfc.detach().view(torch.int32))
    se,sd=opt_ext.state[p_ext],opt_dfc.state[p_dfc]
    assert torch.equal(semantic_high16_bits(se['exp_avg']), semantic_high16_bits(sd['exp_avg']))
    assert torch.equal(semantic_high16_bits(se['exp_avg_sq']), semantic_high16_bits(sd['exp_avg_sq']))


def test_paired_trajectory():
    pair=_make_pair(); gen=torch.Generator().manual_seed(991)
    for _ in range(30): _one_step(pair, torch.randn(pair[0].shape,generator=gen))
    assert pair[4].allocated_bytes == pair[0].numel()*4
    assert pair[5].allocated_bytes == 0


def _save(pair, path):
    p_ext,p_dfc,opt_ext,opt_dfc,ef_ext,ef_dfc=pair
    torch.save({
        'p_ext':p_ext.detach().clone(),'p_dfc':p_dfc.detach().clone(),
        'opt_ext':opt_ext.state_dict(),'opt_dfc':opt_dfc.state_dict(),
        'ef_ext':ef_ext.state_dict(),'ef_dfc':ef_dfc.state_dict(),
    },path)


def _load(path):
    state=torch.load(path,weights_only=False)
    pair=_make_pair(n=state['p_ext'].numel())
    p_ext,p_dfc,opt_ext,opt_dfc,ef_ext,ef_dfc=pair
    p_ext.data.copy_(state['p_ext']);p_dfc.data.copy_(state['p_dfc'])
    opt_ext.load_state_dict(state['opt_ext']);opt_dfc.load_state_dict(state['opt_dfc'])
    ef_ext.load_state_dict(state['ef_ext']);ef_dfc.load_state_dict(state['ef_dfc'])
    return pair


def test_checkpoint_resume_preserves_fiber_payload():
    gen=torch.Generator().manual_seed(2026); grads=[torch.randn(257,generator=gen) for _ in range(25)]
    uninterrupted=_make_pair(); resumed=_make_pair()
    for g in grads: _one_step(uninterrupted,g)
    for g in grads[:11]: _one_step(resumed,g)
    buf=io.BytesIO(); _save(resumed,buf); buf.seek(0); resumed=_load(buf)
    for g in grads[11:]: _one_step(resumed,g)
    for a,b in zip(uninterrupted[:2],resumed[:2]): assert torch.equal(a.detach().view(torch.int32),b.detach().view(torch.int32))
    assert_bits_equal(uninterrupted[4].residuals[0],resumed[4].residuals[0])
    assert_bits_equal(uninterrupted[5].decoded_residuals()[0],resumed[5].decoded_residuals()[0])


if __name__=='__main__':
    test_pack_roundtrip_and_semantics(); test_paired_trajectory(); test_checkpoint_resume_preserves_fiber_payload(); print('DFC-EF CPU preflight: PASS')
