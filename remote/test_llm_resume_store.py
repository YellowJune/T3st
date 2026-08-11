from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from torch_fiber import DFCAdamW,TorchSignFiberChannel
from llm_continual_qwen import CombinedByteChannel,ReservoirStore,RECORD_BYTES
from llm_continual_qwen_partial_resume import _store_without_init,_load_optimizer_fp32_exact


def test_resume_store_does_not_clobber_fiber():
    p=torch.nn.Parameter(torch.zeros(10000,dtype=torch.float32))
    opt=DFCAdamW([p],enable_fiber=True)
    fiber=TorchSignFiberChannel(opt);channel=CombinedByteChannel(512,fiber);rng=np.random.default_rng(13);store=ReservoirStore(channel,rng)
    for i in range(20):store.insert(bytes([i%251])*RECORD_BYTES)
    digest,count,seen=store.digest(),store.count,store.seen
    opt_state=opt.state_dict();external=bytes(channel.external);rng_state=rng.bit_generator.state
    q=torch.nn.Parameter(torch.zeros_like(p));opt2=DFCAdamW([q],enable_fiber=True);_load_optimizer_fp32_exact(opt2,opt_state,[q],torch.device('cpu'))
    fiber2=TorchSignFiberChannel(opt2);channel2=CombinedByteChannel(512,fiber2);channel2.external[:]=external
    rng2=np.random.default_rng();rng2.bit_generator.state=rng_state;store2=_store_without_init(channel2,rng2)
    assert store2.digest()==digest
    assert store2.count==count and store2.seen==seen


def test_fp16_parameter_resume_keeps_fp32_moment_bits():
    p=torch.nn.Parameter(torch.zeros(4099,dtype=torch.float16))
    opt=DFCAdamW([p],enable_fiber=True)
    state=opt.state[p]
    g=torch.Generator().manual_seed(771)
    first=torch.randn(4099,generator=g,dtype=torch.float32)
    second=torch.rand(4099,generator=g,dtype=torch.float32)
    payload=(torch.randint(0,2,(4099,),generator=g,dtype=torch.int32)<<31)
    second_bits=(second.view(torch.int32)&0x7fffffff)|payload
    state['exp_avg'].copy_(first);state['exp_avg_sq'].copy_(second_bits.view(torch.float32));state['step']=17
    saved=opt.state_dict()
    q=torch.nn.Parameter(torch.zeros(4099,dtype=torch.float16));opt2=DFCAdamW([q],enable_fiber=True)
    _load_optimizer_fp32_exact(opt2,saved,[q],torch.device('cpu'))
    restored=opt2.state[q]
    assert restored['exp_avg'].dtype==torch.float32 and restored['exp_avg_sq'].dtype==torch.float32
    assert torch.equal(restored['exp_avg'].view(torch.int32),first.view(torch.int32))
    assert torch.equal(restored['exp_avg_sq'].view(torch.int32),second_bits)
    assert restored['step']==17

if __name__=='__main__':
    test_resume_store_does_not_clobber_fiber();test_fp16_parameter_resume_keeps_fp32_moment_bits();print('LLM replay-store and FP32 optimizer resume: PASS')
