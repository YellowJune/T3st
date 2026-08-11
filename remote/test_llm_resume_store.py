from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from torch_fiber import DFCAdamW,TorchSignFiberChannel
from llm_continual_qwen import CombinedByteChannel,ReservoirStore,RECORD_BYTES
from llm_continual_qwen_partial_resume import _store_without_init


def test_resume_store_does_not_clobber_fiber():
    p=torch.nn.Parameter(torch.zeros(10000,dtype=torch.float32))
    opt=DFCAdamW([p],enable_fiber=True)
    fiber=TorchSignFiberChannel(opt);channel=CombinedByteChannel(512,fiber);rng=np.random.default_rng(13);store=ReservoirStore(channel,rng)
    for i in range(20):store.insert(bytes([i%251])*RECORD_BYTES)
    digest,count,seen=store.digest(),store.count,store.seen
    opt_state=opt.state_dict();external=bytes(channel.external);rng_state=rng.bit_generator.state
    q=torch.nn.Parameter(torch.zeros_like(p));opt2=DFCAdamW([q],enable_fiber=True);opt2.load_state_dict(opt_state)
    fiber2=TorchSignFiberChannel(opt2);channel2=CombinedByteChannel(512,fiber2);channel2.external[:]=external
    rng2=np.random.default_rng();rng2.bit_generator.state=rng_state;store2=_store_without_init(channel2,rng2)
    assert store2.digest()==digest
    assert store2.count==count and store2.seen==seen

if __name__=='__main__':
    test_resume_store_does_not_clobber_fiber();print('LLM replay-store resume: PASS')
