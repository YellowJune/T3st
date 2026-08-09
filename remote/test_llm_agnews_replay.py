from __future__ import annotations
import unittest
import numpy as np
import torch
from llm_continual_qwen_agnews import ReplayCodec, ReservoirStore, CombinedByteChannel, RECORD_BYTES
from torch_fiber import DFCAdamW, TorchSignFiberChannel

class AGNewsReplayCodecTests(unittest.TestCase):
    def test_round_trip_and_crc(self):
        ids=torch.arange(24,dtype=torch.int64)+100
        mask=torch.tensor([1]*19+[0]*5,dtype=torch.int64)
        logits=torch.tensor([-.5,.25,1.5,2.0],dtype=torch.float32)
        raw=ReplayCodec.encode(ids,mask,3,3,logits)
        self.assertEqual(len(raw),RECORD_BYTES)
        decoded=ReplayCodec.decode(raw)
        self.assertTrue(torch.equal(decoded['input_ids'],ids))
        self.assertTrue(torch.equal(decoded['attention_mask'],mask))
        self.assertEqual(decoded['label'],3)
        self.assertTrue(torch.allclose(decoded['logits'],logits,atol=1e-3,rtol=0))
        corrupt=bytearray(raw); corrupt[15]^=1
        with self.assertRaises(RuntimeError): ReplayCodec.decode(bytes(corrupt))

    def test_store_survives_optimizer_step(self):
        p=torch.nn.Parameter(torch.zeros(16384,dtype=torch.float32))
        opt=DFCAdamW([p],lr=1e-3,enable_fiber=True)
        fiber=TorchSignFiberChannel(opt)
        store=ReservoirStore(CombinedByteChannel(512,fiber),np.random.default_rng(5))
        ids=torch.arange(24,dtype=torch.int64)+100
        mask=torch.ones(24,dtype=torch.int64)
        logits=torch.tensor([0.,1.,2.,3.])
        for i in range(min(store.capacity_records,12)):
            store.insert(ReplayCodec.encode(ids+i,mask,i%4,i%4,logits+i))
        digest=store.digest(); count=store.count
        p.grad=torch.randn_like(p); opt.step()
        self.assertEqual(store.count,count)
        self.assertEqual(store.digest(),digest)

if __name__=='__main__': unittest.main()
