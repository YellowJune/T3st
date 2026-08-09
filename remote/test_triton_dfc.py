"""CUDA-only exactness test for matched fused AdamW and DFC-SIGN kernels."""

from __future__ import annotations
import unittest
import torch
from triton_dfc_adamw import dfc_adamw_step, reference_adamw_step, triton

@unittest.skipUnless(torch.cuda.is_available() and triton is not None, "CUDA/Triton unavailable")
class TritonDFCNoninterferenceTests(unittest.TestCase):
    def test_dfc_matches_fused_reference_and_preserves_payload(self):
        torch.manual_seed(71); size=65_537
        parameter_ref=torch.randn(size,device="cuda",dtype=torch.float32); parameter_dfc=parameter_ref.clone(); gradient=torch.randn(size,device="cuda",dtype=torch.float32)
        first_ref=torch.randn(size,device="cuda",dtype=torch.float32); first_dfc=first_ref.clone(); magnitude=torch.rand(size,device="cuda",dtype=torch.float32); words=magnitude.view(torch.int32)&0x7FFFFFFF; payload=torch.randint(0,2,(size,),device="cuda",dtype=torch.int32)<<31; second_ref=magnitude.clone(); second_dfc=(words|payload).view(torch.float32).clone()
        kwargs=dict(step=17,lr=4e-4,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01)
        reference_adamw_step(parameter_ref,gradient,first_ref,second_ref,**kwargs); dfc_adamw_step(parameter_dfc,gradient,first_dfc,second_dfc,**kwargs); torch.cuda.synchronize()
        self.assertTrue(torch.equal(parameter_ref,parameter_dfc)); self.assertTrue(torch.equal(first_ref,first_dfc)); self.assertTrue(torch.equal(second_ref.view(torch.int32),second_dfc.view(torch.int32)&0x7FFFFFFF)); self.assertTrue(torch.equal(second_dfc.view(torch.int32)&-2147483648,payload))

    def test_payload_choice_cannot_change_decoded_trajectory(self):
        torch.manual_seed(73); size=131_071
        parameter_a=torch.randn(size,device="cuda",dtype=torch.float32); parameter_b=parameter_a.clone(); gradient=torch.randn(size,device="cuda",dtype=torch.float32); first_a=torch.randn(size,device="cuda",dtype=torch.float32); first_b=first_a.clone(); magnitude=torch.rand(size,device="cuda",dtype=torch.float32); words=magnitude.view(torch.int32)&0x7FFFFFFF; payload_a=torch.randint(0,2,(size,),device="cuda",dtype=torch.int32)<<31; payload_b=torch.randint(0,2,(size,),device="cuda",dtype=torch.int32)<<31; second_a=(words|payload_a).view(torch.float32).clone(); second_b=(words|payload_b).view(torch.float32).clone()
        kwargs=dict(step=23,lr=7e-4,betas=(0.9,0.999),eps=1e-8,weight_decay=0.02)
        dfc_adamw_step(parameter_a,gradient,first_a,second_a,**kwargs); dfc_adamw_step(parameter_b,gradient,first_b,second_b,**kwargs); torch.cuda.synchronize()
        self.assertTrue(torch.equal(parameter_a,parameter_b)); self.assertTrue(torch.equal(first_a,first_b)); self.assertTrue(torch.equal(second_a.view(torch.int32)&0x7FFFFFFF,second_b.view(torch.int32)&0x7FFFFFFF)); self.assertTrue(torch.equal(second_a.view(torch.int32)&-2147483648,payload_a)); self.assertTrue(torch.equal(second_b.view(torch.int32)&-2147483648,payload_b))

if __name__=="__main__": unittest.main()
