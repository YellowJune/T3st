from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import torch
from torch_fiber import DFCLow16AdamW
from dfc_ef_residual import ExternalErrorFeedback, DFCLow16ErrorFeedback, semantic_high16_bits


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--coordinates',type=int,default=1_000_003);ap.add_argument('--updates',type=int,default=25);ap.add_argument('--output',required=True);a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    d=torch.device('cuda');torch.manual_seed(4441);torch.cuda.manual_seed_all(4441)
    p0=torch.linspace(-.25,.25,a.coordinates,device=d,dtype=torch.float32);p1=torch.nn.Parameter(p0.clone());p2=torch.nn.Parameter(p0.clone());del p0
    oe=DFCLow16AdamW([p1],lr=2e-3,enable_fiber=False);od=DFCLow16AdamW([p2],lr=2e-3,enable_fiber=True);ee=ExternalErrorFeedback([p1]);ed=DFCLow16ErrorFeedback(od,[p2])
    residual_fail=grad_fail=param_fail=semantic_fail=0;started=time.perf_counter()
    gen=torch.Generator(device=d);gen.manual_seed(9281)
    for _ in range(a.updates):
        g=torch.randn(a.coordinates,device=d,dtype=torch.float32,generator=gen)
        p1.grad=g.clone();p2.grad=g
        ee.compress_grads_();ed.compress_grads_();rd=ed.decoded_residuals()[0]
        residual_fail+=int(torch.count_nonzero(ee.residuals[0].view(torch.int32)!=rd.view(torch.int32)).item())
        grad_fail+=int(torch.count_nonzero(p1.grad.view(torch.int32)!=p2.grad.view(torch.int32)).item())
        oe.step();od.step();oe.zero_grad(set_to_none=True);od.zero_grad(set_to_none=True)
        param_fail+=int(torch.count_nonzero(p1.detach().view(torch.int32)!=p2.detach().view(torch.int32)).item())
        se,sd=oe.state[p1],od.state[p2]
        semantic_fail+=int(torch.count_nonzero(semantic_high16_bits(se['exp_avg'])!=semantic_high16_bits(sd['exp_avg'])).item())
        semantic_fail+=int(torch.count_nonzero(semantic_high16_bits(se['exp_avg_sq'])!=semantic_high16_bits(sd['exp_avg_sq'])).item())
    torch.cuda.synchronize();prop=torch.cuda.get_device_properties(0)
    result={'schema_version':1,'protocol':'dfc-low16-ef-gpu-preflight-v1','gpu':prop.name,'compute_capability':[prop.major,prop.minor],'coordinates':a.coordinates,'updates':a.updates,'residual_bit_failures':residual_fail,'compressed_gradient_bit_failures':grad_fail,'parameter_bit_failures':param_fail,'semantic_moment_bit_failures':semantic_fail,'external_ef_bytes':ee.allocated_bytes,'dfc_external_ef_bytes':ed.allocated_bytes,'wall_seconds':time.perf_counter()-started}
    canonical=json.dumps(result,sort_keys=True,separators=(',',':')).encode();result['result_sha256']=hashlib.sha256(canonical).hexdigest()
    if any(result[k] for k in ['residual_bit_failures','compressed_gradient_bit_failures','parameter_bit_failures','semantic_moment_bit_failures']):raise RuntimeError(json.dumps(result))
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
