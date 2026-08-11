from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import torch
import torch.distributed as dist
from block_topk_ef import block_topk_dfc_inplace_, block_topk_external_inplace_
from chunked_low16_adamw import DFCLow16AdamWChunked
from dfc_ef import PackedFP32Residual
from torch_fiber import HIGH16_MASK_I32


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def sem_sha(opt, p):
    h=hashlib.sha256(); st=opt.state[p]
    for key in ('exp_avg','exp_avg_sq'):
        b=(st[key].view(torch.int32)&HIGH16_MASK_I32).contiguous().numpy().tobytes(); h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--method',choices=['external','dfc'],required=True); ap.add_argument('--coordinates',type=int,default=250000); ap.add_argument('--steps',type=int,default=32); ap.add_argument('--output-dir',required=True); args=ap.parse_args()
    dist.init_process_group('gloo'); rank=dist.get_rank(); world=dist.get_world_size()
    if world < 2:
        raise RuntimeError('distributed validation requires at least two ranks')
    torch.set_num_threads(1); torch.manual_seed(7000)
    p=torch.nn.Parameter(torch.randn(args.coordinates,dtype=torch.float32))
    opt=DFCLow16AdamWChunked([p],lr=3e-4,enable_fiber=args.method=='dfc',chunk_coordinates=65536)
    residual=torch.zeros_like(p,dtype=torch.float32) if args.method=='external' else None
    ch=PackedFP32Residual(opt) if args.method=='dfc' else None
    if ch: ch.zero_()
    sent=0
    for step in range(args.steps):
        gen=torch.Generator().manual_seed(100000 + rank*1000 + step)
        g=torch.randn(args.coordinates,generator=gen,dtype=torch.float32)
        if args.method=='external': sent+=block_topk_external_inplace_(g,residual,keep_ratio=.125,chunk_coordinates=65536)
        else: sent+=block_topk_dfc_inplace_(p,g,ch,keep_ratio=.125,chunk_coordinates=65536)
        dist.all_reduce(g,op=dist.ReduceOp.SUM); g.div_(world)
        p.grad=g; opt.step()
    logical=residual if residual is not None else ch.read_for_parameter(p)
    row={'schema_version':1,'protocol':'dfc-ef-torch-distributed-gloo-v1','method':args.method,'rank':rank,'world_size':world,'coordinates':args.coordinates,'steps':args.steps,'transmitted_values_local':sent,'parameter_sha256':sha(p.view(torch.int32)),'semantic_optimizer_sha256':sem_sha(opt,p),'logical_residual_sha256':sha(logical.view(torch.int32)),'external_residual_bytes':int(residual.numel()*4) if residual is not None else 0,'dfc_fiber_capacity_bytes':int(4*p.numel()) if ch is not None else 0}
    root=Path(args.output_dir); root.mkdir(parents=True,exist_ok=True); (root/f'{args.method}_rank{rank}.json').write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
    dist.barrier(); dist.destroy_process_group()

if __name__=='__main__': main()
