"""Secondary memory-efficient EF baseline with an external FP16 residual.

This intentionally changes logical EF precision and therefore is NOT a
placement-equivalent DFC comparison. It exists to test the practical objection
that an external residual can itself be stored at lower precision.
"""
from __future__ import annotations
import hashlib, json
import torch
import llm_dfc_ef_qwen as core
from block_topk_ef import _k_for_block


def _alloc(params):
    return [torch.zeros_like(p, dtype=torch.float16, memory_format=torch.preserve_format) for p in params]

@torch.no_grad()
def _compress(method, params, *, external_residuals, channel, keep_ratio, chunk_coordinates):
    if method != 'external_ef':
        return core._original_fp16_baseline_compress(method, params, external_residuals=external_residuals, channel=channel, keep_ratio=keep_ratio, chunk_coordinates=chunk_coordinates)
    sent=0
    for i,p in enumerate(params):
        g=core.prepare_gradient(p)
        if g is None: continue
        r=external_residuals[i].view(-1); gf=g.view(-1)
        for st in range(0,gf.numel(),chunk_coordinates):
            en=min(gf.numel(),st+chunk_coordinates); gs=gf[st:en]; rs=r[st:en]
            compensated=gs.float()+rs.float(); k=_k_for_block(compensated.numel(),keep_ratio)
            if k==compensated.numel():
                communicated=compensated.to(dtype=g.dtype); gs.copy_(communicated); rs.copy_((compensated-communicated.float()).to(torch.float16)); sent+=k; continue
            idx=torch.topk(compensated.abs(),k,sorted=False).indices; chosen=compensated[idx]; communicated=chosen.to(dtype=g.dtype)
            residual_new=compensated; residual_new[idx]=chosen-communicated.float(); rs.copy_(residual_new.to(torch.float16)); gs.zero_(); gs[idx]=communicated; sent+=k
    return int(sent)

def _residual_digest(residuals):
    h=hashlib.sha256()
    for r in residuals: h.update(r.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

core.allocate_external_residuals=_alloc
core._original_fp16_baseline_compress=core.compress_gradients
core.compress_gradients=_compress
core.external_residual_digest=_residual_digest
_orig_run=core.run

def _run(args):
    r=_orig_run(args); P=r['trainable_parameters']; r['protocol']='qwen-external-fp16-ef-blocktopk-v1'; r['method']='external_fp16_ef'; r['external_residual_dtype']='float16'; r['actual_external_residual_bytes']=2*P; r['model_scale_external_residual_removed_bytes']=0; r.pop('result_sha256',None); raw=json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=True).encode(); r['result_sha256']=hashlib.sha256(raw).hexdigest(); return r
core.run=_run

if __name__=='__main__': core.main()
