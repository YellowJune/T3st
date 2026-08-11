from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def load(p): return json.loads(Path(p).read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--memory',required=True);ap.add_argument('--gpu-preflight',required=True);ap.add_argument('--qwen-external',required=True);ap.add_argument('--qwen-dfc',required=True);ap.add_argument('--partial-dir',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    mem=load(a.memory);pre=load(a.gpu_preflight);qe=load(a.qwen_external);qd=load(a.qwen_dfc)
    gates={}
    gates['gpu_exactness_zero_failures']=all(pre.get(k,1)==0 for k in ['residual_bit_failures','compressed_gradient_bit_failures','parameter_bit_failures','semantic_moment_bit_failures'])
    bpc=mem['measured_eliminated_bytes_per_coordinate'];gates['measured_memory_approximately_4B_per_coordinate']=abs(bpc-4.0)<=0.01
    gates['qwen_same_protocol']=all(qe[k]==qd[k] for k in ['model','requested_revision','resolved_hub_revision','total_trainable_parameters','steps','seq_len'])
    P=qe['total_trainable_parameters'];expected=4*P
    gates['qwen_external_allocates_full_fp32_ef']=qe['external_ef_allocated_bytes']==expected
    gates['qwen_dfc_allocates_zero_external_ef']=qd['external_ef_allocated_bytes']==0 and qd['dfc_low16_capacity_bytes']==expected
    after_saved=qe['after_ef_allocated_bytes']-qd['after_ef_allocated_bytes'];peak_saved=qe['peak_hbm_bytes']-qd['peak_hbm_bytes']
    gates['qwen_post_ef_hbm_saved_at_least_95pct']=after_saved>=0.95*expected
    gates['qwen_peak_hbm_saved_positive']=peak_saved>0
    loss_diff=max(abs(x-y) for x,y in zip(qe['losses'],qd['losses']))
    gates['qwen_loss_trajectory_match']=loss_diff<=1e-4
    gates['qwen_model_digest_match']=qe['model_sha256']==qd['model_sha256']
    rows=[];pdir=Path(a.partial_dir)
    for p in sorted(pdir.glob('*.json')):
        d=load(p)
        if d.get('complete'): rows.append(d)
    by={(r['method'],r['seed']):r for r in rows};seeds=sorted(set(r['seed'] for r in rows));paired=[]
    for s in seeds:
        e=by.get(('derpp',s));d=by.get(('dfc_sign_derpp',s))
        if e and d:paired.append({'seed':s,'final_diff':d['final_average_accuracy']-e['final_average_accuracy'],'forgetting_diff':d['average_forgetting']-e['average_forgetting'],'current_diff':d['current_task_accuracy']-e['current_task_accuracy']})
    gates['partial_three_paired_seeds']=len(paired)>=3
    if paired:
        mf=float(np.mean([x['final_diff'] for x in paired]));mfg=float(np.mean([x['forgetting_diff'] for x in paired]));mc=float(np.mean([x['current_diff'] for x in paired]))
    else: mf=mfg=mc=float('nan')
    gates['partial_final_gain_gate']=bool(paired) and mf>=0.05
    gates['partial_forgetting_gate']=bool(paired) and mfg<=-0.05
    gates['partial_current_task_gate']=bool(paired) and mc>=-0.05
    passed=all(gates.values())
    result={'schema_version':1,'protocol':'dfc-kaggle-final-aggregate-v1','pass':passed,'gates':gates,'memory_eliminated_bytes_per_coordinate':bpc,'qwen05b_expected_external_ef_bytes':expected,'qwen05b_after_ef_hbm_saved_bytes':after_saved,'qwen05b_peak_hbm_saved_bytes':peak_saved,'qwen05b_max_loss_abs_diff':loss_diff,'partial_paired':paired,'partial_mean_final_diff':mf,'partial_mean_forgetting_diff':mfg,'partial_mean_current_diff':mc,'claim_boundary':'7B/30B figures remain projections unless separately executed; this aggregate seals only executed Kaggle rows.'}
    canonical=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=True).encode();result['result_sha256']=hashlib.sha256(canonical).hexdigest();Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True,allow_nan=True))
if __name__=='__main__':main()
