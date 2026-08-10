# DFC-SIGN v2.1 validation report

This document records only executed evidence or executable hardware gates on branch `dfc-v2.1-validation`. It is intended to be the manuscript-integration source of truth.

## 1. Full-FP32 Qwen continual adaptation: sealed partial-finetuning result

**Status: PASS.**

Source experiment:
- model: `Qwen/Qwen2.5-0.5B`
- pinned model revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- source workflow run: `31315481945`
- source commit: `08962f67fe8e2d78ccfce2f2383024d933f02d8f`
- protocol: `qwen-causal-partial-v1`
- five sealed seeds: `1901, 1931, 1951, 1987, 2017`
- adaptation: full-FP32 AdamW state, partial fine-tuning of the final transformer block (14,912,384 trainable parameters)
- external allocation: exactly 512 bytes for both external DER++ and DFC-SIGN+DER++
- 4 sequential tasks, 128 optimizer steps/task, 512 updates total, identical two-prompt-slot update shape
- replay record: 104 bytes

The original aggregate job failed before analysis because NumPy was absent from the aggregate-only runner. All ten method-by-seed experiment jobs had already completed successfully and uploaded immutable rows. Recovery workflow run `31368127648` downloaded those original artifacts, recomputed every accuracy matrix metric, re-audited physical-resource identities, and applied the original predeclared gate. Recovery completed successfully. Its summary artifact has GitHub artifact digest `sha256:5cd3061ef533276b05c2a144461d0deccb3ee2b6e399d1bb7661415dc659bb7c` and result digest `c75ccc686fd36cc004fcafe0d9035afef3896e5437cc64820d636c468eb1c01a`.

| Metric | External DER++ | DFC-SIGN + DER++ | Paired DFC - DER++ |
|---|---:|---:|---:|
| Final average accuracy | 36.250% | 79.375% | **+43.125 p.p.** |
| Average forgetting | 84.167 p.p. | 25.833 p.p. | **-58.333 p.p.** |
| Current-task accuracy | 99.375% | 98.125% | **-1.250 p.p.** |
| Replay-record capacity | 4 | 17,928 | +17,924 |
| Mean wall time | 252.31 s | 242.90 s | -9.41 s |

Per-seed final-accuracy gains are `+34.375, +53.125, +50.000, +50.000, +28.125` percentage points. Per-seed forgetting differences are all favorable: `-45.833, -70.833, -70.833, -62.500, -41.667` points. The current-task differences are `-3.125, 0, -3.125, +3.125, -3.125` points.

The predeclared paired gate was:
- mean final-accuracy gain >= +10 p.p.;
- mean forgetting difference <= -5 p.p.;
- mean current-task difference >= -5 p.p.

All three conditions pass. The sign fiber contributes 1,864,048 addressable bytes because the 14,912,384 ordinary FP32 second-moment coordinates contribute one exact sign bit each. No extra optimizer array is allocated. External DER++ and DFC-SIGN+DER++ both retain the same 512-byte external allocation and the same ordinary FP32 moment allocation; DFC-SIGN changes only which equivalent physical representatives are addressable.

This is a controlled sequential-task partial-finetuning experiment. It is not presented as a natural-domain corpus benchmark.

## 2. Real-data AG News LoRA stress test

**Status: retention improvement, predeclared plasticity gate FAIL.**

A separate five-seed sealed run used `fancyzhx/ag_news` with Qwen2.5-0.5B LoRA, four class-incremental tasks, 512 external bytes, and seeds `911, 929, 947, 971, 997`. All five seed jobs completed; the original aggregate job again failed only because NumPy was absent. Recovery run `31368332240` reconstructed the summary and exposed the actual scientific gate result.

| Metric | External DER++ | DFC-SIGN + DER++ | Paired DFC - DER++ |
|---|---:|---:|---:|
| Final average accuracy | 27.734% | 37.109% | **+9.375 p.p.** |
| Average forgetting | 82.500 p.p. | 61.667 p.p. | **-20.833 p.p.** |
| Current-task accuracy | 89.609% | 79.219% | **-10.391 p.p.** |
| Replay-record capacity | 4 | 54 | +50 |

The predeclared final-gain and forgetting conditions pass, but the current-task condition required a difference no worse than -5 p.p. and therefore fails. This row is retained as a falsifier/stress-test result rather than promoted to the main positive claim.

A development-only v2 protocol then reduced replay to 25% of each batch (3 current : 1 replay). The single development seed still showed a large current-task tradeoff (DER++ 98.438% vs DFC-SIGN 63.672%), so no post-hoc sealed claim is made from that pilot.

## 3. Fused Triton DFC-SIGN AdamW

**Implementation status: matched fused-vs-fused benchmark ready; actual GPU timing not yet executed on an available CUDA runner.**

`remote/triton_dfc_adamw.py` contains both a reference fused AdamW Triton kernel and a DFC-SIGN fused AdamW Triton kernel. The arithmetic order and p/g/m/v memory footprint are matched; DFC-SIGN adds only sign extraction, magnitude decode, and sign re-embedding around the same FP32 update.

`remote/benchmark_triton_dfc.py` first verifies bitwise parameter/first-moment/decoded-second-moment equality and payload persistence, then measures both fused kernels using CUDA Events. It reports per-size reference latency, DFC latency, percent overhead, effective bandwidth, median overhead, and maximum overhead for 1M, 4M, 16M, and 32M coordinates.

`.github/workflows/dfc-gpu-validation.yml` is the actual-hardware gate. It requires a CUDA-capable self-hosted runner, checks `nvidia-smi`, runs Triton exactness tests, and then executes the matched benchmark. No valid CUDA runner has completed this workflow yet. The older `DFC Triton actual-GPU gate` job remains queued for a missing GPU runner label. Therefore no GPU timing number is admitted to the paper at this point.

The manuscript may state that the matched fused benchmark and exactness gate are implemented and ready, but **must not claim a measured 1--5% overhead until the JSON artifact is produced by a real CUDA run**.

## 4. Manuscript integration decision

The first requested paradigm-strengthening criterion is now empirically satisfied in the strongest clean form currently available: full-FP32 DFC-SIGN on an actual 0.5B-parameter Qwen model with partial fine-tuning, five sealed seeds, equal external bytes, equal update/batch shape, and a large retention improvement while passing the predeclared current-task plasticity bound.

The second criterion is reduced to a hardware-execution dependency rather than an implementation dependency. The fused-vs-fused code path, exactness guard, timing harness, and GPU workflow exist; only an available CUDA runner is missing.

For claim discipline, the primary new empirical statement should use the sealed partial-finetuning PASS. The AG News LoRA experiment should appear as a stress-test/falsifier result showing that memory capacity alone does not remove the replay-plasticity tradeoff. This separation strengthens rather than weakens the evidence hierarchy because negative rows are preserved instead of tuned away after inspection.
