# DFC GitHub CPU Frontier Validation

This directory contains validation that is deliberately executable on ordinary GitHub-hosted CPU runners. It extends the `dfc-v2.1-validation` evidence without requiring an H100/H200/B200/B300 or any CUDA device.

## Publication gates

The suite checks:

- adversarial DFC-SIGN payload patterns (all-zero, all-one, alternating, random) while preserving the exact semantic AdamW trajectory;
- FP32/BF16/FP16 parameter-dtype coverage using ordinary FP32 optimizer state;
- payload reads/writes spanning optimizer-tensor boundaries;
- checkpoint -> restore -> continued-training trajectory and payload preservation for DFC-SIGN and DFC-LOW16;
- multiple AdamW parameter groups, weight decay, accumulated gradients, and deliberately missing gradients;
- closed-form capacity identities across boundary sizes;
- finite IEEE-754 subnormal / sign-payload boundary behavior;
- multi-million-coordinate DFC-SIGN and DFC-LOW16 stress trajectories;
- two-rank CPU/Gloo DDP with different hidden payloads on each rank while semantic parameters remain bit-identical.

A non-contiguous-storage probe is recorded as a diagnostic rather than a publication gate, because the current optimizer contract does not claim arbitrary optimizer-state strides.

## Explicitly excluded

No result from this workflow is used to claim:

- GPU latency, throughput, or kernel speedup;
- Nsight DRAM/L2 hardware counters;
- H100/H200/B200/B300 architecture generalization;
- CUDA/Triton runtime overhead.

Those claims require actual accelerator hardware and must remain separately gated.

## Evidence contract

Every run emits JSON plus SHA-256 hashes. The workflow runs the single-process suite on Python 3.11 and 3.12, runs a two-process Gloo DDP gate, uploads all raw JSON, aggregates it, and persists the aggregate on an isolated `result-<run>-cpu-frontier` branch. A failed publication gate fails CI rather than being silently converted into a positive result.
