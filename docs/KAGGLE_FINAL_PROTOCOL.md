# DFC Kaggle-only final validation protocol

This branch is the predeclared execution package for a free-accelerator final validation. It is designed so that the expensive rows are generated on Kaggle while CPU-only exactness/checkpoint logic is already gated by GitHub Actions.

## Scientific claim boundary

Executed Kaggle rows may support only what they actually measure. In particular:

- `qwen05b_full_ef_kaggle.py` is a **real full-parameter Qwen2.5-0.5B** execution of DFC-LOW16 error feedback versus an external FP32 residual.
- `dfc_ef_memory_benchmark.py` is an **actual GPU state-scale allocation** benchmark and measures the eliminated bytes per coordinate.
- 7B = 28,000,000,000 bytes and 30B = 120,000,000,000 bytes are **projections from the exact 4-byte/coordinate law**, not executed 7B/30B model runs unless a separate artifact explicitly says otherwise.
- Communication compression belongs to the compressor. The DFC claim is elimination of its model-scale auxiliary error-feedback allocation.
- The TPU row is cross-substrate decoder-fiber evidence. It is not an optimizer-memory-saving row.

## GPU kernel phases

`kaggle/dfc_kaggle_final_gpu.py` performs the following in order and seals every completed phase before proceeding:

1. environment and GPU capture;
2. independent CPU rerun of ReLU30, PERM/KV, hidden automaton, and DRAM/mmap persistence;
3. CUDA ReLU30 and PERM exactness/timing;
4. DFC-LOW16 error-feedback paired CUDA exactness;
5. actual GPU memory-scaling row measuring the four eliminated external bytes per coordinate;
6. matched fused Triton DFC-SIGN AdamW benchmark when the assigned accelerator supports Triton;
7. real full-parameter Qwen2.5-0.5B external-EF and DFC-EF executions;
8. three paired seeds of crash-resumable Qwen2.5-1.5B partial continual adaptation;
9. predeclared fail-closed aggregate gates;
10. SHA-256 manifest and a single result ZIP.

The 1.5B jobs checkpoint every 48 updates. Checkpoints include trainable tensors, the exact FP32 optimizer words carrying DFC payload, the external 512-byte envelope, replay-store integrity metadata, and Python/NumPy/Torch/CUDA RNG states. Resume uses a custom bit-preserving FP32 optimizer restore so FP16 trainable parameters cannot down-cast the physical DFC state.

## TPU phase

`kaggle_tpu/dfc_kaggle_tpu.py` independently executes ReLU30 and permutation fibers under JAX on the assigned TPU, including synchronized exactness/timing rows.

## Authentication and launch

Never place a Kaggle token in this repository. With the current official Kaggle CLI, supply a generated access token through the `KAGGLE_API_TOKEN` environment variable or the CLI token file, then push the two private kernels from the corresponding folders.

Example accelerator choices are controlled by the CLI `--accelerator` option. Use an accelerator actually available to the account. The GPU harness automatically uses a second GPU when Kaggle exposes one; otherwise it runs the paired jobs sequentially.

## Acceptance gates

The aggregate is deliberately strict. It requires zero CUDA exactness failures, approximately 4.00 measured eliminated bytes per coordinate, a real Qwen0.5B external allocation equal to `4P`, zero DFC external EF allocation, at least 95% of the expected post-EF HBM allocation difference, positive peak-HBM saving, matching Qwen loss/model trajectory, and three paired 1.5B continual seeds passing final-accuracy/forgetting/current-task gates.

A failed gate remains a negative result. It must not be silently retuned into a sealed positive claim.
