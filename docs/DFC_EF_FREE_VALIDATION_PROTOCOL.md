# DFC-EF Free-Compute Validation Protocol (predeclared)

Status: **frozen before any Kaggle GPU/TPU result is inspected**.

This document fixes the primary claims, controls, metrics, and pass/fail gates for
DFC-EF before the free-compute campaign is launched.  Later edits that change a
threshold or protocol must be versioned as post-hoc and must not overwrite this
record.

## 1. Claim under test

DFC-EF does **not** claim that decoder fibers create gradient compression.
Blockwise top-k (learning path) or stride-k (systems microbenchmark) supplies the
compression.  DFC-EF claims only that an FP32 error-feedback residual that would
normally require one additional model-sized tensor can be represented inside the
low-word decoder fibers of the two already-allocated Adam moment tensors.

For `P` coordinates:

- common Adam moment allocation: `8P` bytes (two FP32 containers),
- conventional external EF residual: `4P` bytes,
- DFC-EF model-scale external residual: `0` bytes,
- DFC low-word fiber capacity: `4P` bytes (16+16 physical payload bits).

The relevant numerical contract is **BF16-high moment semantics in FP32
containers**, not ordinary full-FP32 AdamW.  Every placement-only external-EF vs
DFC-EF comparison therefore uses the same LOW16 semantic optimizer.

## 2. Frozen primary hypotheses

### H1 — exact representation / semantic invisibility

For arbitrary finite FP32 residual words, splitting the low and high 16-bit
halves across the low words of `exp_avg` and `exp_avg_sq` must recover the
logical residual bit-for-bit while the decoded optimizer moments are unchanged.

**PASS:** bitwise residual roundtrip and decoded-moment equality hold for all
CPU and CUDA exactness tests, including checkpoint/restart.

### H2 — placement-only trajectory equivalence

Given identical model, seed, batch order, gradients, compressor, LOW16 optimizer
semantics, hyperparameters, and update count, conventional external-EF and
DFC-EF must produce the same logical residual, same communicated/decompressed
gradient, same decoded optimizer state, and same parameters.

**PASS:** paired CUDA/Qwen jobs have identical SHA-256 digests for parameters,
decoded optimizer state, and logical residual. Any digest mismatch is a hard
failure; learning-score similarity cannot substitute for this gate.

### H3 — model-scale allocation substitution

The conventional path owns a separate `4P`-byte FP32 residual tensor. DFC-EF
must not own a model-sized decoded residual tensor; its persistent residual state
must reside only in the two moment low-word fibers. Temporary workspace is
bounded by the configured chunk size and is independent of `P`.

**PASS:** the resource ledger reports exactly `4P` external bytes for the
external-EF path and zero model-scale external residual bytes for DFC-EF, while
DFC fiber capacity is exactly `4P`. CUDA allocation telemetry must show the
placement difference in the expected direction.

### H4 — feasible-state frontier movement on a real CUDA device

The state-only benchmark allocates the common two FP32 moment tensors for both
methods and additionally the conventional FP32 EF residual only for the
external path.

The theoretical asymptotic coordinate frontiers are proportional to `1/12` and
`1/8` bytes-per-coordinate contracts respectively, giving an ideal ratio of
`1.5x` before allocator/runtime overhead.

**Primary gate:** measured DFC maximum successful coordinate count must exceed
external-EF maximum by at least **1.25x**, and the benchmark must find at least
one tested coordinate count where external-EF fails allocation while DFC
succeeds. The exact measured ratio is reported, not replaced by the 1.5x theory.

### H5 — communication accounting is matched

For every placement-only pair, the compressor, keep ratio, communicated value
count, and update schedule must be equal. No communication-volume reduction may
be attributed to DFC itself.

**PASS:** resource contract and transmitted-value audits match exactly between
external-EF and DFC-EF.

## 3. Secondary / falsifying hypotheses

These results are reported whether favorable or unfavorable and are not allowed
to redefine H1–H5 after execution.

### S1 — low-word numerical contract cost

`fp32_dense` vs `low16_dense` measures the learning effect of changing moment
semantics independently of compression and DFC placement. This is a boundary
experiment, not a placement-only DFC comparison.

### S2 — error-feedback utility

`low16_noef` vs `external_ef` measures whether EF is useful under the chosen
blockwise top-k workload. This determines whether the hidden state has practical
learning value in that workload. A negative result is retained.

### S3 — implementation overhead

CUDA end-to-end compressor+optimizer timing is reported as measured. The current
implementation is a bounded-workspace PyTorch reference path, not a fused kernel,
so **no 1–5% overhead gate is predeclared**. High overhead does not invalidate
H1–H5 but limits the systems-performance claim and becomes a target for fusion.

### S4 — cross-substrate execution

The JAX/XLA TPU experiment must reproduce bitwise representation and
external-vs-fiber trajectory equality. TPU timing is descriptive only; it is not
used to claim TPU HBM savings.

## 4. Actual-model campaign

Primary actual-model workload:

- `Qwen/Qwen2.5-0.5B`, pinned revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`,
- FP16 model parameters on CUDA,
- all transformer blocks trainable for the strongest free-GPU run; embeddings,
  final head, and other frozen parameters remain common across methods,
- blockwise top-k, keep ratio `0.125`,
- 128 optimizer updates,
- seeds `1901, 1931, 1951`,
- paired external-EF and DFC-EF runs.

A smaller checkpointed smoke pair is executed first. If remaining free-compute
budget permits, Qwen2.5-1.5B and Qwen2.5-3B partial-training placement pairs are
run as scale extensions. Failure/OOM of a scale extension is retained rather
than silently retuned.

## 5. Checkpoint / interruption protocol

The actual-model runner supports atomic checkpoint writes, RNG-state capture,
optimizer state, trainable parameters, and progress JSON. Conventional
external-EF checkpoints must serialize the separate residual tensor. DFC-EF
must recover the logical residual from optimizer state without a separate
model-sized residual object. Successful large checkpoints may be deleted after
the byte count and hashes are sealed to respect free-storage quotas.

## 6. Scaling statements allowed after validation

If H1–H5 pass, larger sizes may be shown only as **algebraic projections from the
exact 4P storage law**, clearly separated from measured GPU points:

- 7B coordinates -> 28 GB decimal external FP32 residual,
- 30B coordinates -> 120 GB decimal,
- 70B coordinates -> 280 GB decimal.

These are not to be described as measured 7B/30B/70B HBM savings unless an
actual run at that scale is later performed.

## 7. Evidence preservation

Every result bundle records source commit, device/runtime information, JSON raw
rows, SHA-256 digests, and failures. A scientific failure is preserved; it is
not converted to a passing row through retuning after inspection.
