# DFC-EF Kaggle CUDA validation — sealed measured results

This document reports results obtained **after** the criteria in
`docs/DFC_EF_FREE_VALIDATION_PROTOCOL.md` were fixed.  It does not modify the
predeclared criteria.

## Execution provenance

- Platform: Kaggle free GPU
- Device: Tesla P100-PCIE-16GB
- Reported total HBM: 17,059,545,088 bytes
- PyTorch: 2.7.1+cu126
- CUDA runtime: 12.6
- Source commit measured by the kernel: `edc10ad783aabc2354ef5f1a906634122816d659`
- GitHub launch/recovery run: `31545709052`
- Evidence artifact: `9122505694`
- Evidence artifact SHA-256: `acb21bd18be52e7d68b23e49c779ba4d427a5c4b3eeb2490e5f500d9b4192820`

The P100 required an explicit official PyTorch 2.7.1 CUDA-12.6 wheel because
Kaggle's then-current default wheel omitted `sm_60`.  The final environment
probe confirmed compute capability 6.0 and `sm_60` support before any result was
accepted.

## H1: CUDA exactness and restart path

`gpu_dfc_ef_exactness.py` passed for both FP32 and FP16 parameters at
1,048,579 coordinates, 32 updates, and a 131,071-coordinate workspace chunk.
The path includes DFC payload preservation and checkpoint/reload.  A real bug
found during development was retained and fixed: PyTorch's generic optimizer
loader casts floating optimizer state toward parameter dtype, which is invalid
for DFC's physical FP32 moment containers.  The accepted source therefore uses
a direct bit-preserving state reload and a regression test for this contract.

- FP32 DFC checkpoint: 12,585,249 bytes
- FP16 DFC checkpoint: 10,488,097 bytes
- CUDA exactness gate: **PASS**

## H3: measured HBM allocation law

At all five measured scales, the steady allocated-memory difference between the
external-FP32-EF baseline and DFC-EF is **exactly `4P` bytes**, while the final
parameter and logical-residual trajectory digests match.

| Coordinates P | Measured steady bytes saved | Exact 4P law | Peak bytes saved | External median ms | DFC median ms | DFC overhead |
|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 4,194,304 | 4,194,304 | -4,194,304 | 0.6813 | 0.9443 | +38.60% |
| 4,194,304 | 16,777,216 | 16,777,216 | 8,388,608 | 2.2241 | 3.4634 | +55.73% |
| 16,777,216 | 67,108,864 | 67,108,864 | 58,720,256 | 8.4081 | 13.2998 | +58.18% |
| 33,554,432 | 134,217,728 | 134,217,728 | 125,829,120 | 16.4633 | 26.5798 | +61.45% |
| 67,108,864 | 268,435,456 | 268,435,456 | 260,046,848 | 32.9196 | 53.1638 | +61.50% |

The small-P peak row is intentionally retained even though DFC peaks 4 MiB
higher there because temporary bounded-workspace tensors dominate at that size.
The steady-state allocation law is exact at every measured scale.

## H4: measured OOM frontier

The state-only benchmark allocates two FP32 moment containers plus an external
FP32 EF residual for the baseline (`12P` bytes) and only the two FP32 physical
moment containers for DFC-EF (`8P` bytes).  Every allocation probe runs in a
fresh subprocess, and only an actual CUDA OOM return code is classified as an
OOM; infrastructure failures fail the benchmark instead of being counted as
scientific observations.

- External maximum successful coordinates: **1,371,537,408**
- DFC maximum successful coordinates: **2,059,403,264**
- Measured frontier ratio: **1.501529x**
- Predeclared threshold: **>= 1.25x**
- DFC hidden residual capacity at its measured frontier: **8,237,613,056 bytes**

Verified crossover:

- Coordinates: **1,375,731,712**
- External theoretical state: 16,508,780,544 bytes -> **CUDA OOM**
- DFC theoretical state: 11,005,853,696 bytes
- DFC measured allocated state: **11,005,853,696 bytes** -> **success**
- DFC hidden payload capacity at crossover: 5,502,926,848 bytes

Therefore both the predeclared ratio gate and the predeclared
external-OOM/DFC-success crossover gate pass.  **H4: PASS.**

## Performance disclosure

The current bounded-workspace PyTorch reference path is **not** a deployment-
speed result.  Across the measured microbenchmark range DFC is approximately
38.6% to 61.5% slower than the matched external-EF path.  This negative result
is retained.  It does not alter H1/H3/H4, which test exactness and physical
state substitution, but it makes fused/kernel optimization necessary before a
claim of production-speed practicality.

## Claim boundary

These are real CUDA/P100 measurements.  The H4 frontier is intentionally
state-only: model weights, gradients, and activations are common allocations
and are excluded by the predeclared protocol.  Communication reduction belongs
to the matched compressor, not to DFC.  Any 7B/30B/70B memory number remains an
algebraic projection unless that scale is separately measured.
