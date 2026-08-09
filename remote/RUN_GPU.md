# Sealed DFC-Sign actual-GPU launch

This file launches the predeclared Triton benchmark on the configured
`gpu-t4-4-core` runner. The run is valid only when `/dev/nvidia0`, `nvidia-smi`,
CUDA PyTorch, and Triton all prove physical GPU execution. The primary point is
8,388,608 FP32 coordinates, with nine alternating paired rounds, 40 warmups,
and 200 CUDA-event repetitions. DFC-Sign must preserve an arbitrary payload and
match the payload-free Triton AdamW trajectory bitwise before the measured
median paired overhead may be tested against the sealed 5.0% ceiling.
