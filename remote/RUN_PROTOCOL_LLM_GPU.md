# Sealed full-FP32 DFC-Sign LLM and GPU protocol

## Qwen continual adaptation matrix

The first sealed matrix uses `Qwen/Qwen2.5-0.5B` at one immutable Hugging Face commit resolved before any learning cell starts. It freezes the base model and inserts rank-8, alpha-16 FP32 LoRA matrices into every attention `q_proj` and `v_proj`. Sequential, external DER++, and full-FP32 DFC-Sign+DER++ use identical base weights, LoRA parameters, two FP32 Adam states, actual 2,048-byte external arrays, batch shape, fixed sequence length 48, updates, and counted dense Transformer FLOPs.

The stream contains four codebook adaptation tasks (`orion`, `cedar`, `marble`, `saffron`) with conflicting key-to-token assignments. Four prompt forms train each mapping and two unseen prompt forms evaluate it. A 256-byte record carries the actual token sequence, answer location, target, four FP16 dark logits, type/version fields, and CRC32. External DER++ can address seven records. DFC-Sign composes the same external array with exactly `floor(p/8)` bytes in the signs of the full-FP32 Adam second moments; it adds no tensor and clears every sign before arithmetic.

Fresh sealed seeds are `901`, `907`, and `919`. Each of four tasks receives 32 updates, batch 2, one current and one replay sequence for both replay methods, one insertion per update, learning rate 0.008, and dark-logit coefficient 0.05. Sequential receives two current sequences in the same dense batch. A checkpoint round trip after task two must reproduce the payload digest. The accepted aggregate must satisfy all four mean conditions against external DER++: final average accuracy no lower, average forgetting no higher, current-task accuracy no more than one percentage point lower, and final average answer NLL no higher. Any source/revision, matrix, metric, capacity, checkpoint, or paired-resource mismatch rejects the entire matrix.

## Triton actual-GPU gate

The GPU run is admitted only on a runner exposing `/dev/nvidia0`, successful `nvidia-smi`, and `torch.cuda.is_available() == True`; CPU or interpreter emulation is rejected. At 1,048,576, 8,388,608, and 33,554,432 FP32 coordinates, the matched payload-free Triton AdamW and DFC-Sign kernel use the same block size and floating-point operation order. Before timing, eight updates must be bitwise identical in parameters, first moments, and second-moment magnitudes while an independent arbitrary payload remains bitwise unchanged.

Timing uses nine alternating-order paired rounds, 40 warmups and 200 CUDA-event repetitions per method and size. PyTorch `AdamW(fused=True)` is a secondary library anchor. The predeclared primary point is 8,388,608 coordinates; median paired DFC/reference overhead must not exceed 5.0%. Complete per-round samples, GPU model, driver, clocks, framework versions, source hashes, and the workflow source commit are mandatory evidence.
