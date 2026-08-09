# Sealed full-FP32 DFC-Sign LLM and GPU protocol

## Qwen continual adaptation matrix

Run `31298252568` is a rejected acquisition pilot. Its first completed sealed seed passed every integrity, checkpoint, capacity, and resource assertion, and DFC-Sign reduced final answer NLL relative to external DER++; however all three methods remained at the eight-way 12.5% chance accuracy. None of its rows may enter a performance table. The revised gate below adds an explicit acquisition floor and nontrivial effect-size minima so an NLL-only chance solution cannot be accepted.

The next sealed matrix uses `Qwen/Qwen2.5-0.5B` at one immutable Hugging Face commit resolved before any learning cell starts. It freezes the base model and inserts rank-8, alpha-16 FP32 LoRA matrices into every attention `q_proj`, `v_proj`, `o_proj`, and MLP `down_proj`. Sequential, external DER++, and full-FP32 DFC-Sign+DER++ use identical base weights, LoRA parameters, two FP32 Adam states, actual 2,048-byte external arrays, batch shape, fixed sequence length 48, updates, and counted dense Transformer FLOPs.

The stream contains four codebook adaptation tasks (`orion`, `cedar`, `marble`, `saffron`) with conflicting four-key token assignments and two fixed prompt forms. A 256-byte record carries the actual token sequence, answer location, target, four FP16 dark logits, type/version fields, and CRC32. External DER++ can address seven records. DFC-Sign composes the same external array with exactly `floor(p/8)` bytes in the signs of the full-FP32 Adam second moments; it adds no tensor and clears every sign before arithmetic.

Fresh sealed seeds are `941`, `947`, and `953`. Each of four tasks receives 64 updates, batch 2, one current and one replay sequence for both replay methods, one insertion per update, learning rate 0.01, and dark-logit coefficient 0.1. Sequential receives two current sequences in the same dense batch. A checkpoint round trip after task two must reproduce the payload digest. Acceptance requires DFC mean acquisition accuracy at least 75%, at least +5 final-accuracy points and +5 forgetting-reduction points versus external DER++, current-task accuracy no more than one point lower, and final answer NLL no higher. Any source/revision, matrix, metric, capacity, checkpoint, or paired-resource mismatch rejects the entire matrix.

## Triton actual-GPU gate

The GPU run is admitted only on a runner exposing `/dev/nvidia0`, successful `nvidia-smi`, and `torch.cuda.is_available() == True`; CPU or interpreter emulation is rejected. At 1,048,576, 8,388,608, and 33,554,432 FP32 coordinates, the matched payload-free Triton AdamW and DFC-Sign kernel use the same block size and floating-point operation order. Before timing, eight updates must be bitwise identical in parameters, first moments, and second-moment magnitudes while an independent arbitrary payload remains bitwise unchanged.

Timing uses nine alternating-order paired rounds, 40 warmups and 200 CUDA-event repetitions per method and size. PyTorch `AdamW(fused=True)` is a secondary library anchor. The predeclared primary point is 8,388,608 coordinates; median paired DFC/reference overhead must not exceed 5.0%. Complete per-round samples, GPU model, driver, clocks, framework versions, source hashes, and the workflow source commit are mandatory evidence.
