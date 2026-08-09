# Sealed full-FP32 DFC-Sign LLM and GPU protocol

## Qwen continual adaptation matrix

Runs `31298252568`, `31299088264`, `31301234609`, `31302105163`, and
`31302986996` are rejected chance-level LoRA or classification-head pilots.
Run `31303768550` switched to two-block partial fine-tuning but its
`[0.50,1.00,0.75,0.375]` learning diagonal failed acquisition. Run
`31304365772` lengthened every task to 512 updates and reduced the two-block
and head learning rates to 0.00002 and 0.0005. Its immutable diagonal was
`[0.625,1.00,0.75,0.375]`, mean 0.6875, so that entire controlled eight-example
suite is rejected. No row from any rejected run may enter a performance table.

The replacement is a new, predeclared public-data protocol. It loads immutable
revisions of `Qwen/Qwen2.5-0.5B`, AG News, DAIR Emotion, MTEB Banking77, and
TREC before learning and records every resolved commit, split schema, row count,
and manifest hash. Four deterministic SHA256-ranked, disjoint classification
subsets are used: all four AG News classes; sadness/joy/anger/fear from Emotion;
activate-card/cancel-transfer/country-support/forgotten-passcode from Banking77;
and entity/description/human/numeric from TREC. Each task has 32 training and 32
test examples per class. Prompts state the dataset-specific four-class ontology
but never the answer. Tokenization is capped at 48 tokens and the exact selected
row hashes are recorded.

The model masked-mean pools all non-padding Qwen states and trains the final
transformer block, final norm, and an FP32 LayerNorm--256--GELU--4 head. The
block and head learning rates are 0.000001 and 0.001, respectively, with global
gradient-norm ceiling 1. Each task receives 512 updates. Sequential, external
DER++, and full-FP32 DFC-Sign+DER++ use identical model tensors, trainable
parameters, two FP32 Adam states, actual 2,048-byte external arrays, batch two,
sequence length 48, update count, and counted dense Transformer FLOPs.

A 256-byte replay record carries the actual token sequence, target, all four
FP16 dark logits, type/version fields, and CRC32. External DER++ can address
seven records. DFC-Sign composes the same 2,048-byte array with exactly
`floor(p/8)` bytes in the sign fiber of the ordinary full-FP32 Adam second
moments. No tensor is added, and every sign is decoded before numerical
arithmetic. Both replay methods use one current and one replay sequence and one
insertion per update; sequential uses two current sequences in the identical
dense batch. A checkpoint round trip after task two must preserve the payload
digest exactly.

Untouched seed `1123` is acquisition-only and can never enter the comparative
table. It is admitted only when every public-dataset diagonal accuracy is at
least 80% and the four-task mean is at least 90%. If it passes, the final
untouched seeds are `1129`, `1151`, and `1171`. Final acceptance requires
DFC mean learning accuracy at least 90%, at least +10 final-accuracy points and
+10 forgetting-reduction points versus external DER++, current-task accuracy no
more than one point lower, and final NLL no higher. Any source, immutable
revision, dataset subset, matrix, metric, capacity, checkpoint, physical-byte,
or counted-FLOP mismatch rejects the matrix. Manifests and raw results are
persisted on isolated branches before any acceptance assertion.

## Triton actual-GPU gate

The GPU run is admitted only on a runner exposing `/dev/nvidia0`, successful `nvidia-smi`, and `torch.cuda.is_available() == True`; CPU or interpreter emulation is rejected. At 1,048,576, 8,388,608, and 33,554,432 FP32 coordinates, the matched payload-free Triton AdamW and DFC-Sign kernel use the same block size and floating-point operation order. Before timing, eight updates must be bitwise identical in parameters, first moments, and second-moment magnitudes while an independent arbitrary payload remains bitwise unchanged.

Timing uses nine alternating-order paired rounds, 40 warmups and 200 CUDA-event repetitions per method and size. PyTorch `AdamW(fused=True)` is a secondary library anchor. The predeclared primary point is 8,388,608 coordinates; median paired DFC/reference overhead must not exceed 5.0%. Complete per-round samples, GPU model, driver, clocks, framework versions, source hashes, and the workflow source commit are mandatory evidence.
