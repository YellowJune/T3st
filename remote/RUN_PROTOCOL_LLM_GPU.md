# Sealed full-FP32 DFC-Sign LLM and GPU protocol

## Qwen continual adaptation matrix

All prior controlled-codebook, random-head, LoRA-classifier, and partial-MLP
pilots are rejected. In particular, public-data run `31305802075` used 128
training and 128 held-out examples per task on immutable AG News, Emotion,
Banking77, and TREC revisions. Its one-block masked-mean MLP diagonal was
`[0.859375,0.3125,0.734375,0.640625]`, mean 0.63671875, so the entire random
classification-head family is discarded. No row from any rejected run may
enter a performance table.

The replacement removes the auxiliary classifier. It loads the immutable
`Qwen/Qwen2.5-0.5B-Instruct` model and uses Qwen's own next-token logits over
four natural-language verbalizer tokens per task. The same immutable public
dataset revisions and deterministic SHA256-ranked subsets are retained: 32
training and 32 test examples per class for four classes on AG News, Emotion,
Banking77, and TREC. Each prompt names the four allowed verbalizer words without
revealing the answer. The answer suffix is preserved under the fixed 48-token
limit, every verbalizer is proved to be one tokenizer token, and classification
cross-entropy is computed only over the four predeclared candidate tokens.

Rank-8, alpha-16 FP32 LoRA is inserted in every q, v, o, and MLP down
projection. Each task receives 512 updates at learning rate 0.0005 and gradient
norm ceiling 1. Sequential, external DER++, and full-FP32 DFC-Sign+DER++ share
identical base weights, LoRA tensors, two FP32 Adam states, actual 2,048-byte
external arrays, batch two, fixed sequence length, update count, and counted
dense Transformer FLOPs.

A 256-byte replay record carries the actual token sequence, task, four-way
target, all four FP16 candidate dark logits, type/version fields, and CRC32.
External DER++ addresses seven records. DFC-Sign composes the same external
array with exactly `floor(p/8)` bytes in the sign fiber of the ordinary
full-FP32 Adam second moments, adds no tensor, and decodes every sign before
arithmetic. Both replay methods use one current and one replay sequence;
sequential uses two current sequences in the identical dense batch. A
checkpoint round trip after task two must preserve the payload digest.

Untouched seed `1201` is acquisition-only and can never enter the comparative
table. It is admitted only when every public-dataset diagonal accuracy is at
least 80% and their mean is at least 90%. If it passes, the final untouched
seeds are `1213`, `1223`, and `1231`. Final acceptance requires DFC mean
learning accuracy at least 90%, at least +10 final-accuracy points and +10
forgetting-reduction points versus external DER++, current-task accuracy no
more than one point lower, and final four-way NLL no higher. Any source,
revision, dataset subset, verbalizer, matrix, metric, capacity, checkpoint,
physical-byte, or counted-FLOP mismatch rejects the matrix. Manifests and raw
results are persisted before every acceptance assertion.

## Triton actual-GPU gate

The GPU run is admitted only on a runner exposing `/dev/nvidia0`, successful `nvidia-smi`, and `torch.cuda.is_available() == True`; CPU or interpreter emulation is rejected. At 1,048,576, 8,388,608, and 33,554,432 FP32 coordinates, the matched payload-free Triton AdamW and DFC-Sign kernel use the same block size and floating-point operation order. Before timing, eight updates must be bitwise identical in parameters, first moments, and second-moment magnitudes while an independent arbitrary payload remains bitwise unchanged.

Timing uses nine alternating-order paired rounds, 40 warmups and 200 CUDA-event repetitions per method and size. PyTorch `AdamW(fused=True)` is a secondary library anchor. The predeclared primary point is 8,388,608 coordinates; median paired DFC/reference overhead must not exceed 5.0%. Complete per-round samples, GPU model, driver, clocks, framework versions, source hashes, and the workflow source commit are mandatory evidence.
