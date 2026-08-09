# Sealed full-FP32 DFC-Sign LLM and GPU protocol

## Qwen continual adaptation: semantic multiple-choice replacement

All controlled-codebook, random-head, masked-mean classifier, partial-block,
synthetic-semantic, and natural-word verbalizer pilots are permanently rejected.
No metric from those acquisition runs may enter a comparative table. The final
natural-word pilot was run `31307278646` on immutable public data. Its diagonal
was `[0.7890625, 0.7734375, 0.984375, 0.78125]` (mean 0.83203125), final
average accuracy was 0.646484375, and average forgetting was 0.25. It failed
the learning gate, so the verbalizer family and its prepared 3x3 matrix were
discarded before execution.

The replacement uses the immutable `Qwen/Qwen2.5-0.5B-Instruct` causal LM
without an auxiliary classification head. AG News, Emotion, Banking77, and
TREC retain their immutable Hugging Face revisions and deterministic
SHA256-ranked subsets: 32 training and 32 held-out test examples for each of
four classes per task. Every prompt declares dataset-specific option semantics
and requests exactly one of `A/B/C/D`. The four answer strings must map to
four distinct single tokenizer tokens. The answer suffix is retained under the
fixed 48-token record limit, and both optimization and evaluation use Qwen's
actual next-token vocabulary logits restricted to those four tokens.

Rank-16, alpha-32 FP32 LoRA is inserted into every q, v, o, and MLP down
projection (96 modules). The base model, adapters, gradients, and both Adam
moments are full FP32. Each task receives 1,024 updates at learning rate
0.0005, batch two, gradient-norm ceiling one, and fixed dense length 48.
Sequential, external DER++, and DFC-Sign+DER++ must share the exact model
revision, deterministic examples, trainable tensors, two FP32 Adam states,
actual 2,048-byte external allocation, update count, dense token count, and
counted Transformer FLOPs.

Each 256-byte replay record contains the actual token sequence, task and target,
all four FP16 candidate dark logits, type/version fields, and CRC32. External
DER++ can address seven records in the common 2,048-byte envelope. DFC-Sign
composes that identical envelope with exactly `floor(p/8)` bytes carried by
the sign fiber of the already allocated FP32 Adam second moments. It allocates
no new tensor and applies the absolute-value decoder before every optimizer
arithmetic operation. Both replay methods use one current plus one replay
sequence; sequential uses two current sequences in the identical dense batch.
A checkpoint round trip after task two must preserve every payload byte.

Seed `1301` is an acquisition-only semantic-learning run and is forbidden
from the final comparison. Its predeclared gate is: all four diagonal
accuracies at least 85% and their mean at least 92%. Only if it passes may the
unchanged code and hyperparameters proceed to untouched final seeds `1319`,
`1321`, and `1327` for all three methods. Final acceptance requires DFC
mean learning accuracy at least 92%, at least +10 final-accuracy points and
+10 forgetting-reduction points versus physical-envelope-matched external
DER++, current-task accuracy no more than one point lower, and final four-way
NLL no higher. Any source, revision, dataset subset, answer-token mapping,
matrix, metric, capacity, checkpoint, physical-byte, or counted-FLOP mismatch
rejects the complete matrix. Manifests and raw outputs are pushed to immutable
result branches before any gate assertion.

## Triton actual-GPU gate

The GPU run is admitted only on a runner exposing `/dev/nvidia0`, successful
`nvidia-smi`, and `torch.cuda.is_available() == True`; CPU or interpreter
emulation is rejected. At 1,048,576, 8,388,608, and 33,554,432 FP32
coordinates, the matched payload-free Triton AdamW and DFC-Sign kernel use the
same block size and floating-point operation order. Before timing, eight
updates must be bitwise identical in parameters, first moments, and
second-moment magnitudes while an independent arbitrary payload remains
bitwise unchanged.

Timing uses nine alternating-order paired rounds, 40 warmups and 200 CUDA-event
repetitions per method and size. PyTorch `AdamW(fused=True)` is a secondary
library anchor. The predeclared primary point is 8,388,608 coordinates; median
paired DFC/reference overhead must not exceed 5.0%. Complete per-round samples,
GPU model, driver, clocks, framework versions, source hashes, and the workflow
source commit are mandatory evidence.
