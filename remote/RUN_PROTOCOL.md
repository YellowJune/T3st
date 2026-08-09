# Audited DFC v2 remote runs

The accepted remote evidence must satisfy the PyTorch bitwise trajectory/payload/checkpoint gate before any vision cell runs. CIFAR-100 is downloaded from the immutable mirror and verified against SHA-256 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`.

Run `31289144091` validated the original 32x32/4-bit reference codec and all nine execution paths. Its CIFAR accuracy rows were rejected after DFC-Sign+ER failed the empirical dominance gate; they remain preserved as negative evidence and are not used in the paper's performance claims.

The tuned pilot uses 16x16/4-bit CRC-checked records, CIFAR normalization, a seen-class single-head loss, 300 updates per task, 32 replay replacements in a fixed batch of 64, and seed 301. A final three-seed run is authorized only if the pilot passes both exactness and DFC-versus-external-ER performance gates.
