#!/usr/bin/env bash
set -euo pipefail

# Requires the official Kaggle CLI and KAGGLE_API_TOKEN in the environment.
# This script never prints the token.
: "${KAGGLE_API_TOKEN:?Set KAGGLE_API_TOKEN in the environment first}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pip install -q --upgrade kaggle

# T4 is preferred over P100 for the Triton row (Volta-or-newer requirement).
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4 --timeout 43200

# Separate TPU cross-substrate gate; it does not consume GPU quota.
kaggle kernels push -p kaggle_tpu --accelerator TpuV38 --timeout 21600

printf '\nGPU kernel: yellwojune083/dfc-final-kaggle-validation\n'
printf 'TPU kernel: yellwojune083/dfc-universal-tpu-validation\n'
printf 'Check with: kaggle kernels status <owner/kernel-slug>\n'
