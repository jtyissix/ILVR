#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CONFIG=${1:-configs/interaction/interaction_ce.json}
if [[ $# -gt 0 ]]; then shift; fi
torchrun --standalone --nproc_per_node="${NUM_GPUS:-8}" \
  --module src.interaction.train --config "$CONFIG" "$@"
