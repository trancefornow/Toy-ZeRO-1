#!/usr/bin/env bash
set -euo pipefail

NUM_PROCESSES="${NUM_PROCESSES:-1}"

VISIBLE_GPUS="$(python - <<'PY'
import torch

print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

if [[ "${VISIBLE_GPUS}" -eq 0 ]]; then
  echo "CUDA is not available. This smoke test expects at least one GPU for the NCCL backend." >&2
  exit 1
fi

if [[ "${NUM_PROCESSES}" -gt "${VISIBLE_GPUS}" ]]; then
  echo "Requested NUM_PROCESSES=${NUM_PROCESSES}, but only ${VISIBLE_GPUS} CUDA device(s) are visible." >&2
  echo "For a single-GPU machine, run: NUM_PROCESSES=1 bash scripts/run_smoke_distributed.sh" >&2
  exit 1
fi

torchrun --standalone --nproc_per_node="${NUM_PROCESSES}" scripts/smoke_distributed.py
