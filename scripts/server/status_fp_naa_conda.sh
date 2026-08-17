#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="${CARE_ASD_CONDA_ENV:-care-asd-fp-naa}"
CONDA_EXE="${CARE_ASD_CONDA_EXE:-$HOME/miniconda3/bin/conda}"

printf 'CONDA ENVIRONMENT:\n'
"$CONDA_EXE" env list | grep -E "(^#|${CONDA_ENV_NAME})" || true
printf '\nPINNED RUNTIME:\n'
"$CONDA_EXE" list -n "$CONDA_ENV_NAME" | grep -E '^(python|torch|torchaudio|triton|nvidia-cudnn-cu11)[[:space:]]' || true
printf '\nCUDA PROBE:\n'
bash scripts/server/fp_naa_conda_run.sh python -c "import torch, torchaudio; print(f'torch={torch.__version__} torchaudio={torchaudio.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()} available={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

