#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
cd "$REPO_DIR"

if pgrep -af '[r]un_fp_naa_' >/dev/null; then
    printf 'Refusing CUDA repair while an FP-NAA job is running.\n'
    exit 1
fi

printf 'Synchronizing the frozen Torch 2.6.0 CUDA 11.8 environment.\n'
env -u LD_LIBRARY_PATH -u LD_PRELOAD uv sync --frozen --extra full --extra official-alignment
bash scripts/server/setup_fp_naa_beats.sh
printf 'FP-NAA CUDA repair complete; cuDNN convolution probe passed.\n'
