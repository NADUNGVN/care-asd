#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
cd "$REPO_DIR"

if pgrep -af '[r]un_fp_naa_' >/dev/null; then
    printf 'Refusing CUDA repair while an FP-NAA job is running.\n'
    exit 1
fi

printf 'The FP-NAA runtime is now managed by a dedicated Conda environment.\n'
exec bash scripts/server/setup_fp_naa_conda.sh
