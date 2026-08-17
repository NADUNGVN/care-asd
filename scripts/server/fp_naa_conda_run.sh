#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="${CARE_ASD_CONDA_ENV:-care-asd-fp-naa}"
CONDA_EXE="${CARE_ASD_CONDA_EXE:-$HOME/miniconda3/bin/conda}"

if [ ! -x "$CONDA_EXE" ]; then
    printf 'Conda executable not found: %s\n' "$CONDA_EXE" >&2
    exit 1
fi
if ! "$CONDA_EXE" run -n "$CONDA_ENV_NAME" python --version >/dev/null 2>&1; then
    printf 'Conda environment %s is unavailable; run scripts/server/setup_fp_naa_conda.sh first.\n' "$CONDA_ENV_NAME" >&2
    exit 1
fi

exec env -u LD_LIBRARY_PATH -u LD_PRELOAD "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV_NAME" "$@"

