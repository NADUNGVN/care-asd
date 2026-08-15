#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
cd "$REPO_DIR"

if [ ! -x .venv/bin/python ]; then
    printf 'Missing .venv/bin/python; run the server environment setup first.\n'
    exit 1
fi

.venv/bin/python -c 'import torch; expected="2.6.0+cu118"; actual=str(torch.__version__); runtime=str(torch.version.cuda); available=torch.cuda.is_available(); print(f"torch={actual} cuda_runtime={runtime} cuda_available={available}"); assert actual == expected and runtime == "11.8" and available, "Expected the verified torch 2.6.0+cu118 GPU runtime; run scripts/server/setup_phase5_torch.sh"'
