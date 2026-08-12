#!/usr/bin/env bash
# Install the pinned CUDA-enabled PyTorch runtime used by Phase 5, then record it.
set -uo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
cd "$REPO_DIR"
RUN_ID="server02_phase5_torch_$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/neural/$RUN_ID.log"
REPORT="reports/server/$RUN_ID.md"
mkdir -p outputs/neural reports/server

uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu118 'torch==2.6.0+cu118' >"$LOG_PATH" 2>&1
TASK_STATUS=$?
if [ "$TASK_STATUS" -eq 0 ]; then
    .venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(f"torch={torch.__version__}"); print(f"cuda_runtime={torch.version.cuda}"); print(f"gpu={torch.cuda.get_device_name(0)}")' >>"$LOG_PATH" 2>&1
    TASK_STATUS=$?
fi

{
    printf '# Phase 5 GPU runtime report\n\nrun_id=%s\ntask_status=%s\n\n## Diagnostic\n\n```text\n' "$RUN_ID" "$TASK_STATUS"
    tail -n 40 "$LOG_PATH"
    printf '```\n'
} >"$REPORT"

LD_LIBRARY_PATH="" git add "$REPORT"
LD_LIBRARY_PATH="" git commit -m "report: add $RUN_ID"
LD_LIBRARY_PATH="" git pull --rebase origin main
LD_LIBRARY_PATH="" git push origin main
printf 'run_id=%s task_status=%s\n' "$RUN_ID" "$TASK_STATUS"
exit "$TASK_STATUS"
