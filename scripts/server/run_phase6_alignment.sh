#!/usr/bin/env bash
# Build/reuse exact official vectors, run internal alignment, and push evidence.
set -uo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
DATA_ROOT="${CARE_ASD_DATA_ROOT:-$HOME/Dung_TDTU/data/CARE_ASD}"
cd "$REPO_DIR"
# Required by PyTorch when the pinned official deterministic setting reaches
# CUDA cuBLAS. It must exist before the Python process starts.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN_ID="server02_phase6_alignment_$(date -u +%Y%m%dT%H%M%SZ)"
CACHE_DIR="${CARE_ASD_OFFICIAL_CACHE_DIR:-$(find "$DATA_ROOT/official_vector_cache" -mindepth 2 -maxdepth 2 -type f -name cache.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)}"
REPORT_DIR="reports/alignment/$RUN_ID"
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/alignment/$RUN_ID"
LOG_DIR="outputs/alignment/$RUN_ID"
SERVER_REPORT="reports/server/$RUN_ID.md"
mkdir -p "$DATA_ROOT/official_vector_cache" "$CHECKPOINT_DIR" "$LOG_DIR" reports/server

LOG_PATH="$LOG_DIR/alignment.log"
TASK_STATUS=0
if ! .venv/bin/python -c 'import librosa' >>"$LOG_PATH" 2>&1; then
    uv pip install --python .venv/bin/python 'librosa>=0.10,<0.12' >>"$LOG_PATH" 2>&1 || TASK_STATUS=$?
fi
if [ "$TASK_STATUS" -eq 0 ] && [ -z "$CACHE_DIR" ]; then
    CACHE_DIR="$DATA_ROOT/official_vector_cache/$RUN_ID"
    uv run --extra official-alignment care-asd data cache-official-vectors --manifest data/manifests/dcase2026_dev.parquet --audio-root "$DATA_ROOT/raw/dcase2026/dev/extracted" --output-dir "$CACHE_DIR" --workers 16 >>"$LOG_PATH" 2>&1 || TASK_STATUS=$?
fi
if [ "$TASK_STATUS" -eq 0 ]; then
    uv run --extra official-alignment care-asd official-alignment-dev --cache-dir "$CACHE_DIR" --output-dir "$REPORT_DIR" --checkpoint-dir "$CHECKPOINT_DIR" --config configs/experiment/phase6_official_alignment.yaml >>"$LOG_PATH" 2>&1 || TASK_STATUS=$?
fi
printf 'official_alignment=%s\n' "$TASK_STATUS" >"$LOG_DIR/status.txt"
{
    printf '# Phase 6 official-alignment report\n\nrun_id=%s\ncache_dir=%s\ncheckpoint_dir=%s\ntask_status=%s\n\n## Status\n\n```text\n' "$RUN_ID" "$CACHE_DIR" "$CHECKPOINT_DIR" "$TASK_STATUS"
    cat "$LOG_DIR/status.txt"
    printf '```\n\n## Final log tail\n\n```text\n'
    tail -n 60 "$LOG_PATH" 2>/dev/null || true
    printf '```\n'
} >"$SERVER_REPORT"

LD_LIBRARY_PATH="" git add "$SERVER_REPORT"
[ -d "$REPORT_DIR" ] && LD_LIBRARY_PATH="" git add "$REPORT_DIR"
LD_LIBRARY_PATH="" git commit -m "report: add $RUN_ID"
LD_LIBRARY_PATH="" git pull --rebase origin main
LD_LIBRARY_PATH="" git push origin main
printf 'run_id=%s task_status=%s\n' "$RUN_ID" "$TASK_STATUS"
exit "$TASK_STATUS"
