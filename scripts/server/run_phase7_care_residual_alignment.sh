#!/usr/bin/env bash
# Phase 7 B01: cache bounded CARE residuals and evaluate with the locked official AE.
set -uo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
DATA_ROOT="${CARE_ASD_DATA_ROOT:-$HOME/Dung_TDTU/data/CARE_ASD}"
cd "$REPO_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN_ID="server02_phase7_care_residual_$(date -u +%Y%m%dT%H%M%SZ)"
CACHE_DIR="${CARE_ASD_CARE_RESIDUAL_CACHE_DIR:-$(find "$DATA_ROOT/care_residual_vector_cache" -mindepth 2 -maxdepth 2 -type f -name cache.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)}"
REFERENCE_SCORE="${CARE_ASD_PHASE6_REFERENCE_SCORE:-$(find reports/alignment -mindepth 2 -maxdepth 2 -type f -name scores.csv -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)}"
REPORT_DIR="reports/alignment/$RUN_ID"
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/care_residual_alignment/$RUN_ID"
LOG_DIR="outputs/alignment/$RUN_ID"
SERVER_REPORT="reports/server/$RUN_ID.md"
mkdir -p "$DATA_ROOT/care_residual_vector_cache" "$CHECKPOINT_DIR" "$LOG_DIR" reports/server

LOG_PATH="$LOG_DIR/care_residual_alignment.log"
TASK_STATUS=0
if ! .venv/bin/python -c 'import librosa' >>"$LOG_PATH" 2>&1; then
    uv pip install --python .venv/bin/python 'librosa>=0.10,<0.12' >>"$LOG_PATH" 2>&1 || TASK_STATUS=$?
fi
if [ "$TASK_STATUS" -eq 0 ] && [ -z "$CACHE_DIR" ]; then
    CACHE_DIR="$DATA_ROOT/care_residual_vector_cache/$RUN_ID"
    uv run --extra official-alignment care-asd data cache-care-residual-vectors --manifest data/manifests/dcase2026_dev.parquet --audio-root "$DATA_ROOT/raw/dcase2026/dev/extracted" --output-dir "$CACHE_DIR" --config configs/experiment/phase7_care_residual_alignment.yaml --workers 16 >>"$LOG_PATH" 2>&1 || TASK_STATUS=$?
fi
if [ "$TASK_STATUS" -eq 0 ] && [ -z "$REFERENCE_SCORE" ]; then
    printf 'Missing committed Phase 6 reference scores.\n' >>"$LOG_PATH"
    TASK_STATUS=1
fi
if [ "$TASK_STATUS" -eq 0 ]; then
    uv run --extra official-alignment care-asd care-residual-alignment-dev --cache-dir "$CACHE_DIR" --output-dir "$REPORT_DIR" --checkpoint-dir "$CHECKPOINT_DIR" --config configs/experiment/phase7_care_residual_alignment.yaml >>"$LOG_PATH" 2>&1 || TASK_STATUS=$?
fi
if [ "$TASK_STATUS" -eq 0 ]; then
    uv run --extra official-alignment care-asd mvp-bootstrap --reference-scores "$REFERENCE_SCORE" --candidate-scores "$REPORT_DIR/scores.csv" --output "$REPORT_DIR/paired_bootstrap_b01_vs_b00.json" --iterations 5000 --seed 2026 >>"$LOG_PATH" 2>&1 || TASK_STATUS=$?
fi
printf 'care_residual_alignment=%s\n' "$TASK_STATUS" >"$LOG_DIR/status.txt"
{
    printf '# Phase 7 CARE residual-alignment report\n\nrun_id=%s\ncache_dir=%s\nreference_score=%s\ncheckpoint_dir=%s\ntask_status=%s\n\n## Status\n\n```text\n' "$RUN_ID" "$CACHE_DIR" "$REFERENCE_SCORE" "$CHECKPOINT_DIR" "$TASK_STATUS"
    cat "$LOG_DIR/status.txt"
    printf '```\n\n## Final log tail\n\n```text\n'
    tail -n 80 "$LOG_PATH" 2>/dev/null || true
    printf '```\n'
} >"$SERVER_REPORT"

LD_LIBRARY_PATH="" git add "$SERVER_REPORT"
[ -d "$REPORT_DIR" ] && LD_LIBRARY_PATH="" git add "$REPORT_DIR"
LD_LIBRARY_PATH="" git commit -m "report: add $RUN_ID"
LD_LIBRARY_PATH="" git pull --rebase origin main
LD_LIBRARY_PATH="" git push origin main
printf 'run_id=%s task_status=%s\n' "$RUN_ID" "$TASK_STATUS"
exit "$TASK_STATUS"
