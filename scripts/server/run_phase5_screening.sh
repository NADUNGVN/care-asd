#!/usr/bin/env bash
# Run all pre-registered Phase 5 GPU ablations sequentially and push evidence.
set -uo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
DATA_ROOT="${CARE_ASD_DATA_ROOT:-$HOME/Dung_TDTU/data/CARE_ASD}"
cd "$REPO_DIR"
RUN_ID="server02_phase5_screening_$(date -u +%Y%m%dT%H%M%SZ)"
CACHE_DIR="${CARE_ASD_CACHE_DIR:-$(find "$DATA_ROOT/neural_cache" -mindepth 1 -maxdepth 1 -type f -name cache.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)}"
REPORT_DIR="reports/neural/$RUN_ID"
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/neural/$RUN_ID"
LOG_DIR="outputs/neural/$RUN_ID"
SERVER_REPORT="reports/server/$RUN_ID.md"
mkdir -p "$REPORT_DIR" "$CHECKPOINT_DIR" "$LOG_DIR" reports/server

TASK_STATUS=0
for ABLATION in a00_near a01_near_far a02_care_multiview; do
    LOG_PATH="$LOG_DIR/$ABLATION.log"
    uv run care-asd mvp-neural-dev --cache-dir "$CACHE_DIR" --output-dir "$REPORT_DIR/$ABLATION" --checkpoint-dir "$CHECKPOINT_DIR" --ablation "$ABLATION" --config configs/experiment/phase5_screening.yaml >"$LOG_PATH" 2>&1
    STATUS=$?
    printf '%s=%s\n' "$ABLATION" "$STATUS" >>"$LOG_DIR/status.txt"
    if [ "$STATUS" -ne 0 ]; then TASK_STATUS="$STATUS"; break; fi
done

if [ "$TASK_STATUS" -eq 0 ]; then
    { head -n1 "$REPORT_DIR/a00_near/summary.csv"; for ABLATION in a00_near a01_near_far a02_care_multiview; do tail -n +2 "$REPORT_DIR/$ABLATION/summary.csv"; done; } >"$REPORT_DIR/screening_summary.csv"
fi
{
    printf '# Phase 5 GPU screening report\n\nrun_id=%s\ncache_dir=%s\ncheckpoint_dir=%s\ntask_status=%s\n\n## Per-ablation status\n\n```text\n' "$RUN_ID" "$CACHE_DIR" "$CHECKPOINT_DIR" "$TASK_STATUS"
    cat "$LOG_DIR/status.txt" 2>/dev/null || true
    printf '```\n\n## Tail of final log\n\n```text\n'
    tail -n 40 "$(find "$LOG_DIR" -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)" 2>/dev/null || true
    printf '```\n'
} >"$SERVER_REPORT"

LD_LIBRARY_PATH="" git add "$REPORT_DIR" "$SERVER_REPORT"
LD_LIBRARY_PATH="" git commit -m "report: add $RUN_ID"
LD_LIBRARY_PATH="" git pull --rebase origin main
LD_LIBRARY_PATH="" git push origin main
printf 'run_id=%s task_status=%s\n' "$RUN_ID" "$TASK_STATUS"
exit "$TASK_STATUS"
