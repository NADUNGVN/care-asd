#!/usr/bin/env bash
# Replicate the retained Phase 5 views, ensemble three seeds, and bootstrap.
set -uo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
DATA_ROOT="${CARE_ASD_DATA_ROOT:-$HOME/Dung_TDTU/data/CARE_ASD}"
cd "$REPO_DIR"
RUN_ID="server02_phase5_replication_$(date -u +%Y%m%dT%H%M%SZ)"
CACHE_DIR="${CARE_ASD_CACHE_DIR:-$(find "$DATA_ROOT/neural_cache" -mindepth 2 -maxdepth 2 -type f -name cache.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)}"
SCREENING_DIR="$(find reports/neural -type f -path '*/a00_near/scores.csv' -printf '%T@ %h/..\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
REPORT_DIR="reports/neural/$RUN_ID"
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/neural/$RUN_ID"
LOG_DIR="outputs/neural/$RUN_ID"
SERVER_REPORT="reports/server/$RUN_ID.md"
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" reports/server

LOG_PATH="$LOG_DIR/replication.log"
uv run care-asd mvp-neural-replication-dev --cache-dir "$CACHE_DIR" --output-dir "$REPORT_DIR" --checkpoint-dir "$CHECKPOINT_DIR" --config configs/experiment/phase5_screening.yaml --seeds 42,2026 --ablations a00_near,a02_care_multiview --preload-workers 16 >"$LOG_PATH" 2>&1
TASK_STATUS=$?
if [ "$TASK_STATUS" -eq 0 ]; then
    for ABLATION in a00_near a02_care_multiview; do
        ENSEMBLE_DIR="$REPORT_DIR/${ABLATION}_ensemble"
        if ! uv run care-asd mvp-ensemble --scores "$SCREENING_DIR/$ABLATION/scores.csv" --scores "$REPORT_DIR/seed42/$ABLATION/scores.csv" --scores "$REPORT_DIR/seed2026/$ABLATION/scores.csv" --output "$ENSEMBLE_DIR/scores.csv" --model-id "mvp_${ABLATION}_three_seed_ensemble" --experiment-id "$RUN_ID" >>"$LOG_PATH" 2>&1; then TASK_STATUS=1; break; fi
        if ! uv run care-asd baseline metrics --scores "$ENSEMBLE_DIR/scores.csv" --output "$ENSEMBLE_DIR/metrics.json" >>"$LOG_PATH" 2>&1; then TASK_STATUS=1; break; fi
    done
    if [ "$TASK_STATUS" -eq 0 ]; then
        uv run care-asd mvp-bootstrap --reference-scores "$REPORT_DIR/a00_near_ensemble/scores.csv" --candidate-scores "$REPORT_DIR/a02_care_multiview_ensemble/scores.csv" --output "$REPORT_DIR/paired_bootstrap_a02_vs_a00.json" --iterations 5000 --seed 2026 >>"$LOG_PATH" 2>&1 || TASK_STATUS=$?
    fi
fi
printf 'replication_and_analysis=%s\n' "$TASK_STATUS" >"$LOG_DIR/status.txt"
{
    printf '# Phase 5 retained-model replication report\n\nrun_id=%s\ncache_dir=%s\nscreening_dir=%s\ncheckpoint_dir=%s\ntask_status=%s\n\n## Status\n\n```text\n' "$RUN_ID" "$CACHE_DIR" "$SCREENING_DIR" "$CHECKPOINT_DIR" "$TASK_STATUS"
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
