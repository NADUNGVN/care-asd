#!/usr/bin/env bash
# Phase 8: post-hoc analysis of frozen B00/B01 scores and feature displacement.
set -uo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
DATA_ROOT="${CARE_ASD_DATA_ROOT:-$HOME/Dung_TDTU/data/CARE_ASD}"
cd "$REPO_DIR"
RUN_ID="server02_phase8_care_residual_analysis_$(date -u +%Y%m%dT%H%M%SZ)"
NEAR_CACHE="${CARE_ASD_NEAR_CACHE_DIR:-$(find "$DATA_ROOT/official_vector_cache" -mindepth 2 -maxdepth 2 -type f -name cache.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)}"
RESIDUAL_CACHE="${CARE_ASD_RESIDUAL_CACHE_DIR:-$(find "$DATA_ROOT/care_residual_vector_cache" -mindepth 2 -maxdepth 2 -type f -name cache.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)}"
B00_SCORES="${CARE_ASD_B00_SCORES:-$(find reports/alignment -path '*server02_phase6_alignment*' -type f -name scores.csv -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)}"
B01_SCORES="${CARE_ASD_B01_SCORES:-$(find reports/alignment -path '*server02_phase7_care_residual*' -type f -name scores.csv -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)}"
REPORT_DIR="reports/analysis/$RUN_ID"
LOG_DIR="outputs/analysis/$RUN_ID"
SERVER_REPORT="reports/server/$RUN_ID.md"
mkdir -p "$LOG_DIR" reports/analysis reports/server
LOG_PATH="$LOG_DIR/analysis.log"
TASK_STATUS=0
if [ -z "$NEAR_CACHE" ] || [ -z "$RESIDUAL_CACHE" ] || [ -z "$B00_SCORES" ] || [ -z "$B01_SCORES" ]; then
    printf 'Missing near/residual cache or frozen B00/B01 score input.\n' >>"$LOG_PATH"
    TASK_STATUS=1
fi
if [ "$TASK_STATUS" -eq 0 ]; then
    uv run --extra official-alignment care-asd care-residual-analysis-dev --near-cache-dir "$NEAR_CACHE" --residual-cache-dir "$RESIDUAL_CACHE" --reference-scores "$B00_SCORES" --candidate-scores "$B01_SCORES" --output-dir "$REPORT_DIR" >>"$LOG_PATH" 2>&1 || TASK_STATUS=$?
fi
printf 'care_residual_analysis=%s\n' "$TASK_STATUS" >"$LOG_DIR/status.txt"
{
    printf '# Phase 8 CARE residual analysis report\n\nrun_id=%s\nnear_cache=%s\nresidual_cache=%s\nb00_scores=%s\nb01_scores=%s\ntask_status=%s\n\n## Status\n\n```text\n' "$RUN_ID" "$NEAR_CACHE" "$RESIDUAL_CACHE" "$B00_SCORES" "$B01_SCORES" "$TASK_STATUS"
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
