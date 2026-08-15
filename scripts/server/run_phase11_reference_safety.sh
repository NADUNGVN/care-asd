#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
DATA_ROOT="${CARE_ASD_DATA_ROOT:-$HOME/Dung_TDTU/data/CARE_ASD}"
RUN_ID="${1:?run id required}"
STAGE="${2:?stage required}"
cd "$REPO_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SOURCE_RUN="$(find reports/reference_safety -path '*/simulation/policy.yaml' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- | xargs -r dirname | xargs -r dirname | xargs -r basename)"
CACHE_DIR="$DATA_ROOT/reference_safety_cache/dev/$SOURCE_RUN"
POLICY="reports/reference_safety/$SOURCE_RUN/simulation/policy.yaml"
SIM_GATE="reports/reference_safety/$SOURCE_RUN/simulation/gate.json"
JOB_DIR="outputs/reference_safety/$RUN_ID"
STATE="$JOB_DIR/state.env"
LOG="$JOB_DIR/phase11.log"
REPORT_DIR="reports/reference_safety/$RUN_ID/development"
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/reference_safety/$RUN_ID"
SERVER_REPORT="reports/server/$RUN_ID.md"
CONFIG="configs/experiment/phase10_reference_safety.yaml"
mkdir -p "$JOB_DIR" "$CHECKPOINT_DIR" reports/server

write_state() {
    local status="$1" stage="$2" code="$3"
    printf 'run_id=%s\nstatus=%s\nstage=%s\ntask_status=%s\nupdated_utc=%s\nlog=%s\n' "$RUN_ID" "$status" "$stage" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LOG" >"$STATE.tmp"
    mv "$STATE.tmp" "$STATE"
}

TASK_STATUS=0
write_state RUNNING "$STAGE" 99
printf 'Phase 11 SAFE-REF %s started at %s\n' "$STAGE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$LOG"
if [ -z "$SOURCE_RUN" ] || [ ! -f "$POLICY" ] || [ ! -f "$SIM_GATE" ] || [ ! -f "$CACHE_DIR/cache.json" ]; then
    printf 'No completed Phase 10 cache/policy found.\n' >>"$LOG"
    TASK_STATUS=1
elif ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$SIM_GATE"; then
    printf 'Phase 10 simulation gate did not pass; development is blocked.\n' >>"$LOG"
    TASK_STATUS=1
fi
if [ "$TASK_STATUS" -eq 0 ]; then
    uv run --extra official-alignment --extra ml care-asd reference-safety dev --cache-dir "$CACHE_DIR" --policy "$POLICY" --output-dir "$REPORT_DIR" --checkpoint-dir "$CHECKPOINT_DIR" --config "$CONFIG" --stage "$STAGE" >>"$LOG" 2>&1 || TASK_STATUS=$?
fi

FINAL_STATUS=DONE
[ "$TASK_STATUS" -eq 0 ] || FINAL_STATUS=FAILED
write_state "$FINAL_STATUS" complete "$TASK_STATUS"
{
    printf '# Phase 11 SAFE-REF %s report\n\nrun_id=%s\nsource_run=%s\ncache=%s\ntask_status=%s\n\n## Gate\n\n```json\n' "$STAGE" "$RUN_ID" "$SOURCE_RUN" "$CACHE_DIR" "$TASK_STATUS"
    [ -f "$REPORT_DIR/gate.json" ] && cat "$REPORT_DIR/gate.json" || true
    printf '\n```\n\n## Log tail\n\n```text\n'
    tail -n 80 "$LOG" 2>/dev/null || true
    printf '\n```\n'
} >"$SERVER_REPORT"
PUSH_STATUS=0
[ -d "$REPORT_DIR" ] && LD_LIBRARY_PATH="" git add "$REPORT_DIR"
LD_LIBRARY_PATH="" git add "$SERVER_REPORT" && LD_LIBRARY_PATH="" git commit -m "report: add $RUN_ID" && LD_LIBRARY_PATH="" git pull --rebase origin main && LD_LIBRARY_PATH="" git push origin main || PUSH_STATUS=$?
printf 'run_id=%s task_status=%s push_status=%s\n' "$RUN_ID" "$TASK_STATUS" "$PUSH_STATUS" >>"$LOG"
exit "$TASK_STATUS"
