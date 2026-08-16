#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
BRANCH="research/ap-care-v2"
WORKERS="${CARE_ASD_AP_G1_WORKERS:-16}"
RUN_ID="${1:?run id required}"
cd "$REPO_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

JOB_DIR="outputs/ap_care/$RUN_ID"
STATE="$JOB_DIR/state.env"
PROGRESS="$JOB_DIR/progress.env"
LOG="$JOB_DIR/ap_care_g1.log"
REPORT_DIR="reports/ap_care/$RUN_ID"
SERVER_REPORT="reports/server/$RUN_ID.md"
CONFIG="configs/experiment/ap_care_v2.yaml"
SOURCE_SHA="$(git rev-parse HEAD)"
CONFIG_FILE_SHA="$(sha256sum "$CONFIG" | cut -d' ' -f1)"
mkdir -p "$JOB_DIR" reports/ap_care reports/server

write_state() {
    local status="$1" stage="$2" task_status="$3" simulation_status="$4" gate="$5" push_status="$6"
    printf 'run_id=%s\nstatus=%s\nstage=%s\ntask_status=%s\nsimulation_exit_status=%s\ngate_passed=%s\npush_status=%s\nworkers=%s\nsource_git_sha=%s\nupdated_utc=%s\nlog=%s\nreport=%s\n' \
        "$RUN_ID" "$status" "$stage" "$task_status" "$simulation_status" "$gate" \
        "$push_status" "$WORKERS" "$SOURCE_SHA" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$LOG" "$SERVER_REPORT" >"$STATE.tmp"
    mv "$STATE.tmp" "$STATE"
}

TASK_STATUS=0
SIMULATION_STATUS=99
PUSH_STATUS=99
GATE_PASSED=unknown
write_state RUNNING simulation 99 "$SIMULATION_STATUS" "$GATE_PASSED" "$PUSH_STATUS"
printf 'AP-CARE G1 started at %s\nrun_id=%s\nsource_git_sha=%s\nworkers=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ID" "$SOURCE_SHA" "$WORKERS" >"$LOG"

uv run --no-sync care-asd ap-care simulate \
    --config "$CONFIG" \
    --output-dir "$REPORT_DIR" \
    --cases 512 \
    --workers "$WORKERS" \
    --progress-file "$PROGRESS" >>"$LOG" 2>&1
SIMULATION_STATUS=$?

if [ -f "$REPORT_DIR/gate.json" ] && [ -f "$REPORT_DIR/run.json" ]; then
    if grep -Eq '"passed": true' "$REPORT_DIR/gate.json"; then
        GATE_PASSED=true
    else
        GATE_PASSED=false
    fi
    if { [ "$SIMULATION_STATUS" -eq 0 ] && [ "$GATE_PASSED" = true ]; } \
        || { [ "$SIMULATION_STATUS" -eq 2 ] && [ "$GATE_PASSED" = false ]; }; then
        TASK_STATUS=0
    else
        TASK_STATUS="$SIMULATION_STATUS"
        [ "$TASK_STATUS" -ne 0 ] && [ "$TASK_STATUS" -ne 2 ] || TASK_STATUS=1
    fi
else
    TASK_STATUS="$SIMULATION_STATUS"
    [ "$TASK_STATUS" -ne 0 ] || TASK_STATUS=1
fi

write_state RUNNING report "$TASK_STATUS" "$SIMULATION_STATUS" "$GATE_PASSED" "$PUSH_STATUS"
{
    printf '# AP-CARE v2 G1 controlled mechanism report\n\n'
    printf 'run_id=%s\nbranch=%s\nsource_git_sha=%s\nconfig=%s\nconfig_file_sha256=%s\ncases=512\nworkers=%s\ntask_status=%s\nsimulation_exit_status=%s\ngate_passed=%s\n\n' \
        "$RUN_ID" "$BRANCH" "$SOURCE_SHA" "$CONFIG" "$CONFIG_FILE_SHA" "$WORKERS" \
        "$TASK_STATUS" "$SIMULATION_STATUS" "$GATE_PASSED"
    printf 'A simulation exit status of 2 is a completed scientific gate failure, not a runtime failure.\n\n'
    printf '## Gate\n\n```json\n'
    [ -f "$REPORT_DIR/gate.json" ] && cat "$REPORT_DIR/gate.json" || true
    printf '\n```\n\n## Run provenance\n\n```json\n'
    [ -f "$REPORT_DIR/run.json" ] && cat "$REPORT_DIR/run.json" || true
    printf '\n```\n\n## Log tail\n\n```text\n'
    tail -n 80 "$LOG" 2>/dev/null || true
    printf '\n```\n'
} >"$SERVER_REPORT"

if [ -d "$REPORT_DIR" ]; then
    LD_LIBRARY_PATH="" git add "$REPORT_DIR"
fi
LD_LIBRARY_PATH="" git add "$SERVER_REPORT"
if LD_LIBRARY_PATH="" git commit -m "report: add $RUN_ID" \
    && LD_LIBRARY_PATH="" git pull --rebase origin "$BRANCH" \
    && LD_LIBRARY_PATH="" git push origin "$BRANCH"; then
    PUSH_STATUS=0
else
    PUSH_STATUS=$?
fi

FINAL_STATUS=DONE
FINAL_STAGE=complete
FINAL_CODE="$TASK_STATUS"
if [ "$TASK_STATUS" -ne 0 ]; then
    FINAL_STATUS=FAILED
    FINAL_STAGE=simulation
elif [ "$PUSH_STATUS" -ne 0 ]; then
    FINAL_STATUS=FAILED
    FINAL_STAGE=push
    FINAL_CODE=3
fi
write_state "$FINAL_STATUS" "$FINAL_STAGE" "$FINAL_CODE" "$SIMULATION_STATUS" "$GATE_PASSED" "$PUSH_STATUS"
printf 'run_id=%s task_status=%s simulation_exit_status=%s gate_passed=%s push_status=%s\n' \
    "$RUN_ID" "$TASK_STATUS" "$SIMULATION_STATUS" "$GATE_PASSED" "$PUSH_STATUS" >>"$LOG"
exit "$FINAL_CODE"
