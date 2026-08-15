#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
DATA_ROOT="${CARE_ASD_DATA_ROOT:-$HOME/Dung_TDTU/data/CARE_ASD}"
WORKERS="${CARE_ASD_WORKERS:-12}"
RUN_ID="${1:?run id required}"
cd "$REPO_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DEV_RUN="$(find reports/reference_safety -path '*replication*/development/gate.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- | xargs -r dirname | xargs -r dirname | xargs -r basename)"
DEV_GATE="reports/reference_safety/$DEV_RUN/development/gate.json"
DEV_RUN_JSON="reports/reference_safety/$DEV_RUN/development/run.json"
POLICY="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["policy"])' "$DEV_RUN_JSON" 2>/dev/null || true)"
CONFIG="configs/experiment/phase10_reference_safety.yaml"
DEV_MANIFEST="data/manifests/dcase2026_dev.parquet"
ADD_MANIFEST="$DATA_ROOT/manifests/dcase2026_additional.parquet"
EVAL_MANIFEST="$DATA_ROOT/manifests/dcase2026_evaluation.parquet"
ADD_AUDIO="$DATA_ROOT/raw/dcase2026/additional/extracted"
EVAL_AUDIO="$DATA_ROOT/raw/dcase2026/evaluation/extracted"
CACHE_DIR="$DATA_ROOT/reference_safety_cache/evaluation/$RUN_ID"
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/reference_safety/$RUN_ID"
JOB_DIR="outputs/reference_safety/$RUN_ID"
STATE="$JOB_DIR/state.env"
LOG="$JOB_DIR/phase12.log"
EVAL_DIR="$JOB_DIR/evaluation"
REPORT_DIR="reports/reference_safety/$RUN_ID"
FREEZE="$REPORT_DIR/freeze.yaml"
SERVER_REPORT="reports/server/$RUN_ID.md"
mkdir -p "$JOB_DIR" "$REPORT_DIR" "$CHECKPOINT_DIR" "$DATA_ROOT/reference_safety_cache/evaluation" reports/server

write_state() {
    local status="$1" stage="$2" code="$3"
    printf 'run_id=%s\nstatus=%s\nstage=%s\ntask_status=%s\nupdated_utc=%s\nlog=%s\n' "$RUN_ID" "$status" "$stage" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LOG" >"$STATE.tmp"
    mv "$STATE.tmp" "$STATE"
}

TASK_STATUS=0
write_state RUNNING preflight 99
printf 'Phase 12 SAFE-REF evaluation started at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$LOG"
if [ -z "$DEV_RUN" ] || [ -z "$POLICY" ] || [ ! -f "$DEV_GATE" ]; then
    printf 'No passed replication gate/policy found.\n' >>"$LOG"
    TASK_STATUS=1
elif ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$DEV_GATE"; then
    printf 'Latest replication gate did not pass.\n' >>"$LOG"
    TASK_STATUS=1
fi

if [ "$TASK_STATUS" -eq 0 ]; then
    write_state RUNNING data 99
    uv run care-asd data download --split additional --data-root "$DATA_ROOT" >>"$LOG" 2>&1 || TASK_STATUS=$?
fi
if [ "$TASK_STATUS" -eq 0 ]; then uv run care-asd data extract --split additional --data-root "$DATA_ROOT" >>"$LOG" 2>&1 || TASK_STATUS=$?; fi
if [ "$TASK_STATUS" -eq 0 ] && [ ! -f "$ADD_MANIFEST" ]; then uv run care-asd data manifest --split additional --data-root "$DATA_ROOT" >>"$LOG" 2>&1 || TASK_STATUS=$?; fi
if [ "$TASK_STATUS" -eq 0 ]; then uv run care-asd data validate --split additional --data-root "$DATA_ROOT" >>"$LOG" 2>&1 || TASK_STATUS=$?; fi
if [ "$TASK_STATUS" -eq 0 ]; then uv run care-asd data download --split evaluation --data-root "$DATA_ROOT" --accept-eval-policy >>"$LOG" 2>&1 || TASK_STATUS=$?; fi
if [ "$TASK_STATUS" -eq 0 ]; then uv run care-asd data extract --split evaluation --data-root "$DATA_ROOT" >>"$LOG" 2>&1 || TASK_STATUS=$?; fi
if [ "$TASK_STATUS" -eq 0 ] && [ ! -f "$EVAL_MANIFEST" ]; then uv run care-asd data manifest --split evaluation --data-root "$DATA_ROOT" >>"$LOG" 2>&1 || TASK_STATUS=$?; fi
if [ "$TASK_STATUS" -eq 0 ]; then uv run care-asd data validate --split evaluation --data-root "$DATA_ROOT" >>"$LOG" 2>&1 || TASK_STATUS=$?; fi

if [ "$TASK_STATUS" -eq 0 ]; then
    write_state RUNNING cache 99
    uv run --no-sync care-asd data cache-reference-safety-vectors --train-manifest "$ADD_MANIFEST" --train-audio-root "$ADD_AUDIO" --test-manifest "$EVAL_MANIFEST" --test-audio-root "$EVAL_AUDIO" --output-dir "$CACHE_DIR" --config "$CONFIG" --workers "$WORKERS" >>"$LOG" 2>&1 || TASK_STATUS=$?
fi
if [ "$TASK_STATUS" -eq 0 ]; then
    write_state RUNNING freeze 99
    uv run --no-sync care-asd reference-safety freeze --policy "$POLICY" --development-gate "$DEV_GATE" --development-manifest "$DEV_MANIFEST" --output "$FREEZE" --config "$CONFIG" >>"$LOG" 2>&1 || TASK_STATUS=$?
fi
if [ "$TASK_STATUS" -eq 0 ]; then
    LD_LIBRARY_PATH="" git add "$FREEZE" && LD_LIBRARY_PATH="" git commit -m "experiment: freeze $RUN_ID" && LD_LIBRARY_PATH="" git pull --rebase origin main && LD_LIBRARY_PATH="" git push origin main || TASK_STATUS=$?
fi
if [ "$TASK_STATUS" -eq 0 ]; then
    write_state RUNNING scoring 99
    uv run --no-sync care-asd reference-safety eval --cache-dir "$CACHE_DIR" --policy "$POLICY" --freeze-file "$FREEZE" --output-dir "$EVAL_DIR" --checkpoint-dir "$CHECKPOINT_DIR" --config "$CONFIG" >>"$LOG" 2>&1 || TASK_STATUS=$?
fi
if [ "$TASK_STATUS" -eq 0 ]; then
    write_state RUNNING evaluator 99
    uv run care-asd baseline checkout >>"$LOG" 2>&1 || TASK_STATUS=$?
fi
if [ "$TASK_STATUS" -eq 0 ]; then
    uv run --no-sync care-asd reference-safety official-score --evaluation-output-dir "$EVAL_DIR" --evaluator-dir external/dcase2026_task2_evaluator --output-dir "$REPORT_DIR/official" >>"$LOG" 2>&1 || TASK_STATUS=$?
fi
if [ "$TASK_STATUS" -eq 0 ]; then
    mkdir -p "$REPORT_DIR/frozen_scores"
    cp "$EVAL_DIR/score_complete.json" "$EVAL_DIR/decisions.csv" "$REPORT_DIR/frozen_scores/"
    for SYSTEM in near unconditional_refsub safe_ref; do
        mkdir -p "$REPORT_DIR/frozen_scores/$SYSTEM"
        cp "$EVAL_DIR/$SYSTEM/scores.csv" "$REPORT_DIR/frozen_scores/$SYSTEM/"
    done
fi

FINAL_STATUS=DONE
[ "$TASK_STATUS" -eq 0 ] || FINAL_STATUS=FAILED
write_state "$FINAL_STATUS" complete "$TASK_STATUS"
{
    printf '# Phase 12 frozen SAFE-REF evaluation\n\nrun_id=%s\ndevelopment_run=%s\ncache=%s\nworkers=%s\ntask_status=%s\nfreeze=%s\n\n## Log tail\n\n```text\n' "$RUN_ID" "$DEV_RUN" "$CACHE_DIR" "$WORKERS" "$TASK_STATUS" "$FREEZE"
    tail -n 100 "$LOG" 2>/dev/null || true
    printf '\n```\n'
} >"$SERVER_REPORT"
PUSH_STATUS=0
LD_LIBRARY_PATH="" git add "$REPORT_DIR" "$SERVER_REPORT" && LD_LIBRARY_PATH="" git commit -m "report: add $RUN_ID" && LD_LIBRARY_PATH="" git pull --rebase origin main && LD_LIBRARY_PATH="" git push origin main || PUSH_STATUS=$?
printf 'run_id=%s task_status=%s push_status=%s\n' "$RUN_ID" "$TASK_STATUS" "$PUSH_STATUS" >>"$LOG"
exit "$TASK_STATUS"
