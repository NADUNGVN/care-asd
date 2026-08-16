#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
cd "$REPO_DIR"
RUN_FILE="outputs/ap_care/latest_run_id.txt"
if [ ! -f "$RUN_FILE" ]; then
    printf 'No AP-CARE G1 run has been started.\n'
    exit 0
fi

RUN_ID="$(cat "$RUN_FILE")"
JOB_DIR="outputs/ap_care/$RUN_ID"
REPORT_DIR="reports/ap_care/$RUN_ID"
printf 'RUN_ID=%s\n\nSTATE:\n' "$RUN_ID"
[ -f "$JOB_DIR/state.env" ] && cat "$JOB_DIR/state.env" || printf 'state file not created yet\n'
printf '\nPROGRESS:\n'
[ -f "$JOB_DIR/progress.env" ] && cat "$JOB_DIR/progress.env" || printf 'progress file not created yet\n'
printf '\nPROCESS:\n'
pgrep -af '[r]un_ap_care_g1.sh|care-asd ap-care simulate' || printf 'no matching process\n'
printf '\nRESOURCE SNAPSHOT:\n'
ps -eo pid,etime,stat,pcpu,pmem,args | grep -E '[r]un_ap_care_g1.sh|[c]are-asd ap-care simulate' || printf 'no matching process\n'
printf '\nLOG TAIL:\n'
[ -f "$JOB_DIR/ap_care_g1.log" ] && tail -n 30 "$JOB_DIR/ap_care_g1.log" || printf 'log not created yet\n'
if [ -f "$REPORT_DIR/gate.json" ]; then
    printf '\nGATE:\n'
    cat "$REPORT_DIR/gate.json"
fi
