#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
BRANCH="research/ap-care-v2"
cd "$REPO_DIR"

CURRENT_BRANCH="$(git branch --show-current)"
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    printf 'Refusing to start: expected branch %s, found %s.\n' "$BRANCH" "$CURRENT_BRANCH"
    exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    printf 'Refusing to start: server worktree is not clean.\n'
    git status --short
    exit 1
fi
if pgrep -af '[r]un_ap_care_g1.sh' >/dev/null; then
    printf 'AP-CARE G1 is already running.\n'
    exit 1
fi

RUN_ID="server02_ap_care_g1_$(date -u +%Y%m%dT%H%M%SZ)"
JOB_DIR="outputs/ap_care/$RUN_ID"
mkdir -p "$JOB_DIR"
printf '%s\n' "$RUN_ID" >outputs/ap_care/latest_run_id.txt
printf 'run_id=%s\nstatus=STARTING\nstage=launcher\ntask_status=99\nupdated_utc=%s\n' \
    "$RUN_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$JOB_DIR/state.env"

nohup setsid bash scripts/server/run_ap_care_g1.sh "$RUN_ID" \
    >"$JOB_DIR/launcher.log" 2>&1 </dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
printf 'AP-CARE G1 started: run_id=%s pid=%s workers=%s\n' \
    "$RUN_ID" "$PID" "${CARE_ASD_AP_G1_WORKERS:-16}"
