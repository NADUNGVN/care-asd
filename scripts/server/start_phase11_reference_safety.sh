#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"; STAGE="${1:-screening}"; cd "$REPO_DIR"
if [ -n "$(git status --porcelain)" ]; then printf 'Refusing to start: server worktree is not clean.\n'; git status --short; exit 1; fi
case "$STAGE" in screening|replication) ;; *) printf 'stage must be screening or replication\n'; exit 1 ;; esac
if pgrep -af '[r]un_phase11_reference_safety.sh' >/dev/null; then printf 'Phase 11 is already running.\n'; exit 1; fi
RUN_ID="server02_phase11_safe_ref_${STAGE}_$(date -u +%Y%m%dT%H%M%SZ)"; JOB_DIR="outputs/reference_safety/$RUN_ID"; mkdir -p "$JOB_DIR"; printf '%s\n' "$RUN_ID" > outputs/reference_safety/latest_run_id.txt
nohup setsid bash scripts/server/run_phase11_reference_safety.sh "$RUN_ID" "$STAGE" >"$JOB_DIR/launcher.log" 2>&1 </dev/null &
printf 'Phase 11 %s started: run_id=%s pid=%s\n' "$STAGE" "$RUN_ID" "$!"
