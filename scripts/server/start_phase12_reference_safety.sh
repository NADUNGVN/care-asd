#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"; cd "$REPO_DIR"
if [ -n "$(git status --porcelain)" ]; then printf 'Refusing to start: server worktree is not clean.\n'; git status --short; exit 1; fi
if pgrep -af '[r]un_phase12_reference_safety.sh' >/dev/null; then printf 'Phase 12 is already running.\n'; exit 1; fi
RUN_ID="server02_phase12_safe_ref_eval_$(date -u +%Y%m%dT%H%M%SZ)"; JOB_DIR="outputs/reference_safety/$RUN_ID"; mkdir -p "$JOB_DIR"; printf '%s\n' "$RUN_ID" > outputs/reference_safety/latest_run_id.txt
nohup setsid bash scripts/server/run_phase12_reference_safety.sh "$RUN_ID" >"$JOB_DIR/launcher.log" 2>&1 </dev/null &
printf 'Phase 12 evaluation started: run_id=%s pid=%s\n' "$RUN_ID" "$!"
