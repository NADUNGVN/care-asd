#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"; cd "$REPO_DIR"
if [ -n "$(env -u LD_LIBRARY_PATH -u LD_PRELOAD git status --porcelain)" ]; then printf 'Refusing to start: server worktree is not clean.\n'; env -u LD_LIBRARY_PATH -u LD_PRELOAD git status --short; exit 1; fi
if pgrep -af '[r]un_fp_naa_baseline.sh' >/dev/null; then printf 'FP-NAA baseline is already running.\n'; exit 1; fi
RUN_ID="server02_fp_naa_c0_$(date -u +%Y%m%dT%H%M%SZ)"; JOB_DIR="outputs/fp_naa/$RUN_ID"; mkdir -p "$JOB_DIR"; printf '%s\n' "$RUN_ID" > outputs/fp_naa/latest_run_id.txt
nohup setsid bash scripts/server/run_fp_naa_baseline.sh "$RUN_ID" > "$JOB_DIR/launcher.log" 2>&1 < /dev/null &
printf 'FP-NAA C0 started: run_id=%s pid=%s workers=%s\n' "$RUN_ID" "$!" "${CARE_ASD_WORKERS:-12}"

