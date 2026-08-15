#!/usr/bin/env bash
set -uo pipefail
REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"; cd "$REPO_DIR"; RUN_FILE="outputs/reference_safety/latest_run_id.txt"
if [ ! -f "$RUN_FILE" ]; then printf 'No SAFE-REF run has been started.\n'; exit 0; fi
RUN_ID="$(cat "$RUN_FILE")"; JOB_DIR="outputs/reference_safety/$RUN_ID"; printf 'RUN_ID=%s\n\nSTATE:\n' "$RUN_ID"; [ -f "$JOB_DIR/state.env" ] && cat "$JOB_DIR/state.env" || printf 'state file not created yet\n'; printf '\nPROCESS:\n'; pgrep -af '[r]un_phase1[012]_reference_safety.sh|care-asd reference-safety|cache-reference-safety-vectors' || printf 'no matching process\n'; printf '\nLOG TAIL:\n'; find "$JOB_DIR" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- | xargs -r tail -n 30
