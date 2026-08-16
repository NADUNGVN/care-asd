#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"; cd "$REPO_DIR"
if pgrep -af '[r]un_fp_naa_screening.sh' >/dev/null; then exec bash scripts/server/status_fp_naa_screening.sh; fi
if pgrep -af '[r]un_fp_naa_baseline.sh' >/dev/null; then exec bash scripts/server/status_fp_naa_baseline.sh; fi
C0_GATE="$(find reports/fp_naa -path '*/c0_baseline/gate.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
if [ -n "$C0_GATE" ] && grep -q '"passed": true' "$C0_GATE"; then exec bash scripts/server/start_fp_naa_screening.sh; fi
if [ -n "$C0_GATE" ]; then printf 'Latest C0 gate did not pass; candidate screening remains blocked.\n'; cat "$C0_GATE"; exit 2; fi
exec bash scripts/server/start_fp_naa_baseline.sh
