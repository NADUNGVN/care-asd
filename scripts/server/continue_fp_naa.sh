#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"; cd "$REPO_DIR"
if pgrep -af '[r]un_fp_naa_lomo.sh' >/dev/null; then exec bash scripts/server/status_fp_naa_lomo.sh; fi
if pgrep -af '[r]un_fp_naa_screening.sh' >/dev/null; then exec bash scripts/server/status_fp_naa_screening.sh; fi
if pgrep -af '[r]un_fp_naa_baseline.sh' >/dev/null; then exec bash scripts/server/status_fp_naa_baseline.sh; fi
LOMO_GATE="$(find reports/fp_naa -path '*/lomo/gate.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
if [ -n "$LOMO_GATE" ]; then printf 'Latest LOMO gate is complete; report it for confirmatory analysis.\n'; cat "$LOMO_GATE"; exit 0; fi
SCREEN_GATE="$(find reports/fp_naa -path '*/screening/gate.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
if [ -n "$SCREEN_GATE" ] && grep -q '"core_screening": true' "$SCREEN_GATE"; then exec bash scripts/server/start_fp_naa_lomo.sh; fi
if [ -n "$SCREEN_GATE" ]; then printf 'Latest core screening gate did not pass; LOMO remains blocked.\n'; cat "$SCREEN_GATE"; exit 2; fi
C0_GATE="$(find reports/fp_naa -path '*/c0_baseline/gate.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
if [ -n "$C0_GATE" ] && grep -q '"passed": true' "$C0_GATE"; then exec bash scripts/server/start_fp_naa_screening.sh; fi
if [ -n "$C0_GATE" ]; then printf 'Latest C0 gate did not pass; candidate screening remains blocked.\n'; cat "$C0_GATE"; exit 2; fi
exec bash scripts/server/start_fp_naa_baseline.sh
