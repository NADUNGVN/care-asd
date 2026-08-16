#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"; cd "$REPO_DIR"
mkdir -p reports/fp_naa outputs/fp_naa
if pgrep -af '[r]un_fp_naa_reference_safety.sh' >/dev/null; then exec bash scripts/server/status_fp_naa_reference_safety.sh; fi
if pgrep -af '[r]un_fp_naa_confirmatory_lomo.sh' >/dev/null; then exec bash scripts/server/status_fp_naa_confirmatory_lomo.sh; fi
if pgrep -af '[r]un_fp_naa_confirmatory.sh' >/dev/null; then exec bash scripts/server/status_fp_naa_confirmatory.sh; fi
if pgrep -af '[r]un_fp_naa_lomo.sh' >/dev/null; then exec bash scripts/server/status_fp_naa_lomo.sh; fi
if pgrep -af '[r]un_fp_naa_screening.sh' >/dev/null; then exec bash scripts/server/status_fp_naa_screening.sh; fi
if pgrep -af '[r]un_fp_naa_baseline.sh' >/dev/null; then exec bash scripts/server/status_fp_naa_baseline.sh; fi
LOMO_GATE="$(find reports/fp_naa -path '*/lomo/gate.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
CONFIRM_GATE="$(find reports/fp_naa -path '*/confirmatory/gate.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
CONFIRM_LOMO_GATE="$(find reports/fp_naa -path '*/confirmatory_lomo/gate.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
SAFETY_GATE="$(find reports/fp_naa -path '*/reference_safety/gate.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
if [ -n "$SAFETY_GATE" ]; then printf 'Latest FP-NAA reference-safety gate is complete.\n'; cat "$SAFETY_GATE"; exit 0; fi
if [ -n "$CONFIRM_LOMO_GATE" ] && grep -q '"passed": true' "$CONFIRM_LOMO_GATE"; then exec bash scripts/server/start_fp_naa_reference_safety.sh; fi
if [ -n "$CONFIRM_LOMO_GATE" ]; then printf 'Latest confirmatory LOMO gate did not pass; reference safety remains blocked.\n'; cat "$CONFIRM_LOMO_GATE"; exit 2; fi
if [ -n "$CONFIRM_GATE" ] && grep -q '"core_confirmatory": true' "$CONFIRM_GATE"; then exec bash scripts/server/start_fp_naa_confirmatory_lomo.sh; fi
if [ -n "$CONFIRM_GATE" ]; then printf 'Latest five-seed core gate did not pass; confirmatory LOMO remains blocked.\n'; cat "$CONFIRM_GATE"; exit 2; fi
if [ -n "$LOMO_GATE" ] && grep -q '"passed": true' "$LOMO_GATE"; then exec bash scripts/server/start_fp_naa_confirmatory.sh; fi
if [ -n "$LOMO_GATE" ]; then printf 'Latest screening LOMO gate did not pass; confirmatory training remains blocked.\n'; cat "$LOMO_GATE"; exit 2; fi
SCREEN_GATE="$(find reports/fp_naa -path '*/screening/gate.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
if [ -n "$SCREEN_GATE" ] && grep -q '"core_screening": true' "$SCREEN_GATE"; then exec bash scripts/server/start_fp_naa_lomo.sh; fi
if [ -n "$SCREEN_GATE" ]; then printf 'Latest core screening gate did not pass; LOMO remains blocked.\n'; cat "$SCREEN_GATE"; exit 2; fi
C0_GATE="$(find reports/fp_naa -path '*/c0_baseline/gate.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
if [ -n "$C0_GATE" ] && grep -q '"passed": true' "$C0_GATE"; then exec bash scripts/server/start_fp_naa_screening.sh; fi
if [ -n "$C0_GATE" ]; then printf 'Latest C0 gate did not pass; candidate screening remains blocked.\n'; cat "$C0_GATE"; exit 2; fi
exec bash scripts/server/start_fp_naa_baseline.sh
