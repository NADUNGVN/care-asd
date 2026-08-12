#!/usr/bin/env bash
# Build the immutable Phase 5 base-feature cache and push one compact report.
set -uo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
DATA_ROOT="${CARE_ASD_DATA_ROOT:-$HOME/Dung_TDTU/data/CARE_ASD}"
WORKERS="${CARE_ASD_CACHE_WORKERS:-12}"
cd "$REPO_DIR"
RUN_ID="server02_phase5_cache_$(date -u +%Y%m%dT%H%M%SZ)"
CACHE_DIR="$DATA_ROOT/neural_cache/$RUN_ID"
LOG_PATH="outputs/neural/$RUN_ID.log"
REPORT="reports/server/$RUN_ID.md"
mkdir -p "$DATA_ROOT/neural_cache" outputs/neural reports/server

uv run care-asd data cache-neural --manifest data/manifests/dcase2026_dev.parquet --audio-root "$DATA_ROOT/raw/dcase2026/dev/extracted" --output-dir "$CACHE_DIR" --workers "$WORKERS" >"$LOG_PATH" 2>&1
TASK_STATUS=$?
{
    printf '# Phase 5 cache report\n\nrun_id=%s\ncache_dir=%s\ntask_status=%s\n\n## Log\n\n```text\n' "$RUN_ID" "$CACHE_DIR" "$TASK_STATUS"
    tail -n 30 "$LOG_PATH"
    printf '```\n\n## Metadata\n\n```json\n'
    [ -f "$CACHE_DIR/cache.json" ] && cat "$CACHE_DIR/cache.json"
    printf '\n```\n\nnpz_files='
    find "$CACHE_DIR/features" -type f -name '*.npz' 2>/dev/null | wc -l
    printf 'cache_size='
    du -sh "$CACHE_DIR" 2>/dev/null || true
} >"$REPORT"

LD_LIBRARY_PATH="" git add "$REPORT"
LD_LIBRARY_PATH="" git commit -m "report: add $RUN_ID"
LD_LIBRARY_PATH="" git pull --rebase origin main
LD_LIBRARY_PATH="" git push origin main
printf 'run_id=%s task_status=%s\n' "$RUN_ID" "$TASK_STATUS"
exit "$TASK_STATUS"
