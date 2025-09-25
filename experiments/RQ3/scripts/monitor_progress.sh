#!/usr/bin/env bash
set -euo pipefail

# Monitor cache growth and pipeline status at a fixed interval.
# Usage: monitor_progress.sh <pipeline_pid> [interval_seconds]

PID="${1:-}"
INTERVAL="${2:-900}"
CSV="experiments/RQ3/inputs/factory_addresses.csv"
SRC="experiments/RQ3/data/sources"
OUTLOG="experiments/RQ3/logs/progress_monitor.log"

mkdir -p "$(dirname "$OUTLOG")"

chains=("polygon" "ethereum")

total_for_chain() {
  local ch="$1"
  awk -F, -v c="$ch" 'NR>1 && tolower($1)==c {n++} END{print n+0}' "$CSV" 2>/dev/null || echo 0
}

cached_for_chain() {
  local ch="$1"
  ls -1 "$SRC/$ch"/*.json 2>/dev/null | wc -l | tr -d ' '
}

echo "[monitor] PID=${PID:-none} interval=${INTERVAL}s" >> "$OUTLOG"

while true; do
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  line="$ts"
  for ch in "${chains[@]}"; do
    tot=$(total_for_chain "$ch")
    cur=$(cached_for_chain "$ch")
    pct=0
    if [ "$tot" -gt 0 ]; then
      pct=$(awk -v c="$cur" -v t="$tot" 'BEGIN{printf "%.2f", (c/t*100)}')
    fi
    line+=" $ch:$cur/$tot(${pct}%)"
  done
  if [ -n "$PID" ]; then
    if kill -0 "$PID" >/dev/null 2>&1; then
      line+=" pipeline:running"
    else
      line+=" pipeline:stopped"
      echo "$line" | tee -a "$OUTLOG"
      break
    fi
  fi
  echo "$line" | tee -a "$OUTLOG"
  sleep "$INTERVAL"
done

