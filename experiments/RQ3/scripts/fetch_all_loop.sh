#!/usr/bin/env bash
set -euo pipefail

# Robust full-chain fetch with retry loops until no progress, then exit.
# Usage: fetch_all_loop.sh <chain> [max_workers]

chain="$1"; shift || true
workers="${1:-64}"

LOG="experiments/RQ3/logs/fetch_all.log"
SRC_DIR="experiments/RQ3/data/sources/$chain"
mkdir -p "$(dirname "$LOG")" "$SRC_DIR"

count_cached() {
  ls -1 "$SRC_DIR"/*.json 2>/dev/null | wc -l | tr -d ' '
}

prev=$(count_cached)
echo "[$(date -u +%FT%TZ)] start fetch loop chain=$chain workers=$workers prev=$prev" | tee -a "$LOG"

while true; do
  ts=$(date -u +%FT%TZ)
  echo "[$ts] fetch run begin chain=$chain prev=$prev" | tee -a "$LOG"
  python experiments/RQ3/run_rq3.py --log INFO fetch --chains "$chain" --max-workers "$workers" | tee -a "$LOG"
  cur=$(count_cached)
  delta=$((cur - prev))
  echo "[$ts] fetch run end chain=$chain cur=$cur delta=$delta" | tee -a "$LOG"
  if [ "$cur" -le "$prev" ]; then
    echo "[$ts] no progress detected for $chain, exiting loop" | tee -a "$LOG"
    break
  fi
  prev="$cur"
  sleep 60
done

