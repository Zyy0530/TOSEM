#!/usr/bin/env bash
set -euo pipefail

# Run robust full fetch for polygon and ethereum, then cluster both.
# Logs are appended to experiments/RQ3/logs/fetch_all.log

WORKERS="${1:-64}"
LOG="experiments/RQ3/logs/fetch_all.log"
mkdir -p "$(dirname "$LOG")"

echo "[$(date -u +%FT%TZ)] pipeline start workers=$WORKERS" | tee -a "$LOG"
bash experiments/RQ3/scripts/fetch_all_loop.sh polygon "$WORKERS"
bash experiments/RQ3/scripts/fetch_all_loop.sh ethereum "$WORKERS"
echo "[$(date -u +%FT%TZ)] clustering begin" | tee -a "$LOG"
python experiments/RQ3/run_rq3.py --log INFO cluster --chains polygon ethereum | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] pipeline done" | tee -a "$LOG"

