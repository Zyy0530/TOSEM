#!/usr/bin/env bash
set -euo pipefail

# Loop fetching top-N factories (global across given chains) until no progress.
# Usage: fetch_topn_loop.sh <N> <workers> <chains...>

TOPN="${1:-5000}"
WORKERS="${2:-64}"
shift 2 || true
CHAINS=("$@")
if [ ${#CHAINS[@]} -eq 0 ]; then CHAINS=("ethereum" "polygon"); fi

LOG="experiments/RQ3/logs/fetch_top${TOPN}.log"
mkdir -p "$(dirname "$LOG")"

count_cached_top() {
  python - << PY 2>/dev/null || echo 0
import csv, os
from pathlib import Path
CSV=Path('experiments/RQ3/inputs/factory_addresses.csv')
BASE=Path('experiments/RQ3/data/sources')
rows=[]
with open(CSV, newline='', encoding='utf-8') as fh:
    rdr=csv.DictReader(fh)
    for r in rdr:
        ch=r['chain'].lower(); addr=r['factory_address'].lower()
        tc=float(r.get('total_creations','nan')) if r.get('total_creations','') not in ('','NA') else float('nan')
        if tc==tc: rows.append((ch,addr,tc))
rows.sort(key=lambda x:x[2], reverse=True)
rows=rows[:int(${TOPN})]
cnt=0
for ch,addr,_ in rows:
    if (BASE/ch/f"{addr}.json").exists(): cnt+=1
print(cnt)
PY
}

prev=$(count_cached_top)
echo "[$(date -u +%FT%TZ)] start fetch loop topN=${TOPN} chains=${CHAINS[*]} workers=${WORKERS} prev=${prev}" | tee -a "$LOG"

while true; do
  ts=$(date -u +%FT%TZ)
  echo "[$ts] fetch run begin prev=$prev" | tee -a "$LOG"
  python experiments/RQ3/run_rq3.py --log INFO fetch --chains "${CHAINS[@]}" \
    --factory-only --sort-by total_creations --top-n "$TOPN" --global-top --max-workers "$WORKERS" | tee -a "$LOG"
  cur=$(count_cached_top)
  delta=$((cur - prev))
  echo "[$ts] fetch run end cur=$cur delta=$delta" | tee -a "$LOG"
  if [ "$cur" -le "$prev" ]; then
    echo "[$ts] no progress, exiting" | tee -a "$LOG"
    break
  fi
  prev="$cur"
  sleep 60
done

