#!/bin/bash
# Periodically check for completed proteins and generate their summary CSVs
WORKDIR="$(cd "$(dirname "$0")/../.." && pwd)"
WORKDIR="${VENUSREM_ROOT:-$WORKDIR}"
PYTHON="${VENUSREM_PYTHON:-python3}"

while true; do
    generated=0
    for d in $WORKDIR/output/mavedb/*/; do
        protein=$(basename "$d")
        [ -f "$d/${protein}_job_statistics_summary.csv" ] && continue
        count=$(ls "$d"/${protein}_b*_final.outcfg 2>/dev/null | wc -l)
        if [ "$count" -eq 9 ]; then
            $PYTHON $WORKDIR/script/msa/generate_summary.py "$d"
            generated=$((generated+1))
        fi
    done
    [ $generated -gt 0 ] && echo "[$(date)] Generated $generated summaries"
    sleep 300
done
