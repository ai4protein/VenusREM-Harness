#!/bin/bash
# Run evcouplings MSA search locally in parallel with strict resource isolation.
# Each protein job is pinned to dedicated CPU cores via taskset.
# Usage: nohup bash script/msa/run_mavedb_local.sh > log/run_mavedb.log 2>&1 &

CONDA_BIN=$CONDA_PREFIX/bin
DATABASE=$UNIREF100
WORKDIR=.
UNIQUE_LIST=$WORKDIR/data/mavedb/unique_proteins.txt
BITSCORES="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

# --- Resource isolation ---
CORES_PER_JOB=8
PARALLEL=12               # 12 jobs × 8 cores = 96 / 128 cores

export PATH=$CONDA_BIN:$PATH
cd $WORKDIR
mkdir -p log

TOTAL_CORES=$(nproc)
MAX_SLOTS=$(( TOTAL_CORES / CORES_PER_JOB ))
[ $PARALLEL -gt $MAX_SLOTS ] && PARALLEL=$MAX_SLOTS

SLOT_DIR=$(mktemp -d)
acquire_slot() {
    while true; do
        for i in $(seq 0 $(( PARALLEL - 1 ))); do
            if mkdir "$SLOT_DIR/lock_$i" 2>/dev/null; then echo $i; return; fi
        done
        sleep 2
    done
}
release_slot() { rmdir "$SLOT_DIR/lock_$1" 2>/dev/null; }

# Step 1 is serialized: generate all configs first, then run in parallel.
# This avoids billiard workers spawning uncontrolled jackhmmer processes.
generate_configs() {
    local protein=$1
    local outdir="output/mavedb/$protein"
    local logfile="log/evcouplings_${protein}.log"

    [ -f "$outdir/${protein}_job_statistics_summary.csv" ] && return 0
    [ -f "$outdir/${protein}_b0.9_config.txt" ] && return 0

    echo "[CONFIG] $protein $(date)"

    setsid evcouplings \
        -P "$outdir/$protein" \
        -p "$protein" \
        -s "data/mavedb/aa_seq/${protein}.fasta" \
        -d "$DATABASE" \
        -b "${BITSCORES// /,}" \
        -n 5 \
        src/single_config_monomer.txt \
        > "$logfile" 2>&1 &
    local pid=$!

    # Poll for configs (1s intervals, 30s max), kill immediately when ready.
    local waited=0
    local ok=0
    while [ $waited -lt 30 ]; do
        sleep 1
        waited=$((waited+1))
        ok=0
        for b in $BITSCORES; do
            [ -f "${outdir}/${protein}_b${b}_config.txt" ] && ok=$((ok+1))
        done
        [ $ok -eq 9 ] && break
    done

    # kill evcouplings + billiard workers immediately
    kill -- -$pid 2>/dev/null
    sleep 0.5
    kill -9 -- -$pid 2>/dev/null
    wait $pid 2>/dev/null
    # kill any orphaned jackhmmer from billiard
    pkill -9 -f "jackhmmer.*${protein}" 2>/dev/null

    if [ $ok -eq 0 ]; then
        echo "[CONFIG-FAIL] $protein: 0/9 configs after ${waited}s"
        return 1
    fi
    echo "[CONFIG-OK] $protein: $ok/9 configs (${waited}s)"
}

run_one() {
    local protein=$1
    local logfile="log/evcouplings_${protein}.log"
    local outdir="output/mavedb/$protein"

    [ -f "$outdir/${protein}_job_statistics_summary.csv" ] && echo "[SKIP] $protein" && return 0

    # verify configs exist (generated in phase 1)
    local cfg_count=0
    for b in $BITSCORES; do
        [ -f "${outdir}/${protein}_b${b}_config.txt" ] && cfg_count=$((cfg_count+1))
    done
    [ $cfg_count -eq 0 ] && echo "[SKIP-NOCONFIG] $protein" && return 1

    local slot=$(acquire_slot)
    local core_start=$(( slot * CORES_PER_JOB ))
    local core_end=$(( core_start + CORES_PER_JOB - 1 ))

    echo "[RUN] $protein slot=$slot cores=${core_start}-${core_end} $(date)"

    local all_configs=""
    local has_failure=0
    for b in $BITSCORES; do
        local bcfg="${outdir}/${protein}_b${b}_config.txt"
        [ ! -f "$bcfg" ] && continue
        all_configs="$all_configs $bcfg"
        echo "[RUN] $protein b${b} $(date)" >> "$logfile"
        taskset -c ${core_start}-${core_end} evcouplings_runcfg "$bcfg" >> "$logfile" 2>&1
        [ $? -ne 0 ] && echo "[FAIL-SUB] $protein b${b}" >> "$logfile" && has_failure=1
    done

    local global_cfg="${outdir}/${protein}_config.txt"
    if [ -f "$global_cfg" ] && [ -n "$all_configs" ]; then
        taskset -c ${core_start}-${core_end} evcouplings_summarize protein_monomer "$outdir/$protein" $all_configs >> "$logfile" 2>&1
    fi

    release_slot $slot

    [ $has_failure -eq 0 ] && echo "[DONE] $protein $(date)" || echo "[PARTIAL] $protein $(date)"
}

export -f run_one acquire_slot release_slot
export CONDA_BIN DATABASE WORKDIR BITSCORES PARALLEL CORES_PER_JOB SLOT_DIR

echo "========================================"
echo "  evcouplings MSA search (isolated)"
echo "========================================"
echo "Proteins:       $(wc -l < $UNIQUE_LIST)"
echo "Parallel:       $PARALLEL"
echo "Cores/job:      $CORES_PER_JOB"
echo "Total cores:    $(nproc)"
echo "Total mem:      $(free -g | awk '/Mem:/{print $2}')G"
echo "Database:       $DATABASE"
echo "Time:           $(date)"
echo "========================================"

# Phase 1: generate all configs sequentially (fast, <5s each, no resource contention)
echo "=== Phase 1: generating configs ==="
while IFS= read -r protein; do
    generate_configs "$protein"
done < "$UNIQUE_LIST"
echo "=== Phase 1 done $(date) ==="

# Kill all orphaned jackhmmer from Phase 1 billiard workers
echo "=== Cleaning orphaned jackhmmer ==="
killall -9 jackhmmer 2>/dev/null
sleep 2
killall -9 jackhmmer 2>/dev/null
echo "jackhmmer remaining: $(pgrep -c jackhmmer 2>/dev/null || echo 0)"

# Phase 2: run jackhmmer + plmc in parallel with CPU isolation
echo "=== Phase 2: running evcouplings_runcfg ==="
cat $UNIQUE_LIST | xargs -I {} -P $PARALLEL bash -c 'run_one "$@"' _ {}

rm -rf "$SLOT_DIR"
echo "=== All done $(date) ==="
