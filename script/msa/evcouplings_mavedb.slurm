#!/bin/bash
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=168:00:00
#SBATCH --output=log/%x.o
#SBATCH --error=log/%x.e

WORKDIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_BIN="${VENUSREM_CONDA_BIN:-${CONDA_PREFIX:+$CONDA_PREFIX/bin}}"
DATABASE="${UNIREF100:?Set UNIREF100 to a UniRef FASTA path}"
BITSCORES="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

if [ -n "$CONDA_BIN" ]; then
    export PATH="$CONDA_BIN:$PATH"
fi
cd "$WORKDIR"

outdir="output/mavedb/$protein"

# Step 1: generate per-bitscore configs (kill after configs are written to avoid billiard workers)
timeout 15 evcouplings \
    -P "$outdir/$protein" \
    -p "$protein" \
    -s "data/mavedb/aa_seq/${protein}.fasta" \
    -d "$DATABASE" \
    -b "${BITSCORES// /,}" \
    -n 5 \
    script/msa/single_config_monomer.txt > /dev/null 2>&1 || true

# Step 2: run each bitscore sequentially (no resource contention)
all_configs=""
for b in $BITSCORES; do
    bcfg="${outdir}/${protein}_b${b}_config.txt"
    [ -f "$bcfg" ] && all_configs="$all_configs $bcfg"
done

for b in $BITSCORES; do
    bcfg="${outdir}/${protein}_b${b}_config.txt"
    [ -f "$bcfg" ] || continue
    echo "[RUN] $protein b${b} $(date)"
    evcouplings_runcfg "$bcfg"
    [ $? -ne 0 ] && echo "[FAIL] $protein b${b}"
done

# Step 3: summarize
global_cfg="${outdir}/${protein}_config.txt"
if [ -f "$global_cfg" ] && [ -n "$all_configs" ]; then
    evcouplings_summarize protein_monomer "$outdir/$protein" $all_configs
fi

echo "[DONE] $protein $(date)"
