#!/bin/bash
# Submit EVcouplings MSA jobs for MaveDB via SLURM, deduplicating by sequence.
# Only one job per unique sequence; duplicates share the first protein's output.
# Usage: bash script/msa/submit_mavedb.sh

cd .
mkdir -p log

declare -A seq_representative

for p in data/mavedb/aa_seq/*.fasta
do
    name=$(basename $p .fasta)
    seq_md5=$(grep -v "^>" $p | tr -d '\n' | md5sum | awk '{print $1}')

    if [ -z "${seq_representative[$seq_md5]}" ]; then
        seq_representative[$seq_md5]=$name

        if [ -d output/mavedb/$name/ ]; then
            # skip if summary already exists (fully completed)
            if [ -f "output/mavedb/$name/${name}_job_statistics_summary.csv" ]; then
                echo ">>> skip (done) $name"
                continue
            fi
            # remove if all 9 bitscores failed
            failed_count=$(ls output/mavedb/$name/ 2>/dev/null | grep -c ".failed")
            if [ "$failed_count" -eq 9 ]; then
                rm -rf output/mavedb/$name/
                echo ">>> remove all-failed $name"
            fi
        fi

        echo ">>> submit $name"
        sbatch --export=protein=$name \
               --job-name=$name \
               script/msa/evcouplings_mavedb.slurm
    else
        echo ">>> skip (dup of ${seq_representative[$seq_md5]}) $name"
    fi
done
