#!/bin/bash
database=data/uniref100.fasta

for p in data/mavedb/aa_seq/*.fasta
do
    p=$(basename $p .fasta)
    if [ -d output/mavedb/$p/ ]; then
        if [ $(ls output/mavedb/$p/ | grep -c ".failed") -eq 9 ]; then
            rm -rf output/mavedb/$p/
            echo ">>> remove $p"
        else
            echo ">>> skip $p"
            continue
        fi
    fi

    echo ">>> submit $p"
    qsub -v protein=$p,database=$database -N $p script/msa/evcouplings_mavedb.pbs
    echo "============== done $p =============="
done
