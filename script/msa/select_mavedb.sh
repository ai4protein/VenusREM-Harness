#!/bin/bash
# Step 1: select best MSA for each unique protein that has output
python script/msa/select_msa.py \
    --is_multi \
    --input_dir output/mavedb \
    --output_dir data/mavedb

# Step 2: symlink duplicates (same sequence shares the same a2m)
python -c "
from Bio import SeqIO
from collections import defaultdict
import os

aa_dir = 'data/mavedb/aa_seq'
a2m_dir = 'data/mavedb/aa_seq_aln_a2m'
raw_dir = 'data/mavedb/aa_seq_aln_a2m_raw'

seq_to_files = defaultdict(list)
for f in sorted(os.listdir(aa_dir)):
    if not f.endswith('.fasta'): continue
    name = f.replace('.fasta', '')
    for rec in SeqIO.parse(os.path.join(aa_dir, f), 'fasta'):
        seq_to_files[str(rec.seq)].append(name)
        break

linked = 0
for seq, files in seq_to_files.items():
    rep = files[0]
    rep_a2m = os.path.join(a2m_dir, f'{rep}.a2m')
    rep_raw = os.path.join(raw_dir, f'{rep}.fasta')
    if not os.path.exists(rep_a2m):
        continue
    for dup in files[1:]:
        dst_a2m = os.path.join(a2m_dir, f'{dup}.a2m')
        dst_raw = os.path.join(raw_dir, f'{dup}.fasta')
        if not os.path.exists(dst_a2m):
            os.symlink(os.path.abspath(rep_a2m), dst_a2m)
            linked += 1
        if os.path.exists(rep_raw) and not os.path.exists(dst_raw):
            os.symlink(os.path.abspath(rep_raw), dst_raw)

print(f'Linked {linked} duplicate a2m files')
"
