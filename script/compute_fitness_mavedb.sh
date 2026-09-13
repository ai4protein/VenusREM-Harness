#!/bin/bash
# MaveDB ProSST-2048 baseline scoring
# Step 1: generate structure sequences from PDB files
# Step 2: run ProSST scoring (alpha=0, no retrieval)

export HF_ENDPOINT=https://hf-mirror.com
protein_dir=mavedb

# Step 1: Generate struc_seq from pdbs (skips already processed)
python script/structure/get_struc_seq.py \
    --pdb_dir data/$protein_dir/pdbs \
    --output_dir data/$protein_dir/struc_seq \
    --num_processes 24 \
    --num_threads 24 \
    --max_batch_nodes 15000 \
    --cache_subgraph_dir data/$protein_dir/cache_subgraph

# Step 2: ProSST-2048 baseline (alpha=0)
CUDA_VISIBLE_DEVICES=0 python compute_fitness.py \
    --base_dir data/$protein_dir \
    --out_scores_dir result/${protein_dir}_prosst \
    --alpha 0 \
    --model_out_name ProSST-2048
