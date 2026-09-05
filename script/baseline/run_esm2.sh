#!/bin/bash
set -e

BASE_DIR="./data/proteingym_v1"
MODEL_NAME="facebook/esm2_t33_650M_UR50D"

# A: Vanilla (no orbit, no alignment prior)
echo "=== ESM-2 650M: Vanilla ==="
python compute_fitness.py \
    --model_name "$MODEL_NAME" \
    --model_out_name "ESM2-650M" \
    --backbone_mode plain_mlm \
    --alpha 0 \
    --scoring_mode log_odds \
    --base_dir "$BASE_DIR" \
    --out_scores_dir result/baseline_esm2_vanilla \
    "$@"

# B: +Orbit (MSA retriever, linear_alpha, alpha=0.8)
echo "=== ESM-2 650M: +Orbit ==="
python compute_fitness.py \
    --model_name "$MODEL_NAME" \
    --model_out_name "ESM2-650M-Orbit" \
    --backbone_mode plain_mlm \
    --alpha 0.8 \
    --orbit_enable \
    --retriever msa \
    --fusion linear_alpha \
    --logit_mode aa_seq_aln \
    --scoring_mode log_odds \
    --base_dir "$BASE_DIR" \
    --out_scores_dir result/baseline_esm2_orbit \
    "$@"

# C: +Orbit+CCD (calibrated_margin scoring head)
echo "=== ESM-2 650M: +Orbit+CCD ==="
python compute_fitness.py \
    --model_name "$MODEL_NAME" \
    --model_out_name "ESM2-650M-OrbitCCD" \
    --backbone_mode plain_mlm \
    --alpha 0.8 \
    --orbit_enable \
    --retriever msa \
    --fusion linear_alpha \
    --logit_mode aa_seq_aln \
    --scoring_mode calibrated_margin \
    --base_dir "$BASE_DIR" \
    --out_scores_dir result/baseline_esm2_orbit_ccd \
    "$@"

echo "=== ESM-2 650M: All runs complete ==="
