#!/bin/bash
set -e

BASE_DIR="./data/proteingym_v1"
MODEL_NAME="facebook/esm1v_t33_650M_UR90S_1"

# A: Vanilla (no VenusREM2, no alignment prior)
echo "=== ESM-1v (5-seed ensemble): Vanilla ==="
python compute_fitness.py \
    --model_name "$MODEL_NAME" \
    --model_out_name "ESM1v-Ensemble" \
    --baseline_type esm1v \
    --backbone_mode plain_mlm \
    --alpha 0 \
    --scoring_mode log_odds \
    --base_dir "$BASE_DIR" \
    --out_scores_dir result/baseline_esm1v_vanilla \
    "$@"

# B: +VenusREM2 (MSA retriever, linear_alpha, alpha=0.8)
echo "=== ESM-1v (5-seed ensemble): +VenusREM2 ==="
python compute_fitness.py \
    --model_name "$MODEL_NAME" \
    --model_out_name "ESM1v-Ensemble-VenusREM2" \
    --baseline_type esm1v \
    --backbone_mode plain_mlm \
    --alpha 0.8 \
    --retriever msa \
    --fusion linear_alpha \
    --logit_mode aa_seq_aln \
    --scoring_mode log_odds \
    --base_dir "$BASE_DIR" \
    --out_scores_dir result/baseline_esm1v_venusrem2 \
    "$@"

# C: +VenusREM2+CCD (calibrated_margin scoring head)
echo "=== ESM-1v (5-seed ensemble): +VenusREM2+CCD ==="
python compute_fitness.py \
    --model_name "$MODEL_NAME" \
    --model_out_name "ESM1v-Ensemble-VenusREM2CCD" \
    --baseline_type esm1v \
    --backbone_mode plain_mlm \
    --alpha 0.8 \
    --retriever msa \
    --fusion linear_alpha \
    --logit_mode aa_seq_aln \
    --scoring_mode calibrated_margin \
    --base_dir "$BASE_DIR" \
    --out_scores_dir result/baseline_esm1v_venusrem2_ccd \
    "$@"

echo "=== ESM-1v: All runs complete ==="
