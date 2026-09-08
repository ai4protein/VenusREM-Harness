#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_DIR="${PROTEINGYM_DIR:-$ROOT/data/proteingym_v1}"
MODEL_NAME="facebook/esm2_t33_650M_UR50D"

# A: Vanilla (no VenusREM2, no alignment prior)
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

# B: +VenusREM2 (MSA retriever, linear_alpha, alpha=0.8)
echo "=== ESM-2 650M: +VenusREM2 ==="
python compute_fitness.py \
    --model_name "$MODEL_NAME" \
    --model_out_name "ESM2-650M-VenusREM2" \
    --backbone_mode plain_mlm \
    --alpha 0.8 \
    --retriever msa \
    --fusion linear_alpha \
    --logit_mode aa_seq_aln \
    --scoring_mode log_odds \
    --base_dir "$BASE_DIR" \
    --out_scores_dir result/baseline_esm2_venusrem2 \
    "$@"

# C: +VenusREM2+CCD (calibrated_margin scoring head)
echo "=== ESM-2 650M: +VenusREM2+CCD ==="
python compute_fitness.py \
    --model_name "$MODEL_NAME" \
    --model_out_name "ESM2-650M-VenusREM2CCD" \
    --backbone_mode plain_mlm \
    --alpha 0.8 \
    --retriever msa \
    --fusion linear_alpha \
    --logit_mode aa_seq_aln \
    --scoring_mode calibrated_margin \
    --base_dir "$BASE_DIR" \
    --out_scores_dir result/baseline_esm2_venusrem2_ccd \
    "$@"

echo "=== ESM-2 650M: All runs complete ==="
