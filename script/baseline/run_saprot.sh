#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_DIR="${PROTEINGYM_DIR:-$ROOT/data/proteingym_v1}"
MODEL_NAME="westlake-repl/SaProt_650M_AF2"
FOLDSEEK_BIN="${FOLDSEEK_BIN:-foldseek}"

# A: Vanilla (no VenusREM2, no alignment prior)
echo "=== SaProt 650M: Vanilla ==="
python compute_fitness.py \
    --model_name "$MODEL_NAME" \
    --model_out_name "SaProt-650M" \
    --baseline_type saprot \
    --foldseek_bin "$FOLDSEEK_BIN" \
    --backbone_mode plain_mlm \
    --alpha 0 \
    --scoring_mode log_odds \
    --base_dir "$BASE_DIR" \
    --out_scores_dir result/baseline_saprot_vanilla \
    "$@"

# B: +VenusREM2 (MSA retriever, linear_alpha, alpha=0.8)
echo "=== SaProt 650M: +VenusREM2 ==="
python compute_fitness.py \
    --model_name "$MODEL_NAME" \
    --model_out_name "SaProt-650M-VenusREM2" \
    --baseline_type saprot \
    --foldseek_bin "$FOLDSEEK_BIN" \
    --backbone_mode plain_mlm \
    --alpha 0.8 \
    --retriever msa \
    --fusion linear_alpha \
    --logit_mode aa_seq_aln \
    --scoring_mode log_odds \
    --base_dir "$BASE_DIR" \
    --out_scores_dir result/baseline_saprot_venusrem2 \
    "$@"

# C: +VenusREM2+CCD (calibrated_margin scoring head)
echo "=== SaProt 650M: +VenusREM2+CCD ==="
python compute_fitness.py \
    --model_name "$MODEL_NAME" \
    --model_out_name "SaProt-650M-VenusREM2CCD" \
    --baseline_type saprot \
    --foldseek_bin "$FOLDSEEK_BIN" \
    --backbone_mode plain_mlm \
    --alpha 0.8 \
    --retriever msa \
    --fusion linear_alpha \
    --logit_mode aa_seq_aln \
    --scoring_mode calibrated_margin \
    --base_dir "$BASE_DIR" \
    --out_scores_dir result/baseline_saprot_venusrem2_ccd \
    "$@"

echo "=== SaProt 650M: All runs complete ==="
