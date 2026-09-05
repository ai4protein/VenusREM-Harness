#!/bin/bash
set -e

BASE_DIR="./data/proteingym_v1"
MODEL_NAME="westlake-repl/SaProt_650M_AF2"
FOLDSEEK_BIN="foldseek"

# A: Vanilla (no orbit, no alignment prior)
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

# B: +Orbit (MSA retriever, linear_alpha, alpha=0.8)
echo "=== SaProt 650M: +Orbit ==="
python compute_fitness.py \
    --model_name "$MODEL_NAME" \
    --model_out_name "SaProt-650M-Orbit" \
    --baseline_type saprot \
    --foldseek_bin "$FOLDSEEK_BIN" \
    --backbone_mode plain_mlm \
    --alpha 0.8 \
    --orbit_enable \
    --retriever msa \
    --fusion linear_alpha \
    --logit_mode aa_seq_aln \
    --scoring_mode log_odds \
    --base_dir "$BASE_DIR" \
    --out_scores_dir result/baseline_saprot_orbit \
    "$@"

# C: +Orbit+CCD (calibrated_margin scoring head)
echo "=== SaProt 650M: +Orbit+CCD ==="
python compute_fitness.py \
    --model_name "$MODEL_NAME" \
    --model_out_name "SaProt-650M-OrbitCCD" \
    --baseline_type saprot \
    --foldseek_bin "$FOLDSEEK_BIN" \
    --backbone_mode plain_mlm \
    --alpha 0.8 \
    --orbit_enable \
    --retriever msa \
    --fusion linear_alpha \
    --logit_mode aa_seq_aln \
    --scoring_mode calibrated_margin \
    --base_dir "$BASE_DIR" \
    --out_scores_dir result/baseline_saprot_orbit_ccd \
    "$@"

echo "=== SaProt 650M: All runs complete ==="
