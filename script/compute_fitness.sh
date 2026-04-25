# Basic inference recipes only (no leaderboard sweep / ablation tuning).

# 1) VenusREM zero-shot with aa sequence alignment
export HF_ENDPOINT=https://hf-mirror.com
protein_dir=proteingym_v1
CUDA_VISIBLE_DEVICES=0 python compute_fitness.py \
    --base_dir data/$protein_dir \
    --out_scores_dir result/$protein_dir

# 2) Baseline-style run (ProSST-2048)
export HF_ENDPOINT=https://hf-mirror.com
alpha=0
protein_dir=proteingym_v1
CUDA_VISIBLE_DEVICES=0 python compute_fitness.py \
    --base_dir data/$protein_dir \
    --out_scores_dir result/${protein_dir}_prosst \
    --alpha $alpha \
    --model_out_name ProSST-2048

# 3) Structure-sequence alignment mode
export HF_ENDPOINT=https://hf-mirror.com
alpha=0.8
protein_dir=proteingym_v1
CUDA_VISIBLE_DEVICES=0 python compute_fitness.py \
    --logit_mode struc_seq_aln \
    --alpha $alpha \
    --model_out_name ProtREM-struc \
    --base_dir data/$protein_dir \
    --out_scores_dir result/${protein_dir}_struc

# 4) VenusREM-Orbit basic run (fixed recipe, non-sweep)
export HF_ENDPOINT=https://hf-mirror.com
alpha=0.8
protein_dir=proteingym_v1
CUDA_VISIBLE_DEVICES=1 python compute_fitness.py \
    --base_dir data/$protein_dir \
    --out_scores_dir result/${protein_dir}_orbit \
    --orbit_enable \
    --print_compare_spearman \
    --alpha $alpha \
    --logit_mode aa_seq_aln \
    --model_out_name VenusREM-Orbit \
    --retriever msa \
    --fusion linear_alpha
