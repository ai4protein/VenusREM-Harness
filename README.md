# Dynamic spatial gating aligns evolutionary retrieval with protein language models

## 🚀 Introduction

This repository now has two paper tracks:

- **VenusREM-Orbit**  
  *Dynamic spatial gating aligns evolutionary retrieval with protein language models*
- **VenusREM-v1 (published, ISMB/ECCB 2025):**  
  *From high-throughput evaluation to wet-lab studies: advancing mutation effect prediction with a retrieval-enhanced model*

## 🧭 Directory Navigation

### General
- [Repository tracks](#repository-tracks)
- [Quick start by track](#quick-start-by-track)
- [Requirements](#-requirement)
- [News and downloads](#-results)
- [Common directories and components](#-common-directories-and-components)
- [VenusREM-Orbit usage](#-venusrem-orbit-usage)
- [VenusREM (v1) usage](#-venusrem-v1-usage)
- [Citation](#-citation)

### Repository tracks

- `v1.0.0`: Published VenusREM paper release.
- `venusrem-v1-maintenance`: Bug-fix only branch for v1.
- `main`: Active VenusREM-Orbit development line.

### Quick start by track

- **Part I (VenusREM-v1):** use `v1.0.0` or `venusrem-v1-maintenance` for published baseline reproduction.
- **Part II (VenusREM-Orbit):** use `main` for dynamic gating + dual-retriever Orbit experiments.

## 📑 Results

### News

- [2026.04.25] VenusREM-Orbit mainline upgraded to **Orbit-PlusV2** strategy (dynamic spatial gating + dual-layer evolutionary retrieval).
- [2026.04.01] VenusREM has been integrated into [VenusFactory2](https://github.com/ai4protein/VenusFactory2). Welcome to use it! Here is the [web server](https://venusfactory.cn/playground/) and [technical report](https://arxiv.org/abs/2603.27303).
- [2025.07.21] Our paper was online at [Bioinformatics](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i401/8199374).
- [2025.04.19] We rank 1st in the [ProteinGym](https://proteingym.org/benchmarks) substitution leaderboard!
- [2025.04.09] Congratulations! Our paper was accepted by *ISMB/ECCB 2025*! See you in Liverpool, England.

### Downloads

- ProteinGym a2m homology sequences (EVCouplings): https://huggingface.co/datasets/AI4Protein/VenusREM/resolve/main/aa_seq_aln_a2m.tar.gz. The original a2m files are downloaded at [ProteinGym](https://github.com/OATML-Markslab/ProteinGym).
- ProteinGym a3m homology sequences (ColabFold): https://huggingface.co/datasets/AI4Protein/VenusREM/resolve/main/aa_seq_aln_a3m.tar.gz
- Uniref 100 database: https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz

## 🛫 Requirement

### Conda Enviroment

Please make sure you have installed **[Anaconda3](https://www.anaconda.com/download)** or **[Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)**.

```
conda env create -f environment.yml
conda activate venusrem

# We need HMMER and EVCouplings for MSA
# pip install hmmer
# pip install https://github.com/debbiemarkslab/EVcouplings/archive/develop.zip
```

### Other Requirement

Install plmc and change the path in `src/single_config_monomer.txt`
```shell
git clone https://github.com/debbiemarkslab/plmc.git
cd plmc
make all-openmp
```

### Hardware

- For direct use of inference, we recommend at least 10G of graphics memory, such as RTX 3080
- For searching homology sequences, 8 cores cpu.

## 🧱 Common Directories and Components

Common branches:
- `v1.0.0`: published VenusREM release
- `venusrem-v1-maintenance`: maintenance for v1
- `main`: VenusREM-Orbit mainline

Common scripts/components:
- `compute_fitness.py`: shared scoring entrypoint
- `script/compute_fitness.sh`: shared basic runs
- `script/compute_fitness_advance.sh`: shared advanced configurable run
- `script/compute_fitness_calibrated.sh`: Orbit-PlusV2 + calibrated-margin scoring run
- `data/proteingym_v1`: common benchmark input root
- `result/`: common output root
- `output/`: shared intermediate artifacts (e.g., homolog search outputs)

Common dataset workflow (for your own proteins):
1) Prepare `aa_seq`, `substitutions`, optional `pdbs`
2) Build MSA (`evcouplings` + `src/data/select_msa.py`)
3) Build structure sequence (`src/data/get_struc_seq.py`)
4) Run `compute_fitness.py` with either v1 or Orbit args

Expected structure:
```shell
data/<your_protein_dir_name>
|——aa_seq
|——|——protein1.fasta
|——|——protein2.fasta
|——aa_seq_aln_a2m
|——|——protein1.a2m
|——|——protein2.a2m
|——pdbs
|——|——protein1.pdb
|——|——protein2.pdb
|——struc_seq
|——|——protein1.fasta
|——|——protein2.fasta
|——substitutions
|——|——protein1.csv
|——|——protein2.csv
```

## 🛰 VenusREM-Orbit Usage

Recommended branch:
- `main`

Model/parameter profile:
- Enable Orbit via `--orbit_enable`
- Standard Orbit: `retriever=msa`, `fusion=linear_alpha`, `alpha=0.8`
- Orbit-PlusV2: dual retriever + dual-stage gating (`retriever2`, `fusion2`, PSALOR and gate hyperparameters)

Standard Orbit run:
```shell
protein_dir=proteingym_v1
python compute_fitness.py \
    --base_dir data/$protein_dir \
    --out_scores_dir result/${protein_dir}_orbit \
    --orbit_enable \
    --print_compare_spearman \
    --alpha 0.8 \
    --logit_mode aa_seq_aln \
    --model_out_name VenusREM-Orbit \
    --retriever msa \
    --fusion linear_alpha
```

Orbit with homolog hits metadata:
```shell
protein_dir=proteingym_v1
python compute_fitness.py \
    --base_dir data/$protein_dir \
    --out_scores_dir result/${protein_dir}_orbit_hits \
    --orbit_enable \
    --print_compare_spearman \
    --alpha 0.8 \
    --logit_mode aa_seq_aln \
    --model_out_name VenusREM-Orbit-Hits \
    --retriever hits \
    --fusion adaptive_gate \
    --hits_dir output/$protein_dir
```

Orbit-PlusV2 SOTA recipe:
```shell
protein_dir=proteingym_v1
python compute_fitness.py \
    --base_dir data/$protein_dir \
    --out_scores_dir result/${protein_dir}_orbit_plus_v2_full \
    --orbit_enable \
    --print_compare_spearman \
    --alpha 0.35 \
    --logit_mode aa_seq_aln \
    --model_out_name VenusREM-Orbit-PlusV2 \
    --retriever psalor_exact \
    --retriever2 psalor_variant \
    --fusion adaptive_gate \
    --fusion2 two_stage_learnable_gate \
    --adaptive_min_gate 0.0 \
    --adaptive_max_gate 0.85 \
    --psalor_mix 0.35 \
    --psalor_exact_weight 0.7 \
    --psalor_variant_weight 0.3 \
    --rsa_mode rsa \
    --pdb_dir DMS_ProteinGym_substitutions_pdbs \
    --dedup_identity_threshold 0.85 \
    --max_sequences_for_clustering 2000 \
    --layer2_weight 0.55 \
    --gate_temperature 1.4 \
    --alpha_family 0.85 \
    --hits_dir output/$protein_dir \
    --enable_gate_diagnostics
```

Orbit-PlusV2 with calibrated-margin scoring:
```shell
protein_dir=proteingym_v1
python compute_fitness.py \
    --base_dir data/$protein_dir \
    --out_scores_dir result/${protein_dir}_orbit_plus_v2_calibrated \
    --orbit_enable \
    --print_compare_spearman \
    --alpha 0.35 \
    --logit_mode aa_seq_aln \
    --model_out_name VenusREM-Orbit-PlusV2-Calibrated \
    --scoring_mode calibrated_margin \
    --background_weight 0.25 \
    --uncertainty_weight 0.15 \
    --wt_confidence_weight 0.1 \
    --retriever psalor_exact \
    --retriever2 psalor_variant \
    --fusion adaptive_gate \
    --fusion2 two_stage_learnable_gate \
    --adaptive_min_gate 0.0 \
    --adaptive_max_gate 0.85 \
    --psalor_mix 0.35 \
    --psalor_exact_weight 0.7 \
    --psalor_variant_weight 0.3 \
    --rsa_mode rsa \
    --pdb_dir DMS_ProteinGym_substitutions_pdbs \
    --dedup_identity_threshold 0.85 \
    --max_sequences_for_clustering 2000 \
    --layer2_weight 0.55 \
    --gate_temperature 1.4 \
    --alpha_family 0.85 \
    --hits_dir output/$protein_dir \
    --enable_gate_diagnostics
```

v1 baseline notes:
- `v1` in compare tables is a hard-locked baseline: `alpha=0.8` + `aa_seq_aln` only.
- Orbit hyperparameters do not change how `v1` is computed.
- `--alpha` controls Orbit / Orbit-cal fusion strength only.
- `--enhance_on_v1 --enhance_lambda <x>` enables v1-anchor enhancement:
  final logits are blended as `(1-x) * v1_logits(fixed baseline) + x * orbit_logits`.

Compare backbone/v1/orbit/orbit-cal summaries:
```shell
python tools/evaluate_orbit_proteingym.py \
    --prosst_summary result/proteingym_v1/summary_performance.csv \
    --venus_summary result/proteingym_v1_orbit_plus_v2_full/summary_performance.csv \
    --orbit_summary result/proteingym_v1_orbit/summary_performance.csv \
    --calibrated_summary result/proteingym_v1_orbit_plus_v2_calibrated/summary_performance.csv \
    --prosst_col ProSST-2048 \
    --venus_col VenusREM \
    --orbit_col VenusREM-Orbit \
    --calibrated_col VenusREM-Orbit-PlusV2-Calibrated \
    --out_file result/proteingym_v1_orbit/comparison.csv
```

Backbone mode options:
- `--backbone_mode prosst`: ProSST-style with structure tokens if available
- `--backbone_mode plain_mlm`: sequence-only backbone logits
- `--backbone_mode auto`: automatic mode selection (default)

## 🧬 VenusREM (v1) Usage

Recommended branch/tag:
- `v1.0.0` or `venusrem-v1-maintenance`

Model/parameter profile:
- Use `compute_fitness.py` without `--orbit_enable`
- Typical baseline: `--logit_mode aa_seq_aln --alpha 0.8`
- ProSST-style baseline: `--alpha 0 --model_out_name ProSST-2048`

ProteinGym baseline:
```shell
protein_dir=proteingym_v1
python compute_fitness.py \
    --base_dir data/$protein_dir \
    --out_scores_dir result/$protein_dir
```

Optional ProSST-style baseline:
```shell
protein_dir=proteingym_v1
python compute_fitness.py \
    --base_dir data/$protein_dir \
    --out_scores_dir result/${protein_dir}_prosst \
    --alpha 0 \
    --model_out_name ProSST-2048
```

## 🙌 Citation

Please cite our work if you have used our code or data.

### VenusREM-Orbit (mainline, in preparation)

```
```

### VenusREM-v1 (published)

```
@article{tan2025venusrem,
    author = {Tan, Yang and Wang, Ruilin and Wu, Banghao and Hong, Liang and Zhou, Bingxin},
    title = {From high-throughput evaluation to wet-lab studies: advancing mutation effect prediction with a retrieval-enhanced model},
    journal = {Bioinformatics},
    volume = {41},
    number = {Supplement_1},
    pages = {i401-i409},
    year = {2025},
    month = {07},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btaf189},
    url = {https://doi.org/10.1093/bioinformatics/btaf189},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/41/Supplement\_1/i401/63745466/btaf189.pdf},
}
```

