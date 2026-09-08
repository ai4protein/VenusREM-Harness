# VenusREM2

Training-free calibration for protein language models on variant-effect prediction.

Given a frozen PLM, **rem2** recalibrates substitution scores with homolog frequencies (MSA), amino-acid background bias (CCD), and structure-derived weights (RSA, pLDDT). No fine-tuning. On ProteinGym substitutions, the full recipe raises ESM-2 mean Spearman from 0.444 to 0.499 and SaProt from 0.473 to 0.511.

| | |
|---|---|
| **rem2** | Calibration recipe. Applies to ESM-2, SaProt, ProSST, ProteinMPNN, … CLI: `rem2`. |
| **VenusREM2** | rem2 on the official ProSST ensemble only (`--model venusrem2`). A single ESM-2 or ProSST-2048 run is rem2, not VenusREM2. |

Python import: `venusrem2`. Default backbone: ESM-2 650M.

```bash
rem2 doctor
rem2 demo
rem2 --model esm2 --fasta prot.fasta --mutants mutants.csv
rem2 --model venusrem2 --base_dir data/proteingym_v1
```

## News

- **2026.09** CLI is `rem2`. Defaults: entropy-α, β = 1 − α, CCD on raw logits. Experimental PDBs skip pLDDT. `--model venusrem2` is the official ProSST ensemble.
- **2026.08** Weight cache standardized at `~/.cache/venusrem2/weights`.
- **2026.07** Installable `venusrem2` package. VenusREM-v1 frozen on `v1` / `v1.0.0`.
- **2025.07** VenusREM-v1 in [Bioinformatics](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i401/8199374).
- **2025.04** Ranked 1st on the [ProteinGym](https://proteingym.org/benchmarks) substitution leaderboard.

## Installation

Install a CUDA [PyTorch](https://pytorch.org/get-started/locally/) wheel first so pip does not pull a CPU build. ESM-2 650M needs roughly ≥10 GB VRAM; `rem2 demo` uses ESM-2 8M and can run on CPU.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[recommended]"
rem2 doctor
rem2 demo
```

| Extra | Use |
|-------|-----|
| `[recommended]` | ESM-2 / FASTA / PDB (biotite) |
| `[prosst]` | Official VenusREM2 (ProSST ensemble); needs `struc_seq/` |
| `[carp]`, `[esm3]`, `[s3f]` | Other backbones (`[s3f]` requires Python &lt; 3.11) |
| `[dev]` | pytest / ruff |

Portable conda stub: `environment-minimal.yml`. `environment.yml` is a pinned lab snapshot, not a user recipe.

`rem2 doctor --strict` exits 1 if `torch` or `biopython` is missing.

## Quick start

```bash
# single protein
rem2 --fasta prot.fasta --mutants mutants.csv
rem2 --fasta prot.fasta --mutants mutants.csv --pdb prot.pdb

# single-site saturation (omit --mutants)
rem2 --fasta prot.fasta

# dataset
rem2 --base_dir data/my_assay
rem2 --model venusrem2 --base_dir data/proteingym_v1
```

```python
from venusrem2 import score

df = score("prot.fasta", mutants="mutants.csv", pdb="prot.pdb")
summary = score(base_dir="data/my_assay")
```

Outputs (default `result/`):

```
result/scores/<protein>.csv    # mutants + rem2 column
result/summary_performance.csv # Spearman vs DMS_score, if present
result/run_meta.json
```

Score column: `{backbone}__rem2` (e.g. `esm2_t33_650M_UR50D__rem2`). VenusREM2 writes per-K columns plus a z-mean `VenusREM2`. Higher = more preferred by the calibrated model. Use for ranking; this is not a ΔΔG.

## Data

**Single protein.** Mutants CSV needs a `mutant` column (`A42G`; multi-site `A42G:L10M`). Optional `DMS_score` is used only for Spearman.

**Dataset** (`--base_dir`):

```
data/my_assay/
  aa_seq/              # wild-type FASTA, one file per protein
  substitutions/       # mutant CSV, same stem as the FASTA
  aa_seq_aln_a2m/      # optional MSA (a2m / a3m)
  pdbs/                # optional structures
  struc_seq/           # ProSST structure tokens (VenusREM2 / --model prosst)
```

`--base_dir` also accepts ProteinGym layout names (`aa_seq_aln_a2m_af2cf/`, `pdbs_af2_assay_resolved_full/`, …) and uses the first match.

MSA / PDB / `struc_seq` are optional. Missing inputs drop the corresponding terms and emit a warning; rem2 does not silently claim the full recipe.

ProteinGym alignments: [a2m](https://huggingface.co/datasets/AI4Protein/VenusREM/resolve/main/aa_seq_aln_a2m.tar.gz), [a3m](https://huggingface.co/datasets/AI4Protein/VenusREM/resolve/main/aa_seq_aln_a3m.tar.gz).

**Structures.** RSA is computed for any PDB. pLDDT decay uses the B-factor column and is applied only to predicted models (AlphaFold / ColabFold / ESMFold). Crystal, NMR, and cryo-EM structures skip pLDDT (B-factor is a temperature factor) and print a warning.

**Combinatorial libraries.** Double/triple mutants require an explicit site list:

```bash
rem2 --fasta prot.fasta --pdb prot.pdb \
    --mutant_sites 1,2,3 --positions 10,11,12,13,14
```

Libraries larger than `--max_mutants` (default 1e6) are refused.

## Method

Default recipe (all terms on when the files exist):

1. **MSA fusion** — mix column frequencies into the logits. α is entropy-adaptive (`--alpha entropy`). No MSA → α = 0. `--alpha 0.8` is the fixed-blend ablation.
2. **CCD** — subtract amino-acid background bias computed on raw logits; β = 1 − α.
3. **RSA** — down-weight solvent-exposed positions (stability-like assays). `--task_type surface` reverses the sign (binding / surface phenotypes).
4. **pLDDT** — down-weight low-confidence predicted structure. Skipped on experimental PDBs.

Scoring mode: `calibrated_margin`. Formula: [`docs/scoring_formula.md`](docs/scoring_formula.md).

Raw PLM baseline (no rem2 extras):

```bash
rem2 --fasta prot.fasta --mutants m.csv --alpha 0 --scoring_mode log_odds \
    --no_rsa_decay --no_plddt_decay --background_weight 0
```

Reuse one forward pass across heads:

```bash
CACHE="--logits_cache_dir cache/esm2 --reuse_logits_cache --write_logits_cache --logits_cache_tag esm2_v1"
rem2 --base_dir data/my_assay $CACHE --out_scores_dir result/raw --alpha 0
rem2 --base_dir data/my_assay $CACHE --out_scores_dir result/rem2
```

## Models

`rem2 --list-models`. Override the default with `VENUSREM2_MODEL` / `REM2_MODEL`. First ESM-2 650M download is ~2.5 GB (`~/.cache/venusrem2/weights`, or `$VENUSREM2_CACHE`).

| `--model` | Backbone | Requirements |
|-----------|----------|--------------|
| `esm2` | ESM-2 650M (default) | FASTA |
| `esm2-8m` | ESM-2 8M (`rem2 demo`) | FASTA |
| `esm1v` | ESM-1v 5-seed ensemble | FASTA |
| `venusrem2` | official ProSST ensemble + rem2 | `struc_seq/` + `[prosst]` |
| `prosst` | single ProSST + rem2 | `struc_seq/` |
| `saprot` | SaProt | PDB (Foldseek on first use) |
| `protein_mpnn` / `esm_if` | inverse folding | PDB |
| `auto` | any Hugging Face masked LM | `--model_id` |

ProSST does not read a raw PDB; it needs precomputed structure tokens. Inverse-folding and causal LMs cannot use `--scoring_strategy masked-marginals`.

| Flag | Default |
|------|---------|
| `--model` | `esm2` |
| `--alpha` | `entropy` |
| `--task_type` | `default` |
| `--out_scores_dir` | `result` |
| `--no_auto_download` | off (download if missing) |

Offline: `--no_auto_download`. Full flag list: `rem2 --help`. `python compute_fitness.py` accepts the same arguments.

To register another PLM, implement `forward_log_probs` → `[L, V]` log-probs (`venusrem2.models`).

## ProteinGym (217 proteins)

Tables use fixed α = 0.8. Package default is entropy-α.

### ESM-2 (650M)

| Variant | Scoring | Alpha | RSA | pLDDT | Mean Spearman |
|---------|---------|-------|-----|-------|---------------|
| Raw backbone | log_odds | 0.0 | - | - | 0.4437 |
| + MSA | log_odds | 0.8 | - | - | 0.4643 |
| + MSA + CCD | calibrated_margin | 0.8 | - | - | 0.4791 |
| + MSA + CCD + RSA | calibrated_margin | 0.8 | above_mean | - | 0.4979 |
| + MSA + CCD + pLDDT | calibrated_margin | 0.8 | - | above_mean | 0.4846 |
| **Full recipe** | calibrated_margin | 0.8 | above_mean | above_mean | **0.4991** |

### SaProt (AF-650M)

| Variant | Scoring | Alpha | RSA | pLDDT | Mean Spearman |
|---------|---------|-------|-----|-------|---------------|
| Raw backbone | log_odds | 0.0 | - | - | 0.4734 |
| + MSA | log_odds | 0.8 | - | - | 0.4919 |
| + MSA + CCD | calibrated_margin | 0.8 | - | - | 0.5017 |
| + MSA + CCD + RSA | calibrated_margin | 0.8 | above_mean | - | 0.5108 |
| + MSA + CCD + pLDDT | calibrated_margin | 0.8 | - | above_mean | 0.5030 |
| **Full recipe** | calibrated_margin | 0.8 | above_mean | above_mean | **0.5109** |

### ESM-1v (5-seed ensemble)

| Variant | Scoring | Alpha | RSA | pLDDT | Mean Spearman |
|---------|---------|-------|-----|-------|---------------|
| Raw backbone | log_odds | 0.0 | - | - | 0.4362 |
| + MSA | log_odds | 0.8 | - | - | 0.4561 |
| + MSA + CCD | calibrated_margin | 0.8 | - | - | 0.4716 |
| + MSA + CCD + RSA | calibrated_margin | 0.8 | above_mean | - | 0.4898 |
| Full recipe | calibrated_margin | 0.8 | above_mean | above_mean | *in progress* |

## VenusREM v1

Published VenusREM is frozen on **`v1.0.0`**. Closest v1-style command on this tree:

```bash
rem2 --model prosst --base_dir data/proteingym_v1 \
    --alpha 0.8 --scoring_mode log_odds
```

## Development

```bash
pip install -e ".[prosst,dev]"
pytest test/ -v
```

## Citation

VenusREM2 (in preparation):

```bibtex
@article{tan2026venusrem2,
    title={Beyond Fine-Tuning: Calibrate Any Protein Language Model for Variant Effect Prediction},
    author={Tan, Yang and others},
    year={2026},
}
```

VenusREM-v1:

```bibtex
@article{tan2025venusrem,
    author = {Tan, Yang and Wang, Ruilin and Wu, Banghao and Hong, Liang and Zhou, Bingxin},
    title = {From high-throughput evaluation to wet-lab studies: advancing mutation effect prediction with a retrieval-enhanced model},
    journal = {Bioinformatics},
    volume = {41},
    number = {Supplement_1},
    pages = {i401-i409},
    year = {2025},
    month = {07},
    doi = {10.1093/bioinformatics/btaf189},
    url = {https://doi.org/10.1093/bioinformatics/btaf189},
}
```

Related: [VenusFactory2](https://github.com/ai4protein/VenusFactory2), [web server](https://venusfactory.cn/playground/), [technical report](https://arxiv.org/abs/2603.27303).

## License

[CC-BY-NC-ND 4.0](LICENSE).
