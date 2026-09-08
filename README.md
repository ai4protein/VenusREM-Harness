# VenusREM2

From Thinking Globally to Ranking Locally: An Adaptive and Model-Agnostic Readout Boosts Protein Mutation Prediction

Given a frozen PLM, **rem2** recalibrates substitution scores with homolog frequencies (MSA), amino-acid background bias (CCD), and structure-derived weights (RSA, pLDDT). No fine-tuning. Default mix is **dynamic α** (entropy-adaptive per protein, β = 1 − α); `--alpha 0.8` is the fixed-blend ablation. On ProteinGym substitutions (official 217-assay Average Spearman), rem2 raises ESM-2 650M from 0.418 to 0.468 and SaProt from 0.424 to 0.454; VenusREM2 from 0.524 to 0.556 (0.038 above the best public score).

| | |
|---|---|
| **rem2** | Calibration recipe. Applies to ESM-2, SaProt, ProSST, ProteinMPNN, … CLI: `rem2`. |
| **VenusREM2** | rem2 on the six official ProSST checkpoints (`--model venusrem2`). A single ESM-2 or ProSST-2048 run is rem2, not VenusREM2. |

Python import: `rem2`. Default backbone: ESM-2 650M.

```bash
rem2 doctor
rem2 demo
rem2 --model esm2 --fasta prot.fasta --mutants mutants.csv
rem2 --model venusrem2 --base_dir data/proteingym_v1
```

## News

- **2026.09** Package and CLI are `rem2`. Defaults: dynamic α, β = 1 − α, CCD on raw logits. Experimental PDBs skip pLDDT. `--model venusrem2` is the official ProSST ensemble.
- **2026.08** Weight cache standardized at `~/.cache/rem2/weights` (old `~/.cache/venusrem2` still read).
- **2026.07** Installable package. VenusREM frozen on `v1` / `v1.0.0`.
- **2025.07** VenusREM in [Bioinformatics](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i401/8199374).
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
| (core) | ESM-2 and other HF sequence LMs |
| `[recommended]` | biotite (RSA / PDB helpers) |
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
from rem2 import score

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
  struc_seq/           # ProSST structure tokens (VenusREM2 / --model prosst-2048)
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

1. **MSA fusion** — mix column frequencies into the logits. α is **dynamic** (entropy-adaptive per protein; `--alpha entropy`). No MSA → α = 0. Compute fail → α = 0.8, β = 0.2. `--alpha 0.8` is the fixed-blend ablation.
2. **CCD** — z-score AA background on raw logits, gated by `bg_scale = max(0, bg_consistency)`; β = 1 − α.
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

Keys below are the ProteinGym ablation set (`experiments/rem2_iclr_20260823`). `rem2 --list-models` prints the live table plus extras (`auto`, `protgpt2`, `s2f`). Override the default with `REM2_MODEL` (or `VENUSREM2_MODEL`). First ESM-2 650M download is ~2.5 GB (`~/.cache/rem2/weights`, or `$REM2_CACHE`).

`--scoring_strategy` is `wt` (default), `mask` (ESM-2 / SaProt / ProtSSN / …), or `tf` (ProteinMPNN teacher-force). ProSST / VenusREM2 are wt only. Experiment CSV `*_mask` / `*_wt` is this flag, not a second `--model`.

| `--model` | Backbone | Requirements |
|-----------|----------|--------------|
| `venusrem2` | official ProSST ensemble (K=20/128/512/1024/2048/4096) | `struc_seq/` + `[prosst]` |
| `prosst`, `prosst-20`, `prosst-128`, `prosst-512`, `prosst-1024`, `prosst-2048`, `prosst-4096` | single ProSST-K (aliases: `prosst_k4096`, …) | `struc_seq/` + `[prosst]` |
| `esm2` | ESM-2 650M (default). Also `esm2-8m`, `esm2-35m`, `esm2-150m`, `esm2-3b` | FASTA |
| `esm1b`, `esm1v` | ESM-1b; ESM-1v 5-seed | FASTA |
| `saprot`, `saprot-35m-af2`, `saprot-650m-pdb` | SaProt AF2 650M / 35M / PDB 650M | PDB (Foldseek on first use) |
| `carp` | CARP-640M | FASTA + `[carp]` |
| `esm3`, `esmc`, `esmc-600m` | ESM3 small / ESM-C 300M / 600M | FASTA + `[esm3]` |
| `protssn`, `protssn-ensemble` | ProtSSN 9-model ensemble (`--protssn_no_ensemble` for one) | PDB |
| `s3f` | S3F | PDB + `[s3f]` |
| `esm_if`, `esmif`, `mifst` | ESM-IF1; MIF-ST | PDB (`[carp]` for MIF-ST) |
| `protein_mpnn`, `proteinmpnn-020` | ProteinMPNN `v_48_020` (tf). Also `proteinmpnn-002` / `010` / `030` and `proteinmpnn-soluble-*` | PDB |
| `progen2`, `progen2-s`, `progen2-m`, `progen2-b`, `progen2-xl` | ProGen2 (default L) | FASTA |
| `progen3`, `progen3-112m`, `progen3-219m`, `progen3-339m`, `progen3-762m`, `progen3-3b` | ProGen3 (default 1B) | FASTA |
| `rita`, `rita-s`, `rita-m`, `rita-l` | RITA (default XL) | FASTA |
| `auto` | any Hugging Face masked LM | `--model_id` |

ProSST does not read a raw PDB; it needs precomputed structure tokens. Inverse-folding and causal LMs cannot use `--scoring_strategy masked-marginals`.

| Flag | Default |
|------|---------|
| `--model` | `esm2` |
| `--scoring_strategy` | `wt-marginals` (`masked-marginals` on MASK=yes models) |
| `--alpha` | `entropy` (dynamic α, per protein) |
| `--background_weight` | `one_minus_alpha` (β = 1 − α) |
| `--scoring_mode` | `calibrated_margin` |
| `--calibrate_on_raw` | on (CCD on raw logits; coherence gate on) |
| `--rsa_decay_mode` / `--plddt_decay_mode` | `above_mean` |
| `--task_type` | `default` |
| `--out_scores_dir` | `result` |
| `--no_auto_download` | off (download if missing) |

`--disable_adaptive_ccd` turns the coherence gate off (`bg_scale = 1`). `--no_rsa_decay` / `--no_plddt_decay` drop those terms. ProteinGym ablation stages:

```bash
# raw
rem2 --alpha 0 --scoring_mode log_odds --background_weight 0 --no_rsa_decay --no_plddt_decay
# + MSA (dynamic α)
rem2 --alpha entropy --scoring_mode log_odds --background_weight 0 --no_rsa_decay --no_plddt_decay
# + MSA + gated CCD
rem2 --no_rsa_decay --no_plddt_decay
# full rem2 (package default)
rem2
```

Offline: `--no_auto_download`. Full flag list: `rem2 --help`. `python compute_fitness.py` accepts the same arguments.

To register another PLM, implement `forward_log_probs` → `[L, V]` log-probs (`rem2.models`).

## ProteinGym (217 proteins)

Package default: **dynamic α**, β = 1 − α, `calibrated_margin`. `--alpha 0.8` is a fixed-blend ablation, not the main recipe.

### VenusREM2 (ProSST ensemble)

| Variant | Scoring | Alpha | RSA | pLDDT | Mean Spearman |
|---------|---------|-------|-----|-------|---------------|
| Raw backbone | log_odds | — | - | - | 0.524 |
| + MSA | log_odds | dynamic | - | - | 0.542 |
| + MSA + CCD | calibrated_margin | dynamic | - | - | 0.550 |
| + MSA + CCD + RSA | calibrated_margin | dynamic | above_mean | - | 0.554 |
| **Full rem2** | calibrated_margin | dynamic | above_mean | above_mean | **0.556** |

### rem2 on ESM-2 (650M, wt-marginals)

| Variant | Scoring | Alpha | RSA | pLDDT | Mean Spearman |
|---------|---------|-------|-----|-------|---------------|
| Raw backbone | log_odds | — | - | - | 0.418 |
| + MSA | log_odds | dynamic | - | - | 0.429 |
| + MSA + CCD | calibrated_margin | dynamic | - | - | 0.440 |
| + MSA + CCD + RSA | calibrated_margin | dynamic | above_mean | - | 0.465 |
| **Full rem2** | calibrated_margin | dynamic | above_mean | above_mean | **0.468** |

### rem2 on SaProt (AF-650M, wt-marginals)

| Variant | Scoring | Alpha | RSA | pLDDT | Mean Spearman |
|---------|---------|-------|-----|-------|---------------|
| Raw backbone | log_odds | — | - | - | 0.424 |
| + MSA | log_odds | dynamic | - | - | 0.427 |
| + MSA + CCD | calibrated_margin | dynamic | - | - | 0.433 |
| + MSA + CCD + RSA | calibrated_margin | dynamic | above_mean | - | 0.452 |
| **Full rem2** | calibrated_margin | dynamic | above_mean | above_mean | **0.454** |

### rem2 on ESM-1v (5-seed, wt-marginals)

| Variant | Scoring | Alpha | RSA | pLDDT | Mean Spearman |
|---------|---------|-------|-----|-------|---------------|
| Raw backbone | log_odds | — | - | - | 0.410 |
| + MSA | log_odds | dynamic | - | - | 0.419 |
| + MSA + CCD | calibrated_margin | dynamic | - | - | 0.433 |
| + MSA + CCD + RSA | calibrated_margin | dynamic | above_mean | above_mean | **0.455 |
| **Full rem2** | calibrated_margin | dynamic | above_mean | above_mean | **0.457** |

## VenusREM

Published VenusREM is frozen on **`v1.0.0`**. Closest v1-style command on this tree:

```bash
rem2 --model prosst-2048 --base_dir data/proteingym_v1 \
    --alpha 0.8 --scoring_mode log_odds
```

## Development

```bash
pip install -e ".[prosst,dev]"
pytest test/ -v
```

## Citation

VenusREM2:

```bibtex
@article{tan2026venusrem2,
    title={From Thinking Globally to Ranking Locally: An Adaptive and Model-Agnostic Readout Boosts Protein Mutation Prediction},
}
```

VenusREM:

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

Related: [VenusFactory2](https://github.com/ai4protein/VenusFactory2), [web server](https://venusfactory.bio/), [technical report](https://arxiv.org/abs/2603.27303).

## License

[CC-BY-NC-ND 4.0](LICENSE).
