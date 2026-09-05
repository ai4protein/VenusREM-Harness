# Beyond Fine-Tuning: Calibrate Any Protein Language Model for Variant Effect Prediction

## Introduction

**Orbit** is a training-free calibration recipe that improves any protein language model (PLM) for variant effect prediction. It works as a post-hoc scoring pipeline on top of frozen PLM logits, requiring no gradient updates or fine-tuning.

The Orbit recipe consists of four additive components:

| Component | Flag | What it does |
|-----------|------|-------------|
| **MSA fusion** | `--alpha 0.8` | Blends PLM log-odds with evolutionary counts from MSA alignments |
| **CCD** | `--scoring_mode calibrated_margin` | Calibrated margin scoring with background correction and wild-type confidence bonus |
| **RSA decay** | `--use_rsa_decay --rsa_decay_mode above_mean` | Suppresses surface-exposed positions where PLMs over-predict effects |
| **pLDDT decay** | `--use_plddt_decay --plddt_decay_mode above_mean` | Suppresses disordered regions where structure predictions are unreliable |

Each component is independently toggleable. The full recipe stacks all four.

### Related work

- **VenusREM (v1)** — *From high-throughput evaluation to wet-lab studies: advancing mutation effect prediction with a retrieval-enhanced model* (ISMB/ECCB 2025). Frozen on branch / tag below.

## Repository layout (v1 archive → Orbit)

This codebase is developed **on top of [ai4protein/VenusREM](https://github.com/ai4protein/VenusREM)**. Published VenusREM-v1 is **frozen**; active development is the Orbit package on `main`.

| Ref | Purpose | Layout |
|-----|---------|--------|
| **`v1` / `venusrem-v1-maintenance`** (and tag **`v1.0.0`**) | **Archive** of published VenusREM-v1 — do not rewrite history | Monolithic `src/` + `compute_fitness.py` |
| **`main`** (this tree) | Orbit: pluggable PLMs, pip package, tiered deps | Package `venus_orbit/` + thin `compute_fitness.py` shim |

```bash
# Published VenusREM-v1 (paper reproduction)
git checkout v1.0.0
# or: git checkout venusrem-v1-maintenance

# Current Orbit development (default)
git checkout main
pip install -e ".[prosst,structure]"
```

### What changed from v1 → Orbit (`main`)

| Area | VenusREM-v1 (archived) | Orbit (`main`) |
|------|------------------------|----------------|
| Package | Ad-hoc `src/` imports | Installable **`venus-orbit`** (`import venus_orbit`) |
| CLI | `python compute_fitness.py …` | `venus-orbit …` (same entry still works) |
| Backbones | ProSST (+ a few scripts) | **Registry of 17+ models** (`--model esm2|prosst|saprot|…`) |
| Weights | Manual download / fixed paths | Auto-download into `~/.cache/venus_orbit/weights` |
| Dependencies | One fat environment | **Tiered extras** (`.[prosst]`, `.[s3f]`, …) — core install stays small |
| Scoring | MSA α fusion (VenusREM) | Orbit modules: MSA + CCD + RSA + pLDDT (all optional flags) |
| Compatibility | — | `compute_fitness.py` remains a **6-line shim** → `venusrem_orbit.cli` / `venus_orbit.cli` |

### Tracked vs local-only (this checkout)

| In git | Local only (gitignored) |
|--------|-------------------------|
| `venus_orbit/`, `venusrem_orbit/`, `test/`, `pyproject.toml` | `data/*` datasets & weight dumps (except two ProteinGym CSVs) |
| `script/` (baseline / smoke / msa / data_prep / structure) | `result/`, `experiments/`, `log/`, `tools/` |
| `README.md`, `docs/scoring_formula.md` | other `docs/`, `script/iclr/` |
| ProSST `venus_orbit/.../static/*` (~165MB, needed for ProSST) | HF / cache downloads under `~/.cache/venus_orbit/` |

### Migration cheat-sheet

```bash
# v1-style ProSST + MSA α=0.8 (closest to published VenusREM scoring)
venus-orbit --model prosst \
    --base_dir data/proteingym_v1 \
    --struc_seq_dir struc_seq_af2_assay_resolved_full \
    --structure_vocab_subdir 2048 \
    --aa_seq_aln_dir aa_seq_aln_a2m \
    --alpha 0.8 --scoring_mode log_odds \
    --out_scores_dir result/venusrem_like

# Full Orbit recipe on any backbone
venus-orbit --model esm2 \
    --base_dir data/proteingym_v1 \
    --pdb_dir pdbs_af2_assay_resolved_full \
    --aa_seq_aln_dir aa_seq_aln_a2m \
    --alpha 0.8 --scoring_mode calibrated_margin \
    --use_rsa_decay --rsa_decay_mode above_mean \
    --use_plddt_decay --plddt_decay_mode above_mean \
    --out_scores_dir result/orbit_full
```

Old scripts that call `python compute_fitness.py …` **keep working** on `main` (shim unchanged). Prefer `venus-orbit --model …` for new work.

If you only need the paper pipeline and not Orbit, stay on **`v1.0.0` / `venusrem-v1-maintenance`** and use the archived README there.

## News

- [2026.07] Orbit packaging: `venus_orbit` installable package, model registry, tiered deps; VenusREM-v1 frozen on `v1` / `v1.0.0`.
- [2026.05] Orbit recipe generalization: verified on ESM-2, ESM-1v, and SaProt backbones.
- [2026.04] VenusREM has been integrated into [VenusFactory2](https://github.com/ai4protein/VenusFactory2). [Web server](https://venusfactory.cn/playground/) | [Technical report](https://arxiv.org/abs/2603.27303).
- [2025.07] VenusREM-v1 published at [Bioinformatics](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i401/8199374).
- [2025.04] Ranked 1st on the [ProteinGym](https://proteingym.org/benchmarks) substitution leaderboard.

## Install

Install is **tiered**. Core is enough for Orbit on HF sequence models (ESM-2 / ESM-1b / …).
Heavy stacks (ProSST graph bits, S3F/TorchDrug, CARP, ESM3) are **optional extras**.

```bash
# 1) Install a CUDA-matching PyTorch yourself (do this first)
#    https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2) Core package (~ torch + transformers + scoring code)
pip install venus-orbit
# or from this repo:
pip install -e .

# 3) Optional extras (only what you need)
pip install -e ".[prosst]"      # ProSST structure tokens (torch-geometric, biotite)
pip install -e ".[structure]"   # PDB helpers for ESM-IF / RSA / ProteinMPNN I/O
pip install -e ".[msa]"         # BioPython MSA utilities
pip install -e ".[carp]"        # CARP (sequence_models)
pip install -e ".[esm3]"        # EvolutionaryScale ESM-3 / ESM-C
pip install -e ".[s3f]"         # full S3F / S2F (TorchDrug + pykeops; large; see note)
pip install -e ".[baselines]"   # prosst+structure+carp+esm3+msa (no S3F)
pip install -e ".[all]"         # everything including S3F (Python <3.11 recommended)
```

**S3F note:** TorchDrug only publishes wheels for **Python &lt; 3.11**. On 3.11/3.12:

```bash
bash script/install_s3f_deps.sh
# or:
pip install --ignore-requires-python --no-deps \
  "git+https://github.com/DeepGraphLearning/torchdrug.git"
pip install 'rdkit==2023.9.6' 'numpy<2' lmdb robust-laplacian biopython
```

Inference with precomputed surface `.pkl` (**does not need PyKeOps**). Provide surfaces via `--s3f_surface_dir` (default `<base_dir>/s3f_surfaces_af2_assay_resolved_full`). Checkpoint: `data/s3f_weights/s3f.pth` or `S3F_CHECKPOINT`.

On-the-fly surface generation from PDB still needs `pip install -e ".[s3f-surface-gen]"` (pykeops + CUDA toolkit headers).

| Install | Pulls | Typical models |
|---------|-------|----------------|
| `venus-orbit` (core) | torch, transformers, numpy, pandas, scipy, huggingface-hub | `esm2`, `esm1b`, `esm1v`, `progen2`, `rita`, `protgpt2`, `tranception`, `protein_mpnn`, `protssn`, `saprot`\* |
| `.[prosst]` | + torch-geometric, biotite | `prosst` / VenusREM backbone |
| `.[structure]` | + biotite, biopython | `esm_if` coordinate I/O |
| `.[s3f]` | + torchdrug, pykeops, … | full `s3f` / `s2f` (not lightweight ESM2 fallback) |
| `.[carp]` / `.[esm3]` | sequence_models / `esm` | `carp`, `esm3` |

\* Foldseek binary is **auto-downloaded** for SaProt (not a pip dependency). Checkpoints for ProtSSN / ProteinMPNN / CARP are **auto-downloaded** into `~/.cache/venus_orbit/weights` on first use.

**You do not need TorchDrug / pykeops / sequence_models for the default Orbit + ESM-2 / ProSST path.** Those only appear when you opt into the corresponding extras.

CLI:

```bash
venus-orbit --help
venus-orbit --list-models

# Legacy script entry (unchanged)
python compute_fitness.py --help
```

Python:

```python
import venus_orbit
from venus_orbit.models import get_model, list_models
from venus_orbit.scoring import score_protein
```

### Environment (optional conda)

```bash
conda env create -f environment.yml
conda activate venusrem
pip install -e ".[prosst,structure]"
# optional: pip install -e ".[s3f]"
```

### External tools (not pip-installed)

| Tool | Used for | Notes |
|------|----------|--------|
| HMMER + EVcouplings | MSA generation | `pip install hmmer` + EVcouplings from GitHub |
| plmc | Evolutionary couplings | Build from source; set path in `venus_orbit/single_config_monomer.txt` |
| Foldseek | SaProt structure tokens | Auto-downloaded from HF (`tyang816/Foldseek_bin`); override with `--foldseek_bin` or `PATH` |

### Hardware

- GPU: >= 10 GB VRAM (e.g., RTX 3080)
- CPU: >= 8 cores (for MSA search)

## Quick start

Data layout:

```
data/<dataset_name>/
  aa_seq/              # wild-type FASTA (one per protein)
  substitutions/       # mutation CSV (columns: mutant, DMS_score)
  aa_seq_aln_a2m/      # optional MSA (a2m/a3m)
  pdbs/                # optional PDB (RSA / pLDDT / structure models)
  struc_seq/           # optional structure tokens (ProSST)
```

```bash
# Default backbone: ProSST (weights download from Hugging Face on first run)
venus-orbit \
    --model prosst \
    --base_dir data/proteingym_v1 \
    --out_scores_dir result/prosst_full \
    --alpha 0.8 \
    --scoring_mode calibrated_margin \
    --use_rsa_decay --rsa_decay_mode above_mean \
    --use_plddt_decay --plddt_decay_mode above_mean

# ESM-2
venus-orbit \
    --model esm2 \
    --base_dir data/proteingym_v1 \
    --out_scores_dir result/esm2_full \
    --alpha 0.8 \
    --scoring_mode calibrated_margin \
    --use_rsa_decay --rsa_decay_mode above_mean \
    --use_plddt_decay --plddt_decay_mode above_mean
```

`python compute_fitness.py` accepts the same flags.

### Single protein (no `base_dir`)

Score one FASTA; auto-generate an n-point saturation library with `--mutant_sites`
(`1`=single, `2`=double, `3`=triple). Orders ≥2 require `--positions` or `--residue_range`.

```bash
# Full-length single-site saturation
venus-orbit --model esm2 --fasta prot.fasta \
    --mutant_sites 1 \
    --alpha 0 \
    --out_scores_dir result/single_scan

# Single + double + triple on selected sites
venus-orbit --model esm2 --fasta prot.fasta --pdb prot.pdb \
    --mutant_sites 1,2,3 \
    --positions 10,11,12,13,14 \
    --alpha 0 \
    --out_scores_dir result/npoint_scan

# Use an existing mutants CSV instead of generating
venus-orbit --model esm2 --fasta prot.fasta --mutants mutants.csv \
    --out_scores_dir result/custom_mutants
```

Generated CSVs are written to `<out_scores_dir>/generated_mutants/` and
`<out_scores_dir>/_inputs/substitutions/`. Libraries larger than `--max_mutants`
(default 1,000,000) are rejected.

## Supported models

List at runtime: `venus-orbit --list-models`

| `--model` | Default weights | Auto-download | Needs PDB | Notes |
|-----------|-----------------|---------------|-----------|--------|
| `prosst` | `AI4Protein/ProSST-2048` | yes (HF) | no* | *uses `struc_seq/`; extras: `prosst` |
| `auto` | pass `--model_id` | yes (HF) | no | Any `AutoModelForMaskedLM` |
| `esm2` | `facebook/esm2_t33_650M_UR50D` | yes (HF) | no | |
| `esm1b` | `facebook/esm1b_t33_650M_UR50S` | yes (HF) | no | |
| `esm1v` | 5-seed ensemble | yes (HF) | no | `--esm1v_seeds` |
| `saprot` | `westlake-repl/SaProt_650M_AF2` | yes (HF) | yes | Foldseek auto-downloaded |
| `protssn` | HF `tyang816/ProtSSN` | yes | yes | cache `protssn/` |
| `esm_if` | fair-esm ESM-IF1 | yes | yes | isolated from ESM3 package |
| `protein_mpnn` | GitHub `v_48_020.pt` | yes | yes | cache `protein_mpnn/` |
| `progen2` | `hugohrban/progen2-large` | yes (HF) | no | |
| `progen3` | `Profluent-Bio/progen3-1b` | yes (HF) | no | heavy optional deps |
| `protgpt2` | `nferruz/ProtGPT2` | yes (HF) | no | |
| `rita` | `lightonai/RITA_xl` | yes (HF) | no | |
| `esm3` | `esmc_300m` | yes | no | `pip install esm` |
| `tranception` | `OATML-Markslab/Tranception_Large` | yes (HF) | no | |
| `carp` | Zenodo CARP | yes | no | needs `sequence_models` |
| `s2f` | lightweight ESM2 if no ckpt | partial | no | full mode: `--s2f_checkpoint` |
| `s3f` | `data/s3f_weights/s3f.pth` | yes* | yes | *local or `S3F_CHECKPOINT` / HF `tyang816/S3F_weights` |

ProteinGym reference (raw → full Orbit): ESM-2 0.444 → 0.499; SaProt 0.473 → 0.511.

## CLI reference

Preferred flags:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `prosst` (inferred) | Backbone key from the table above |
| `--model_id` | model default | HF id / local path override |
| `--cache_dir` | `~/.cache/venus_orbit/weights` | Weight cache (`$VENUS_ORBIT_CACHE`) |
| `--base_dir` | — | Dataset root (`aa_seq/`, `substitutions/`, …) |
| `--fasta` / `--pdb` | — | Single-protein mode (no `base_dir`) |
| `--mutant_sites` | — | Auto-generate n-point library: `1` / `1,2` / `1,2,3` |
| `--positions` / `--residue_range` | — | Residue set for mutagenesis (required if orders ≥2) |
| `--mutants` | — | Existing mutants CSV (skips generation) |
| `--max_mutants` | `1000000` | Cap on auto-generated library size |
| `--out_scores_dir` | — | Output directory |
| `--alpha` | `0.8` | MSA fusion weight (`0` disables) |
| `--scoring_mode` | `log_odds` | `log_odds` / `calibrated_margin` / … |
| `--use_rsa_decay` | off | RSA position decay |
| `--use_plddt_decay` | off | pLDDT disorder decay |

Legacy flags (`--baseline_type`, `--model_name`, `--progen2_model_name_or_path`, …) still work.

### More examples

```bash
# ESM-1v ensemble
venus-orbit --model esm1v --base_dir data/proteingym_v1 --out_scores_dir result/esm1v_full \
    --alpha 0.8 --scoring_mode calibrated_margin \
    --use_rsa_decay --rsa_decay_mode above_mean \
    --use_plddt_decay --plddt_decay_mode above_mean

# SaProt (Foldseek auto-downloaded if missing)
venus-orbit --model saprot --base_dir data/proteingym_v1 --out_scores_dir result/saprot_full \
    --scoring_strategy masked-marginals \
    --alpha 0.8 --scoring_mode calibrated_margin \
    --use_rsa_decay --rsa_decay_mode above_mean \
    --use_plddt_decay --plddt_decay_mode above_mean

# ProteinMPNN (checkpoint auto-downloaded)
venus-orbit --model protein_mpnn --base_dir data/proteingym_v1 --out_scores_dir result/mpnn \
    --alpha 0

# Arbitrary HF MLM
venus-orbit --model auto --model_id facebook/esm2_t30_150M_UR50D \
    --base_dir data/proteingym_v1 --out_scores_dir result/esm2_150m --alpha 0
```

### Orbit ablations (ESM-2)

```bash
BASE="--model esm2 --base_dir data/proteingym_v1 --pdb_dir pdbs"

venus-orbit $BASE --out_scores_dir result/esm2_raw --alpha 0
venus-orbit $BASE --out_scores_dir result/esm2_msa --alpha 0.8
venus-orbit $BASE --out_scores_dir result/esm2_msa_ccd --alpha 0.8 --scoring_mode calibrated_margin
venus-orbit $BASE --out_scores_dir result/esm2_full --alpha 0.8 --scoring_mode calibrated_margin \
    --use_rsa_decay --rsa_decay_mode above_mean \
    --use_plddt_decay --plddt_decay_mode above_mean
```

### Logits caching

```bash
BASE="--model esm2 --base_dir data/proteingym_v1"
CACHE="--logits_cache_dir cache/esm2 --reuse_logits_cache --write_logits_cache --logits_cache_stage raw --logits_cache_tag esm2_v1"

venus-orbit $BASE $CACHE --out_scores_dir result/esm2_raw --alpha 0
venus-orbit $BASE $CACHE --out_scores_dir result/esm2_msa --alpha 0.8
```

## Weights & auto-download

Missing weights are downloaded into:

```
~/.cache/venus_orbit/weights/
```

Override with `--cache_dir` or `VENUS_ORBIT_CACHE`. HF models use the normal Hugging Face cache as well.

## Pluggable custom models

Any backbone that can produce **`[L, V]` log-probs** can plug into the same Orbit pipeline.

```python
from venus_orbit.models import ModelAdapter, ModelSpec, register_model

@register_model
class MyAdapter(ModelAdapter):
    spec = ModelSpec(
        name="my_plm",
        description="My custom PLM",
        default_model_id="org/my-model",
        needs_pdb=False,
        auto_download=True,
        baseline_type="auto",  # reuse HF MLM dispatch, or implement load fully
    )

    @classmethod
    def load(cls, model_id, device, cache_dir, args, logger):
        # Load weights (auto-download if needed), return cls(state, args, device)
        ...

    def forward_log_probs(self, sequence, *, pdb_path=None, structure_fasta=None, **kw):
        # Return Tensor[L, V] log-probs (project to ESM vocab if needed)
        ...
```

Third-party packages can register via entry points:

```toml
[project.entry-points."venus_orbit.models"]
my_plm = "mypkg.adapters:MyAdapter"
```

Then:

```bash
venus-orbit --model my_plm --base_dir data/foo --out_scores_dir out/
```

## Tests

Integration tests use a real mini-protein fixture (trp-cage sequence + PDB):

```bash
pip install -e ".[prosst,dev]"
pytest test/ -v
```

- `test/test_registry.py` — model registry / CLI flags / fixtures
- `test/test_cli.py` — `venus-orbit --list-models`
- `test/test_models_forward.py` — every registered `--model` loads and returns `[L, V]` log-probs on the fixture FASTA/PDB; a subset also runs `score_protein` end-to-end

Models missing optional deps (e.g. `torchdrug` for full S3F) are reported as **skipped**, not silent passes.

## Data preparation

### Downloads (ProteinGym benchmark)

- [EVCouplings a2m alignments](https://huggingface.co/datasets/AI4Protein/VenusREM/resolve/main/aa_seq_aln_a2m.tar.gz)
- [ColabFold a3m alignments](https://huggingface.co/datasets/AI4Protein/VenusREM/resolve/main/aa_seq_aln_a3m.tar.gz)

## Results on ProteinGym (217 proteins)

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

## Citation

### VenusREM-Orbit (in preparation)

```
@article{tan2026orbit,
    title={Beyond Fine-Tuning: Calibrate Any Protein Language Model for Variant Effect Prediction},
    author={Tan, Yang and others},
    year={2026},
}
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
}
```
