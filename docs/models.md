# vrh models and forwards

`--scoring_strategy` chooses how the backbone produces \(\ell_{\mathrm{raw}}\).
`--scoring_mode` is the vrh head on top of those logits. They are not the same flag.

`vrh --list-models` prints each backbone and its allowed forwards (`wt` / `mask` / `tf`).
An unsupported strategy is refused; vrh does not silently fall back.

## vrh vs VenusREM2

| Name | Meaning |
|------|---------|
| **vrh** | Calibration recipe on any backbone. CLI: `vrh` / `remharness`. |
| **VenusREM2** | vrh on the six official ProSST checkpoints (`--model venusrem2`). |

A single ESM-2 or ProSST-2048 run is vrh, not VenusREM2.

## Score columns

Matches `vrh/naming.py`. `--scoring_strategy` is not part of the column name.

| Run | Column |
|-----|--------|
| One backbone | `{HF-basename}__vrh` (also `{HF-basename}__raw_backbone`) |
| Official ensemble, per K | `VenusREM2__ProSST-{K}` |
| Official ensemble, combined | `VenusREM2` (z-mean of the six members) |

Experiment CSVs that end in `_wt` / `_mask` are this flag, not a second `--model`.

## `--scoring_strategy`

Default: `wt` (`wt-marginals`). Short names and aliases:

| Short | Canonical | Also accepted |
|-------|-----------|----------------|
| `wt` | `wt-marginals` | `wild-type` |
| `mask` | `masked-marginals` | `masked` |
| `tf` | `teacher-force` | `teacher_force` |

### `wt` — one unmasked forward

One pass on the wild-type sequence. Site \(i\) uses the logits at position \(i\).
This is the default for every backbone.

On causal LMs (ProGen2 / ProGen3 / ProtGPT2 / RITA) and inverse-folding models
(ESM-IF, MIF-ST) this *is* the teacher-forced / unmasked likelihood. vrh still
calls that `wt`, not `tf`. `mask` is refused.

MIF-ST: ProteinGym’s name looks like a masked model; vrh scores it unmasked
and refuses `--scoring_strategy mask`.

### `mask` — per-site mask (Meier / ESM-1v)

\(L\) forwards: mask site \(i\), read the distribution at \(i\). Needs a real
mask token. Allowed on: ESM-2 / ESM-1b / ESM-1v, SaProt, ProtSSN, CARP, ESM3 /
ESM-C, S3F, and `--model auto` when the tokenizer has `mask_token_id`.

Refused on: ProSST / VenusREM2, ProteinMPNN, ESM-IF, MIF-ST, S2F, and all
causal LMs.

### `tf` — ProteinMPNN teacher-force

ProteinMPNN only. The decoder sees the full WT sequence; the decode order is
the historical deterministic zeros path in this repo.

`--scoring_strategy wt` on ProteinMPNN is the **same** pass (`tf` is accepted
as the explicit name). `mask` is refused.

`--protein_mpnn_scoring_mode random_order` is a ProteinMPNN-only extra
(average random decode orders). It is not `--scoring_strategy`.

## Who allows what

| Backbones | `wt` | `mask` | `tf` |
|-----------|------|--------|------|
| ESM-2, ESM-1b, ESM-1v, SaProt, ProtSSN, CARP, ESM3 / ESM-C, S3F, `auto` | yes | yes | no |
| ProSST, VenusREM2 | yes | no | no |
| ProteinMPNN (`proteinmpnn-020`, …) | yes (= tf) | no | yes |
| ESM-IF, MIF-ST, ProGen2 / 3, ProtGPT2, RITA, S2F | yes | no | no |

`--model auto --model_id <repo>` loads a generic Hugging Face MLM. Custom
`modeling_*.py` from that repo is **not** executed unless you also pass
`--trust_remote_code`. Built-in keys (`esm2`, `progen3`, `rita`, …) keep
the code their adapters already need.

## `--scoring_mode` (the vrh head)

After \(\ell_{\mathrm{raw}}\) is built, vrh is the same for every strategy.
See [`scoring_formula.md`](scoring_formula.md).

| Mode | What it scores |
|------|----------------|
| `calibrated_margin` (default) | fused \(\Delta\) minus gated CCD, then RSA / pLDDT |
| `log_odds` | fused \(\Delta\) only (`--alpha 0` → raw PLM) |

## Variant dump campaign (2026-09-12)

Named singles that were missing from the three-benchmark logit caches:

| `--model` | Catalog key | Notes |
|-----------|-------------|--------|
| `protssn-k{10,20,30}-h{512,768,1280}` | `protssn_k*_h*` | One GNN each. The board row `protssn` stays the 9-model ensemble. |
| `carp-600k` / `carp-38m` / `carp-76m` | `carp_600k` / `carp_38m` / `carp_76m` | Size variants. `carp` / `carp_640m` already cached. |
| `esm1b` (`wt` or `mask`) | `esm1b_wt` / `esm1b_mask` | Re-dump into the new layout; skip if that folder is already complete. |

Caches:

- ProteinGym / VenusMutHub → `experiments/full_recipe_wc0/extra_seq_gnn_variants/{pg,vmh}/cache/logits/{key}/` (same `.pt` payload as `extra_structure_models`)
- VenusViroHub → `experiments/viro_clinvar/cache/logits/viro90/{key}/` via `dump_logits.py`

```bash
BENCHMARK=pg bash script/baseline/dump_logits_cache.sh carp-600k
BENCHMARK=vmh bash script/baseline/dump_logits_cache.sh protssn-k20-h512
BENCHMARK=viro bash script/baseline/dump_logits_cache.sh carp_38m
bash experiments/full_recipe_wc0/extra_seq_gnn_variants/scripts/dump_only.sh
```

