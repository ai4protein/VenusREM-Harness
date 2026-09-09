# rem2 models and forwards

`--scoring_strategy` chooses how the backbone produces \(\ell_{\mathrm{raw}}\).
`--scoring_mode` is the rem2 head on top of those logits. They are not the same flag.

`rem2 --list-models` prints each backbone and its allowed forwards (`wt` / `mask` / `tf`).
An unsupported strategy is refused; rem2 does not silently fall back.

## rem2 vs VenusREM2

| Name | Meaning |
|------|---------|
| **rem2** | Calibration recipe on any backbone. CLI: `rem2`. |
| **VenusREM2** | rem2 on the six official ProSST checkpoints (`--model venusrem2`). |

A single ESM-2 or ProSST-2048 run is rem2, not VenusREM2.

## Score columns

Matches `rem2/naming.py`. `--scoring_strategy` is not part of the column name.

| Run | Column |
|-----|--------|
| One backbone | `{HF-basename}__rem2` (also `{HF-basename}__raw_backbone`) |
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
(ESM-IF, MIF-ST) this *is* the teacher-forced / unmasked likelihood. rem2 still
calls that `wt`, not `tf`. `mask` is refused.

MIF-ST: ProteinGym’s name looks like a masked model; rem2 scores it unmasked
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

## `--scoring_mode` (the rem2 head)

After \(\ell_{\mathrm{raw}}\) is built, rem2 is the same for every strategy.
See [`scoring_formula.md`](scoring_formula.md).

| Mode | What it scores |
|------|----------------|
| `calibrated_margin` (default) | fused \(\Delta\) minus gated CCD, then RSA / pLDDT |
| `log_odds` | fused \(\Delta\) only (`--alpha 0` → raw PLM) |
