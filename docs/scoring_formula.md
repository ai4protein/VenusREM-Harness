# vrh scoring formula

Package default (`vrh`, `--alpha entropy`, `--scoring_mode calibrated_margin`).
`--alpha 0.8` is a fixed-blend ablation, not this recipe.

**vrh** is the calibration head and can sit on any backbone.
**VenusREM2** is vrh on the official ProSST ensemble only (`--model venusrem2`).

## Score columns

| Run | Column |
|-----|--------|
| Single backbone | `{HF-basename}__vrh` (e.g. `esm2_t33_650M_UR50D__vrh`, `ProSST-2048__vrh`) |
| Same run, raw PLM | `{HF-basename}__raw_backbone` |
| Official ensemble, per K | `VenusREM2__ProSST-{K}` |
| Official ensemble, combined | `VenusREM2` (z-mean of the six `VenusREM2__*` members) |

`--scoring_strategy` is **not** a column name. `wt` / `mask` / `tf` only change how \(\ell_{\mathrm{raw}}\) is obtained (`docs/models.md`). The vrh head below is the same.

## Notation

- \(\ell_{\mathrm{raw}}\in\mathbb{R}^{L\times|V|}\): backbone log-probs (default: one `wt` pass; `mask` / `tf` in [`models.md`](models.md))
- \(C\in\mathbb{R}^{L\times|V|}\): MSA count matrix after row-normalize + \(\log\mathrm{softmax}\)
- \(\ell\): fused logits used for \(\Delta\)
- \(L\): sequence length; \(V_{20}=\{\mathrm{A},\mathrm{C},\ldots,\mathrm{Y}\}\)
- \(\mathcal{M}\): mutated positions in one mutant string (`A42G` or `A42G:D10E`)

CCD terms are computed on \(\ell_{\mathrm{raw}}\) (`--calibrate_on_raw`). \(\Delta\) uses \(\ell\).

## 1. Dynamic \(\alpha\) (per protein)

Needs a usable MSA. No MSA → \(\alpha=0\) (CCD / RSA / pLDDT still run). If the feature computation raises → fallback \(\alpha=0.8\), \(\beta=0.2\).

On the aligned span, using the 20 standard amino acids:

\[
\rho_\pi=\mathrm{clip}\!\left(\frac{-\overline{\log p_{\mathrm{PLM}}(\mathrm{WT})}}{\log 20},\,0,1\right),\qquad
\bar H=\frac{\overline{H(\mathrm{softmax}(\ell_{\mathrm{raw}}))}}{\log 20}
\]

\[
\rho_{\mathrm{rot}}=\tfrac12\bigl(1-\overline{\mathrm{pearson}}(\ell_{\mathrm{raw}},C)\bigr),\qquad
\rho=(1-\bar H)\,\rho_\pi+\bar H\,\rho_{\mathrm{rot}}
\]

\(s_P\) / \(s_M\) are the std of non-WT native-margin scores on \(\ell_{\mathrm{raw}}\) and \(C\). Then

\[
\alpha=\frac{\rho\,s_P}{\rho\,s_P+(1-\rho)\,s_M},\qquad
\beta=1-\alpha
\]

(`--background_weight one_minus_alpha`). `--alpha 0.8` skips this and sets \(\alpha=0.8\).

## 2. MSA fusion

On the aligned span (default path: `aa_seq_aln`):

\[
\ell=(1-\alpha)\,\ell_{\mathrm{raw}}+\alpha\,C,\qquad
C=\log\mathrm{softmax}(f_{\mathrm{MSA}})
\]

\(\alpha=0\) leaves \(\ell=\ell_{\mathrm{raw}}\).

## 3. CCD (z-score background + coherence gate)

All of the following use \(\ell_{\mathrm{raw}}\), not \(\ell\).

\[
b_v=\mathrm{logsumexp}_i(\ell^{\mathrm{raw}}_{i,v})-\log L
\]

\(b^z\) is the z-score of \(b\) over the 20 AA columns (other vocab slots stay 0).

\[
g_i=\mathrm{pearson}\bigl(\ell^{\mathrm{raw}}_{i,V_{20}},\,b_{V_{20}}\bigr),\qquad
\mathrm{bg\_scale}=\max\bigl(0,\,\overline{g_i}\bigr)
\]

`--disable_adaptive_ccd` forces \(\mathrm{bg\_scale}=1\) (ungated CCD ablation). This is a coherence gate, not a plain subtract-\(b\) correction.

## 4. `calibrated_margin`

\[
\Delta_i=\ell_{i,\mathrm{mt}}-\ell_{i,\mathrm{wt}}
\]

\[
s_i=\Delta_i-\mathrm{bg\_scale}\cdot\beta\cdot(b^z_{\mathrm{mt}}-b^z_{\mathrm{wt}})
\]

`--scoring_mode log_odds` is just \(\Delta_i\) (raw or fused, depending on \(\alpha\)).

## 5. RSA above-mean decay (on)

RSA from the PDB (Shrake–Rupley / max ASA). Mean is over positions with \(\mathrm{RSA}>0\).

\[
d_i=\max\bigl(0,\,\mathrm{RSA}_i-\overline{\mathrm{RSA}}\bigr)
\]

Default `--task_type default`: \(s_i\leftarrow s_i(1-d_i)\).
`--task_type surface` (binding / surface phenotypes): \(s_i\leftarrow s_i(1+d_i)\).

## 6. pLDDT above-mean decay (on, predicted structures only)

Skipped when the PDB looks experimental (crystal / NMR / cryo-EM; B-factors are not pLDDT).

\[
q_i=\max\bigl(0,\,(1-p_i)-(1-\bar p)\bigr)
\]

\[
s_i\leftarrow s_i\,(1-q_i)
\]

## 7. Multi-mutant

\[
S=\sum_{i\in\mathcal{M}} s_i
\]

## 8. VenusREM2 ensemble

Each of the six official ProSST checkpoints (\(K\in\{20,128,512,1024,2048,4096\}\)) is scored with vrh, then

\[
\mathrm{VenusREM2}=\mathrm{mean}_K\;\mathrm{zscore}(\mathrm{VenusREM2\_\_ProSST\text{-}K})
\]

z-score is over mutants of that protein.

## Defaults

| Flag | Default |
|------|---------|
| `--scoring_strategy` | `wt-marginals` |
| `--alpha` | `entropy` (per protein) |
| `--background_weight` | `one_minus_alpha` (\(\beta=1-\alpha\)) |
| `--scoring_mode` | `calibrated_margin` |
| `--calibrate_on_raw` | on |
| `--rsa_decay_mode` / `--plddt_decay_mode` | `above_mean` |

VenusREM v1 on this tree: `--model prosst-2048 --alpha 0.8 --scoring_mode log_odds`.

## ProteinGym (217 proteins)

Numbers are mean Spearman from `experiments/rem2_iclr_20260823/summaries_beta1ma/staged_ablation_59.csv` (`pg_*` columns). Full vrh = entropy \(\alpha\), \(\beta=1-\alpha\), gated CCD, RSA + pLDDT `above_mean`.

### VenusREM2 (ProSST ensemble)

| Variant | Scoring | Alpha | RSA | pLDDT | Mean Spearman |
|---------|---------|-------|-----|-------|---------------|
| Raw backbone | log_odds | — | - | - | 0.5245 |
| + MSA | log_odds | dynamic | - | - | 0.5424 |
| + MSA + CCD | calibrated_margin | dynamic | - | - | 0.5499 |
| + MSA + CCD + RSA | calibrated_margin | dynamic | above_mean | - | 0.5543 |
| **Full vrh** | calibrated_margin | dynamic | above_mean | above_mean | **0.5556** |

### vrh on ESM-2 (650M, wt-marginals)

| Variant | Scoring | Alpha | RSA | pLDDT | Mean Spearman |
|---------|---------|-------|-----|-------|---------------|
| Raw backbone | log_odds | — | - | - | 0.4175 |
| + MSA | log_odds | dynamic | - | - | 0.4287 |
| + MSA + CCD | calibrated_margin | dynamic | - | - | 0.4405 |
| + MSA + CCD + RSA | calibrated_margin | dynamic | above_mean | - | 0.4647 |
| **Full vrh** | calibrated_margin | dynamic | above_mean | above_mean | **0.4677** |

### vrh on SaProt (AF-650M, wt-marginals)

| Variant | Scoring | Alpha | RSA | pLDDT | Mean Spearman |
|---------|---------|-------|-----|-------|---------------|
| Raw backbone | log_odds | — | - | - | 0.4242 |
| + MSA | log_odds | dynamic | - | - | 0.4273 |
| + MSA + CCD | calibrated_margin | dynamic | - | - | 0.4326 |
| + MSA + CCD + RSA | calibrated_margin | dynamic | above_mean | - | 0.4522 |
| **Full vrh** | calibrated_margin | dynamic | above_mean | above_mean | **0.4537** |

### vrh on ESM-1v (5-seed, wt-marginals)

| Variant | Scoring | Alpha | RSA | pLDDT | Mean Spearman |
|---------|---------|-------|-----|-------|---------------|
| Raw backbone | log_odds | — | - | - | 0.4099 |
| + MSA | log_odds | dynamic | - | - | 0.4190 |
| + MSA + CCD | calibrated_margin | dynamic | - | - | 0.4331 |
| + MSA + CCD + RSA | calibrated_margin | dynamic | above_mean | - | 0.4548 |
| **Full vrh** | calibrated_margin | dynamic | above_mean | above_mean | **0.4567** |
