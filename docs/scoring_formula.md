# VenusREM2 Scoring Formula

## Notation

- $\ell_{\text{raw}} \in \mathbb{R}^{L \times |V|}$: raw backbone (ProSST) logits
- $\mathbf{f}_{\text{MSA}} \in \mathbb{R}^{L \times |V|}$: MSA frequency matrix
- $L$: sequence length
- $|V|$: vocabulary size
- $\mathcal{M}$: set of mutated positions

## Step 1: Logit Fusion

$$\ell_{\text{fused}} = (1 - \alpha) \cdot \ell_{\text{raw}} + \alpha \cdot \log \operatorname{softmax}(\mathbf{f}_{\text{MSA}})$$

$\alpha$ is entropy-adaptive per assay (rem2 default, main column `rem2_entropy`):
$\rho = (1-\bar{H})\rho_\pi + \bar{H}\rho_{\text{rot}}$, then
$\alpha = \rho\, s_P / (\rho\, s_P + (1-\rho)\, s_M)$;
the fixed variant $\alpha = 0.8$ is kept as the `rem2_fixed08` ablation column.

## Step 2: Background Bias

$$b_v = \operatorname{logsumexp}_{i}(\ell^{\text{raw}}_{i,v}) - \log L$$

$b_v$ is the position-averaged raw logit for amino acid $v$, capturing the backbone model's intrinsic bias toward certain amino acids regardless of context.

## Step 3: Wild-Type Confidence

$$c_i = \sigma(\ell^{\text{raw}}_{i, \text{wt}})$$

$c_i$ measures how confident the backbone model is that position $i$ should be the wild-type residue. Higher confidence implies the model "understands" this position, making its predictions more trustworthy.

## Step 4: Calibrated Margin

$$s_i = \underbrace{(\ell_{\text{fused}}[i, \text{mt}] - \ell_{\text{fused}}[i, \text{wt}])}_{\Delta \text{log-odds}} - \underbrace{w_b \cdot (b_{\text{mt}} - b_{\text{wt}})}_{\text{background penalty}} + \underbrace{w_c \cdot c_i}_{\text{confidence bonus}}$$

The calibration terms (background, confidence) are computed from $\ell_{\text{raw}}$, while the delta log-odds uses $\ell_{\text{fused}}$, avoiding double-penalty between CCD calibration and MSA fusion.

## Step 5: RSA Above-Mean Decay

$$d_i = \max(0, \; \text{RSA}_i - \overline{\text{RSA}})$$

$$\text{score}_i = s_i \cdot (1 - d_i)$$

Only positions with solvent exposure above the protein-level mean are penalized. This avoids over-penalizing small proteins (which have systematically higher mean RSA).

## Step 6: Multi-Mutant Aggregation

$$S = \sum_{i \in \mathcal{M}} \text{score}_i$$

## Parameters

rem2 defaults (supersedes the v1-era values
$w_b{=}0.15$, $w_c{=}0.05$ previously listed here):

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\alpha$ | entropy-adaptive (per assay); fixed variant 0.8 | MSA fusion weight |
| $\beta$ ($w_b$) | **$\beta = 1-\alpha$** (package default); fixed \(0.2\) is the \(\alpha=0.8\) special case | Background penalty weight |
| $\gamma$ ($w_c$) | 0.0 | Wild-type confidence bonus weight |

Dynamic-$\beta$ rationale: the background term $b$ is computed from the raw
(PLM-channel) logits only, and the PLM channel enters the fused score with
weight $(1-\alpha)$, so the correction strength scales with the actual PLM
contribution. Fixed $\beta = 0.2$ is the special case of this rule at
$\alpha = 0.8$. On VenusViroHub (59 models), $\beta = 1-\alpha$ improves the
hierarchical mean for 56/59 models (mean $+0.0068$); see
`experiments/viro_clinvar/results/venusvirohub_dynbeta/README.md`.

## Performance (ProteinGym, 217 proteins)

| Method | Mean Spearman |
|--------|:---:|
| ProSST backbone | 0.5252 |
| VenusREM v1 | 0.5357 |
| VenusREM2 Single-Anchor | 0.5406 |
| Raw-CCD (no RSA) | 0.5423 |
| **Raw-CCD + RSA above-mean** | **0.5500** |
