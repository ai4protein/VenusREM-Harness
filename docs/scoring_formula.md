# VenusREM-Orbit Scoring Formula

## Notation

- $\ell_{\text{raw}} \in \mathbb{R}^{L \times |V|}$: raw backbone (ProSST) logits
- $\mathbf{f}_{\text{MSA}} \in \mathbb{R}^{L \times |V|}$: MSA frequency matrix
- $L$: sequence length
- $|V|$: vocabulary size
- $\mathcal{M}$: set of mutated positions

## Step 1: Logit Fusion

$$\ell_{\text{fused}} = (1 - \alpha) \cdot \ell_{\text{raw}} + \alpha \cdot \log \operatorname{softmax}(\mathbf{f}_{\text{MSA}})$$

where $\alpha = 0.8$.

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

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\alpha$ | 0.8 | MSA fusion weight |
| $w_b$ | 0.15 | Background penalty weight |
| $w_c$ | 0.05 | Wild-type confidence bonus weight |

## Performance (ProteinGym, 217 proteins)

| Method | Mean Spearman |
|--------|:---:|
| ProSST backbone | 0.5252 |
| VenusREM v1 | 0.5357 |
| Orbit Single-Anchor | 0.5406 |
| Raw-CCD (no RSA) | 0.5423 |
| **Raw-CCD + RSA above-mean** | **0.5500** |
