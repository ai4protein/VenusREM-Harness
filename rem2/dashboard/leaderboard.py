"""ProteinGym substitution board shown on New Job (no network)."""

from __future__ import annotations

from typing import Any

# Mean Spearman, 217 substitution assays. Same numbers as README / docs/scoring_formula.md
# (experiments/rem2_iclr_20260823/summaries_beta1ma/staged_ablation_59.csv).
_ROWS = [
    {
        "name": "VenusREM2",
        "model": "venusrem2",
        "recipe": "full",
        "score": 0.5556,
        "inputs": ["pdb", "msa"],
        "note": "Official ProSST ensemble + full rem2",
        "highlight": True,
    },
    {
        "name": "rem2 · ESM-2 650M",
        "model": "esm2",
        "recipe": "full",
        "score": 0.4677,
        "inputs": ["pdb", "msa"],
        "note": "wt-marginals",
        "highlight": False,
    },
    {
        "name": "rem2 · ESM-1v",
        "model": "esm1v",
        "recipe": "full",
        "score": 0.4567,
        "inputs": ["pdb", "msa"],
        "note": "5-seed ensemble",
        "highlight": False,
    },
    {
        "name": "rem2 · SaProt",
        "model": "saprot",
        "recipe": "full",
        "score": 0.4537,
        "inputs": ["pdb", "msa"],
        "note": "AF-650M, wt-marginals",
        "highlight": False,
    },
]


def proteingym_board() -> dict[str, Any]:
    ranked = []
    for i, row in enumerate(sorted(_ROWS, key=lambda item: -float(item["score"])), start=1):
        item = dict(row)
        item["rank"] = i
        item["score"] = round(float(item["score"]), 3)
        ranked.append(item)
    return {
        "id": "proteingym_substitutions",
        "title": "ProteinGym substitutions",
        "metric": "Mean Spearman",
        "n": 217,
        "url": "https://proteingym.org/benchmarks",
        "source": "rem2 staged ablation on 217 ProteinGym substitution assays",
        "note": "Published full rem2 numbers used a PDB and an MSA. Sequence-only runs skip structure terms and set α=0.",
        "rows": ranked,
    }
