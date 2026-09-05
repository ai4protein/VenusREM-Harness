#!/usr/bin/env python3
"""Recompute alignment verdicts for smoke_single_dms records vs prior leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "result" / "smoke_single_dms_20260722"
PROTEIN = "PIN1_HUMAN_Tsuboyama_2023_1I6C"

PRIOR_RAW_COL = {
    "prosst": "ProSST (K=2048)",
    "esm2": "esm2_650m_wt",
    "esm1b": "esm1b_wt",
    "esm1v": "esm1v_wt",
    "saprot": "saprot_mask",
    "protssn": "protssn",
    "esm_if": "esmif",
    "protein_mpnn": "ProteinMPNN",
    "progen2": "ProGen2-L",
    "progen3": "progen3",
    "rita": "rita_xl",
    "esm3": "esmc",
    "carp": "carp_640m",
    "s3f": "S3F (wt raw)",
}
PRIOR_MSA_COL = {"prosst": "VenusREM"}
PRIOR_FULL_ORBIT_COL = {
    "prosst": "ProSST (Orbit)",
    "esm2": "esm2_650m_wt",
    "esm1b": "esm1b_wt",
    "esm1v": "esm1v_wt",
    "saprot": "saprot_mask",
    "protssn": "protssn",
    "esm_if": "esmif",
    "protein_mpnn": "ProteinMPNN",
    "progen2": "ProGen2-L",
    "progen3": "progen3",
    "rita": "rita_xl",
    "esm3": "esmc",
    "carp": "carp_640m",
    "s3f": "S3F (wt Orbit)",
}

MODEL_ORDER = [
    "prosst", "esm2", "esm1b", "esm1v", "saprot", "protssn", "esm_if",
    "protein_mpnn", "progen2", "progen3", "protgpt2", "rita", "esm3",
    "tranception", "carp", "s2f", "s3f",
]


def load_prior():
    base = (
        ROOT / "results_and_figures" / "proteingym_raw_vs_orbit_official"
        / "performance" / "Spearman"
    )
    out = {}
    for kind, fname in [
        ("raw", "DMS_substitutions_Raw_Spearman_DMS_level.csv"),
        ("orbit", "DMS_substitutions_Orbit_Spearman_DMS_level.csv"),
    ]:
        df = pd.read_csv(base / fname)
        row = df[df["DMS ID"].astype(str) == PROTEIN]
        out[kind] = row.iloc[0].to_dict() if not row.empty else {}
    return out


def spearman(df, col):
    if col not in df.columns:
        return None
    try:
        corr = spearmanr(df["DMS_score"], df[col]).correlation
    except Exception:
        return None
    if corr is None or pd.isna(corr):
        return None
    return float(corr)


def read_smoke_scores(model: str):
    path = OUT / "scores" / model / "scores" / f"{PROTEIN}.csv"
    if not path.is_file():
        return None, None
    df = pd.read_csv(path)
    msa = spearman(df, model)
    raw = None
    for c in df.columns:
        if c.endswith("__raw_backbone"):
            raw = spearman(df, c)
            break
    return raw, msa


def verdict_raw(delta):
    if delta is None:
        return "N/A"
    ad = abs(delta)
    if ad < 0.02:
        return "PASS"
    if ad < 0.05:
        return "REVIEW"
    return "FAIL"


def delta(a, b):
    if a is None or b is None:
        return None
    try:
        return round(float(a) - float(b), 4)
    except Exception:
        return None


def fmt(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def main():
    records_path = OUT / "records.jsonl"
    by_model = {}
    if records_path.is_file():
        for line in records_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                by_model[rec["model"]] = rec
    prior = load_prior()

    # Preserve known s3f failure if scores missing.
    if "s3f" not in by_model or by_model["s3f"].get("status") != "ok":
        by_model["s3f"] = {
            "model": "s3f",
            "status": "failed",
            "error": "TorchDrug not installed",
            "elapsed_sec": None,
        }

    rows = []
    for m in MODEL_ORDER:
        r = by_model.get(m)
        if r is None:
            continue
        smoke_raw, smoke_msa = read_smoke_scores(m)
        prior_raw = prior["raw"].get(PRIOR_RAW_COL[m]) if m in PRIOR_RAW_COL else None
        prior_msa = prior["orbit"].get(PRIOR_MSA_COL[m]) if m in PRIOR_MSA_COL else None
        prior_full = prior["orbit"].get(PRIOR_FULL_ORBIT_COL[m]) if m in PRIOR_FULL_ORBIT_COL else None
        d_raw = delta(smoke_raw, prior_raw)
        d_msa = delta(smoke_msa, prior_msa)
        d_full = delta(smoke_msa, prior_full)
        status = r.get("status")
        rows.append(
            {
                "model": m,
                "status": status,
                "smoke_raw": smoke_raw,
                "prior_raw": prior_raw,
                "delta_raw": d_raw,
                "verdict_raw": verdict_raw(d_raw) if status == "ok" else status,
                "smoke_msa": smoke_msa,
                "prior_msa_ref": prior_msa,
                "delta_msa_ref": d_msa,
                "prior_full_orbit": prior_full,
                "delta_vs_full_orbit": d_full,
                "error": r.get("error"),
                "elapsed_sec": r.get("elapsed_sec"),
            }
        )

    pd.DataFrame(rows).to_csv(OUT / "alignment_verdict.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": r["model"],
                "status": r["status"],
                "spearman_msa_a08": r["smoke_msa"],
                "spearman_raw_backbone": r["smoke_raw"],
                "prior_raw": r["prior_raw"],
                "prior_msa_ref": r["prior_msa_ref"],
                "prior_full_orbit": r["prior_full_orbit"],
                "delta_vs_prior_raw": r["delta_raw"],
                "delta_vs_msa_ref": r["delta_msa_ref"],
                "delta_vs_full_orbit": r["delta_vs_full_orbit"],
                "verdict_raw": r["verdict_raw"],
                "elapsed_sec": r["elapsed_sec"],
                "error": r["error"],
            }
            for r in rows
        ]
    ).to_csv(OUT / "summary.csv", index=False)

    md = [
        f"# Alignment verdict — `{PROTEIN}`",
        "",
        "Primary gate: **smoke_raw vs prior_raw** (`|Δ|<0.02` PASS).",
        "MSA gate: only **prosst** has MSA-only prior (`VenusREM`); full Orbit is gap monitor.",
        "",
        "| model | status | smoke_raw | prior_raw | Δraw | verdict | smoke_msa | msa_ref | Δmsa | full_orbit | Δfull |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['status']} | {fmt(r['smoke_raw'])} | {fmt(r['prior_raw'])} | "
            f"{fmt(r['delta_raw'])} | {r['verdict_raw']} | {fmt(r['smoke_msa'])} | "
            f"{fmt(r['prior_msa_ref'])} | {fmt(r['delta_msa_ref'])} | "
            f"{fmt(r['prior_full_orbit'])} | {fmt(r['delta_vs_full_orbit'])} |"
        )

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_pass = sum(1 for r in rows if r["verdict_raw"] == "PASS")
    n_review = sum(1 for r in rows if r["verdict_raw"] == "REVIEW")
    n_fail = sum(1 for r in rows if r["verdict_raw"] == "FAIL")
    n_failed = sum(1 for r in rows if r["status"] == "failed")
    md += [
        "",
        f"- ok runs: **{n_ok}** / {len(rows)} (failed: {n_failed})",
        f"- raw PASS: **{n_pass}**, REVIEW: **{n_review}**, FAIL: **{n_fail}**",
        "",
        "### Failures / notes",
        "",
        "- **s3f**: full TorchDrug path; surface↔residue KNN uses pure PyTorch (no PyKeOps).",
        "  On-the-fly surface generation still needs `.[s3f-surface-gen]` / pykeops.",
        "- **s2f**: lightweight ESM2 fallback (scores match esm2); not in official prior table",
        "- **protgpt2 / tranception**: no official prior column (smoke-only regression)",
        "- **smoke_msa vs full Orbit**: expected gap (~0.03–0.10); smoke is MSA α=0.8 `log_odds` only",
        "",
        "Artifacts: `scores/{model}/scores/PIN1_….csv`, `records.jsonl`, `summary.csv`, `logs/`",
        "",
        "See `ALIGNMENT_GUIDE.md` for protocol details.",
    ]
    text = "\n".join(md) + "\n"
    (OUT / "ALIGNMENT_VERDICT.md").write_text(text)
    (OUT / "SUMMARY.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
