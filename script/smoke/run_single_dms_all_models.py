#!/usr/bin/env python3
"""Single-DMS smoke test for every registered backbone (MSA-fused VenusREM2 path).

Runs one ProteinGym assay with AF2 structure + a2m MSA, records status / Spearman,
and compares against prior official-format DMS-level results when available.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTEIN = "PIN1_HUMAN_Tsuboyama_2023_1I6C"
DEFAULT_OUT = ROOT / "result" / "smoke_single_dms_20260722"

# Maps --model key -> prior leaderboard column names (Raw / VenusREM2 DMS-level CSVs).
# pro sst MSA-only (α=0.8 log_odds) aligns to VenusREM, not full ProSST (VenusREM2).
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
PRIOR_VENUSREM2_COL = {
    "prosst": "VenusREM",  # MSA α=0.8 log_odds; full VenusREM2 is ProSST (VenusREM2)
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
    "s3f": "S3F (wt VenusREM2)",
}
# Alias for summary display
PRIOR_COL = {**PRIOR_RAW_COL}

# Prefer models that match default CLI weights / strategy used in prior tables.
MODELS = [
    "prosst",
    "esm2",
    "esm1b",
    "esm1v",
    "saprot",
    "protssn",
    "esm_if",
    "protein_mpnn",
    "progen2",
    "progen3",
    "protgpt2",
    "rita",
    "esm3",
    "carp",
    "s2f",
    "s3f",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_prior(protein: str) -> dict:
    base = ROOT / "results_and_figures" / "proteingym_raw_vs_venusrem2_official" / "performance" / "Spearman"
    out = {"raw": {}, "venusrem2": {}}
    for kind, fname in [
        ("raw", "DMS_substitutions_Raw_Spearman_DMS_level.csv"),
        ("venusrem2", "DMS_substitutions_VenusREM2_Spearman_DMS_level.csv"),
    ]:
        path = base / fname
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        row = df[df["DMS ID"].astype(str) == protein]
        if row.empty:
            continue
        out[kind] = row.iloc[0].to_dict()
    return out


def _spearman_from_scores(csv_path: Path, score_col: str) -> float | None:
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    if score_col not in df.columns or "DMS_score" not in df.columns:
        return None
    try:
        corr = spearmanr(df["DMS_score"], df[score_col]).correlation
    except Exception:
        return None
    if corr is None or pd.isna(corr):
        return None
    return float(corr)


def _raw_backbone_spearman(csv_path: Path) -> float | None:
    """Prefer any ``*__raw_backbone`` column (CLI may mislabel basename)."""
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    for c in df.columns:
        if c.endswith("__raw_backbone"):
            return _spearman_from_scores(csv_path, c)
    return None


def build_cmd(model: str, protein: str, out_dir: Path, base_dir: Path) -> list[str]:
    scores_dir = out_dir / "scores" / model
    cmd = [
        sys.executable,
        str(ROOT / "compute_fitness.py"),
        "--model",
        model,
        "--model_out_name",
        model,
        "--base_dir",
        str(base_dir),
        "--pdb_dir",
        "pdbs_af2_assay_resolved_full",
        "--struc_seq_dir",
        "struc_seq_af2_assay_resolved_full",
        "--structure_vocab_subdir",
        "2048",
        "--aa_seq_aln_dir",
        "aa_seq_aln_a2m",
        "--protein_list",
        protein,
        "--alpha",
        "0.8",
        "--logit_mode",
        "aa_seq_aln",
        "--scoring_mode",
        "log_odds",
        "--print_compare_spearman",
        "--disable_tqdm",
        "--log_level",
        "info",
        "--out_scores_dir",
        str(scores_dir),
    ]
    # Align scoring strategy with prior "mask" columns when relevant.
    if model == "saprot":
        cmd.extend(["--scoring_strategy", "masked-marginals"])
    if model == "esm1v":
        # full ensemble for alignment with esm1v_wt prior
        cmd.extend(["--esm1v_seeds", "1", "2", "3", "4", "5"])
    return cmd


def run_one(model: str, protein: str, out_dir: Path, base_dir: Path, prior: dict, timeout: int) -> dict:
    scores_dir = out_dir / "scores" / model
    scores_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / f"{model}.log"
    meta = {
        "model": model,
        "protein": protein,
        "started_at": _utc_now(),
        "status": "running",
        "protocol": {
            "alpha": 0.8,
            "logit_mode": "aa_seq_aln",
            "scoring_mode": "log_odds",
            "msa": "aa_seq_aln_a2m",
            "pdb": "pdbs_af2_assay_resolved_full",
            "struc_seq": "struc_seq_af2_assay_resolved_full/2048",
        },
        "prior_col_raw": PRIOR_RAW_COL.get(model),
        "prior_col_venusrem2": PRIOR_VENUSREM2_COL.get(model),
        "prior_raw": None,
        "prior_venusrem2": None,
        "spearman_msa": None,
        "spearman_raw_backbone": None,
        "delta_vs_prior_raw": None,
        "delta_vs_prior_venusrem2": None,
        "elapsed_sec": None,
        "returncode": None,
        "error": None,
        "log": str(log_path),
        "scores_csv": str(scores_dir / "scores" / f"{protein}.csv"),
    }
    raw_col = PRIOR_RAW_COL.get(model)
    venusrem2_col = PRIOR_VENUSREM2_COL.get(model)
    if raw_col:
        meta["prior_raw"] = prior.get("raw", {}).get(raw_col)
    if venusrem2_col:
        meta["prior_venusrem2"] = prior.get("venusrem2", {}).get(venusrem2_col)

    cmd = build_cmd(model, protein, out_dir, base_dir)
    meta["cmd"] = cmd
    t0 = time.time()
    try:
        with open(log_path, "w") as logf:
            logf.write("CMD: " + " ".join(cmd) + "\n\n")
            logf.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        meta["returncode"] = proc.returncode
        meta["status"] = "ok" if proc.returncode == 0 else "failed"
        if proc.returncode != 0:
            meta["error"] = f"exit={proc.returncode}"
    except subprocess.TimeoutExpired:
        meta["status"] = "timeout"
        meta["error"] = f"timeout after {timeout}s"
        meta["returncode"] = -1
    except Exception as exc:
        meta["status"] = "error"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["returncode"] = -1
        with open(log_path, "a") as logf:
            logf.write("\n" + traceback.format_exc())

    meta["elapsed_sec"] = round(time.time() - t0, 2)

    score_csv = scores_dir / "scores" / f"{protein}.csv"
    if score_csv.is_file():
        meta["spearman_msa"] = _spearman_from_scores(score_csv, model)
        meta["spearman_raw_backbone"] = _raw_backbone_spearman(score_csv)

    for key, prior_key, smoke_key in [
        ("delta_vs_prior_raw", "prior_raw", "spearman_raw_backbone"),
        ("delta_vs_prior_venusrem2", "prior_venusrem2", "spearman_msa"),
    ]:
        p = meta.get(prior_key)
        s = meta.get(smoke_key)
        if p is not None and s is not None:
            try:
                meta[key] = round(float(s) - float(p), 4)
            except Exception:
                pass

    meta["finished_at"] = _utc_now()
    return meta


def write_summary(records: list[dict], out_dir: Path) -> None:
    jsonl = out_dir / "records.jsonl"
    with open(jsonl, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rows = []
    for r in records:
        rows.append(
            {
                "model": r["model"],
                "status": r["status"],
                "spearman_msa_a08": r.get("spearman_msa"),
                "spearman_raw_backbone": r.get("spearman_raw_backbone"),
                "prior_raw": r.get("prior_raw"),
                "prior_venusrem2": r.get("prior_venusrem2"),
                "delta_vs_prior_raw": r.get("delta_vs_prior_raw"),
                "delta_vs_prior_venusrem2": r.get("delta_vs_prior_venusrem2"),
                "prior_col_raw": r.get("prior_col_raw"),
                "prior_col_venusrem2": r.get("prior_col_venusrem2"),
                "elapsed_sec": r.get("elapsed_sec"),
                "error": r.get("error"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)

    md = out_dir / "SUMMARY.md"
    lines = [
        f"# Single-DMS smoke: `{records[0]['protein'] if records else ''}`",
        "",
        f"- Generated: `{_utc_now()}`",
        f"- Protocol: α=0.8 MSA (`aa_seq_aln_a2m`), AF2 PDB/struc tokens, `log_odds`",
        f"- Prior: `results_and_figures/proteingym_raw_vs_venusrem2_official/performance/Spearman/`",
        "",
        "| model | status | smoke_msa | smoke_raw | prior_raw | prior_venusrem2 | Δvenusrem2 | sec | error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        lines.append(
            "| {model} | {status} | {spearman_msa} | {spearman_raw_backbone} | {prior_raw} | {prior_venusrem2} | {delta_vs_prior_venusrem2} | {elapsed_sec} | {error} |".format(
                model=r["model"],
                status=r["status"],
                spearman_msa=_fmt(r.get("spearman_msa")),
                spearman_raw_backbone=_fmt(r.get("spearman_raw_backbone")),
                prior_raw=_fmt(r.get("prior_raw")),
                prior_venusrem2=_fmt(r.get("prior_venusrem2")),
                delta_vs_prior_venusrem2=_fmt(r.get("delta_vs_prior_venusrem2")),
                elapsed_sec=r.get("elapsed_sec"),
                error=(r.get("error") or "")[:80].replace("|", "/"),
            )
        )
    md.write_text("\n".join(lines) + "\n")


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default=DEFAULT_PROTEIN)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--base_dir", type=Path, default=ROOT / "data" / "proteingym_v1")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--timeout", type=int, default=1800, help="Per-model timeout seconds")
    ap.add_argument("--only", nargs="*", default=None, help="Subset of models")
    args = ap.parse_args()

    models = args.only or args.models
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)

    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "protein": args.protein,
                "base_dir": str(args.base_dir),
                "models": models,
                "started_at": _utc_now(),
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            },
            indent=2,
        )
    )

    prior = _load_prior(args.protein)
    (out_dir / "prior_pin1_snapshot.json").write_text(json.dumps(prior, indent=2, default=str))

    records: list[dict] = []
    # resume: keep previous records (ok and failed); only re-run missing/failed if --retry-failed
    jsonl_path = out_dir / "records.jsonl"
    done_ok = set()
    if jsonl_path.is_file():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            records.append(rec)
            if rec.get("status") == "ok":
                done_ok.add(rec["model"])

    for model in models:
        if model in done_ok:
            print(f"[skip] {model} already ok", flush=True)
            continue
        # drop previous failed record for this model
        records = [r for r in records if r.get("model") != model]
        print(f"[run] {model} ...", flush=True)
        rec = run_one(model, args.protein, out_dir, args.base_dir, prior, args.timeout)
        records.append(rec)
        write_summary(records, out_dir)
        print(
            f"[{rec['status']}] {model} msa={rec.get('spearman_msa')} "
            f"raw={rec.get('spearman_raw_backbone')} Δvenusrem2={rec.get('delta_vs_prior_venusrem2')} "
            f"({rec.get('elapsed_sec')}s)",
            flush=True,
        )

    write_summary(records, out_dir)
    print(f"Done. Summary: {out_dir / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
