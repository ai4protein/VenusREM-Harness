#!/usr/bin/env python3
"""Package VenusMutHub Raw/+VRH dashboard pairs from paper artifacts.

Sources (never invent scores):

* ``docs/figure/data/fig4_venusmuthub_multimetric.csv`` — overall Spearman /
  NDCG / AUC / MCC / Top recall for each of the 59 configurations.
* ``docs/0overleaf/tables/table_leaderboard_vmh_category.tex`` — assay-macro
  Spearman for Overall + the five MutHub tasks (Raw and VRH rows).
* ``data/VenusMutHub/assay_manifest.csv`` — assay / property counts.

Spearman (overall + tasks) comes from the printed tex leaderboard so the
dashboard matches the paper table. Non-Spearman metrics are overall-only
from the Fig.4 CSV, rounded to 3 decimals like ProteinGym. Per-task
NDCG / AUC / MCC / Top recall are omitted because the tex table does not
print them.

Orbit ``Leaderboard_Compare.csv`` (Accuracy / F1) is not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data" / "VenusMutHub" / "assay_manifest.csv"
FIG4_CSV = REPO / "docs" / "figure" / "data" / "fig4_venusmuthub_multimetric.csv"
VMH_TEX = REPO / "docs" / "0overleaf" / "tables" / "table_leaderboard_vmh_category.tex"
DASHBOARD_DATA = REPO / "vrh" / "dashboard" / "data"
CATALOG_PATH = DASHBOARD_DATA / "venusmuthub_catalog.json"
VRH_PATH = DASHBOARD_DATA / "venusmuthub_vrh.json"

METRICS = [
    {"id": "spearman", "label": "Spearman"},
    {"id": "ndcg", "label": "NDCG"},
    {"id": "auc", "label": "AUC"},
    {"id": "mcc", "label": "MCC"},
    {"id": "top_recall", "label": "Top recall"},
]
COMPARE_METRICS = {
    "spearman": "Spearman",
    "ndcg": "NDCG",
    "auc": "AUC",
    "mcc": "MCC",
    "top_recall": "Top_recall",
}
PROPERTY_LABELS = {
    "overall": "Overall",
    "stability": "Stability",
    "activity": "Activity",
    "ppi_binding": "PPI binding",
    "selectivity": "Selectivity",
    "dti_binding": "DTI binding",
}
PREFERRED_TASK_ORDER = (
    "stability",
    "activity",
    "ppi_binding",
    "selectivity",
    "dti_binding",
)
TEX_PROPERTY_ORDER = (
    "overall",
    "stability",
    "activity",
    "ppi_binding",
    "selectivity",
    "dti_binding",
)
# Paper display name → ProteinGym snapshot key.
DISPLAY_TO_KEY = {
    "ProSST-Ensemble (K=all)": "prosst_ensemble",
    "VenusREM2": "prosst_ensemble",
    "CARP-640M": "carp_640m",
    "ESM-1b (mask)": "esm1b_mask",
    "ESM-1b (wt)": "esm1b_wt",
    "ESM-1v (mask)": "esm1v_mask",
    "ESM-1v (wt)": "esm1v_wt",
    "ESM-2 150M (mask)": "esm2_150m_mask",
    "ESM-2 150M (wt)": "esm2_150m_wt",
    "ESM-2 35M (mask)": "esm2_35m_mask",
    "ESM-2 35M (wt)": "esm2_35m_wt",
    "ESM-2 3B (mask)": "esm2_3b_mask",
    "ESM-2 3B (wt)": "esm2_3b_wt",
    "ESM-2 650M (mask)": "esm2_650m_mask",
    "ESM-2 650M (wt)": "esm2_650m_wt",
    "ESM-2 8M (mask)": "esm2_8m_mask",
    "ESM-2 8M (wt)": "esm2_8m_wt",
    "ESM3": "esm3",
    "ESMC-300M": "esmc",
    "ESMC-600M": "esmc_600m",
    "ESM-IF1": "esmif",
    "ProGen2-L": "progen2",
    "ProGen2-B": "progen2_b",
    "ProGen2-M": "progen2_m",
    "ProGen2-S": "progen2_s",
    "ProGen2-XL": "progen2_xl",
    "ProGen3-1B": "progen3",
    "ProGen3-112M": "progen3_112m",
    "ProGen3-219M": "progen3_219m",
    "ProGen3-339M": "progen3_339m",
    "ProGen3-3B": "progen3_3b",
    "ProGen3-762M": "progen3_762m",
    "ProSST (K=2048)": "prosst_k2048",
    "ProSST (K=20)": "prosst_k20",
    "ProSST (K=128)": "prosst_k128",
    "ProSST (K=512)": "prosst_k512",
    "ProSST (K=1024)": "prosst_k1024",
    "ProSST (K=4096)": "prosst_k4096",
    "ProteinMPNN (v_48_020)": "proteinmpnn",
    "ProteinMPNN (v_48_002)": "pmpnn_v_48_002",
    "ProteinMPNN (v_48_010)": "pmpnn_v_48_010",
    "ProteinMPNN (v_48_030)": "pmpnn_v_48_030",
    "ProteinMPNN-Soluble (v_48_002)": "pmpnn_soluble_v_48_002",
    "ProteinMPNN-Soluble (v_48_010)": "pmpnn_soluble_v_48_010",
    "ProteinMPNN-Soluble (v_48_020)": "pmpnn_soluble_v_48_020",
    "ProteinMPNN-Soluble (v_48_030)": "pmpnn_soluble_v_48_030",
    "ProtSSN": "protssn",
    "RITA-L": "rita_l",
    "RITA-M": "rita_m",
    "RITA-S": "rita_s",
    "RITA-XL": "rita_xl",
    "SaProt (mask)": "saprot_mask",  # legacy alias of SaProt (650M_AF2, mask)
    "SaProt (wt)": "saprot_wt",  # legacy alias of SaProt (650M_AF2, wt)
    "SaProt (650M_AF2, mask)": "saprot_mask",
    "SaProt (650M_AF2, wt)": "saprot_wt",
    "S3F (wt)": "s3f_wt",
    "S3F (mask)": "s3f_mask",
    "MIF-ST": "mifst",
    "SaProt (35M_AF2, mask)": "saprot35m_af2_mask",
    "SaProt (35M_AF2, wt)": "saprot35m_af2_wt",
    "SaProt (650M_PDB, mask)": "saprot650m_pdb_mask",
    "SaProt (650M_PDB, wt)": "saprot650m_pdb_wt",
}
CANONICAL_DISPLAY = {
    "VenusREM2": "ProSST-Ensemble (K=all)",
    "prosst_ensemble": "ProSST-Ensemble (K=all)",
}


def _int_field(row: dict[str, str], key: str) -> int | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    return int(float(raw))


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def task_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        task = (row.get("task") or "").strip()
        if task:
            counts[task] += 1
    return dict(counts)


def property_entries(counts: dict[str, int], n_assays: int) -> list[dict[str, object]]:
    properties = [{"id": "overall", "label": PROPERTY_LABELS["overall"], "n": n_assays}]
    seen: set[str] = set()
    for task_id in (*PREFERRED_TASK_ORDER, *sorted(counts)):
        if task_id in seen or task_id not in counts:
            continue
        seen.add(task_id)
        properties.append(
            {
                "id": task_id,
                "label": PROPERTY_LABELS.get(task_id, task_id.replace("_", " ")),
                "n": counts[task_id],
            }
        )
    return properties


def catalog_snapshot(rows: list[dict[str, str]], source: Path) -> dict[str, object]:
    counts = task_counts(rows)
    n_assays = len(rows)
    mutants = [_int_field(row, "n_mutants") for row in rows]
    lengths = [_int_field(row, "seq_len") for row in rows]
    mutant_values = [value for value in mutants if value is not None]
    length_values = [value for value in lengths if value is not None]
    properties = property_entries(counts, n_assays)
    return {
        "id": "venusmuthub",
        "benchmark": "VenusMutHub",
        "label": "VenusMutHub",
        "title": "VenusMutHub substitutions",
        "status": "catalog",
        "assays": n_assays,
        "n": n_assays,
        "source": "VenusMutHub assay_manifest.csv",
        "setting": "Zero-shot · substitutions",
        "description": (
            f"{n_assays} substitution assays across stability, activity, PPI binding, "
            "selectivity, and DTI binding. Same-backbone comparison of raw model "
            "scores and the full vrh recipe."
        ),
        "metrics": METRICS,
        "properties": properties,
        "pairs": [],
        "task_counts": {item["id"]: item["n"] for item in properties[1:]},
        "n_mutants": sum(mutant_values),
        "median_seq_len": statistics.median(length_values) if length_values else None,
        "paired_score_table": None,
        "manifest": _relative_to_repo(source),
    }


def benchmark_inputs(model_key: str) -> list[str]:
    """Foundation-model inputs; VRH adds evolutionary evidence in the UI."""
    key = model_key.lower()
    if key.startswith(("proteinmpnn", "pmpnn_")) or key == "esmif":
        return ["str"]
    if key.startswith(("prosst", "saprot", "s3f")) or key in {"protssn", "mifst"}:
        return ["seq", "str"]
    return ["seq"]


def score_triple(base: float, recipe: float) -> dict[str, float]:
    return {
        "base": base,
        "vrh": recipe,
        "delta": round(recipe - base, 3),
    }


def round_score(value: float) -> float:
    return round(float(value), 3)


_TWO_ARG_CMD = re.compile(r"\\[a-zA-Z]+(?:\[[^\]]*\])?\{([^{}]*)\}\{([^{}]*)\}")
_ONE_ARG_CMD = re.compile(r"\\[a-zA-Z]+(?:\[[^\]]*\])?\{([^{}]*)\}")
_TEX_NUMBER = re.compile(r"-?\d+\.\d+")


def strip_latex_commands(text: str) -> str:
    text = (text or "").replace(r"\_", "_")
    while True:
        two = list(_TWO_ARG_CMD.finditer(text))
        if two:
            match = two[-1]
            text = text[: match.start()] + match.group(2) + text[match.end() :]
            continue
        one = list(_ONE_ARG_CMD.finditer(text))
        if one:
            match = one[-1]
            text = text[: match.start()] + match.group(1) + text[match.end() :]
            continue
        break
    return re.sub(r"\s+", " ", text).strip()


def normalize_display_name(name: str) -> str:
    text = strip_latex_commands(name)
    return CANONICAL_DISPLAY.get(text, text)


def model_key_for(name: str) -> str:
    display = normalize_display_name(name)
    if display in DISPLAY_TO_KEY:
        return DISPLAY_TO_KEY[display]
    raise KeyError(f"no ProteinGym key for MutHub model {name!r} ({display!r})")


def display_name_for(name: str) -> str:
    display = normalize_display_name(name)
    return CANONICAL_DISPLAY.get(display, display)


def parse_tex_number(cell: str) -> float:
    matches = _TEX_NUMBER.findall(strip_latex_commands(cell))
    if not matches:
        raise ValueError(f"no number in tex cell: {cell!r}")
    return float(matches[-1])


def parse_vrh_flag(cell: str) -> str:
    text = cell.strip()
    if r"\checkmark" in text:
        return "vrh"
    if r"\times" in text or r"$\times$" in text:
        return "raw"
    raise ValueError(f"unrecognized VRH flag: {cell!r}")


def parse_tex_leaderboard(path: Path) -> dict[str, dict[str, object]]:
    """Return {key: {display, readout, raw: {prop: score}, vrh: {prop: score}}}."""
    rows: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not re.match(r"^\d+\s*&", stripped):
            continue
        body = stripped.rstrip("\\").strip()
        cells = [cell.strip() for cell in body.split("&")]
        if len(cells) < 10:
            raise SystemExit(f"expected 10 tex columns, got {len(cells)}: {stripped}")
        display = display_name_for(cells[1])
        key = model_key_for(cells[1])
        readout = re.sub(r"\\[a-zA-Z]+", "", cells[2]).strip()
        side = parse_vrh_flag(cells[3])
        scores = {
            prop: parse_tex_number(cells[4 + index])
            for index, prop in enumerate(TEX_PROPERTY_ORDER)
        }
        item = rows.setdefault(
            key,
            {"display": display, "readout": readout, "raw": None, "vrh": None},
        )
        if item[side] is not None:
            raise SystemExit(f"duplicate {side} row for {key}")
        item[side] = scores
        if readout:
            item["readout"] = readout
    missing = [key for key, item in rows.items() if item["raw"] is None or item["vrh"] is None]
    if missing:
        raise SystemExit(f"tex table missing Raw/VRH pair for: {', '.join(sorted(missing))}")
    return rows


def read_fig4(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 59:
        raise SystemExit(f"expected 59 Fig.4 MutHub rows, got {len(rows)}")
    return rows


def pairs_from_paper(fig4_rows: list[dict[str, str]], tex_rows: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    pairs = []
    seen: set[str] = set()
    for row in fig4_rows:
        display = display_name_for(row["Model"])
        key = model_key_for(row["Model"])
        if key in seen:
            raise SystemExit(f"duplicate Fig.4 model key {key}")
        seen.add(key)
        if key not in tex_rows:
            raise SystemExit(f"Fig.4 model {display} ({key}) missing from tex table")
        tex = tex_rows[key]
        raw_spearman = tex["raw"]
        vrh_spearman = tex["vrh"]
        assert isinstance(raw_spearman, dict) and isinstance(vrh_spearman, dict)
        spearman_properties = {
            prop: score_triple(float(raw_spearman[prop]), float(vrh_spearman[prop]))
            for prop in TEX_PROPERTY_ORDER
        }
        metrics = {
            "spearman": dict(spearman_properties["overall"]),
        }
        properties_by_metric: dict[str, dict[str, dict[str, float]]] = {
            "spearman": spearman_properties,
        }
        for metric_id, column in COMPARE_METRICS.items():
            if metric_id == "spearman":
                continue
            base = round_score(float(row[f"{column}_Raw"]))
            recipe = round_score(float(row[f"{column}_REM2"]))
            triple = score_triple(base, recipe)
            metrics[metric_id] = triple
            properties_by_metric[metric_id] = {"overall": dict(triple)}
        pairs.append(
            {
                "key": key,
                "family": display,
                "base_name": display,
                "enhanced_name": f"{display}, vrh",
                "vrh_name": f"{display}, vrh",
                "notes": tex.get("readout") or "",
                "metrics": metrics,
                "properties": spearman_properties,
                "properties_by_metric": properties_by_metric,
                "inputs": benchmark_inputs(key),
            }
        )
    extra = sorted(set(tex_rows) - seen)
    if extra:
        raise SystemExit(f"tex table has models missing from Fig.4: {', '.join(extra)}")
    if len(pairs) != 59:
        raise SystemExit(f"expected 59 MutHub pairs, got {len(pairs)}")
    return pairs


def vrh_snapshot(rows: list[dict[str, str]], pairs: list[dict[str, object]], sources: list[Path]) -> dict[str, object]:
    catalog = catalog_snapshot(rows, MANIFEST)
    catalog["status"] = "ready"
    catalog["pairs"] = pairs
    catalog["source"] = "; ".join(_relative_to_repo(path) for path in sources)
    catalog["paired_score_table"] = catalog["source"]
    catalog["description"] = (
        f"{len(rows)} substitution assays across stability, activity, PPI binding, "
        "selectivity, and DTI binding. Same-backbone comparison of raw model "
        "scores and the full vrh recipe."
    )
    return catalog


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--fig4", type=Path, default=FIG4_CSV)
    parser.add_argument("--tex", type=Path, default=VMH_TEX)
    parser.add_argument("--out", type=Path, default=VRH_PATH)
    parser.add_argument("--catalog-out", type=Path, default=CATALOG_PATH)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if not rows:
        raise SystemExit(f"empty manifest: {args.manifest}")

    if not args.fig4.is_file() or not args.tex.is_file():
        payload = catalog_snapshot(rows, args.manifest)
        write_json(args.catalog_out, payload)
        print(f"wrote catalog snapshot ({payload['n']} assays) to {args.catalog_out}")
        print("paper score tables: missing")
        return

    fig4_rows = read_fig4(args.fig4)
    tex_rows = parse_tex_leaderboard(args.tex)
    pairs = pairs_from_paper(fig4_rows, tex_rows)
    payload = vrh_snapshot(rows, pairs, [args.fig4, args.tex])
    write_json(args.out, payload)
    write_json(args.catalog_out, catalog_snapshot(rows, args.manifest))
    print(f"wrote ready snapshot ({len(pairs)} pairs) to {args.out}")
    print(f"paper tables: {args.fig4} ; {args.tex}")


if __name__ == "__main__":
    main()
