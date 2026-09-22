#!/usr/bin/env python3
"""Rebuild dashboard leaderboard snapshots for the current 71 configurations.

Spearman category scores come from the paper category tables. The other four
metrics are overall-only and come from the checked 71-configuration table.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from vrh.dashboard.leaderboard import _benchmark_inputs


REPO = Path(__file__).resolve().parents[2]
CANONICAL = (
    REPO
    / "docs/audits/experiment_cleanup_20260919/alignment_71/canonical_multimetric_current.csv"
)
MANIFEST = (
    REPO
    / "docs/audits/experiment_cleanup_20260919/alignment_71/configuration_manifest_71.csv"
)
TABLES = REPO / "docs/0overleaf/tables"
DATA = REPO / "vrh/dashboard/data"

METRIC_COLUMNS = (
    ("spearman", "Spearman"),
    ("ndcg", "NDCG"),
    ("auc", "AUC"),
    ("mcc", "MCC"),
    ("top_recall", "Top_recall"),
)
DISPLAY_ALIASES = {
    "saprot (650m af2, mask)": "saprot (mask)",
    "saprot (650m af2, wt)": "saprot (wt)",
}
HUBS = {
    "pg": {
        "snapshot": DATA / "proteingym_rem2.json",
        "table": TABLES / "table_leaderboard_pg_category.tex",
        "dataset": "pg",
        "properties": (
            "overall",
            "activity",
            "binding",
            "expression",
            "organismal",
            "stability",
        ),
    },
    "vmh": {
        "snapshot": DATA / "venusmuthub_vrh.json",
        "table": TABLES / "table_leaderboard_vmh_category.tex",
        "dataset": "vmh",
        "properties": (
            "overall",
            "stability",
            "activity",
            "ppi_binding",
            "selectivity",
            "dti_binding",
        ),
    },
    "viro": {
        "snapshot": DATA / "venusvirohub_rem2.json",
        "table": TABLES / "table_leaderboard_vvh_category.tex",
        "dataset": "viro",
        "properties": (
            "overall",
            "fitness",
            "activity",
            "expression",
            "cell_entry",
            "binding",
            "stability",
            "immune_escape",
        ),
    },
}


def _unwrap(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\textcolor\{[^{}]*\}\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", text)
    return text


def _cell(text: str) -> str:
    cleaned = _unwrap(text.strip())
    cleaned = cleaned.replace(r"\_", "_")
    cleaned = cleaned.replace(r"\checkmark", "vrh")
    cleaned = cleaned.replace(r"$\times$", "raw")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _score_triple(base: float, recipe: float) -> dict[str, float]:
    rounded_base = round(float(base), 3)
    rounded_recipe = round(float(recipe), 3)
    return {
        "base": rounded_base,
        "vrh": rounded_recipe,
        "delta": round(rounded_recipe - rounded_base, 3),
    }


def _table_rows(path: Path, properties: tuple[str, ...]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not re.match(r"\d+\s*&", stripped):
            continue
        body = stripped[:-2].strip() if stripped.endswith(r"\\") else stripped
        cells = [_cell(part) for part in body.split("&")]
        if len(cells) != 4 + len(properties):
            raise SystemExit(f"{path.name}: expected {4 + len(properties)} cells, got {len(cells)} in {stripped}")
        model = cells[1]
        variant = cells[3]
        if variant not in {"raw", "vrh"}:
            raise SystemExit(f"{path.name}: unknown variant {variant!r} for {model}")
        scores = {key: float(cells[4 + index]) for index, key in enumerate(properties)}
        grouped.setdefault(model, {})[variant] = scores
    return grouped


def _manifest() -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    if len(rows) != 71:
        raise SystemExit(f"expected 71 manifest rows, got {len(rows)}")
    by_display = {}
    for row in rows:
        key = row["display"].strip().lower()
        if key in by_display:
            raise SystemExit(f"duplicate display name {row['display']}")
        by_display[key] = row
    return by_display


def _canonical() -> dict[tuple[str, str], dict[str, str]]:
    rows = list(csv.DictReader(CANONICAL.open(encoding="utf-8", newline="")))
    indexed = {(row["dataset"], row["model"]): row for row in rows}
    if len(indexed) != 213:
        raise SystemExit(f"expected 213 canonical rows, got {len(indexed)}")
    return indexed


def _previous_pairs() -> dict[str, dict]:
    saved: dict[str, dict] = {}
    for spec in HUBS.values():
        payload = json.loads(spec["snapshot"].read_text(encoding="utf-8"))
        for pair in payload.get("pairs") or []:
            current = saved.setdefault(pair["key"], {})
            if pair.get("notes") and not current.get("notes"):
                current["notes"] = pair["notes"]
            if pair.get("inputs") and not current.get("inputs"):
                current["inputs"] = list(pair["inputs"])
    return saved


def _build_pairs(hub: str, previous: dict[str, dict], manifest: dict[str, dict[str, str]], canonical: dict) -> list[dict]:
    spec = HUBS[hub]
    readout = _readouts(spec["table"])
    table = _table_rows(spec["table"], spec["properties"])
    if len(table) != 71:
        raise SystemExit(f"{hub}: expected 71 models in the category table, got {len(table)}")
    pairs = []
    for display, sides in table.items():
        row = manifest.get(DISPLAY_ALIASES.get(display.lower(), display.lower()))
        if row is None:
            raise SystemExit(f"{hub}: no manifest row for {display!r}")
        if set(sides) != {"raw", "vrh"}:
            raise SystemExit(f"{hub}: {display} is missing raw/vrh ({sorted(sides)})")
        model = row["model"]
        source = canonical.get((spec["dataset"], model))
        if source is None:
            raise SystemExit(f"{hub}: no canonical row for {model}")
        properties = {
            key: _score_triple(sides["raw"][key], sides["vrh"][key])
            for key in spec["properties"]
        }
        spearman = properties["overall"]
        canonical_spearman = _score_triple(source["Spearman_Raw"], source["Spearman_REM2"])
        if canonical_spearman != spearman:
            raise SystemExit(
                f"{hub} {model} Spearman {canonical_spearman} != paper table {spearman}"
            )
        metrics = {"spearman": spearman}
        by_metric = {"spearman": properties}
        for metric_id, column in METRIC_COLUMNS:
            if metric_id == "spearman":
                continue
            overall = _score_triple(source[f"{column}_Raw"], source[f"{column}_REM2"])
            metrics[metric_id] = overall
            by_metric[metric_id] = {"overall": overall}
        prior = previous.get(model, {})
        label = display
        enhanced_name = label if model == "prosst_ensemble" else f"{label}, VRH"
        pairs.append(
            {
                "key": model,
                "family": label,
                "base_name": label,
                "enhanced_name": enhanced_name,
                "vrh_name": enhanced_name,
                "notes": prior.get("notes") or readout[label],
                "inputs": list(prior.get("inputs") or _benchmark_inputs(model)),
                "base": spearman["base"],
                "enhanced": spearman["vrh"],
                "delta": spearman["delta"],
                "metrics": metrics,
                "properties": properties,
                "properties_by_metric": by_metric,
            }
        )
    pairs.sort(key=lambda item: (-item["enhanced"], item["key"]))
    if len({item["key"] for item in pairs}) != 71:
        raise SystemExit(f"{hub}: expected 71 unique keys")
    return pairs


def _readouts(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not re.match(r"\d+\s*&", stripped):
            continue
        body = stripped[:-2].strip() if stripped.endswith(r"\\") else stripped
        cells = [_cell(part) for part in body.split("&")]
        found.setdefault(cells[1], cells[2])
    return found


def main() -> None:
    manifest = _manifest()
    canonical = _canonical()
    previous = _previous_pairs()
    for hub, spec in HUBS.items():
        payload = json.loads(spec["snapshot"].read_text(encoding="utf-8"))
        payload["pairs"] = _build_pairs(hub, previous, manifest, canonical)
        if hub == "vmh" and "VenusMutHub paper tables" not in str(payload.get("source") or ""):
            payload["source"] = "VenusMutHub paper tables · 71 configurations · 905 assays"
        spec["snapshot"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        ensemble = next(row for row in payload["pairs"] if row["key"] == "prosst_ensemble")
        print(
            hub,
            len(payload["pairs"]),
            ensemble["key"],
            ensemble["base"],
            ensemble["enhanced"],
            ensemble["metrics"]["ndcg"],
        )


if __name__ == "__main__":
    main()
