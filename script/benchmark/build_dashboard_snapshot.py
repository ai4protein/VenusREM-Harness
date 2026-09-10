#!/usr/bin/env python3
"""Enrich and verify the packaged ProteinGym Raw/REM2 dashboard snapshot.

The workbook is an OOXML file.  This reader intentionally uses only the Python
standard library so release checks do not depend on pandas/openpyxl.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
PROPERTY_COLUMNS = {
    "activity": "Function_Activity",
    "binding": "Function_Binding",
    "expression": "Function_Expression",
    "organismal": "Function_OrganismalFitness",
    "stability": "Function_Stability",
}
COMPARE_METRICS = {
    "spearman": "Spearman",
    "ndcg": "NDCG",
    "auc": "AUC",
    "mcc": "MCC",
    "top_recall": "Top_recall",
}
METRIC_SHEETS = {
    "spearman": ("Spearman", "Average_Spearman"),
    "ndcg": ("NDCG", "Average_NDCG"),
    "auc": ("AUC", "Average_AUC"),
    "mcc": ("MCC", "Average_MCC"),
    "top_recall": ("Top_recall", "Average_Top_recall"),
}


def column_index(cell_ref: str) -> int:
    result = 0
    for char in re.match(r"[A-Z]+", cell_ref).group(0):
        result = result * 26 + ord(char) - 64
    return result - 1


def sheet_records(workbook: Path, sheet_name: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(workbook) as archive:
        shared = [
            "".join(node.text or "" for node in item.iter(MAIN_NS + "t"))
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))
        ]
        relationships = {
            item.attrib["Id"]: "xl/" + item.attrib["Target"]
            for item in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet_path = next(
            relationships[item.attrib[REL_ID]]
            for item in book.find(MAIN_NS + "sheets")
            if item.attrib["name"] == sheet_name
        )
        rows = []
        for row in ET.fromstring(archive.read(sheet_path)).iter(MAIN_NS + "row"):
            values: dict[int, str] = {}
            for cell in row.findall(MAIN_NS + "c"):
                value = cell.find(MAIN_NS + "v")
                if value is None:
                    continue
                text = shared[int(value.text)] if cell.attrib.get("t") == "s" else value.text
                values[column_index(cell.attrib["r"])] = text
            rows.append(values)
    header = [rows[0].get(index, "") for index in range(max(rows[0]) + 1)]
    return [{header[index]: value for index, value in row.items()} for row in rows[1:]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("rem2/dashboard/data/proteingym_rem2.json"),
    )
    args = parser.parse_args()

    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    compare = sheet_records(args.workbook, "Leaderboard_Compare")
    compare_index = {row["model_key"]: row for row in compare if row.get("model_key")}
    summary_indexes = {
        metric_key: {
            (row["Variant"], row["Model_name"]): row
            for row in sheet_records(args.workbook, sheet_name)
        }
        for metric_key, (sheet_name, _average_column) in METRIC_SHEETS.items()
    }

    # The paired aggregate sheet contains 59 pairs plus the standalone VenusREM row.
    if len(snapshot["pairs"]) != 59 or len(compare) != 60:
        raise SystemExit(
            f"expected 59 snapshot pairs and 60 compare rows, got "
            f"{len(snapshot['pairs'])} and {len(compare)}"
        )

    checked = 0
    checked_values = 0
    for pair in snapshot["pairs"]:
        comparison = compare_index[pair["key"]]
        for metric_key, column_prefix in COMPARE_METRICS.items():
            expected_metric = pair["metrics"][metric_key]
            for json_key, column_suffix in (("base", "raw"), ("rem2", "rem2"), ("delta", "delta")):
                actual_value = float(comparison[f"{column_prefix}_{column_suffix}"])
                expected_value = float(expected_metric[json_key])
                if abs(actual_value - expected_value) > 1e-9:
                    raise SystemExit(
                        f"{metric_key} {json_key} mismatch for {pair['key']}: "
                        f"{actual_value} != {expected_value}"
                    )
                checked_values += 1
        pair["properties_by_metric"] = {}
        for metric_key, (_sheet_name, average_column) in METRIC_SHEETS.items():
            raw = summary_indexes[metric_key][("Raw", pair["base_name"])]
            rem2 = summary_indexes[metric_key][("rem2", pair["rem2_name"])]
            expected = pair["metrics"][metric_key]
            actual = (float(raw[average_column]), float(rem2[average_column]))
            if actual != (float(expected["base"]), float(expected["rem2"])):
                raise SystemExit(f"{metric_key} mismatch for {pair['key']}: {actual} != {expected}")
            properties = {
                "overall": {"base": actual[0], "rem2": actual[1]},
                **{
                    key: {
                        "base": float(raw[column]),
                        "rem2": float(rem2[column]),
                    }
                    for key, column in PROPERTY_COLUMNS.items()
                }
            }
            for values in properties.values():
                values["delta"] = round(values["rem2"] - values["base"], 3)
            pair["properties_by_metric"][metric_key] = properties
        # Preserve the original field for older dashboard clients.
        pair["properties"] = pair["properties_by_metric"]["spearman"]
        checked += 1

    snapshot["properties"] = [
        {"id": "overall", "label": "Overall"},
        {"id": "activity", "label": "Activity"},
        {"id": "binding", "label": "Binding"},
        {"id": "expression", "label": "Expression"},
        {"id": "organismal", "label": "Organismal fitness"},
        {"id": "stability", "label": "Stability"},
    ]
    args.snapshot.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(
        f"verified {checked_values} aggregate cells and enriched "
        f"{checked * len(METRIC_SHEETS) * 6 * 2} property cells across "
        f"{checked} paired models from {args.workbook}"
    )


if __name__ == "__main__":
    main()
