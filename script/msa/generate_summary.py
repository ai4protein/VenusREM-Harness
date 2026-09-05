#!/usr/bin/env python3
"""Generate job_statistics_summary.csv from per-bitscore _final.outcfg files."""
import sys, os, csv
from ruamel.yaml import YAML

def generate_summary(protein_dir):
    protein_dir = protein_dir.rstrip("/")
    protein = os.path.basename(protein_dir)
    yaml = YAML()
    rows = []
    for b in ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"]:
        outcfg = os.path.join(protein_dir, f"{protein}_b{b}_final.outcfg")
        if not os.path.exists(outcfg):
            continue
        with open(outcfg) as f:
            cfg = yaml.load(f)
        rows.append({
            "prefix": f"{protein_dir}/{protein}_b{b}",
            "num_sequences": cfg.get("num_sequences", 0),
            "num_sites": cfg.get("num_sites", 0),
            "effective_sequences": cfg.get("effective_sequences", 0),
            "num_significant": cfg.get("effective_sequences", 0),
        })
    if not rows:
        return False
    out_csv = os.path.join(protein_dir, f"{protein}_job_statistics_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: generate_summary.py <protein_dir> [protein_dir2 ...]")
        sys.exit(1)
    for d in sys.argv[1:]:
        ok = generate_summary(d)
        name = os.path.basename(d)
        print(f"{'OK' if ok else 'SKIP'} {name}")
