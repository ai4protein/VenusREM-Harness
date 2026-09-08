import argparse
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

BASELINES = {
    "ESM-2 650M": {
        "vanilla": ("result/baseline_esm2_vanilla", "ESM2-650M"),
        "venusrem2": ("result/baseline_esm2_venusrem2", "ESM2-650M-VenusREM2"),
        "venusrem2_ccd": ("result/baseline_esm2_venusrem2_ccd", "ESM2-650M-VenusREM2CCD"),
    },
    "ESM-1v (5avg)": {
        "vanilla": ("result/baseline_esm1v_vanilla", "ESM1v-Ensemble"),
        "venusrem2": ("result/baseline_esm1v_venusrem2", "ESM1v-Ensemble-VenusREM2"),
        "venusrem2_ccd": ("result/baseline_esm1v_venusrem2_ccd", "ESM1v-Ensemble-VenusREM2CCD"),
    },
    "SaProt 650M": {
        "vanilla": ("result/baseline_saprot_vanilla", "SaProt-650M"),
        "venusrem2": ("result/baseline_saprot_venusrem2", "SaProt-650M-VenusREM2"),
        "venusrem2_ccd": ("result/baseline_saprot_venusrem2_ccd", "SaProt-650M-VenusREM2CCD"),
    },
}


def load_summary(result_dir, score_col):
    path = os.path.join(result_dir, "summary_performance.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if score_col not in df.columns:
        for col in df.columns:
            if col != "protein":
                score_col = col
                break
    return df[["protein", score_col]].rename(columns={score_col: "spearman"})


def compute_mean_spearman(df):
    if df is None or df.empty:
        return float("nan")
    return df["spearman"].mean()


def main():
    parser = argparse.ArgumentParser(description="Compare baseline PLMs with VenusREM2 enhancement")
    parser.add_argument("--base_dir", type=str, default=".", help="Project root directory")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path for detailed results")
    args = parser.parse_args()

    os.chdir(args.base_dir)

    print()
    print("=" * 85)
    print("  Baseline PLM + VenusREM2 Enhancement Comparison (ProteinGym, 217 DMS)")
    print("=" * 85)
    print()
    print(f"{'Baseline':<16} | {'Vanilla':>8} | {'+VenusREM2':>8} | {'+VenusREM2+CCD':>11} | {'Δ(VenusREM2)':>9} | {'Δ(CCD)':>9}")
    print("-" * 85)

    all_results = []
    for baseline_name, configs in BASELINES.items():
        scores = {}
        for config_key, (result_dir, score_col) in configs.items():
            df = load_summary(result_dir, score_col)
            scores[config_key] = compute_mean_spearman(df)

        vanilla = scores.get("vanilla", float("nan"))
        venusrem2 = scores.get("venusrem2", float("nan"))
        venusrem2_ccd = scores.get("venusrem2_ccd", float("nan"))
        delta_venusrem2 = (
            venusrem2 - vanilla if not (pd.isna(venusrem2) or pd.isna(vanilla)) else float("nan")
        )
        delta_ccd = venusrem2_ccd - vanilla if not (pd.isna(venusrem2_ccd) or pd.isna(vanilla)) else float("nan")

        def fmt(v):
            return f"{v:.4f}" if not pd.isna(v) else "  N/A  "

        def fmt_delta(v):
            if pd.isna(v):
                return "   N/A  "
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.4f}"

        print(
            f"{baseline_name:<16} | {fmt(vanilla):>8} | {fmt(venusrem2):>8} | "
            f"{fmt(venusrem2_ccd):>11} | {fmt_delta(delta_venusrem2):>9} | {fmt_delta(delta_ccd):>9}"
        )

        all_results.append({
            "baseline": baseline_name,
            "vanilla": vanilla,
            "venusrem2": venusrem2,
            "venusrem2_ccd": venusrem2_ccd,
            "delta_venusrem2": delta_venusrem2,
            "delta_ccd": delta_ccd,
        })

    print("-" * 85)
    print()

    if args.output:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

    # Per-protein detail: merge all available results
    merged = None
    for baseline_name, configs in BASELINES.items():
        for config_key, (result_dir, score_col) in configs.items():
            df = load_summary(result_dir, score_col)
            if df is None:
                continue
            col_name = f"{baseline_name}_{config_key}"
            df = df.rename(columns={"spearman": col_name})
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on="protein", how="outer")

    if merged is not None and args.output:
        detail_path = args.output.replace(".csv", "_per_protein.csv")
        merged.to_csv(detail_path, index=False)
        print(f"Per-protein details saved to {detail_path}")


if __name__ == "__main__":
    main()
