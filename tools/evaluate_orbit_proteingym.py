import argparse
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs VenusREM-Orbit summary performance on ProteinGym."
    )
    parser.add_argument(
        "--baseline_summary",
        required=True,
        help="Path to baseline summary_performance.csv",
    )
    parser.add_argument(
        "--orbit_summary",
        required=True,
        help="Path to Orbit summary_performance.csv",
    )
    parser.add_argument(
        "--baseline_col",
        default="ProSST-2048",
        help="Column name in baseline summary.",
    )
    parser.add_argument(
        "--orbit_col",
        default="VenusREM-Orbit",
        help="Column name in Orbit summary.",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        help="Optional output CSV path for merged comparison.",
    )
    args = parser.parse_args()

    baseline_df = pd.read_csv(args.baseline_summary)
    orbit_df = pd.read_csv(args.orbit_summary)
    merged = baseline_df[["protein", args.baseline_col]].merge(
        orbit_df[["protein", args.orbit_col]], on="protein", how="inner"
    )
    merged["delta"] = merged[args.orbit_col] - merged[args.baseline_col]
    merged = merged.sort_values(by="delta", ascending=False)

    print(f"Matched proteins: {len(merged)}")
    print(
        f"Average {args.baseline_col}: {merged[args.baseline_col].mean():.4f}, "
        f"Average {args.orbit_col}: {merged[args.orbit_col].mean():.4f}, "
        f"Delta: {merged['delta'].mean():.4f}"
    )
    print("Top 10 improvements:")
    print(merged.head(10).to_string(index=False))

    if args.out_file:
        out_dir = os.path.dirname(args.out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        merged.to_csv(args.out_file, index=False)
        print(f"Saved comparison to: {args.out_file}")


if __name__ == "__main__":
    main()
