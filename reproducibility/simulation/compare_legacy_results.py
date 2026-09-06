#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LEGACY_TO_REPAIRED = {
    "lru_sim_hit": ("no_prefetch", "legacy_lru"),
    "cos_sim_hit": ("cosine", "cosine"),
    "grp_sim_hit": ("graph_fixed", "graph_fixed"),
    # The old adaptive was query-type-conditioned, so this is the closest
    # methodological mapping—not the new deployable online policy.
    "adp_sim_hit": (
        "adaptive_offline_query_type_oracle",
        "legacy_query_type_adaptive",
    ),
}


def aggregate_legacy(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"config_k", "type"} | set(LEGACY_TO_REPAIRED)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"legacy CSV is missing columns: {sorted(missing)}")
    rows = []
    for query_type, subset in [("all", frame), *frame.groupby("type")]:
        for k, group in subset.groupby("config_k"):
            for column, (repaired_policy, legacy_label) in LEGACY_TO_REPAIRED.items():
                rows.append(
                    {
                        "k": int(k),
                        "query_type": str(query_type),
                        "repaired_policy": repaired_policy,
                        "legacy_policy": legacy_label,
                        "legacy_events": len(group),
                        "legacy_residency_like_rate": float(group[column].mean()),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_repaired(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query_type, subset in [("all", frame), *frame.groupby("query_type")]:
        for (k, policy), group in subset.groupby(["k", "policy"]):
            rows.append(
                {
                    "k": int(k),
                    "query_type": str(query_type),
                    "repaired_policy": policy,
                    "repaired_events": len(group),
                    "repaired_prediction_recall": float(group.prediction_hit.mean()),
                    "repaired_residency_rate": float(group.residency_hit.mean()),
                    "repaired_current_prefetch_hit_rate": float(
                        group.current_prefetch_hit.mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align historical simulator hit columns with repaired outcomes"
    )
    parser.add_argument("legacy_csv")
    parser.add_argument("repaired_output_dir")
    parser.add_argument(
        "--output", default="legacy_vs_repaired_comparison.csv"
    )
    args = parser.parse_args()

    legacy = aggregate_legacy(pd.read_csv(args.legacy_csv))
    result_paths = sorted(Path(args.repaired_output_dir).glob("results_*.csv"))
    if not result_paths:
        raise ValueError("no repaired results_*.csv files found")
    repaired = aggregate_repaired(
        pd.concat([pd.read_csv(path) for path in result_paths], ignore_index=True)
    )
    comparison = legacy.merge(
        repaired,
        on=["k", "query_type", "repaired_policy"],
        how="outer",
        validate="one_to_one",
    )
    comparison["residency_rate_delta_repaired_minus_legacy"] = (
        comparison.repaired_residency_rate
        - comparison.legacy_residency_like_rate
    )
    comparison["comparison_warning"] = (
        "Trend comparison only: repaired targets are held-out/document-safe; "
        "legacy sim-hit and repaired residency are the closest fields but the "
        "protocols are not numerically interchangeable."
    )
    comparison.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(comparison)} rows)")


if __name__ == "__main__":
    main()

