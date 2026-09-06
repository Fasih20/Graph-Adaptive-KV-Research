from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_result_csvs(paths) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["policy", "k"], as_index=False)
        .agg(
            events=("event_id", "count"),
            prediction_recall=("prediction_hit", "mean"),
            residency_hit_rate=("residency_hit", "mean"),
            current_prefetch_hit_rate=("current_prefetch_hit", "mean"),
            candidate_coverage=("candidate_coverage", "mean"),
            mean_candidate_universe=("candidate_universe_size", "mean"),
            mean_proxy_ms=("total_proxy_ms", "mean"),
            mean_prefetch_bytes=("prefetch_bytes_completed", "mean"),
            online_update_rate=("online_updated", "mean"),
        )
        .sort_values(["k", "policy"])
    )


def summarize_by_query_type(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["policy", "k", "query_type"], as_index=False)
        .agg(
            events=("event_id", "count"),
            prediction_recall=("prediction_hit", "mean"),
            residency_hit_rate=("residency_hit", "mean"),
            current_prefetch_hit_rate=("current_prefetch_hit", "mean"),
            mean_proxy_ms=("total_proxy_ms", "mean"),
            mean_prefetch_bytes=("prefetch_bytes_completed", "mean"),
            online_update_rate=("online_updated", "mean"),
        )
        .sort_values(["k", "query_type", "policy"])
    )


def paired_policy_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    policies = sorted(frame.policy.unique())
    for k in sorted(frame.k.unique()):
        subset = frame[frame.k.eq(k)]
        for index, policy_a in enumerate(policies):
            for policy_b in policies[index + 1 :]:
                left = subset[subset.policy.eq(policy_a)].set_index("event_id")
                right = subset[subset.policy.eq(policy_b)].set_index("event_id")
                paired = left.join(right, lsuffix="_a", rsuffix="_b", how="inner")
                if paired.empty:
                    continue
                a_hit = paired.prediction_hit_a.astype(bool)
                b_hit = paired.prediction_hit_b.astype(bool)
                rows.append(
                    {
                        "k": int(k),
                        "policy_a": policy_a,
                        "policy_b": policy_b,
                        "paired_events": len(paired),
                        "a_prediction_wins": int((a_hit & ~b_hit).sum()),
                        "b_prediction_wins": int((~a_hit & b_hit).sum()),
                        "prediction_ties": int((a_hit == b_hit).sum()),
                        "prediction_recall_difference_a_minus_b": float(
                            paired.prediction_hit_a.mean() - paired.prediction_hit_b.mean()
                        ),
                        "residency_difference_a_minus_b": float(
                            paired.residency_hit_a.mean() - paired.residency_hit_b.mean()
                        ),
                        "proxy_ms_difference_a_minus_b": float(
                            paired.total_proxy_ms_a.mean() - paired.total_proxy_ms_b.mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    metric: str,
    clusters: str = "document_id",
    repetitions: int = 2000,
    seed: int = 42,
) -> dict:
    """Paired document bootstrap for a mean policy difference (A minus B)."""

    subset = frame[frame.policy.isin([policy_a, policy_b])]
    pivot = subset.pivot_table(
        index=[clusters, "event_id", "k"], columns="policy", values=metric, aggfunc="first"
    ).dropna(subset=[policy_a, policy_b])
    if pivot.empty:
        raise ValueError("no paired rows for requested comparison")
    pivot["difference"] = pivot[policy_a] - pivot[policy_b]
    cluster_means = pivot.groupby(level=clusters).difference.mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for i in range(repetitions):
        draws[i] = rng.choice(cluster_means, size=len(cluster_means), replace=True).mean()
    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "metric": metric,
        "difference": float(cluster_means.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(cluster_means)),
        "bootstrap_repetitions": int(repetitions),
    }


def write_analysis(frame: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summarize(frame).to_csv(output / "summary.csv", index=False)
    summarize_by_query_type(frame).to_csv(output / "summary_by_query_type.csv", index=False)
    paired_policy_outcomes(frame).to_csv(output / "paired_policy_outcomes.csv", index=False)
    metadata = {
        "rows": int(len(frame)),
        "policies": sorted(frame.policy.unique().tolist()),
        "k_values": sorted(map(int, frame.k.unique().tolist())),
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
