#!/usr/bin/env python3
"""Rebuild aggregate tables, paired deltas, confidence intervals, and charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "prediction_hit",
    "target_rank",
    "candidate_universe_size",
    "prompt_tokens",
    "capacity_bytes",
    "ready_cache_ttft_ms",
    "request_e2e_ms",
    "policy_ms",
    "current_access_ttft_ms",
    "current_access_request_ms",
    "speculative_population_ms",
    "available_overlap_ms",
    "exposed_population_ms",
    "end_to_end_ttft_ms",
    "end_to_end_request_ms",
    "prefetch_bytes_scheduled",
    "prefetch_bytes_completed",
    "prefetch_completion_fraction",
    "shadow_residency_before_prefetch",
    "shadow_residency_hit",
    "shadow_current_prefetch_hit",
    "shadow_cache_bytes_after",
    "lmcache_retrieved_tokens_delta",
    "lmcache_stored_tokens_delta",
    "vllm_prefix_cache_hit_tokens_delta",
    "online_updated",
    "population_request_count",
    "population_prompt_tokens",
    "predicted_count",
    "completed_prefetch_count",
    "current_insert_eviction_count",
    "prefetch_eviction_count",
    "target_insert_eviction_count",
    "retry_count",
    "prefetch_completed_before_demand",
    "request_success",
]


def load_events(root: Path) -> pd.DataFrame:
    rows = []
    for marker in sorted(root.glob("arms/**/COMPLETE.json")):
        info = json.loads(marker.read_text())
        path = root / info["events_path"]
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows:
        raise SystemExit("No completed event files found")
    return pd.DataFrame(rows)


def clustered_ci(frame: pd.DataFrame, metric: str, seed: int = 42, iterations: int = 2000):
    documents = frame.document_id.dropna().unique()
    if len(documents) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    groups = {document: frame.loc[frame.document_id.eq(document), metric].dropna().to_numpy() for document in documents}
    for _ in range(iterations):
        sampled = rng.choice(documents, size=len(documents), replace=True)
        combined = np.concatenate([groups[document] for document in sampled if len(groups[document])])
        if len(combined):
            values.append(float(np.mean(combined)))
    return tuple(np.percentile(values, [2.5, 97.5])) if values else (np.nan, np.nan)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    out = root / "analysis"
    charts = out / "charts"
    out.mkdir(exist_ok=True)
    charts.mkdir(exist_ok=True)
    raw = load_events(root)
    raw.to_csv(out / "all_events_flat.csv", index=False)

    available = [metric for metric in METRICS if metric in raw]
    summaries = []
    for (policy, k, repetition), block in raw.groupby(["policy", "k", "repetition"]):
        for query_type in ["all", *sorted(block.query_type.unique())]:
            part = block if query_type == "all" else block[block.query_type.eq(query_type)]
            row = {"policy": policy, "k": k, "repetition": repetition, "query_type": query_type, "events": len(part)}
            for metric in available:
                values = pd.to_numeric(part[metric], errors="coerce").dropna()
                row[f"mean_{metric}"] = values.mean() if len(values) else np.nan
                row[f"median_{metric}"] = values.median() if len(values) else np.nan
                row[f"std_{metric}"] = values.std(ddof=1) if len(values) > 1 else np.nan
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "summary_all_metrics.csv", index=False)

    comparisons = []
    baselines = ["cosine", "no_prefetch", "graph_fixed", "adaptive_offline_global"]
    keys = ["repetition", "k", "event_id", "document_id", "query_type"]
    for policy in sorted(raw.policy.unique()):
        for baseline in baselines:
            if policy == baseline or baseline not in set(raw.policy):
                continue
            left = raw[raw.policy.eq(policy)][keys + available]
            right = raw[raw.policy.eq(baseline)][keys + available]
            paired = left.merge(right, on=keys, suffixes=("_left", "_right"))
            if paired.empty:
                continue
            for k, block in paired.groupby("k"):
                for query_type in ["all", *sorted(block.query_type.unique())]:
                    part = block if query_type == "all" else block[block.query_type.eq(query_type)]
                    for metric in available:
                        delta_name = f"delta_{metric}"
                        part = part.copy()
                        part[delta_name] = pd.to_numeric(part[f"{metric}_left"], errors="coerce") - pd.to_numeric(part[f"{metric}_right"], errors="coerce")
                        valid = part.dropna(subset=[delta_name])
                        if valid.empty:
                            continue
                        low, high = clustered_ci(valid, delta_name, iterations=args.bootstrap_iterations)
                        comparisons.append({
                            "policy": policy,
                            "baseline": baseline,
                            "k": k,
                            "query_type": query_type,
                            "metric": metric,
                            "pairs": len(valid),
                            "mean_delta": valid[delta_name].mean(),
                            "median_delta": valid[delta_name].median(),
                            "cluster_bootstrap_ci_low": low,
                            "cluster_bootstrap_ci_high": high,
                        })
    deltas = pd.DataFrame(comparisons)
    deltas.to_csv(out / "paired_policy_deltas.csv", index=False)

    cold = summary[(summary.policy == "no_prefetch") & (summary.query_type == "all")]
    diagnostics = {
        "cold_k_ttft_range_ms": float(cold.mean_ready_cache_ttft_ms.max() - cold.mean_ready_cache_ttft_ms.min()) if not cold.empty else None,
        "cold_k_end_to_end_range_ms": float(cold.mean_end_to_end_ttft_ms.max() - cold.mean_end_to_end_ttft_ms.min()) if not cold.empty else None,
        "note": "Large no-prefetch variation across K is an order/resource diagnostic because K does not alter its prompt.",
    }
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="talk")
        headline = summary[summary.query_type.eq("all")].groupby(["policy", "k"], as_index=False).mean(numeric_only=True)
        plots = [
            ("mean_prediction_hit", "Prediction hit rate", True),
            ("mean_ready_cache_ttft_ms", "Ready-cache TTFT (ms)", False),
            ("mean_end_to_end_ttft_ms", "End-to-end TTFT (ms)", False),
            ("mean_speculative_population_ms", "Speculative population cost (ms)", False),
            ("mean_prefetch_bytes_completed", "Completed KV bytes", False),
            ("mean_lmcache_retrieved_tokens_delta", "LMCache retrieved tokens", False),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))
        for ax, (metric, title, percent) in zip(axes.flat, plots):
            if metric not in headline:
                ax.set_visible(False)
                continue
            for policy, block in headline.groupby("policy"):
                values = block[metric] * (100 if percent else 1)
                ax.plot(block.k, values, marker="o", label=policy)
            ax.set_title(title)
            ax.set_xlabel("K")
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(charts / "00_headline_dashboard.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        (out / "chart_warning.txt").write_text(str(exc), encoding="utf-8")
    print(f"Analysis written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
