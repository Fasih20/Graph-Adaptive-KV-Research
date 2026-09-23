#!/usr/bin/env python3
"""CPU-only, leakage-safe M/K sensitivity study for GraphKV revision v2.

This script consumes a workload produced by ``run_revision.py prepare``.  It
never starts vLLM or LMCache.  Offline weights are fitted on the training
documents, online step size is selected on the development documents, M is
selected from development recall, and the held-out test documents are reported
without using them for tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from adaptive_policies import TraceEvent
from revision_io import atomic
from revision_policy import Policy, edges, fit


POLICIES = (
    "cosine",
    "semantic_only",
    "structure",
    "two_hop",
    "offline",
    "online",
)


def load_prepared(root: Path):
    metadata = json.loads((root / "prepared.json").read_text())
    chunks = [
        json.loads(line)
        for line in (root / "workload/chunks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    similarity = np.load(root / "similarity.npy", allow_pickle=False)
    traces = {}
    integer_fields = {"event_id", "step_id", "current_chunk_id", "target_chunk_id"}
    for split in ("train", "dev", "test"):
        with (root / f"workload/trace_{split}.csv").open(newline="", encoding="utf-8") as handle:
            traces[split] = [
                TraceEvent(
                    **{
                        key: int(value) if key in integer_fields else value
                        for key, value in row.items()
                    }
                )
                for row in csv.DictReader(handle)
            ]
    return metadata, chunks, similarity, traces


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--top-m", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    command.add_argument("--k", type=int, nargs="+", default=[6, 10, 16])
    command.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    command.add_argument("--selection-policy", choices=POLICIES, default="offline")
    command.add_argument("--bootstrap-draws", type=int, default=5000)
    command.add_argument("--seed", type=int, default=43)
    return command


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def target_rank(policy_name: str, prediction, similarity: np.ndarray, current: int, target: int):
    if policy_name == "cosine":
        ranked = sorted(
            (candidate for candidate in range(len(similarity)) if candidate != current),
            key=lambda candidate: (-float(similarity[current, candidate]), candidate),
        )
    else:
        ranked = sorted(
            prediction.scores,
            key=lambda candidate: (-float(prediction.scores[candidate]), int(candidate)),
        )
    try:
        return ranked.index(int(target)) + 1
    except ValueError:
        return None


def evaluate_split(split, events, m, k, name, raw_structural, raw_semantic, similarity, fitted):
    graph = raw_semantic if name == "semantic_only" else raw_structural
    policy = Policy(name, graph, similarity, fitted["weights"], fitted["eta"])
    rows = []
    for event in events:
        prediction = policy.predict(event.current_chunk_id, k)
        rank = target_rank(
            name,
            prediction,
            similarity,
            event.current_chunk_id,
            event.target_chunk_id,
        )
        update = policy.observe(event.target_chunk_id, prediction)
        rows.append(
            {
                "split": split,
                "m": int(m),
                "k": int(k),
                "policy": name,
                "event_id": int(event.event_id),
                "session_id": event.session_id,
                "document_id": event.document_id,
                "query_type": event.query_type,
                "current_chunk_id": int(event.current_chunk_id),
                "target_chunk_id": int(event.target_chunk_id),
                "predicted_ids": json.dumps([int(value) for value in prediction.ids]),
                "prediction_hit": int(event.target_chunk_id in prediction.ids),
                "candidate_coverage": int(event.target_chunk_id in prediction.candidate_universe),
                "candidate_universe_size": int(len(prediction.candidate_universe)),
                "target_rank": rank,
                "online_updated": int(bool(update.get("updated", False))),
                "weights_before": json.dumps(update.get("weights_before")),
                "weights_after": json.dumps(update.get("weights_after")),
            }
        )
    return rows


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["split", "m", "k", "policy", "query_type"]
    parts = [events]
    combined = events.copy()
    combined["query_type"] = "all"
    parts.append(combined)
    frame = pd.concat(parts, ignore_index=True)
    summary = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            events=("event_id", "size"),
            prediction_recall=("prediction_hit", "mean"),
            candidate_coverage=("candidate_coverage", "mean"),
            mean_target_rank=("target_rank", "mean"),
            median_target_rank=("target_rank", "median"),
            mean_candidate_universe=("candidate_universe_size", "mean"),
            online_update_rate=("online_updated", "mean"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return summary


def cluster_interval(values_by_document, draws: int, rng: np.random.Generator):
    clusters = [np.asarray(values, dtype=np.float64) for values in values_by_document if len(values)]
    if len(clusters) < 2:
        return None, None
    estimates = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        sampled = np.concatenate([clusters[position] for position in selected])
        estimates[index] = sampled.mean()
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def paired_deltas(events: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    test = events[events.split.eq("test")]
    rows = []
    for (m, k, name), policy_rows in test.groupby(["m", "k", "policy"]):
        if name == "cosine":
            continue
        baseline = test[(test.m.eq(m)) & (test.k.eq(k)) & (test.policy.eq("cosine"))]
        left = policy_rows.set_index("event_id")
        right = baseline.set_index("event_id")
        paired = left[["document_id", "prediction_hit"]].join(
            right[["prediction_hit"]], how="inner", rsuffix="_baseline"
        )
        paired["delta"] = paired.prediction_hit - paired.prediction_hit_baseline
        clusters = [group.delta.to_numpy() for _, group in paired.groupby("document_id")]
        rng = np.random.default_rng(seed + 1009 * int(m) + 9173 * int(k) + sum(map(ord, name)))
        low, high = cluster_interval(clusters, draws, rng)
        rows.append(
            {
                "m": int(m),
                "k": int(k),
                "policy": name,
                "baseline": "cosine",
                "paired_events": int(len(paired)),
                "document_clusters": int(paired.document_id.nunique()),
                "mean_recall_delta": float(paired.delta.mean()),
                "cluster_bootstrap_ci_low": low,
                "cluster_bootstrap_ci_high": high,
            }
        )
    return pd.DataFrame(rows).sort_values(["k", "m", "policy"]).reset_index(drop=True)


def graph_density(raw, node_count: int):
    degrees = np.asarray([len(raw[index]) for index in range(node_count)], dtype=np.float64)
    return {
        "directed_edges": int(degrees.sum()),
        "mean_degree": float(degrees.mean()),
        "median_degree": float(np.median(degrees)),
        "p95_degree": float(np.quantile(degrees, 0.95)),
        "max_degree": int(degrees.max()),
    }


def choose_m(summary: pd.DataFrame, policy: str):
    dev = summary[
        summary.split.eq("dev")
        & summary.query_type.eq("all")
        & summary.policy.eq(policy)
    ]
    if dev.empty:
        raise ValueError(f"Selection policy {policy!r} was not evaluated")
    scores = (
        dev.groupby("m", as_index=False)
        .agg(mean_dev_recall=("prediction_recall", "mean"))
        .sort_values(["mean_dev_recall", "m"], ascending=[False, True])
    )
    selected = int(scores.iloc[0].m)
    return selected, scores


def main() -> int:
    args = parser().parse_args()
    if min(*args.top_m, *args.k, args.bootstrap_draws) < 1:
        raise ValueError("M, K, and bootstrap draws must be positive")
    if len(set(args.top_m)) != len(args.top_m) or len(set(args.k)) != len(args.k):
        raise ValueError("M and K lists must not contain duplicates")
    if not (args.output_dir / "prepared.json").exists():
        raise FileNotFoundError(
            "Run run_revision.py prepare in this output directory before the sensitivity study"
        )

    metadata, chunks, similarity, traces = load_prepared(args.output_dir)
    documents = [chunk["document_id"] for chunk in chunks]
    positions = [int(chunk["position"]) for chunk in chunks]

    all_rows = []
    density_rows = []
    fits = {}
    for m in args.top_m:
        logging.info("Building repaired document-aware graphs for M=%d", m)
        raw_structural = edges(similarity, documents, positions, m, structure=True)
        raw_semantic = edges(similarity, documents, positions, m, structure=False)
        fitted = fit(traces["train"], traces["dev"], raw_structural, similarity, args.k)
        fits[str(m)] = fitted
        density_rows.append({"m": m, "graph": "semantic_plus_structure", **graph_density(raw_structural, len(chunks))})
        density_rows.append({"m": m, "graph": "semantic_only", **graph_density(raw_semantic, len(chunks))})
        for k in args.k:
            for name in args.policies:
                for split in ("dev", "test"):
                    all_rows.extend(
                        evaluate_split(
                            split,
                            traces[split],
                            m,
                            k,
                            name,
                            raw_structural,
                            raw_semantic,
                            similarity,
                            fitted,
                        )
                    )

    events = pd.DataFrame(all_rows)
    summary = summarize(events)
    deltas = paired_deltas(events, args.bootstrap_draws, args.seed)
    selected_m, selection_scores = choose_m(summary, args.selection_policy)
    held_out = summary[
        summary.split.eq("test")
        & summary.query_type.eq("all")
        & summary.m.eq(selected_m)
    ].copy()

    result_dir = args.output_dir / "sensitivity"
    atomic_csv(events, result_dir / "event_predictions.csv")
    atomic_csv(summary, result_dir / "summary_by_m_k_policy.csv")
    atomic_csv(deltas, result_dir / "paired_recall_deltas_vs_cosine.csv")
    atomic_csv(pd.DataFrame(density_rows), result_dir / "graph_density.csv")
    atomic_csv(selection_scores, result_dir / "development_m_selection.csv")
    atomic_csv(held_out, result_dir / "held_out_test_at_selected_m.csv")
    atomic(result_dir / "fitted_policies.json", fits)
    atomic(
        result_dir / "protocol.json",
        {
            "model": metadata["spec"]["model"],
            "dataset": metadata["spec"]["dataset"],
            "prepared_spec": metadata["spec"],
            "top_m": args.top_m,
            "k": args.k,
            "policies": args.policies,
            "selection_policy": args.selection_policy,
            "selected_m_from_development_only": selected_m,
            "bootstrap_draws": args.bootstrap_draws,
            "test_documents_used_for_tuning": False,
            "online_feedback": "oracle target revealed only after each prediction",
            "scope": "CPU prediction-policy sensitivity; no latency, vLLM, LMCache, answer F1, or measured FLOPs",
        },
    )
    print("Development-selected M:", selected_m)
    print(held_out[["m", "k", "policy", "events", "prediction_recall", "candidate_coverage", "online_update_rate"]].to_string(index=False))
    print("Saved:", result_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
