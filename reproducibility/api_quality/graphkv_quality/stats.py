"""Aggregation and paired question-level bootstrap confidence intervals."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def mean(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def paired_bootstrap(values, *, seed: int = 42, resamples: int = 10_000) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) == 0:
        return {"mean_delta": None, "ci_low": None, "ci_high": None, "n": 0}
    if len(values) == 1:
        value = float(values[0])
        return {"mean_delta": value, "ci_low": None, "ci_high": None, "n": 1}
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(values), size=(int(resamples), len(values)))
    estimates = values[draws].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean_delta": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(len(values)),
    }


def summarise(events: list[dict], *, seed: int = 42) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for event in events:
        if event.get("status") == "complete":
            grouped[(int(event["k"]), event["policy"])].append(event)

    summary = []
    for (k, policy), rows in sorted(grouped.items()):
        fresh_latencies = [row.get("api_latency_ms") for row in rows if not row.get("response_cache_hit")]
        summary.append(
            {
                "k": k,
                "policy": policy,
                "questions": len(rows),
                "mean_em": mean(row.get("em") for row in rows),
                "mean_f1": mean(row.get("f1") for row in rows),
                "mean_support_recall": mean(row.get("support_recall") for row in rows),
                "mean_selection_support_recall_before_budget": mean(
                    row.get("selection_support_recall_before_budget") for row in rows
                ),
                "target_hit_rate": mean(row.get("target_hit") for row in rows),
                "selection_target_hit_rate_before_budget": mean(
                    row.get("selection_target_hit_before_budget") for row in rows
                ),
                "mean_reference_context_tokens": mean(row.get("reference_context_tokens") for row in rows),
                "mean_provider_input_tokens": mean(row.get("provider_input_tokens") for row in rows),
                "mean_provider_output_tokens": mean(row.get("provider_output_tokens") for row in rows),
                "mean_api_latency_ms_fresh_calls": mean(fresh_latencies),
                "fresh_api_calls": len(fresh_latencies),
                "response_cache_hits": sum(bool(row.get("response_cache_hit")) for row in rows),
                "mean_selected_nodes": mean(row.get("selected_nodes") for row in rows),
                "mean_fully_delivered_nodes": mean(row.get("fully_delivered_nodes") for row in rows),
                "mean_candidate_universe": mean(row.get("candidate_universe") for row in rows),
                "online_update_rate": mean(row.get("online_updated") for row in rows)
                if policy == "adaptive_online_warm"
                else None,
            }
        )

    by_key = {(int(row["k"]), row["policy"], row["question_id"]): row for row in events if row.get("status") == "complete"}
    policies = sorted({row["policy"] for row in events if row.get("policy") != "cosine"})
    k_values = sorted({int(row["k"]) for row in events if row.get("status") == "complete"})
    deltas = []
    for k in k_values:
        cosine_ids = {question_id for kk, policy, question_id in by_key if kk == k and policy == "cosine"}
        for policy in policies:
            policy_ids = {question_id for kk, name, question_id in by_key if kk == k and name == policy}
            paired_ids = sorted(cosine_ids & policy_ids)
            for metric in ("em", "f1", "support_recall", "target_hit"):
                values = [by_key[(k, policy, qid)][metric] - by_key[(k, "cosine", qid)][metric] for qid in paired_ids]
                result = paired_bootstrap(values, seed=seed + k + sum(ord(c) for c in policy + metric))
                deltas.append({"k": k, "policy": policy, "baseline": "cosine", "metric": metric, **result})
    return summary, deltas
