from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .candidate_paths import enumerate_candidate_paths, rank_candidate_paths
from .models import TraceEvent


def _score_weights(events, raw_edges, weights, k_values) -> float:
    hits = 0
    total = 0
    weight_array = np.asarray(weights, dtype=np.float64)
    for event in events:
        paths = enumerate_candidate_paths(raw_edges, event.current_chunk_id)
        for k in k_values:
            ids, _, _ = rank_candidate_paths(paths, weight_array, int(k))
            hits += int(event.target_chunk_id in ids)
            total += 1
    return hits / total if total else 0.0


def fit_global_weights(
    train_events: Iterable[TraceEvent], raw_edges, k_values=(3, 8, 16), grid_step=0.05
) -> tuple[tuple[float, float], list[dict]]:
    """Fit only prediction recall on training traces; no latency-result leakage."""

    events = list(train_events)
    rows = []
    count = int(round(1.0 / grid_step))
    for i in range(count + 1):
        alpha = i / count
        weights = (alpha, 1.0 - alpha)
        rows.append({"alpha": alpha, "beta": 1.0 - alpha, "train_recall": _score_weights(events, raw_edges, weights, k_values)})
    best = max(rows, key=lambda row: (row["train_recall"], -abs(row["alpha"] - 0.7), -row["alpha"]))
    return (best["alpha"], best["beta"]), rows


def fit_query_type_oracle_weights(
    train_events: Iterable[TraceEvent], raw_edges, k_values=(3, 8, 16), grid_step=0.05
) -> dict[str, tuple[float, float]]:
    """Diagnostic oracle. Do not present it as deployable unless type is known."""

    grouped = defaultdict(list)
    for event in train_events:
        grouped[event.query_type].append(event)
    return {
        query_type: fit_global_weights(events, raw_edges, k_values, grid_step)[0]
        for query_type, events in grouped.items()
    }


def tune_online_eta(
    dev_events: Iterable[TraceEvent],
    raw_edges,
    initial_weights,
    eta_values=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0),
    k=8,
) -> tuple[float, list[dict]]:
    from .models import PolicyContext
    from .policies import OnlineAdaptivePolicy

    events = list(dev_events)
    rows = []
    for eta in eta_values:
        policy = OnlineAdaptivePolicy(raw_edges, initial_weights, eta)
        hits = 0
        post_update_hits = 0
        update_distance = 0.0
        for event in events:
            context = PolicyContext(
                event.event_id, event.session_id, event.step_id, event.document_id,
                event.current_chunk_id, event.query_type,
            )
            prediction = policy.predict(context, k)
            hits += int(event.target_chunk_id in prediction.ids)
            update = policy.observe(context, event.target_chunk_id, prediction)
            update_distance += float(update.get("update_l2", 0.0))
            post_update = policy.predict(context, k)
            post_update_hits += int(event.target_chunk_id in post_update.ids)
        rows.append(
            {
                "eta": float(eta),
                "dev_recall": hits / len(events) if events else 0.0,
                "post_update_diagnostic_recall": (
                    post_update_hits / len(events) if events else 0.0
                ),
                "updates": policy.updates,
                "total_update_l2": update_distance,
            }
        )
    # Prequential dev recall is the selection criterion. Same-event correction
    # is only a deterministic tie-break, followed by closeness to eta=0.1.
    best = max(
        rows,
        key=lambda row: (
            row["dev_recall"],
            row["post_update_diagnostic_recall"],
            -abs(row["eta"] - 0.1),
            -row["eta"],
        ),
    )
    return float(best["eta"]), rows
