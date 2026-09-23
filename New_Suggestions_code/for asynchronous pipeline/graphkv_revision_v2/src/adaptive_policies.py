"""Leakage-safe offline and mistake-driven online graph policies.

This is the same adaptive model used by the repaired simulator.  It is kept
separate from ``graph_algorithms.py`` so the historical fidelity module stays
unchanged and its regression test remains meaningful.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


Feature = tuple[float, float]


@dataclass(frozen=True)
class TraceEvent:
    event_id: int
    session_id: str
    step_id: int
    document_id: str
    current_chunk_id: int
    target_chunk_id: int
    query_type: str


@dataclass
class Prediction:
    ids: list[int]
    scores: dict[int, float] = field(default_factory=dict)
    features: dict[int, Feature] = field(default_factory=dict)
    candidate_universe: set[int] = field(default_factory=set)


def project_simplex(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (2,):
        raise ValueError("GraphKV currently expects exactly two weights")
    ordered = np.sort(value)[::-1]
    cssv = np.cumsum(ordered) - 1.0
    indices = np.arange(1, len(value) + 1)
    valid = ordered - cssv / indices > 0
    if not np.any(valid):
        return np.full(2, 0.5, dtype=np.float64)
    rho = indices[valid][-1]
    theta = cssv[valid][-1] / rho
    return np.maximum(value - theta, 0.0)


def _blend(first: Feature, second: Feature) -> Feature:
    return (
        0.9 * min(first[0], second[0]) + 0.1 * max(first[0], second[0]),
        0.9 * min(first[1], second[1]) + 0.1 * max(first[1], second[1]),
    )


def enumerate_candidate_paths(raw_edges: dict, primary: int) -> dict[int, list[Feature]]:
    paths: dict[int, list[Feature]] = defaultdict(list)
    for neighbour, semantic, structural in raw_edges.get(int(primary), []):
        first = (float(semantic), float(structural))
        paths[int(neighbour)].append(first)
        for candidate, semantic2, structural2 in raw_edges.get(int(neighbour), []):
            candidate = int(candidate)
            if candidate == int(primary):
                continue
            paths[candidate].append(_blend(first, (float(semantic2), float(structural2))))
    return dict(paths)


def rank_paths(paths: dict[int, list[Feature]], weights, k: int) -> Prediction:
    weights = project_simplex(np.asarray(weights, dtype=np.float64))
    scores: dict[int, float] = {}
    features: dict[int, Feature] = {}
    for candidate, alternatives in paths.items():
        if not alternatives:
            continue
        score, feature = max(
            ((float(np.dot(weights, feature)), feature) for feature in alternatives),
            key=lambda item: (item[0], item[1]),
        )
        scores[int(candidate)] = score
        features[int(candidate)] = feature
    ranked = sorted(scores, key=lambda candidate: (-scores[candidate], candidate))
    return Prediction(ranked[: int(k)], scores, features, set(features))


class OfflineAdaptivePolicy:
    name = "adaptive_offline_global"

    def __init__(self, raw_edges: dict, weights=(0.7, 0.3)):
        self.raw_edges = raw_edges
        self.weights = project_simplex(np.asarray(weights, dtype=np.float64))

    def predict(self, primary: int, k: int) -> Prediction:
        return rank_paths(enumerate_candidate_paths(self.raw_edges, primary), self.weights, k)

    def observe(self, target: int, prediction: Prediction) -> dict:
        return {
            "updated": False,
            "update_reason": "offline_policy",
            "weights_before": self.weights.tolist(),
            "weights_after": self.weights.tolist(),
            "update_l2": 0.0,
        }

    def state_dict(self) -> dict:
        return {"weights": self.weights.tolist()}


class OnlineAdaptivePolicy(OfflineAdaptivePolicy):
    name = "adaptive_online_warm"

    def __init__(self, raw_edges: dict, weights=(0.7, 0.3), eta: float = 0.1):
        super().__init__(raw_edges, weights)
        if eta <= 0:
            raise ValueError("eta must be positive")
        self.eta = float(eta)
        self.updates = 0

    def observe(self, target: int, prediction: Prediction) -> dict:
        target = int(target)
        before = self.weights.copy()
        if target in prediction.ids:
            reason = "prediction_hit"
        elif target not in prediction.features:
            reason = "coverage_miss"
        else:
            negative = next((candidate for candidate in prediction.ids if candidate in prediction.features), None)
            if negative is None:
                reason = "no_negative_candidate"
            else:
                direction = np.asarray(prediction.features[target]) - np.asarray(prediction.features[negative])
                norm = float(np.linalg.norm(direction))
                if norm > 1e-12:
                    direction = direction / norm
                self.weights = project_simplex(self.weights + self.eta * direction)
                if np.any(np.abs(before - self.weights) > 1e-15):
                    self.updates += 1
                    reason = "ranking_miss_update"
                else:
                    reason = "ranking_miss_no_effect"
        return {
            "updated": bool(np.any(np.abs(before - self.weights) > 1e-15)),
            "update_reason": reason,
            "weights_before": before.tolist(),
            "weights_after": self.weights.tolist(),
            "update_l2": float(np.linalg.norm(before - self.weights)),
        }

    def state_dict(self) -> dict:
        return {"weights": self.weights.tolist(), "eta": self.eta, "updates": self.updates}


def _recall(events: Iterable[TraceEvent], raw_edges: dict, weights, k_values) -> float:
    hits = total = 0
    policy = OfflineAdaptivePolicy(raw_edges, weights)
    for event in events:
        for k in k_values:
            hits += int(event.target_chunk_id in policy.predict(event.current_chunk_id, int(k)).ids)
            total += 1
    return hits / total if total else 0.0


def fit_offline_weights(events: Iterable[TraceEvent], raw_edges: dict, k_values, step=0.05):
    events = list(events)
    rows = []
    divisions = int(round(1.0 / float(step)))
    for index in range(divisions + 1):
        alpha = index / divisions
        weights = (alpha, 1.0 - alpha)
        rows.append(
            {
                "alpha": alpha,
                "beta": 1.0 - alpha,
                "train_prediction_recall": _recall(events, raw_edges, weights, k_values),
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["train_prediction_recall"],
            -abs(row["alpha"] - 0.7),
            -row["alpha"],
        ),
    )
    return (best["alpha"], best["beta"]), rows


def tune_eta(events: Iterable[TraceEvent], raw_edges: dict, weights, k: int = 6):
    events = list(events)
    rows = []
    for eta in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0):
        policy = OnlineAdaptivePolicy(raw_edges, weights, eta)
        hits = 0
        for event in events:
            prediction = policy.predict(event.current_chunk_id, k)
            hits += int(event.target_chunk_id in prediction.ids)
            policy.observe(event.target_chunk_id, prediction)
        rows.append(
            {
                "eta": eta,
                "dev_prediction_recall": hits / len(events) if events else 0.0,
                "updates": policy.updates,
            }
        )
    best = max(rows, key=lambda row: (row["dev_prediction_recall"], -abs(row["eta"] - 0.1)))
    return float(best["eta"]), rows

