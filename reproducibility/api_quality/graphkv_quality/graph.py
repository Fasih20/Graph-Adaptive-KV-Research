"""Document-aware GraphKV graph construction and retrieval policies.

Every node is one HotpotQA sentence. Structural distance exists only between
sentences from the same source document. Semantic edges use a symmetric top-M
neighbour graph. Graph policies rank all nodes reachable within two hops.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


Feature = tuple[float, float]


@dataclass(frozen=True)
class Node:
    node_id: int
    document_id: str
    position: int
    text: str
    is_support: bool = False


@dataclass
class Prediction:
    ids: list[int]
    scores: dict[int, float] = field(default_factory=dict)
    features: dict[int, Feature] = field(default_factory=dict)
    candidate_universe: set[int] = field(default_factory=set)


def project_simplex(value) -> np.ndarray:
    """Project two non-negative weights onto alpha + beta = 1."""
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (2,):
        raise ValueError("GraphKV expects exactly two weights")
    ordered = np.sort(value)[::-1]
    cssv = np.cumsum(ordered) - 1.0
    indices = np.arange(1, len(value) + 1)
    valid = ordered - cssv / indices > 0
    if not np.any(valid):
        return np.full(2, 0.5, dtype=np.float64)
    rho = indices[valid][-1]
    theta = cssv[valid][-1] / rho
    return np.maximum(value - theta, 0.0)


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalised = embeddings / norms
    return normalised @ normalised.T


def cosine_query(embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    query_embedding = np.asarray(query_embedding, dtype=np.float64)
    row_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    return (embeddings @ query_embedding) / (row_norms * query_norm + 1e-12)


def build_document_graph(
    similarity: np.ndarray,
    nodes: list[Node],
    *,
    top_m: int = 5,
    fixed_weights=(0.7, 0.3),
    max_degree: int = 100,
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, list[tuple[int, float, float]]]]:
    """Build fixed-weight adjacency and unweighted semantic/structural features."""
    n = len(nodes)
    if similarity.shape != (n, n):
        raise ValueError("similarity matrix does not align with nodes")
    if top_m < 1:
        raise ValueError("top_m must be positive")
    fixed_weights = project_simplex(fixed_weights)

    top_sets: list[set[int]] = []
    for left in range(n):
        row = np.asarray(similarity[left]).copy()
        row[left] = -np.inf
        count = min(int(top_m), max(0, n - 1))
        if count == 0:
            top_sets.append(set())
        elif count < n - 1:
            top_sets.append(set(np.argpartition(row, -count)[-count:].tolist()))
        else:
            top_sets.append(set(np.argsort(row)[::-1][:count].tolist()))

    raw: dict[int, list[tuple[int, float, float]]] = {index: [] for index in range(n)}
    for left in range(n):
        for right in range(left + 1, n):
            same_document = nodes[left].document_id == nodes[right].document_id
            distance = abs(nodes[left].position - nodes[right].position) if same_document else None
            adjacent = same_document and distance == 1
            semantic_edge = right in top_sets[left] or left in top_sets[right]
            if not (adjacent or semantic_edge):
                continue
            semantic = float(similarity[left, right]) if semantic_edge else 0.0
            structural = 1.0 / (1.0 + float(distance)) if same_document else 0.0
            raw[left].append((right, semantic, structural))
            raw[right].append((left, semantic, structural))

    adjacency: dict[int, list[tuple[int, float]]] = {}
    for node_id, edges in raw.items():
        scored = [
            (other, float(fixed_weights[0] * semantic + fixed_weights[1] * structural))
            for other, semantic, structural in edges
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        adjacency[node_id] = scored[: int(max_degree)]
    return adjacency, raw


def _blend(first: Feature, second: Feature) -> Feature:
    return (
        0.9 * min(first[0], second[0]) + 0.1 * max(first[0], second[0]),
        0.9 * min(first[1], second[1]) + 0.1 * max(first[1], second[1]),
    )


def enumerate_candidate_paths(raw_edges, primary: int) -> dict[int, list[Feature]]:
    paths: dict[int, list[Feature]] = defaultdict(list)
    for neighbour, semantic, structural in raw_edges.get(int(primary), []):
        first = (float(semantic), float(structural))
        paths[int(neighbour)].append(first)
        for candidate, semantic2, structural2 in raw_edges.get(int(neighbour), []):
            candidate = int(candidate)
            if candidate != int(primary):
                paths[candidate].append(_blend(first, (float(semantic2), float(structural2))))
    return dict(paths)


def rank_paths(paths, weights, k: int) -> Prediction:
    weights = project_simplex(weights)
    scores: dict[int, float] = {}
    features: dict[int, Feature] = {}
    for candidate, alternatives in paths.items():
        score, feature = max(
            ((float(np.dot(weights, feature)), feature) for feature in alternatives),
            key=lambda item: (item[0], item[1]),
        )
        scores[int(candidate)] = score
        features[int(candidate)] = feature
    ranked = sorted(scores, key=lambda candidate: (-scores[candidate], candidate))
    return Prediction(ranked[: int(k)], scores, features, set(features))


def predict_cosine(similarity: np.ndarray, primary: int, k: int) -> Prediction:
    values = np.asarray(similarity[int(primary)]).copy()
    values[int(primary)] = -np.inf
    ranked = sorted(range(len(values)), key=lambda node_id: (-values[node_id], node_id))
    ids = ranked[: int(k)]
    scores = {int(node_id): float(values[node_id]) for node_id in ranked if node_id != int(primary)}
    return Prediction(ids, scores, {}, set(scores))


def predict_fixed(adjacency, primary: int, k: int) -> Prediction:
    candidates: dict[int, float] = {}
    for neighbour, first_weight in adjacency.get(int(primary), []):
        candidates[neighbour] = max(candidates.get(neighbour, -np.inf), first_weight)
        for candidate, second_weight in adjacency.get(neighbour, []):
            if candidate == int(primary):
                continue
            score = 0.9 * min(first_weight, second_weight) + 0.1 * max(first_weight, second_weight)
            candidates[candidate] = max(candidates.get(candidate, -np.inf), score)
    ranked = sorted(candidates, key=lambda node_id: (-candidates[node_id], node_id))
    return Prediction(ranked[: int(k)], candidates, {}, set(candidates))


def predict_adaptive(raw_edges, primary: int, k: int, weights) -> Prediction:
    return rank_paths(enumerate_candidate_paths(raw_edges, primary), weights, k)


def observe_online(weights, target: int, prediction: Prediction, eta: float):
    """Mistake-driven update used by the repaired online policy."""
    before = project_simplex(weights)
    after = before.copy()
    reason = "prediction_hit"
    if int(target) not in prediction.ids:
        if int(target) not in prediction.features:
            reason = "coverage_miss"
        else:
            negative = next((item for item in prediction.ids if item in prediction.features), None)
            if negative is None:
                reason = "no_negative_candidate"
            else:
                direction = np.asarray(prediction.features[int(target)]) - np.asarray(prediction.features[negative])
                norm = float(np.linalg.norm(direction))
                if norm > 1e-12:
                    direction = direction / norm
                after = project_simplex(before + float(eta) * direction)
                reason = "ranking_miss_update" if not np.allclose(before, after) else "ranking_miss_no_effect"
    return after, {
        "updated": bool(not np.allclose(before, after)),
        "update_reason": reason,
        "weights_before": before.tolist(),
        "weights_after": after.tolist(),
        "update_l2": float(np.linalg.norm(after - before)),
    }


def support_recall(selected_ids: list[int], support_ids: set[int]) -> float:
    if not support_ids:
        return 0.0
    return len(set(selected_ids) & set(support_ids)) / len(support_ids)
