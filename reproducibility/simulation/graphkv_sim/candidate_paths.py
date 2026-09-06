from __future__ import annotations

from collections import defaultdict

import numpy as np


Feature = tuple[float, float]


def _blend_components(a: Feature, b: Feature) -> Feature:
    return (
        0.9 * min(a[0], b[0]) + 0.1 * max(a[0], b[0]),
        0.9 * min(a[1], b[1]) + 0.1 * max(a[1], b[1]),
    )


def enumerate_candidate_paths(
    raw_edges: dict[int, list[tuple[int, float, float]]], pri: int
) -> dict[int, list[Feature]]:
    """Return direct/two-hop semantic-structural path features.

    This is new online-policy behavior; it does not replace the frozen legacy
    adaptive scorer. Each candidate may have several latent paths. Ranking
    selects the best path under the current weight vector.
    """

    paths: dict[int, list[Feature]] = defaultdict(list)
    for j, sem, struct in raw_edges.get(pri, []):
        first = (float(sem), float(struct))
        paths[j].append(first)
        for j2, sem2, struct2 in raw_edges.get(j, []):
            if j2 == pri:
                continue
            second = (float(sem2), float(struct2))
            paths[j2].append(_blend_components(first, second))
    return dict(paths)


def rank_candidate_paths(
    paths: dict[int, list[Feature]], weights: np.ndarray, k: int
) -> tuple[list[int], dict[int, float], dict[int, Feature]]:
    if weights.shape != (2,):
        raise ValueError("weights must have shape (2,)")
    best_scores: dict[int, float] = {}
    best_features: dict[int, Feature] = {}
    for candidate, alternatives in paths.items():
        if not alternatives:
            continue
        scored = [(float(np.dot(weights, feature)), feature) for feature in alternatives]
        score, feature = max(scored, key=lambda item: item[0])
        best_scores[candidate] = score
        best_features[candidate] = feature
    ranked = sorted(best_scores, key=lambda c: (-best_scores[c], c))
    return ranked[:k], best_scores, best_features


def project_simplex(value: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum(w) = 1}."""

    v = np.asarray(value, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError("simplex projection expects a vector")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    indices = np.arange(1, len(v) + 1)
    valid = u - cssv / indices > 0
    if not np.any(valid):
        return np.full_like(v, 1.0 / len(v))
    rho = indices[valid][-1]
    theta = cssv[valid][-1] / rho
    return np.maximum(v - theta, 0.0)

