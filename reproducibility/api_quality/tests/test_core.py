from __future__ import annotations

import numpy as np

from graphkv_quality.graph import (
    Node,
    Prediction,
    build_document_graph,
    observe_online,
    project_simplex,
)
from graphkv_quality.scoring import exact_match, max_token_f1
from graphkv_quality.stats import paired_bootstrap


def test_scoring_matches_squad_normalisation():
    assert exact_match("The Eiffel Tower.", ["Eiffel Tower"]) == 1.0
    assert max_token_f1("Paris France", ["Paris"]) == 2 / 3


def test_structural_edges_never_cross_documents():
    nodes = [
        Node(0, "A", 0, "a0"),
        Node(1, "B", 0, "b0"),
        Node(2, "A", 1, "a1"),
        Node(3, "B", 1, "b1"),
    ]
    similarity = np.eye(4)
    similarity[similarity == 0] = -0.5
    _, raw = build_document_graph(similarity, nodes, top_m=1)
    features = {(left, right): (semantic, structural) for left, edges in raw.items() for right, semantic, structural in edges}
    assert features[(0, 2)][1] == 0.5
    assert features[(1, 3)][1] == 0.5
    assert all(structural == 0.0 for (left, right), (_, structural) in features.items() if nodes[left].document_id != nodes[right].document_id)


def test_online_update_moves_toward_target_features():
    prediction = Prediction(
        ids=[1],
        features={1: (0.9, 0.1), 2: (0.1, 0.9)},
        candidate_universe={1, 2},
    )
    after, report = observe_online((0.7, 0.3), 2, prediction, 0.2)
    assert report["updated"]
    assert after[1] > 0.3
    assert np.isclose(after.sum(), 1.0)


def test_simplex_and_bootstrap():
    projected = project_simplex((2.0, -1.0))
    assert np.all(projected >= 0)
    assert np.isclose(projected.sum(), 1.0)
    result = paired_bootstrap([0.1, 0.2, 0.3], seed=1, resamples=500)
    assert np.isclose(result["mean_delta"], 0.2)
    assert result["ci_low"] <= result["mean_delta"] <= result["ci_high"]
