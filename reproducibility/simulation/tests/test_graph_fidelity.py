import logging

import numpy as np

from graphkv_sim.legacy_graph import build_graph_legacy, get_prefetch_graph


def test_legacy_graph_known_ranking_is_frozen():
    similarities = np.array(
        [
            [1.0, .8, .1, .2],
            [.8, 1.0, .7, .1],
            [.1, .7, 1.0, .9],
            [.2, .1, .9, 1.0],
        ]
    )
    adjacency, raw_edges, _ = build_graph_legacy(
        similarities, similarities, logging.getLogger("test"), "topm", 1
    )
    assert get_prefetch_graph(adjacency, 0, 3) == [1, 2]
    assert raw_edges[0][0][0] == 1


def test_single_chunk_graph_is_well_defined():
    values = np.ones((1, 1))
    adjacency, raw_edges, _ = build_graph_legacy(
        values, values, logging.getLogger("test"), "topm", 20
    )
    assert adjacency.nnz == 0 and raw_edges == {0: []}
