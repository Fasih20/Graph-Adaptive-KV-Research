"""
prefetch_policies.py
=====================
Thin class-based interface (per implementation_plan.md Phase 1) around the
verbatim-ported functions in graph_algorithms.py.

IMPORTANT — what is and isn't "ported" here:
  - The actual candidate SELECTION and ORDER (which chunk ids come back, and
    in what order) is produced entirely by the untouched functions in
    graph_algorithms.py. This file does not re-implement or re-derive that
    logic in any way.
  - The canonical functions return `List[int]` (no scores — scores are
    computed internally then discarded). The plan's interface,
    `get_prefetch_candidates(...) -> List[Tuple[chunk_id, score]]`, requires
    scores. Since exposing real internal scores would mean modifying the
    ported functions (which we're avoiding — see graph_algorithms.py's
    docstring), each policy synthesizes a rank-derived placeholder score
    (k - rank) purely so callers that want a `(id, score)` shape can have
    one. This synthetic score carries no information beyond rank order —
    it is NOT the cosine similarity / edge weight — and must not be used
    for anything beyond "is A ranked above B". If you need the real scores,
    use `get_prefetch_ids()`, which returns exactly what the canonical
    function returns, untouched.
  - `warm_chunks`/cache-facing code (cache_manager.py) should generally call
    `get_prefetch_ids()` directly, since that's what feeds
    KVCacheSimulator.prefetch(ids) and must match the simulated harness's
    trial-by-trial behavior exactly.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

import graph_algorithms as ga


class PrefetchPolicy(ABC):
    @abstractmethod
    def get_prefetch_ids(self, primary_chunk_id: int, k: int) -> List[int]:
        """Returns exactly what the canonical get_prefetch_* function returns."""
        raise NotImplementedError

    def get_prefetch_candidates(self, primary_chunk_id: int, k: int) -> List[Tuple[int, float]]:
        """Plan-interface shim. See module docstring re: synthetic scores."""
        ids = self.get_prefetch_ids(primary_chunk_id, k)
        return [(cid, float(len(ids) - rank)) for rank, cid in enumerate(ids)]


class CosinePrefetchPolicy(PrefetchPolicy):
    """Top-K by embedding cosine similarity. Wraps get_prefetch_cosine verbatim."""

    def __init__(self, sim_matrix: np.ndarray):
        self.sim_matrix = sim_matrix

    def get_prefetch_ids(self, primary_chunk_id: int, k: int) -> List[int]:
        return ga.get_prefetch_cosine(self.sim_matrix, primary_chunk_id, k)


class GraphPrefetchPolicy(PrefetchPolicy):
    """Fixed-weight (SEM_WEIGHT/STRUCT_WEIGHT) 2-hop graph retrieval.
    Wraps get_prefetch_graph verbatim."""

    def __init__(self, adj_matrix: sp.csr_matrix):
        self.adj_matrix = adj_matrix

    def get_prefetch_ids(self, primary_chunk_id: int, k: int) -> List[int]:
        return ga.get_prefetch_graph(self.adj_matrix, primary_chunk_id, k)


class AdaptivePrefetchPolicy(PrefetchPolicy):
    """Query-type-specific (alpha, beta) 2-hop graph retrieval.
    Wraps get_prefetch_adaptive verbatim.

    `adaptive_weights` must be supplied by the caller — this class has no
    opinion on fallback-vs-hardcoded; see config.HARDCODED_ADAPTIVE_WEIGHTS
    for the values actually used in the Phase 4 run.
    """

    def __init__(self, raw_edges: dict, adaptive_weights: dict):
        self.raw_edges = raw_edges
        self.adaptive_weights = adaptive_weights

    def get_prefetch_ids(self, primary_chunk_id: int, k: int, q_type: str = "multi-hop") -> List[int]:
        return ga.get_prefetch_adaptive(self.raw_edges, primary_chunk_id, k, q_type, self.adaptive_weights)

    def get_prefetch_candidates(self, primary_chunk_id: int, k: int, q_type: str = "multi-hop") -> List[Tuple[int, float]]:
        ids = self.get_prefetch_ids(primary_chunk_id, k, q_type)
        return [(cid, float(len(ids) - rank)) for rank, cid in enumerate(ids)]
