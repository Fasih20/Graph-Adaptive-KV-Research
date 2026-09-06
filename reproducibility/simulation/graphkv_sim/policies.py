from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp

from .candidate_paths import enumerate_candidate_paths, project_simplex, rank_candidate_paths
from .legacy_graph import get_prefetch_cosine, get_prefetch_graph
from .models import PolicyContext, Prediction


class BasePolicy(ABC):
    name = "base"
    is_oracle = False
    reports_candidate_coverage = False

    @abstractmethod
    def predict(self, context: PolicyContext, k: int) -> Prediction:
        raise NotImplementedError

    def observe(
        self, context: PolicyContext, target_chunk_id: int, prediction: Prediction
    ) -> dict[str, Any]:
        return {"updated": False, "update_reason": "non_learning_policy"}

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError(f"{self.name} has no mutable state")


class NoPrefetchPolicy(BasePolicy):
    name = "no_prefetch"

    def predict(self, context: PolicyContext, k: int) -> Prediction:
        return Prediction([])


class RandomPolicy(BasePolicy):
    name = "random"

    def __init__(self, chunk_ids: list[int], seed: int = 0) -> None:
        self.chunk_ids = np.asarray(sorted(set(map(int, chunk_ids))), dtype=np.int64)
        self.seed = int(seed)

    def predict(self, context: PolicyContext, k: int) -> Prediction:
        candidates = self.chunk_ids[self.chunk_ids != context.current_chunk_id]
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, context.event_id, context.current_chunk_id])
        )
        count = min(int(k), len(candidates))
        ids = rng.choice(candidates, size=count, replace=False).tolist() if count else []
        return Prediction(
            [int(v) for v in ids],
            candidate_universe_ids=set(map(int, candidates.tolist())),
        )


class SequentialPolicy(BasePolicy):
    name = "sequential"

    def __init__(self, document_orders: Mapping[str, list[int]]) -> None:
        self.orders = {str(d): list(map(int, ids)) for d, ids in document_orders.items()}
        self.indices = {
            (document, chunk_id): i
            for document, ids in self.orders.items()
            for i, chunk_id in enumerate(ids)
        }

    def predict(self, context: PolicyContext, k: int) -> Prediction:
        ids = self.orders.get(context.document_id, [])
        index = self.indices.get((context.document_id, context.current_chunk_id))
        if index is None:
            return Prediction([])
        universe = ids[index + 1 :]
        return Prediction(
            universe[: int(k)], candidate_universe_ids=set(universe)
        )


class ChunkIdTransitionPolicy(BasePolicy):
    """Historical exact-ID Markov baseline; not transferable to unseen docs."""

    name = "chunk_id_transition_nontransferable"

    def __init__(self, transitions: Mapping[int, Counter], popularity: Counter) -> None:
        self.transitions = {int(k): Counter(v) for k, v in transitions.items()}
        self.popularity = Counter(popularity)

    @classmethod
    def fit(cls, events) -> "ChunkIdTransitionPolicy":
        transitions: dict[int, Counter] = defaultdict(Counter)
        popularity: Counter = Counter()
        for event in events:
            transitions[event.current_chunk_id][event.target_chunk_id] += 1
            popularity[event.target_chunk_id] += 1
        return cls(transitions, popularity)

    def predict(self, context: PolicyContext, k: int) -> Prediction:
        local = self.transitions.get(context.current_chunk_id, Counter())
        scores = Counter(self.popularity)
        for candidate, count in local.items():
            scores[candidate] += count * (sum(self.popularity.values()) + 1)
        scores.pop(context.current_chunk_id, None)
        ranked = sorted(scores, key=lambda c: (-scores[c], c))[: int(k)]
        return Prediction(
            ranked,
            {int(c): float(scores[c]) for c in scores},
            candidate_universe_ids=set(map(int, scores)),
        )


class RelativeOffsetTransitionPolicy(BasePolicy):
    """Transferable Markov baseline over relative within-document offsets."""

    name = "relative_offset_transition"

    def __init__(self, offset_counts: Counter, document_orders: Mapping[str, list[int]]):
        self.offset_counts = Counter({int(k): int(v) for k, v in offset_counts.items()})
        self.orders = {str(d): list(map(int, ids)) for d, ids in document_orders.items()}
        self.indices = {
            (document, chunk_id): index
            for document, ids in self.orders.items()
            for index, chunk_id in enumerate(ids)
        }

    @classmethod
    def fit(cls, events, document_orders: Mapping[str, list[int]]):
        indices = {
            (document, chunk_id): index
            for document, ids in document_orders.items()
            for index, chunk_id in enumerate(ids)
        }
        counts: Counter = Counter()
        for event in events:
            current = indices.get((event.document_id, event.current_chunk_id))
            target = indices.get((event.document_id, event.target_chunk_id))
            if current is not None and target is not None and target != current:
                counts[target - current] += 1
        return cls(counts, document_orders)

    def predict(self, context: PolicyContext, k: int) -> Prediction:
        order = self.orders.get(context.document_id, [])
        current_index = self.indices.get((context.document_id, context.current_chunk_id))
        if current_index is None:
            return Prediction([])
        universe = {chunk_id for chunk_id in order if chunk_id != context.current_chunk_id}
        ranked_offsets = sorted(
            self.offset_counts,
            key=lambda offset: (-self.offset_counts[offset], abs(offset), -offset),
        )
        ranked: list[int] = []
        scores: dict[int, float] = {}
        for offset in ranked_offsets:
            target_index = current_index + offset
            if 0 <= target_index < len(order):
                candidate = order[target_index]
                if candidate not in ranked:
                    ranked.append(candidate)
                    scores[candidate] = float(self.offset_counts[offset])
        fallback = sorted(
            universe - set(ranked),
            key=lambda chunk_id: (
                abs(self.indices[(context.document_id, chunk_id)] - current_index),
                self.indices[(context.document_id, chunk_id)] < current_index,
                chunk_id,
            ),
        )
        ranked.extend(fallback)
        return Prediction(
            ranked[: int(k)], scores, candidate_universe_ids=universe
        )


# Import compatibility for external code; new experiments use the transferable baseline.
TransitionPolicy = ChunkIdTransitionPolicy


class CosinePolicy(BasePolicy):
    name = "cosine"

    def __init__(self, similarity_matrix: np.ndarray) -> None:
        self.similarity_matrix = similarity_matrix

    def predict(self, context: PolicyContext, k: int) -> Prediction:
        ids = get_prefetch_cosine(self.similarity_matrix, context.current_chunk_id, k)
        universe = set(range(self.similarity_matrix.shape[0])) - {context.current_chunk_id}
        return Prediction(
            ids,
            {i: float(self.similarity_matrix[context.current_chunk_id, i]) for i in ids},
            candidate_universe_ids=universe,
        )


class FixedGraphPolicy(BasePolicy):
    name = "graph_fixed"

    def __init__(self, adjacency: sp.csr_matrix) -> None:
        self.adjacency = adjacency

    def predict(self, context: PolicyContext, k: int) -> Prediction:
        all_candidates = get_prefetch_graph(
            self.adjacency, context.current_chunk_id, self.adjacency.shape[0]
        )
        return Prediction(
            all_candidates[: int(k)], candidate_universe_ids=set(all_candidates)
        )


class LinearGraphPolicy(BasePolicy):
    name = "adaptive_offline_global"
    reports_candidate_coverage = True

    def __init__(self, raw_edges, weights=(0.7, 0.3), name: str | None = None) -> None:
        self.raw_edges = raw_edges
        self.weights = project_simplex(np.asarray(weights, dtype=np.float64))
        if name:
            self.name = name

    def _weights_for(self, context: PolicyContext) -> np.ndarray:
        return self.weights

    def predict(self, context: PolicyContext, k: int) -> Prediction:
        paths = enumerate_candidate_paths(self.raw_edges, context.current_chunk_id)
        ids, scores, features = rank_candidate_paths(paths, self._weights_for(context), int(k))
        return Prediction(ids, scores, features, set(features))


class QueryTypeOraclePolicy(LinearGraphPolicy):
    """Upper-bound diagnostic: query type is assumed known at prediction time."""

    name = "adaptive_offline_query_type_oracle"

    def __init__(self, raw_edges, weights_by_type: Mapping[str, tuple[float, float]], fallback=(0.7, 0.3)):
        super().__init__(raw_edges, fallback, self.name)
        self.weights_by_type = {
            str(key): project_simplex(np.asarray(value, dtype=np.float64))
            for key, value in weights_by_type.items()
        }

    def _weights_for(self, context: PolicyContext) -> np.ndarray:
        return self.weights_by_type.get(context.query_type, self.weights)


class OnlineAdaptivePolicy(LinearGraphPolicy):
    name = "adaptive_online"

    def __init__(
        self,
        raw_edges,
        weights=(0.5, 0.5),
        eta: float = 0.1,
        name: str | None = None,
        normalize_update: bool = True,
    ):
        super().__init__(raw_edges, weights, name or self.name)
        if eta <= 0:
            raise ValueError("eta must be positive")
        self.eta = float(eta)
        self.normalize_update = bool(normalize_update)
        self.updates = 0

    def observe(self, context, target_chunk_id, prediction) -> dict[str, Any]:
        before = self.weights.copy()
        target_chunk_id = int(target_chunk_id)
        if target_chunk_id in prediction.ids:
            reason = "prediction_hit"
        elif target_chunk_id not in prediction.features:
            reason = "coverage_miss"
        else:
            negative = next((i for i in prediction.ids if i in prediction.features), None)
            if negative is None:
                reason = "no_negative_candidate"
            else:
                positive_feature = np.asarray(prediction.features[target_chunk_id])
                negative_feature = np.asarray(prediction.features[negative])
                direction = positive_feature - negative_feature
                direction_norm = float(np.linalg.norm(direction))
                if self.normalize_update and direction_norm > 1e-12:
                    direction = direction / direction_norm
                self.weights = project_simplex(self.weights + self.eta * direction)
                changed = bool(np.any(np.abs(before - self.weights) > 1e-15))
                if changed:
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

    def state_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "eta": self.eta,
            "normalize_update": self.normalize_update,
            "updates": self.updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.weights = project_simplex(np.asarray(state["weights"], dtype=np.float64))
        if abs(float(state["eta"]) - self.eta) > 1e-15:
            raise ValueError("checkpoint eta does not match policy eta")
        if bool(state.get("normalize_update", True)) != self.normalize_update:
            raise ValueError("checkpoint update normalization does not match policy")
        self.updates = int(state.get("updates", 0))


class OneStepTargetOraclePolicy(BasePolicy):
    """Recall upper bound, not a full-horizon optimal cache-placement oracle."""

    name = "one_step_target_oracle"
    is_oracle = True

    def predict(self, context: PolicyContext, k: int) -> Prediction:
        raise RuntimeError("oracle requires predict_with_target and must be reported separately")

    def predict_with_target(self, context: PolicyContext, target_chunk_id: int, k: int) -> Prediction:
        return Prediction([int(target_chunk_id)][: int(k)], uses_future_target=True)


OraclePolicy = OneStepTargetOraclePolicy
