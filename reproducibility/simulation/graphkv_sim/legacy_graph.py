"""Frozen graph and prefetch logic from the canonical adaptive simulator.

Do not clean up tie ordering or traversal order in the functions marked
``legacy``. New behavior belongs in a separate function/module so historical
results remain reproducible.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.sparse as sp

SEM_THRESHOLD = 0.55
SEM_WEIGHT = 0.70
STRUCT_WEIGHT = 0.30
MAX_DEGREE = 100


def build_graph_legacy(
    embeddings: np.ndarray,
    sim_matrix: np.ndarray,
    logger: logging.Logger,
    graph_construction: str = "topm",
    top_m: int = 20,
):
    """Behavior-preserving port of the canonical graph constructor."""

    n = len(embeddings)
    if n == 0:
        raise ValueError("cannot build a graph with zero chunks")
    logger.info("Building legacy graph for %d chunks", n)
    t0 = time.perf_counter()

    topm_sets = None
    if graph_construction == "topm":
        topm_sets = []
        for i in range(n):
            if n == 1:
                topm_sets.append(set())
                continue
            row = sim_matrix[i].copy()
            row[i] = -1.0
            m = min(top_m, n - 1)
            top_idx = (
                np.argpartition(row, -m)[-m:]
                if m < n - 1
                else np.argsort(row)[::-1]
            )
            topm_sets.append(set(top_idx.tolist()))

    rows, cols, data = [], [], []
    raw_edges: dict[int, list[tuple[int, float, float]]] = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            sem = float(sim_matrix[i, j])
            dist = abs(i - j)
            ss = 1.0 / (1.0 + dist)
            is_adj = dist == 1
            if graph_construction == "topm":
                is_sem = (j in topm_sets[i]) or (i in topm_sets[j])
            else:
                is_sem = sem >= SEM_THRESHOLD
            if is_sem or is_adj:
                sem_c = sem if is_sem else 0.0
                weight = SEM_WEIGHT * sem_c + STRUCT_WEIGHT * ss
                rows += [i, j]
                cols += [j, i]
                data += [weight, weight]
                raw_edges[i].append((j, sem_c, ss))
                raw_edges[j].append((i, sem_c, ss))

    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    p_rows, p_cols, p_data = [], [], []
    for i in range(n):
        row_slice = adj.getrow(i)
        if row_slice.nnz == 0:
            continue
        if row_slice.nnz <= MAX_DEGREE:
            p_rows += row_slice.indices.tolist()
            p_cols += [i] * row_slice.nnz
            p_data += row_slice.data.tolist()
        else:
            top_idx = np.argsort(row_slice.data)[::-1][:MAX_DEGREE]
            p_rows += row_slice.indices[top_idx].tolist()
            p_cols += [i] * MAX_DEGREE
            p_data += row_slice.data[top_idx].tolist()

    adj_pruned = sp.csr_matrix((p_data, (p_cols, p_rows)), shape=(n, n))
    adj_pruned.eliminate_zeros()
    elapsed = time.perf_counter() - t0
    return adj_pruned, raw_edges, elapsed


def build_graph_document_aware(
    embeddings: np.ndarray,
    sim_matrix: np.ndarray,
    document_ids: list[str],
    positions: list[int],
    logger: logging.Logger,
    graph_construction: str = "topm",
    top_m: int = 20,
):
    """Boundary-safe variant; Top-M, weights, pruning and two-hop stay intact.

    Semantic edges may cross documents. Structural scores/adjacency never do.
    The legacy function remains available for exact historical reproduction.
    """

    n = len(embeddings)
    if n == 0:
        raise ValueError("cannot build a graph with zero chunks")
    if len(document_ids) != n or len(positions) != n:
        raise ValueError("document_ids and positions must align with embeddings")
    logger.info("Building document-aware graph for %d chunks", n)
    t0 = time.perf_counter()

    topm_sets = None
    if graph_construction == "topm":
        topm_sets = []
        for i in range(n):
            if n == 1:
                topm_sets.append(set())
                continue
            row = sim_matrix[i].copy()
            row[i] = -1.0
            m = min(top_m, n - 1)
            top_idx = (
                np.argpartition(row, -m)[-m:]
                if m < n - 1
                else np.argsort(row)[::-1]
            )
            topm_sets.append(set(top_idx.tolist()))

    rows, cols, data = [], [], []
    raw_edges: dict[int, list[tuple[int, float, float]]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            sem = float(sim_matrix[i, j])
            same_doc = document_ids[i] == document_ids[j]
            local_dist = abs(positions[i] - positions[j]) if same_doc else None
            ss = 1.0 / (1.0 + local_dist) if same_doc else 0.0
            is_adj = bool(same_doc and local_dist == 1)
            if graph_construction == "topm":
                is_sem = (j in topm_sets[i]) or (i in topm_sets[j])
            else:
                is_sem = sem >= SEM_THRESHOLD
            if is_sem or is_adj:
                sem_c = sem if is_sem else 0.0
                weight = SEM_WEIGHT * sem_c + STRUCT_WEIGHT * ss
                rows += [i, j]
                cols += [j, i]
                data += [weight, weight]
                raw_edges[i].append((j, sem_c, ss))
                raw_edges[j].append((i, sem_c, ss))

    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    p_rows, p_cols, p_data = [], [], []
    for i in range(n):
        row_slice = adj.getrow(i)
        if row_slice.nnz == 0:
            continue
        if row_slice.nnz <= MAX_DEGREE:
            top_idx = np.arange(row_slice.nnz)
        else:
            top_idx = np.argsort(row_slice.data)[::-1][:MAX_DEGREE]
        p_rows += row_slice.indices[top_idx].tolist()
        p_cols += [i] * len(top_idx)
        p_data += row_slice.data[top_idx].tolist()

    adj_pruned = sp.csr_matrix((p_data, (p_cols, p_rows)), shape=(n, n))
    adj_pruned.eliminate_zeros()
    return adj_pruned, raw_edges, time.perf_counter() - t0


def get_prefetch_cosine(sim_matrix: np.ndarray, pri: int, k: int) -> list[int]:
    sims = sim_matrix[pri].copy()
    sims[pri] = -1
    return np.argsort(sims)[::-1][:k].tolist()


def get_prefetch_graph(adj_matrix: sp.csr_matrix, pri: int, k: int) -> list[int]:
    candidates: dict[int, float] = {}
    row = adj_matrix.getrow(pri)
    if row.nnz == 0:
        return []

    for idx, nbr in enumerate(row.indices):
        w1 = row.data[idx]
        if nbr not in candidates or w1 > candidates[nbr]:
            candidates[nbr] = w1
        row2 = adj_matrix.getrow(nbr)
        for idx2, nbr2 in enumerate(row2.indices):
            if nbr2 == pri:
                continue
            w2 = row2.data[idx2]
            score = 0.9 * min(w1, w2) + 0.1 * max(w1, w2)
            if nbr2 not in candidates or score > candidates[nbr2]:
                candidates[nbr2] = score

    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in ranked[:k]]


def get_prefetch_adaptive(
    raw_edges: dict[int, list[tuple[int, float, float]]],
    pri: int,
    k: int,
    query_type: str,
    adaptive_weights: dict,
) -> list[int]:
    neighbours = raw_edges.get(pri, [])
    if not neighbours:
        return []
    weights = adaptive_weights.get(
        query_type,
        adaptive_weights.get(
            "multi-hop", {"alpha": SEM_WEIGHT, "beta": STRUCT_WEIGHT}
        ),
    )
    alpha, beta = weights["alpha"], weights["beta"]

    candidates: dict[int, float] = {}
    for j, sem_c, ss in neighbours:
        w1 = alpha * sem_c + beta * ss
        if j not in candidates or w1 > candidates[j]:
            candidates[j] = w1
        for j2, sem_c2, ss2 in raw_edges.get(j, []):
            if j2 == pri:
                continue
            w2 = alpha * sem_c2 + beta * ss2
            score = 0.9 * min(w1, w2) + 0.1 * max(w1, w2)
            if j2 not in candidates or score > candidates[j2]:
                candidates[j2] = score

    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [j for j, _ in ranked[:k]]
