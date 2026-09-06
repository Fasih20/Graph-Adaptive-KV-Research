"""Document-boundary-safe graph construction for new experiments."""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from exp_config import MAX_DEGREE, SEM_THRESHOLD, SEM_WEIGHT, STRUCT_WEIGHT


def build_graph_document_aware(
    embeddings,
    similarity,
    document_ids,
    positions,
    logger,
    graph_construction="topm",
    top_m=20,
):
    n = len(embeddings)
    if n == 0:
        raise ValueError("cannot build an empty graph")
    if not (len(document_ids) == len(positions) == n):
        raise ValueError("document metadata does not align with embeddings")
    started = time.perf_counter()
    logger.info("Building document-aware graph for %d chunks", n)

    top_sets = None
    if graph_construction == "topm":
        top_sets = []
        for index in range(n):
            if n == 1:
                top_sets.append(set())
                continue
            row = similarity[index].copy()
            row[index] = -1.0
            count = min(int(top_m), n - 1)
            selected = (
                np.argpartition(row, -count)[-count:]
                if count < n - 1
                else np.argsort(row)[::-1]
            )
            top_sets.append(set(selected.tolist()))

    rows, cols, values = [], [], []
    raw_edges = {index: [] for index in range(n)}
    for left in range(n):
        for right in range(left + 1, n):
            semantic = float(similarity[left, right])
            same_document = str(document_ids[left]) == str(document_ids[right])
            distance = abs(int(positions[left]) - int(positions[right])) if same_document else None
            structural = 1.0 / (1.0 + distance) if same_document else 0.0
            adjacent = bool(same_document and distance == 1)
            if graph_construction == "topm":
                semantic_edge = right in top_sets[left] or left in top_sets[right]
            else:
                semantic_edge = semantic >= SEM_THRESHOLD
            if not (semantic_edge or adjacent):
                continue
            semantic_component = semantic if semantic_edge else 0.0
            weight = SEM_WEIGHT * semantic_component + STRUCT_WEIGHT * structural
            rows.extend([left, right])
            cols.extend([right, left])
            values.extend([weight, weight])
            raw_edges[left].append((right, semantic_component, structural))
            raw_edges[right].append((left, semantic_component, structural))

    adjacency = sp.csr_matrix((values, (rows, cols)), shape=(n, n))
    p_rows, p_cols, p_values = [], [], []
    for node in range(n):
        row = adjacency.getrow(node)
        if not row.nnz:
            continue
        if row.nnz <= MAX_DEGREE:
            selected = np.arange(row.nnz)
        else:
            selected = np.argsort(row.data)[::-1][:MAX_DEGREE]
        p_rows.extend(row.indices[selected].tolist())
        p_cols.extend([node] * len(selected))
        p_values.extend(row.data[selected].tolist())
    pruned = sp.csr_matrix((p_values, (p_cols, p_rows)), shape=(n, n))
    pruned.eliminate_zeros()
    elapsed = time.perf_counter() - started
    logger.info("Document-aware graph built in %.3fs with %d directed edges", elapsed, pruned.nnz)
    return pruned, raw_edges, elapsed

