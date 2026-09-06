"""
graph_algorithms.py
====================
PORTED VERBATIM from kv_cache_experiment_adaptive_v2(2).py.

Every function below is a byte-for-byte copy of the corresponding function
in the canonical source (build_graph, get_prefetch_cosine, get_prefetch_graph,
get_prefetch_adaptive, generate_pairs, KVCacheSimulator), with only two
mechanical changes:

  1. `logger.info(...)` calls use the standard `logging` module instead of
     the canonical script's custom setup_logging() — same call sites, same
     messages, just pointed at whatever logger the caller passes in.
  2. Nothing else. No renamed variables, no restructured loops, no changed
     tie-break order, no changed insertion order into `candidates` / `raw_edges`.

Do NOT "clean up" or "simplify" this file. Its whole reason for existing
separately from the rest of the codebase is that its behavior must be
byte-for-byte identical to the simulated harness for the same (embeddings,
sim_matrix, seed) input — that's what tests/test_policy_fidelity.py checks
against the original file directly. If you need different behavior, add a
new function elsewhere and call out the difference explicitly; don't edit
these.
"""

import time
import logging
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp

from exp_config import SEM_THRESHOLD, SEM_WEIGHT, STRUCT_WEIGHT, MAX_DEGREE


# ── KV Cache Simulator (verbatim) ──────────────────────────────────────────
class KVCacheSimulator:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache    = OrderedDict()

    def access(self, chunk_id):
        hit = chunk_id in self.cache
        if hit:
            self.cache.move_to_end(chunk_id)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[chunk_id] = True
        return hit

    def prefetch(self, ids):
        for cid in ids:
            if cid not in self.cache:
                if len(self.cache) >= self.capacity:
                    self.cache.popitem(last=False)
                self.cache[cid] = True


# ── Graph construction (verbatim) ──────────────────────────────────────────
def build_graph(embeddings: np.ndarray, sim_matrix: np.ndarray,
                logger: logging.Logger, graph_construction="topm", top_m=20):
    """
    Builds the fixed-weight adj_matrix (used for the 'graph' baseline)
    AND the raw_edges store (used for adaptive re-ranking).

    graph_construction="threshold": semantic edge (i,j) iff sim(i,j) >= SEM_THRESHOLD.
    graph_construction="topm":      semantic edge (i,j) iff j is one of i's top-`top_m`
                                     most similar chunks, OR i is one of j's (symmetric
                                     kNN graph) — decouples degree from the corpus's raw
                                     similarity distribution. This is the construction
                                     that produced the good results in final_v2.py —
                                     v1 of this file was still threshold-only, which is
                                     why it wasn't comparable yet. Default here matches
                                     that: topm, top_m=20.

    raw_edges: dict  i -> list of (j, sem_c, struct_score)
      sem_c       = cosine sim if it qualifies as a semantic edge, else 0.0
      struct_score= 1 / (1 + |i-j|)
    These raw components are stored so adaptive weights can be applied
    at query time without rebuilding the graph.
    """
    n = len(embeddings)
    logger.info(f"  Building graph for {n} chunks ...")
    logger.info(f"  Graph construction: {graph_construction}" +
                (f" (top_m={top_m})" if graph_construction == "topm" else f" (SEM_THRESHOLD={SEM_THRESHOLD})") +
                f" | MAX_DEGREE={MAX_DEGREE}")
    t0 = time.perf_counter()

    topm_sets = None
    if graph_construction == "topm":
        topm_sets = []
        for i in range(n):
            row = sim_matrix[i].copy()
            row[i] = -1.0
            m = min(top_m, n - 1)
            top_idx = np.argpartition(row, -m)[-m:] if m < n - 1 else np.argsort(row)[::-1]
            topm_sets.append(set(top_idx.tolist()))

    rows, cols, data = [], [], []
    raw_edges: dict = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            sem  = float(sim_matrix[i, j])
            dist = abs(i - j)
            ss   = 1.0 / (1.0 + dist)
            is_adj = (dist == 1)
            if graph_construction == "topm":
                is_sem = (j in topm_sets[i]) or (i in topm_sets[j])
            else:
                is_sem = (sem >= SEM_THRESHOLD)
            if is_sem or is_adj:
                sem_c  = sem if is_sem else 0.0
                weight = SEM_WEIGHT * sem_c + STRUCT_WEIGHT * ss
                rows  += [i, j]; cols += [j, i]; data += [weight, weight]
                # Store raw components for adaptive re-ranking
                raw_edges[i].append((j, sem_c, ss))
                raw_edges[j].append((i, sem_c, ss))

    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Prune to MAX_DEGREE (two-pass)
    p_rows, p_cols, p_data = [], [], []
    for i in range(n):
        row_slice = adj.getrow(i)
        if row_slice.nnz == 0: continue
        if row_slice.nnz <= MAX_DEGREE:
            p_rows += row_slice.indices.tolist()
            p_cols += [i] * row_slice.nnz
            p_data += row_slice.data.tolist()
        else:
            top_idx = np.argsort(row_slice.data)[::-1][:MAX_DEGREE]
            p_rows  += row_slice.indices[top_idx].tolist()
            p_cols  += [i] * MAX_DEGREE
            p_data  += row_slice.data[top_idx].tolist()

    adj_pruned = sp.csr_matrix((p_data, (p_cols, p_rows)), shape=(n, n))
    adj_pruned.eliminate_zeros()
    elapsed = time.perf_counter() - t0
    logger.info(f"  Graph: {elapsed:.3f}s | edges={adj_pruned.nnz // 2} | "
                f"RAM≈{adj_pruned.data.nbytes/1024:.1f} KB | raw_edges built ✓")
    return adj_pruned, raw_edges, elapsed


# ── Prefetch policies (verbatim) ───────────────────────────────────────────
def get_prefetch_cosine(sim_matrix: np.ndarray, pri: int, k: int) -> list:
    sims = sim_matrix[pri].copy()
    sims[pri] = -1
    return np.argsort(sims)[::-1][:k].tolist()


def get_prefetch_graph(adj_matrix: sp.csr_matrix, pri: int, k: int) -> list:
    """
    Two-hop graph retrieval, min/max-blended scoring.

    A pure product (w1*w2) crushes real-but-modest paths (e.g. a pure
    structural pri->pri+1->pri+2 chain, weight ~0.15 each hop) relative to
    any 2-hop path through two moderately-strong semantic edges. Pure min()
    fixes that but throws away information about the stronger edge. Blending
    (0.9*min + 0.1*max) keeps both hop-counts on the same scale (so ranking
    reflects path strength, not hop count) while still distinguishing
    0.8->0.8 from 0.8->0.4, which pure min() can't.

    Naturally returns fewer than k if fewer than k candidates are reachable
    (list slicing doesn't pad) — this already IS the "adaptive K" behavior;
    no separate min(k, degree) check needed.
    """
    candidates = {}
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


def get_prefetch_adaptive(raw_edges: dict, pri: int, k: int,
                          q_type: str, adaptive_weights: dict) -> list:
    """
    Adaptive graph prefetch — now two-hop, matching get_prefetch_graph's
    reach, but each hop is scored with query-type-specific (alpha, beta)
    instead of the fixed (0.7, 0.3), then blended the same way (0.9*min +
    0.1*max). Before this version, adaptive only looked at 1-hop neighbors,
    which meant it couldn't win on multi-hop queries even in principle —
    that's the query type where the fixed graph's 2-hop retrieval helps
    most, so adaptive needs the same reach to be a fair comparison.

    No GPU work, no model calls — pure Python lookup.
    """
    neighbours = raw_edges.get(pri, [])
    if not neighbours:
        return []

    weights = adaptive_weights.get(q_type, adaptive_weights.get("multi-hop",
              {"alpha": SEM_WEIGHT, "beta": STRUCT_WEIGHT}))
    alpha, beta = weights["alpha"], weights["beta"]

    candidates = {}
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
    return [j for j, _ in ranked[:k]]  # was ranked[:MAX_DEGREE][:k] — capped every K>10 to 10


def generate_pairs(n_chunks, n_pairs, sim_matrix, seed, logger):
    rng    = np.random.default_rng(seed)
    pairs  = []
    safe_n = max(n_chunks - 3, 1)
    for _ in range(n_pairs):
        pri = int(rng.integers(0, safe_n))
        r   = rng.random()
        if r < 0.25:
            q_type = "semantic"
            sims   = sim_matrix[pri].copy()
            sims[max(0, pri-2):min(n_chunks, pri+3)] = -1
            sec    = int(np.argmax(sims))
        elif r < 0.50:
            q_type = "structural"
            offset = int(rng.choice([1, 2, 3]))
            sec    = min(n_chunks - 1, pri + offset)
        else:
            q_type = "multi-hop"
            if rng.random() < 0.5:
                sec = min(n_chunks - 1, pri + 2)
            else:
                sims = sim_matrix[pri].copy()
                sims[max(0, pri-2):min(n_chunks, pri+3)] = -1
                sec  = int(np.argmax(sims))
        pairs.append((pri, sec, q_type))
    counts = {t: sum(1 for _, _, qt in pairs if qt == t)
              for t in ["semantic", "structural", "multi-hop"]}
    logger.info(f"  Generated {n_pairs} pairs: {counts}")
    return pairs
