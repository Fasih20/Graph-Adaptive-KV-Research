"""
kv_cache_experiment_adaptive_v2.py
===================================
v2 adds an ADAPTIVE GRAPH policy (4th policy) alongside cold/cosine/graph.

The adaptive policy uses query-type-specific (alpha, beta) weights learned
from a grid search on prior experiment results (adaptive_grid_search_v2.py).
At prefetch time, the query type is classified and the corresponding
(alpha, beta) pair re-ranks the graph edges before selecting top-K.

CHANGES vs v1 of this file:
  - Ported Top-M graph construction (--graph_construction topm --top_m 20,
    matching final_v2.py) into build_graph. v1 was still threshold-only,
    which is why its results weren't comparable to the good topm numbers.
  - Ported 2-hop retrieval into BOTH get_prefetch_graph (fixed) and
    get_prefetch_adaptive — v1's adaptive policy was 1-hop only, which
    meant it structurally couldn't win on multi-hop queries, exactly the
    query type the fixed graph's 2-hop retrieval helps most. Scoring uses
    0.9*min(w1,w2) + 0.1*max(w1,w2) (a small refinement over pure min():
    still bottleneck-scored so a weak structural-only chain isn't crushed
    by a product, but distinguishes 0.8->0.8 from 0.8->0.4, which pure
    min() can't).
  - ONE (alpha, beta) per query type, used at every K — not per-K weights.
    See main()'s comment for the reasoning (overfitting risk with only one
    dataset and no held-out split).
  - "semantic" is never learned. It's defined (generate_pairs) as
    sec = argmax(cosine_similarity[pri]) — literally the ranking cosine
    computes, so cosine's top-K is guaranteed to contain it for any K>=1.
    No (alpha, beta) can close that gap in principle. adaptive_grid_search_v2.py
    skips it; this file uses a fixed heavy-alpha default for it instead
    (FALLBACK_ADAPTIVE_WEIGHTS["semantic"]).

New CSV columns vs v3 (unchanged from v1):
  adaptive_ms          — latency under adaptive policy
  adp_sim_hit          — cache hit under adaptive policy
  adaptive_n_tokens    — cached token count under adaptive policy
  flops_saved_adaptive — theoretical FLOPs saved

To run:
  # First run grid search to get adaptive_weights.json:
  python adaptive_grid_search_v2.py --csv_dir ./results \
      --dataset_key hotpot --model_tag Qwen2.5-1.5B \
      --output_dir ./adaptive_results --graph_construction topm --top_m 20

  # Then run this script pointing at the weights file — graph_construction
  # MUST match what the grid search used, or the learned weights are being
  # evaluated against a different graph than they were tuned on:
  python kv_cache_experiment_adaptive_v2.py \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --datasets hotpot --graph_construction topm --top_m 20 \
      --adaptive_weights ./adaptive_results/adaptive_weights_Qwen2.5-1.5B_hotpot.json

  # Without weights — runs fixed graph only (same policy as final_v2.py):
  python kv_cache_experiment_adaptive_v2.py --model Qwen/Qwen2.5-1.5B-Instruct --datasets hotpot

HIT-AWARE TIMING (ported from kv_cache_experiment_v4.py, no changes to the fix
itself — see that file's header for the full writeup):
  Previously every policy (cold/cosine/graph/adaptive) UNCONDITIONALLY ran
  extend_with_cache(sec) every trial regardless of whether the simulator
  reported a hit for sec. That means a "hit" never got cheaper than a miss,
  so cold (shortest resident context, cheapest FLOPs) could only ever look
  equal-or-better than the prefetch policies — never worse — which defeats
  the point of the experiment on hardware where attention FLOPs aren't
  swamped by kernel-launch overhead.

  THE FIX: when {lru,cos,grp,adp}_hit is True for a trial, skip the final
  extend_with_cache(s_text, ...) call entirely and record HIT_LATENCY_MS
  (default 0.0 — a real cache lookup costs no GPU forward pass) instead of
  a fresh measurement. On a miss, behavior is UNCHANGED: a real, measured
  extend_with_cache call. Pass --legacy_timing to reproduce the old
  always-measure behavior for an A/B comparison.

  IMPORTANT: because this changes what cold_ms/cosine_ms/graph_ms/adaptive_ms
  mean, point --output_dir at a NEW folder rather than resuming into an old
  results_adaptive_*.csv.
"""

import os
import sys
import time
import json
import logging
import argparse
import csv
import zipfile
import tempfile
from datetime import datetime
from collections import OrderedDict
from pathlib import Path

#os.environ["CUDA_LAUNCH_BLOCKING"]   = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False

# ── Constants (identical to v3 / Colab) ───────────────────────────────────────
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
SEM_THRESHOLD   = 0.55
SEM_WEIGHT      = 0.7          # fixed-graph default alpha
STRUCT_WEIGHT   = 0.3          # fixed-graph default beta
MAX_DEGREE      = 100   # was 10 — mismatched final_v2.py's 100, and was quietly pruning
                        # the graph baseline itself down to 10 neighbors/node under topm=20
RANDOM_SEED     = 42
N_PAIRS         = 600
SEPARATOR       = "\n\n"
PRIMARY_MAX_LEN = 512
EXTEND_MAX_LEN  = 256
MAX_RETRIES     = 3
N_WARMUP_PASSES = 3
HIT_LATENCY_MS  = 0.0              # v4 — cost charged to a cache HIT (no forward pass
                                    # is run at all, so 0.0 is the honest number; bump this
                                    # only if you want to model block-table/gather overhead)

DATASET_CONFIGS = {
    "hotpot":     {"hf_name": "THUDM/LongBench", "split": "hotpotqa",        "text_key": "context", "label": "HotpotQA"},
    "2wiki":      {"hf_name": "THUDM/LongBench", "split": "2wikimqa",        "text_key": "context", "label": "2WikiMultiHopQA"},
    "musique":    {"hf_name": "THUDM/LongBench", "split": "musique",         "text_key": "context", "label": "MuSiQue"},
    "multifield": {"hf_name": "THUDM/LongBench", "split": "multifieldqa_en", "text_key": "context", "label": "MultiFieldQA-en"},
}

ABLATION_CONFIGS = [
    {"K": 3, "CACHE_CAPACITY": 16},
    {"K": 4, "CACHE_CAPACITY": 16},
    {"K": 5, "CACHE_CAPACITY": 16},
    {"K": 6, "CACHE_CAPACITY": 16},
    # {"K": 7, "CACHE_CAPACITY": 16},
    {"K": 8, "CACHE_CAPACITY": 16},
    {"K": 10, "CACHE_CAPACITY": 16},
    {"K": 12, "CACHE_CAPACITY": 16},
    {"K": 14, "CACHE_CAPACITY": 16},
    {"K": 16, "CACHE_CAPACITY": 16},
]

# Fallback adaptive weights when no JSON is provided, and the fixed value
# always used for "semantic" regardless of what the grid search finds.
#
# Semantic queries are defined (see generate_pairs) as sec = argmax(cosine
# similarity to pri). Cosine's own top-K is therefore GUARANTEED to contain
# that target for any K >= 1 -- it's not a fair comparison, it's asking a
# threshold/topm-gated graph to match the exact ranking being tested against.
# No (alpha, beta) can close this gap in principle, so we don't grid-search
# it (adaptive_grid_search_v2.py skips "semantic" entirely) and just use a
# heavy-alpha default here that mimics cosine's own ranking as closely as
# the graph structure allows, rather than wasting search budget on it.
FALLBACK_ADAPTIVE_WEIGHTS = {
    "semantic":   {"alpha": 0.90, "beta": 0.10},   # fixed, not learned -- see above
    "structural": {"alpha": 0.40, "beta": 0.60},
    "multi-hop":  {"alpha": 0.70, "beta": 0.30},
}


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging(output_dir: Path, model_tag: str) -> logging.Logger:
    log_dir   = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"run_{model_tag}_{timestamp}.log"

    logger = logging.getLogger("kv_cache_v4")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_file}")
    return logger


def cuda_reset(logger):
    for fn in [torch.cuda.synchronize, torch.cuda.empty_cache, torch.cuda.ipc_collect]:
        try: fn()
        except Exception: pass
    logger.warning("  CUDA context flushed after error.")


def safe_sync():
    try: torch.cuda.synchronize()
    except Exception: pass


# ── KV Cache Simulator ─────────────────────────────────────────────────────────
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


# ── Graph construction ─────────────────────────────────────────────────────────
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
    raw_edges: dict[int, list] = {i: [] for i in range(n)}

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


# ── Prefetch policies ──────────────────────────────────────────────────────────
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


# ── KV engine (identical to v3 / Colab) ───────────────────────────────────────
@torch.no_grad()
def compute_kv_cache(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=PRIMARY_MAX_LEN).to(device)
    out = model(**inputs, use_cache=True)
    safe_sync()
    return out.past_key_values, inputs["input_ids"]


@torch.no_grad()
def extend_with_cache(new_text, past_kv, past_ids, tokenizer, model, device,
                      measure=True):
    new_inputs = tokenizer(
        new_text, return_tensors="pt", truncation=True,
        max_length=EXTEND_MAX_LEN, add_special_tokens=False
    ).to(device)
    past_len = past_ids.shape[1]
    pos_ids  = torch.arange(
        past_len, past_len + new_inputs["input_ids"].shape[1], device=device
    ).unsqueeze(0)

    if measure:
        safe_sync()
        t0 = time.perf_counter()

    out = model(input_ids=new_inputs["input_ids"],
                past_key_values=past_kv,
                position_ids=pos_ids,
                use_cache=True)

    if measure:
        safe_sync()
        ms = (time.perf_counter() - t0) * 1000
    else:
        ms = 0.0

    full_ids = torch.cat([past_ids, new_inputs["input_ids"]], dim=1)
    return out.past_key_values, full_ids, ms


def estimate_flops_saved(n_cached, n_new, hidden_dim, n_heads, n_layers):
    L_full  = n_cached + n_new
    full    = 4 * hidden_dim * (L_full ** 2)
    cached  = 4 * hidden_dim * (n_new ** 2 + 2 * n_new * n_cached)
    return (full - cached) * n_layers


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


# ── GPU Warmup ─────────────────────────────────────────────────────────────────
def warmup_gpu(tokenizer, model, device, logger, n_passes=N_WARMUP_PASSES):
    logger.info(f"  Running {n_passes} GPU warmup passes...")
    dummy_p = "The quick brown fox jumps over the lazy dog. " * 15
    dummy_e = "Warmup extension text for CUDA kernel initialisation. " * 8
    for i in range(n_passes):
        try:
            kv, ids = compute_kv_cache(dummy_p, tokenizer, model, device)
            _, _, ms = extend_with_cache(dummy_e, kv, ids, tokenizer, model, device)
            del kv, ids
            torch.cuda.empty_cache()
            logger.info(f"    Warmup {i+1}/{n_passes}: {ms:.1f}ms")
        except Exception as e:
            logger.warning(f"    Warmup {i+1} failed (non-fatal): {e}")
            cuda_reset(logger)
    logger.info("  Warmup complete.")


# ── Single trial ───────────────────────────────────────────────────────────────
def run_trial(pri, sec, q_type,
              chunk_texts, sim_matrix, adj_matrix, raw_edges,
              K, sim_lru, sim_cos, sim_grp, sim_adp,
              adaptive_weights, tokenizer, model, device,
              model_config, logger, retries=MAX_RETRIES, legacy_timing=False):
    """
    Runs one trial with FOUR policies:
      cold / cosine / graph (fixed) / adaptive-graph

    The adaptive policy uses per-query-type (alpha, beta) loaded from
    adaptive_weights JSON. It runs the same GPU measurement as the fixed
    graph — just a different set of chunk IDs is prefetched.

    v4 hit-aware timing: when a policy's simulator reports a hit for sec,
    the final extend_with_cache(sec) is skipped and HIT_LATENCY_MS is
    recorded instead — a real cache hit costs no GPU forward pass. On a
    miss, behavior is unchanged: a real, measured extend_with_cache call.
    """
    p_text = chunk_texts[pri]
    s_text = chunk_texts[sec]

    for attempt in range(1, retries + 1):
        try:
            # ── Simulator hit tracking ──
            sim_lru.access(pri)
            lru_hit = int(sim_lru.access(sec))

            c_ids = get_prefetch_cosine(sim_matrix, pri, K)
            sim_cos.access(pri); sim_cos.prefetch(c_ids)
            cos_hit = int(sim_cos.access(sec))

            g_ids = get_prefetch_graph(adj_matrix, pri, K)
            sim_grp.access(pri); sim_grp.prefetch(g_ids)
            grp_hit = int(sim_grp.access(sec))

            a_ids = get_prefetch_adaptive(raw_edges, pri, K, q_type, adaptive_weights)
            sim_adp.access(pri); sim_adp.prefetch(a_ids)
            adp_hit = int(sim_adp.access(sec))

            c_texts = [chunk_texts[i] for i in c_ids]
            g_texts = [chunk_texts[i] for i in g_ids]
            a_texts = [chunk_texts[i] for i in a_ids]

            # ── Cold ──
            torch.cuda.empty_cache()
            kv, ids    = compute_kv_cache(p_text, tokenizer, model, device)
            cold_n_tok = ids.shape[1]
            if lru_hit and not legacy_timing:
                cold_ms = HIT_LATENCY_MS
            else:
                _, _, cold_ms = extend_with_cache(s_text, kv, ids, tokenizer, model, device)

            # ── Cosine ──
            torch.cuda.empty_cache()
            kv, ids = compute_kv_cache(p_text, tokenizer, model, device)
            if c_texts:
                kv, ids, _ = extend_with_cache(SEPARATOR.join(c_texts), kv, ids,
                                               tokenizer, model, device, measure=False)
            cos_n_tok = ids.shape[1]
            if cos_hit and not legacy_timing:
                cosine_ms = HIT_LATENCY_MS
            else:
                _, _, cosine_ms = extend_with_cache(s_text, kv, ids, tokenizer, model, device)

            # ── Fixed graph ──
            torch.cuda.empty_cache()
            kv, ids = compute_kv_cache(p_text, tokenizer, model, device)
            if g_texts:
                kv, ids, _ = extend_with_cache(SEPARATOR.join(g_texts), kv, ids,
                                               tokenizer, model, device, measure=False)
            grp_n_tok = ids.shape[1]
            if grp_hit and not legacy_timing:
                graph_ms = HIT_LATENCY_MS
            else:
                _, _, graph_ms = extend_with_cache(s_text, kv, ids, tokenizer, model, device)

            # ── Adaptive graph ──
            torch.cuda.empty_cache()
            kv, ids = compute_kv_cache(p_text, tokenizer, model, device)
            if a_texts:
                kv, ids, _ = extend_with_cache(SEPARATOR.join(a_texts), kv, ids,
                                               tokenizer, model, device, measure=False)
            adp_n_tok = ids.shape[1]
            if adp_hit and not legacy_timing:
                adaptive_ms = HIT_LATENCY_MS
            else:
                _, _, adaptive_ms = extend_with_cache(s_text, kv, ids, tokenizer, model, device)

            # ── FLOPs ──
            s_tok = tokenizer(s_text, return_tensors="pt", truncation=True,
                              max_length=EXTEND_MAX_LEN, add_special_tokens=False
                              )["input_ids"].shape[1]

            def flops(n_cached):
                return estimate_flops_saved(n_cached, s_tok, model_config["hidden_dim"],
                                            model_config["n_heads"], model_config["n_layers"]
                                            ) if n_cached > 0 else 0.0

            return {
                "cold_ms":            round(cold_ms,     4),
                "cosine_ms":          round(cosine_ms,   4),
                "graph_ms":           round(graph_ms,    4),
                "adaptive_ms":        round(adaptive_ms, 4),
                "lru_sim_hit":        lru_hit,
                "cos_sim_hit":        cos_hit,
                "grp_sim_hit":        grp_hit,
                "adp_sim_hit":        adp_hit,
                "cold_n_tokens":      cold_n_tok,
                "cosine_n_tokens":    cos_n_tok,
                "graph_n_tokens":     grp_n_tok,
                "adaptive_n_tokens":  adp_n_tok,
                "flops_saved_cosine": round(flops(cos_n_tok), 0),
                "flops_saved_graph":  round(flops(grp_n_tok), 0),
                "flops_saved_adaptive": round(flops(adp_n_tok), 0),
            }

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.warning(f"  Trial attempt {attempt}/{retries} failed: {e}")
            cuda_reset(logger)
            if attempt == retries:
                logger.error(f"  Giving up on trial pri={pri} sec={sec}")
                return None
            time.sleep(2 * attempt)
    return None


# ── Dataset loop ───────────────────────────────────────────────────────────────
def run_dataset(dataset_key, model_name, tokenizer, model, device,
                output_dir, logger, model_config, adaptive_weights,
                legacy_timing=False, graph_construction="topm", top_m=20):
    cfg   = DATASET_CONFIGS[dataset_key]
    label = cfg["label"]
    logger.info("=" * 70)
    logger.info(f"DATASET: {label}  |  MODEL: {model_name}")
    logger.info("=" * 70)
    logger.info(f"Adaptive weights (all K): {adaptive_weights}")

    t0 = time.perf_counter()
    zip_path = hf_hub_download(repo_id=cfg["hf_name"], filename="data.zip",
                               repo_type="dataset")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            target = f"data/{cfg['split']}.jsonl"
            z.extract(target, tmpdir)
            df_raw = pd.read_json(os.path.join(tmpdir, target), lines=True)

    contexts = []
    for ctx in df_raw[cfg["text_key"]]:
        contexts.append(ctx)
        if sum(len(c) for c in contexts) >= 100_000:
            break
    raw = "\n\n".join(contexts)
    logger.info(f"  Raw text: {len(raw):,} chars  ({time.perf_counter()-t0:.1f}s)")

    splitter    = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)
    chunk_texts = splitter.split_text(raw)
    n           = len(chunk_texts)
    logger.info(f"  Chunks: {n}")

    logger.info("Computing embeddings ...")
    t0         = time.perf_counter()
    embedder   = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunk_texts, batch_size=32,
                                 convert_to_numpy=True, show_progress_bar=False)
    embed_time = time.perf_counter() - t0
    logger.info(f"  Embeddings: {embeddings.shape}  ({embed_time:.2f}s)")

    sim_matrix                     = cosine_similarity(embeddings)
    adj_matrix, raw_edges, build_time = build_graph(embeddings, sim_matrix, logger,
                                                     graph_construction=graph_construction, top_m=top_m)

    logger.info("Generating query pairs ...")
    all_pairs = generate_pairs(n, N_PAIRS, sim_matrix, RANDOM_SEED, logger)

    safe_model = model_name.replace("/", "_").replace("-", "_")
    tag        = "legacy" if legacy_timing else "hitaware"
    gc_tag     = graph_construction if graph_construction == "threshold" else f"topm{top_m}"
    csv_path   = output_dir / f"results_adaptive_{safe_model}_{dataset_key}_{tag}_{gc_tag}.csv"
    csv_exists = csv_path.exists()

    completed = {}
    if csv_exists:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (int(row["config_k"]), int(row["config_cap"]))
                completed[key] = completed.get(key, 0) + 1
        logger.info(f"  Resuming — completed rows: {completed}")

    fieldnames = [
        "model_name", "gpu_name", "seed", "dataset", "config_k", "config_cap", "run", "type",
        "cold_ms", "cosine_ms", "graph_ms", "adaptive_ms",
        "lru_sim_hit", "cos_sim_hit", "grp_sim_hit", "adp_sim_hit",
        "cold_n_tokens", "cosine_n_tokens", "graph_n_tokens", "adaptive_n_tokens",
        "flops_saved_cosine", "flops_saved_graph", "flops_saved_adaptive",
        "embed_time_s", "graph_build_time_s",
        "adp_alpha_semantic", "adp_beta_semantic",
        "adp_alpha_structural", "adp_beta_structural",
        "adp_alpha_multihop", "adp_beta_multihop",
    ]
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    skipped_trials = 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        for ablation in ABLATION_CONFIGS:
            K, CAP = ablation["K"], ablation["CACHE_CAPACITY"]
            # adaptive_weights is now fixed across every K (see main() rationale) —
            # no more per-K lookup here.
            key    = (K, CAP)
            start_i = completed.get(key, 0)
            if start_i >= N_PAIRS:
                logger.info(f"  ⏭  K={K} already complete ({start_i} rows)")
                continue

            logger.info(f"\n  ── K={K}, cap={CAP} | adaptive weights: {adaptive_weights} | "
                        f"resuming from run {start_i} ──")
            warmup_gpu(tokenizer, model, device, logger, n_passes=1)

            sim_lru = KVCacheSimulator(CAP)
            sim_cos = KVCacheSimulator(CAP)
            sim_grp = KVCacheSimulator(CAP)
            sim_adp = KVCacheSimulator(CAP)
            config_start = time.perf_counter()

            for run_i, (pri, sec, q_type) in enumerate(all_pairs):
                if run_i < start_i:
                    continue

                result = run_trial(
                    pri, sec, q_type, chunk_texts, sim_matrix, adj_matrix, raw_edges,
                    K, sim_lru, sim_cos, sim_grp, sim_adp,
                    adaptive_weights, tokenizer, model, device, model_config, logger,
                    legacy_timing=legacy_timing,
                )

                if result is None:
                    skipped_trials += 1
                    logger.warning(f"  Skipping run {run_i} (all retries failed)")
                    continue

                writer.writerow({
                    "model_name": model_name, "gpu_name": gpu_name,
                    "seed": RANDOM_SEED, "dataset": label,
                    "config_k": K, "config_cap": CAP, "run": run_i, "type": q_type,
                    "embed_time_s": round(embed_time, 3),
                    "graph_build_time_s": round(build_time, 3),
                    "adp_alpha_semantic":    adaptive_weights.get("semantic",    {}).get("alpha", SEM_WEIGHT),
                    "adp_beta_semantic":     adaptive_weights.get("semantic",    {}).get("beta",  STRUCT_WEIGHT),
                    "adp_alpha_structural":  adaptive_weights.get("structural",  {}).get("alpha", SEM_WEIGHT),
                    "adp_beta_structural":   adaptive_weights.get("structural",  {}).get("beta",  STRUCT_WEIGHT),
                    "adp_alpha_multihop":    adaptive_weights.get("multi-hop",   {}).get("alpha", SEM_WEIGHT),
                    "adp_beta_multihop":     adaptive_weights.get("multi-hop",   {}).get("beta",  STRUCT_WEIGHT),
                    **result,
                })
                f.flush()
                os.fsync(f.fileno())

                if (run_i + 1) % 50 == 0 or run_i == N_PAIRS - 1:
                    elapsed = time.perf_counter() - config_start
                    done    = run_i - start_i + 1
                    eta_s   = (elapsed / done) * (N_PAIRS - run_i - 1) if done > 0 else 0
                    logger.info(
                        f"     K={K} | {run_i+1:>3}/{N_PAIRS} | {elapsed/60:.1f}min | "
                        f"ETA {eta_s/60:.1f}min | "
                        f"cold={result['cold_ms']:.1f} "
                        f"cos={result['cosine_ms']:.1f} "
                        f"grp={result['graph_ms']:.1f} "
                        f"adp={result['adaptive_ms']:.1f} ms"
                    )

            logger.info(f"  ✓ K={K} in {(time.perf_counter()-config_start)/60:.1f} min")

    if skipped_trials:
        logger.warning(f"  Total skipped trials: {skipped_trials}")
    logger.info(f"  CSV: {csv_path}")
    return csv_path


# ── Summary ────────────────────────────────────────────────────────────────────
def print_summary(csv_path: Path, logger: logging.Logger, adaptive_weights: dict):
    df = pd.read_csv(csv_path)
    logger.info("\n" + "=" * 80)
    logger.info(f"SUMMARY (v4 — adaptive)  —  {csv_path.name}")
    logger.info("=" * 80)
    logger.info(f"{'Policy':>12}  {'Description'}")
    logger.info(f"{'cosine':>12}  Top-K by raw cosine similarity (baseline)")
    logger.info(f"{'graph':>12}  Top-K by hybrid weight α=0.70 β=0.30 (fixed)")
    logger.info(f"{'adaptive':>12}  Top-K by query-type-specific α/β (learned; same weight at every K)")
    logger.info("-" * 80)

    for k in sorted(df["config_k"].unique()):
        sub = df[df["config_k"] == k]
        cos_m = sub["cosine_ms"].mean()
        grp_m = sub["graph_ms"].mean()
        adp_m = sub["adaptive_ms"].mean()
        cos_hr = sub["cos_sim_hit"].mean() * 100
        grp_hr = sub["grp_sim_hit"].mean() * 100
        adp_hr = sub["adp_sim_hit"].mean() * 100
        _, p_cg = stats.ttest_rel(sub["cosine_ms"],   sub["graph_ms"])
        _, p_ca = stats.ttest_rel(sub["cosine_ms"],   sub["adaptive_ms"])
        _, p_ga = stats.ttest_rel(sub["graph_ms"],    sub["adaptive_ms"])

        def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"

        logger.info(f"\n  K={k}")
        logger.info(f"    cosine   : {cos_m:.2f}ms  HR={cos_hr:.1f}%")
        logger.info(f"    graph    : {grp_m:.2f}ms  HR={grp_hr:.1f}%  "
                    f"Δ_vs_cos={cos_m-grp_m:+.3f}ms {sig(p_cg)}")
        logger.info(f"    adaptive : {adp_m:.2f}ms  HR={adp_hr:.1f}%  "
                    f"Δ_vs_cos={cos_m-adp_m:+.3f}ms {sig(p_ca)}  "
                    f"Δ_vs_grp={grp_m-adp_m:+.3f}ms {sig(p_ga)}")

        for qt in ["semantic", "structural", "multi-hop"]:
            tdf = sub[sub["type"] == qt]
            if len(tdf) < 2: continue
            w = adaptive_weights.get(qt, {})
            d_cg = tdf["cosine_ms"].mean() - tdf["graph_ms"].mean()
            d_ca = tdf["cosine_ms"].mean() - tdf["adaptive_ms"].mean()
            d_ga = tdf["graph_ms"].mean()  - tdf["adaptive_ms"].mean()
            _, p1 = stats.ttest_rel(tdf["cosine_ms"], tdf["graph_ms"])
            _, p2 = stats.ttest_rel(tdf["cosine_ms"], tdf["adaptive_ms"])
            _, p3 = stats.ttest_rel(tdf["graph_ms"],  tdf["adaptive_ms"])
            logger.info(
                f"    [{qt:>10}] n={len(tdf)}  "
                f"α={w.get('alpha','0.70'):.2f}/β={w.get('beta','0.30'):.2f}  "
                f"grp-cos={d_cg:+.3f}{sig(p1)}  "
                f"adp-cos={d_ca:+.3f}{sig(p2)}  "
                f"adp-grp={d_ga:+.3f}{sig(p3)}"
            )


# ── Args ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()),
                   default=["hotpot"])
    p.add_argument("--output_dir", default="./results_v4")
    p.add_argument("--adaptive_weights", default=None,
                   help="Path to adaptive_weights_*.json from grid search. "
                        "If omitted, uses built-in fallback priors.")
    p.add_argument("--torch_dtype", default="auto",
                   choices=["auto", "float16", "bfloat16"])
    p.add_argument("--load_in_8bit",  action="store_true")
    p.add_argument("--load_in_4bit",  action="store_true")
    p.add_argument("--legacy_timing", action="store_true",
                    help="Reproduce pre-v4 behavior (always re-encode sec, even on a hit). "
                         "Use this ONLY to regenerate the old numbers for an A/B table in the "
                         "paper — leave it off for the corrected, hit-aware run.")
    p.add_argument("--graph_construction", default="topm", choices=["threshold", "topm"],
                    help="Must match whatever adaptive_grid_search_v2.py was run with — "
                         "training weights on one graph and evaluating on another silently "
                         "invalidates the comparison. Default topm matches final_v2.py.")
    p.add_argument("--top_m", type=int, default=20,
                    help="Neighbors per node when --graph_construction topm.")
    return p.parse_args()


# MODEL_CONFIGS = {
#     "Qwen2.5-7B":   {"hidden_dim": 3584, "n_heads": 28, "n_layers": 28},
#     "Qwen2.5-14B":  {"hidden_dim": 5120, "n_heads": 40, "n_layers": 48},
#     "Qwen2.5-1.5B": {"hidden_dim": 1536, "n_heads": 12, "n_layers": 28},
#     "Qwen2.5-3B":   {"hidden_dim": 2048, "n_heads": 16, "n_layers": 36},
#     "default":      {"hidden_dim": 4096, "n_heads": 32, "n_layers": 32},
# }

MODEL_CONFIGS = {
    # Qwen
    "Qwen2.5-1.5B": {"hidden_dim":1536,"n_heads":12,"n_layers":28},
    "Qwen2.5-3B":   {"hidden_dim":2048,"n_heads":16,"n_layers":36},
    "Qwen2.5-7B":   {"hidden_dim":3584,"n_heads":28,"n_layers":28},
    "Qwen2.5-14B":  {"hidden_dim":5120,"n_heads":40,"n_layers":48},

    # Llama
    "Llama-3.2-1B": {"hidden_dim":2048,"n_heads":32,"n_layers":16},
    "Llama-3.2-3B": {"hidden_dim":3072,"n_heads":24,"n_layers":28},

    # Gemma
    "gemma-3-1b": {"hidden_dim":1152,"n_heads":4,"n_layers":26},
    "gemma-3-4b": {"hidden_dim":2560,"n_heads":8,"n_layers":34},

    # Phi
    "Phi-3.5": {"hidden_dim":3072,"n_heads":32,"n_layers":32},
}

def get_model_config(model_name):
    for key, cfg in MODEL_CONFIGS.items():
        if key.lower() in model_name.lower(): return cfg
    return MODEL_CONFIGS["default"]


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag  = args.model.split("/")[-1]
    logger     = setup_logging(output_dir, model_tag)

    # ── Load adaptive weights (per QUERY TYPE only — not per K) ──
    # v1 of this file supported per-K weights (weights_all_k), on the theory
    # that cache-pollution dynamics differ enough across K to need separate
    # tuning at each one. We're deliberately dropping that here: fitting a
    # new (alpha, beta) at every K multiplies the free parameters by 4x on
    # a single dataset with no held-out split, which is exactly the
    # overfitting risk the weight-clamping step (adaptive_grid_search_v2.py)
    # is already trying to guard against -- adding more parameters back in
    # via per-K weights works against that. A single weight per query type
    # is a more falsifiable, more generalizable claim ("structural queries
    # want more structural signal"), and if performance genuinely drifts
    # with K under one fixed weight, that's a real, reportable finding
    # (and a sign the graph construction/scoring has a K-dependent quirk
    # worth understanding directly) rather than something to paper over by
    # refitting a new number at every K.
    if args.adaptive_weights:
        with open(args.adaptive_weights) as f:
            weights_data = json.load(f)
        loaded = weights_data.get("weights", weights_data.get("weights_at_k_sweet", {}))
        adaptive_weights = dict(FALLBACK_ADAPTIVE_WEIGHTS)
        for qt, v in loaded.items():
            if qt == "semantic":
                continue  # semantic is never learned — see FALLBACK_ADAPTIVE_WEIGHTS comment
            adaptive_weights[qt] = {"alpha": v["alpha"], "beta": v["beta"]}
        logger.info(f"Loaded adaptive weights from: {args.adaptive_weights}")
    else:
        adaptive_weights = FALLBACK_ADAPTIVE_WEIGHTS
        logger.info("No adaptive weights file — using built-in fallback priors")

    logger.info(f"Adaptive weights (used at every K): {adaptive_weights}")

    logger.info("=" * 70)
    logger.info("KV Cache Experiment v4 — Adaptive Graph Policy")
    logger.info(f"Model:    {args.model}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Output:   {output_dir.resolve()}")
    if args.legacy_timing:
        logger.warning("  --legacy_timing is ON: reproducing pre-v4 behavior "
                        "(hits are NOT cheaper). Only use this to regenerate old numbers.")
    else:
        logger.info("  Hit-aware timing ON (v4 default): a cache hit skips the real "
                     f"re-encode of sec and records {HIT_LATENCY_MS}ms.")
    logger.info("=" * 70)

    if not torch.cuda.is_available():
        logger.error("No CUDA GPU found. Exiting.")
        sys.exit(1)

    device   = "cuda:0"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name}  ({vram_gb:.1f} GB)")

    if args.torch_dtype == "bfloat16": dtype = torch.bfloat16
    elif args.torch_dtype == "float16": dtype = torch.float16
    else:
        cc    = torch.cuda.get_device_capability(0)
        dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
        logger.info(f"  Auto dtype: {'bfloat16' if dtype==torch.bfloat16 else 'float16'} (cc {cc[0]}.{cc[1]})")

    logger.info(f"Loading {args.model} ...")
    t0        = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": {"": device}, "trust_remote_code": True}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        logger.info("  4-bit quantisation (nf4)")
    elif args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        logger.info("  8-bit quantisation")
    else:
        load_kwargs["torch_dtype"] = dtype

    model     = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    load_time = time.perf_counter() - t0
    vram_used = torch.cuda.memory_allocated(0) / 1e9
    logger.info(f"Loaded in {load_time:.1f}s  |  VRAM: {vram_used:.2f} GB")

    model_config = get_model_config(args.model)
    warmup_gpu(tokenizer, model, device, logger, n_passes=N_WARMUP_PASSES)

    all_csvs    = []
    total_start = time.perf_counter()

    for ds_key in args.datasets:
        csv_path = run_dataset(
            ds_key, args.model, tokenizer, model, device,
            output_dir, logger, model_config, adaptive_weights,
            legacy_timing=args.legacy_timing,
            graph_construction=args.graph_construction, top_m=args.top_m,
        )
        all_csvs.append(csv_path)
        print_summary(csv_path, logger, adaptive_weights)

    total_h = (time.perf_counter() - total_start) / 3600
    logger.info(f"\nAll datasets done in {total_h:.2f} hrs")
    logger.info(f"CSVs: {[str(p) for p in all_csvs]}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
