"""
Graph-Guided KV Cache Chunk Prefetching — University GPU Run
Supports: 7B and 13B/14B models on RTX 5060 Ti / 5080

v4 CHANGES vs v3 — HIT-AWARE TIMING (the actual fix for "cold always wins"):
  v3 (and the Colab reference before it) computed lru_hit/cos_hit/grp_hit
  purely for bookkeeping, then UNCONDITIONALLY ran extend_with_cache(sec)
  every trial regardless of whether sec was a hit. That means every
  condition always paid full cost to re-encode `sec`, attending over an
  EQUAL-or-LONGER prior context for cosine/graph than for cold — so
  cosine/graph could only ever look equal-or-worse than cold, never
  better. There was no code path where a "hit" ever skipped real work.
  This was invisible on Colab's T4 because kernel-launch/Python overhead
  there is large relative to the extra attention FLOPs from a few
  hundred prefetched tokens, so noise partly masked it. On the 5060 Ti,
  attention is cheap and the overhead floor is low, so the FLOPs penalty
  of a longer resident context comes through cleanly — and since no
  condition could ever skip work, cold (shortest context) wins cleanly
  every time. This is a measurement-methodology issue, not a hardware
  quirk, and it applies equally to the 1.5B/3B Colab numbers already in
  the draft (see chat for details).

  THE FIX: when {lru,cos,grp}_hit is True for a trial, skip the final
  extend_with_cache(s_text, ...) call entirely and record HIT_LATENCY_MS
  (default 0.0 — a real cache lookup costs no GPU forward pass) instead
  of a fresh measurement. On a miss, behavior is UNCHANGED from v3: a
  real, measured extend_with_cache call. The hit/miss bookkeeping itself
  (KVCacheSimulator) is untouched — same LRU/eviction/capacity logic as
  before, same columns in the CSV. Only what happens to REAL GPU TIME on
  a hit has changed. Pass --legacy_timing to reproduce exact v3 behavior
  for an A/B comparison in the paper.

  IMPORTANT: because this changes what cold_ms/cosine_ms/graph_ms mean,
  point --output_dir at a NEW folder rather than resuming into an old
  results_*.csv — the resume-by-row-count logic in run_dataset() has no
  way to know old rows used the old (buggy) timing and will happily
  "resume past" them.

v3 CHANGES vs previous WSL script:
  - METHODOLOGY NOW MATCHES THE COLAB REFERENCE EXACTLY:
      * extend_with_cache uses position_ids (not attention_mask)
      * torch.cuda.empty_cache() called before every cold/cosine/graph step
      * SEPARATOR = "\n\n"  (was " [SEP] ")
      * extend truncation max_length=256 (was 512)
    This matters because your 1.5B/3B Colab results and the 7B WSL
    results must be measuring the identical operation for the
    cross-scale comparison in the paper to be valid.
  - Added GPU warmup: 3 dummy forward+extend passes before the
    timing loop starts, AND before each new K config (first-call
    CUDA kernel JIT / cuDNN autotune cost was polluting early
    measurements).
  - Kept WSL stability fixes: single-device placement, retry
    logic on transient CUDA errors, CUDA_LAUNCH_BLOCKING.

Run:
  python kv_cache_experiment_v3.py --model Qwen/Qwen2.5-7B-Instruct --datasets hotpot
"""

import os
import sys
import time
import logging
import argparse
import csv
import zipfile
import tempfile
from datetime import datetime
from collections import OrderedDict
from pathlib import Path

#os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
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

# ── Constants — MATCH COLAB REFERENCE EXACTLY ─────────────────────────────────
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
SEM_THRESHOLD    = 0.55
SEM_WEIGHT       = 0.7
STRUCT_WEIGHT    = 0.3
MAX_DEGREE       = 100
TOP_M_SEMANTIC   = 20              # NEW — used only when --graph_construction topm
RANDOM_SEED      = 42
N_PAIRS          = 600
SEPARATOR        = "\n\n"          # FIXED: was " [SEP] ", now matches Colab
PRIMARY_MAX_LEN  = 512
EXTEND_MAX_LEN   = 256              # FIXED: was 512, now matches Colab
MAX_RETRIES      = 3
N_WARMUP_PASSES  = 3                # NEW
HIT_LATENCY_MS   = 0.0              # NEW v4 — cost charged to a cache HIT (no forward pass
                                     # is run at all, so 0.0 is the honest number; bump this
                                     # only if you want to model block-table/gather overhead)

DATASET_CONFIGS = {
    "hotpot":     {"hf_name": "THUDM/LongBench", "split": "hotpotqa",         "text_key": "context", "label": "HotpotQA"},
    "2wiki":      {"hf_name": "THUDM/LongBench", "split": "2wikimqa",         "text_key": "context", "label": "2WikiMultiHopQA"},
    "musique":    {"hf_name": "THUDM/LongBench", "split": "musique",          "text_key": "context", "label": "MuSiQue"},
    "multifield": {"hf_name": "THUDM/LongBench", "split": "multifieldqa_en",  "text_key": "context", "label": "MultiFieldQA-en"},
}

ABLATION_CONFIGS = [
    {"K": 3, "CACHE_CAPACITY": 16},
    {"K": 4, "CACHE_CAPACITY": 16},
    {"K": 5, "CACHE_CAPACITY": 16},
    {"K": 6, "CACHE_CAPACITY": 16},
    {"K": 8, "CACHE_CAPACITY": 16},
    {"K": 10, "CACHE_CAPACITY": 16},
    {"K": 12, "CACHE_CAPACITY": 16},
    {"K": 14, "CACHE_CAPACITY": 16},
    {"K": 16, "CACHE_CAPACITY": 16},
]


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging(output_dir: Path, model_tag: str) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"run_{model_tag}_{timestamp}.log"

    logger = logging.getLogger("kv_cache")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_file}")
    return logger


def cuda_reset(logger):
    try: torch.cuda.synchronize()
    except Exception: pass
    try: torch.cuda.empty_cache()
    except Exception: pass
    try: torch.cuda.ipc_collect()
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
def build_graph(embeddings, sim_matrix, logger, graph_construction="threshold", top_m=TOP_M_SEMANTIC):
    """
    graph_construction="threshold": semantic edge (i,j) iff sim(i,j) >= SEM_THRESHOLD (original).
    graph_construction="topm":      semantic edge (i,j) iff j is one of i's top-`top_m` most
                                     similar chunks, OR i is one of j's (symmetric kNN graph).
                                     This decouples average degree from the corpus's raw
                                     similarity distribution — MAX_DEGREE stops being a
                                     meaningful lever once it exceeds the natural degree of a
                                     threshold-built graph (that's expected: the threshold,
                                     not MAX_DEGREE, is what's setting density in that mode).
    Structural edges (dist == 1) are added in both modes, unchanged.
    """
    n = len(embeddings)
    logger.info(f"  Building graph for {n} chunks ...")
    logger.info(f"  Graph construction: {graph_construction}" +
                (f" (top_m={top_m})" if graph_construction == "topm" else f" (SEM_THRESHOLD={SEM_THRESHOLD})"))
    t0 = time.perf_counter()
    rows, cols, data = [], [], []
    semantic_edges = 0
    structural_edges = 0
    both_edges = 0

    topm_sets = None
    if graph_construction == "topm":
        topm_sets = []
        for i in range(n):
            row = sim_matrix[i].copy()
            row[i] = -1.0
            m = min(top_m, n - 1)
            top_idx = np.argpartition(row, -m)[-m:] if m < n - 1 else np.argsort(row)[::-1]
            topm_sets.append(set(top_idx.tolist()))

    for i in range(n):
        for j in range(i + 1, n):
            sem  = float(sim_matrix[i, j])
            dist = abs(i - j)
            ss   = 1.0 / (1.0 + dist)

            if graph_construction == "topm":
                is_sem = (j in topm_sets[i]) or (i in topm_sets[j])
            else:
                is_sem = sem >= SEM_THRESHOLD
            is_struct = (dist == 1)

            if is_sem:
                semantic_edges += 1
            if is_struct:
                structural_edges += 1
            if is_sem and is_struct:
                both_edges += 1

            if is_sem or is_struct:
                sem_c = sem if is_sem else 0.0
                weight = SEM_WEIGHT * sem_c + STRUCT_WEIGHT * ss
                rows += [i, j]
                cols += [j, i]
                data += [weight, weight]

    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    logger.info(f"Semantic edges : {semantic_edges}")
    logger.info(f"Structural edges: {structural_edges}")
    logger.info(f"Overlap         : {both_edges}")
    full_degrees = np.diff(adj.indptr)

    logger.info("Before pruning")
    logger.info(f"Average degree : {full_degrees.mean():.2f}")
    logger.info(f"Median degree  : {np.median(full_degrees):.2f}")
    logger.info(f"Maximum degree : {full_degrees.max()}")

    p_rows, p_cols, p_data = [], [], []
    for i in range(n):
        row_slice = adj.getrow(i)
        if row_slice.nnz == 0: continue
        if row_slice.nnz <= MAX_DEGREE:
            p_rows += row_slice.indices.tolist(); p_cols += [i]*row_slice.nnz; p_data += row_slice.data.tolist()
        else:
            top_idx = np.argsort(row_slice.data)[::-1][:MAX_DEGREE]
            p_rows += row_slice.indices[top_idx].tolist(); p_cols += [i]*MAX_DEGREE; p_data += row_slice.data[top_idx].tolist()

    adj_pruned = sp.csr_matrix((p_data, (p_cols, p_rows)), shape=(n, n))
    adj_pruned.eliminate_zeros()

    degrees = np.diff(adj_pruned.indptr)
    logger.info("After pruning")
    logger.info(f"Average degree : {degrees.mean():.2f}")
    logger.info(f"Median degree  : {np.median(degrees):.2f}")
    logger.info(f"Maximum degree : {degrees.max()}")
    logger.info(f"Minimum degree : {degrees.min()}")

    elapsed = time.perf_counter() - t0
    logger.info(f"MAX DEGREE: {MAX_DEGREE} " +
                ("(non-binding — raw max degree above was already <= this)" if full_degrees.max() <= MAX_DEGREE else "(binding — some nodes were pruned)"))
    logger.info(f"  Graph: {elapsed:.3f}s | edges={adj_pruned.nnz // 2} | RAM≈{adj_pruned.data.nbytes/1024:.1f} KB")
    return adj_pruned, elapsed


def get_prefetch_cosine(sim_matrix, pri, k):
    sims = sim_matrix[pri].copy()
    sims[pri] = -1
    return np.argsort(sims)[::-1][:k].tolist()


def get_prefetch_graph(adj_matrix, pri, k):
    """
    Two-hop graph retrieval.

    1. Get all first-hop neighbors.
    2. Add all of their neighbors.
    3. Score nodes by the strongest path weight.
    4. Return the top-k unique nodes.

    Scoring note: 2-hop candidates are scored as min(w1, w2) — the path's weakest
    link — not w1*w2. A product puts 2-hop scores on a different (smaller, roughly
    quadratic) scale than 1-hop scores, which systematically crowds out real but
    modest-weight paths (e.g. a pure-structural pri->pri+1->pri+2 chain, weight
    ~0.15 each hop) in favor of any 2-hop path through two moderately-strong
    semantic edges, even when the latter is a less relevant candidate. min() keeps
    both hop-counts on the same scale so ranking reflects path strength, not hop count.
    """

    candidates = {}

    # ---------- First hop ----------
    row = adj_matrix.getrow(pri)

    if row.nnz == 0:
        return []

    for idx, nbr in enumerate(row.indices):
        w1 = row.data[idx]

        if nbr not in candidates or w1 > candidates[nbr]:
            candidates[nbr] = w1

        # ---------- Second hop ----------
        row2 = adj_matrix.getrow(nbr)

        for idx2, nbr2 in enumerate(row2.indices):

            if nbr2 == pri:
                continue

            w2 = row2.data[idx2]

            # Score of the path — bottleneck (weakest-link), not product
            score = min(w1, w2)

            if nbr2 not in candidates or score > candidates[nbr2]:
                candidates[nbr2] = score

    ranked = sorted(
        candidates.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [node for node, _ in ranked[:k]]

# ── KV engine — MATCHES COLAB METHODOLOGY EXACTLY ─────────────────────────────
@torch.no_grad()
def compute_kv_cache(text, tokenizer, model, device):
    """Identical to Colab's compute_kv_cache: no attention_mask, standard .to(device)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=PRIMARY_MAX_LEN).to(device)
    out = model(**inputs, use_cache=True)
    safe_sync()
    return out.past_key_values, inputs["input_ids"]


@torch.no_grad()
def extend_with_cache(new_text, past_kv, past_ids, tokenizer, model, device, measure=True):
    """
    Identical to Colab's extend_with_cache:
      - position_ids computed manually from past_len
      - add_special_tokens=False
      - max_length=256 for the extension text
    """
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

    out = model(
        input_ids       = new_inputs["input_ids"],
        past_key_values  = past_kv,
        position_ids     = pos_ids,
        use_cache        = True,
    )

    if measure:
        safe_sync()
        ms = (time.perf_counter() - t0) * 1000
    else:
        ms = 0.0

    full_ids = torch.cat([past_ids, new_inputs["input_ids"]], dim=1)
    return out.past_key_values, full_ids, ms


def estimate_flops_saved(n_cached_tokens, n_new_tokens, hidden_dim, n_heads, n_layers):
    L_full, L_new, L_cached = n_cached_tokens + n_new_tokens, n_new_tokens, n_cached_tokens
    full_attn   = 4 * hidden_dim * (L_full ** 2)
    cached_attn = 4 * hidden_dim * (L_new ** 2 + 2 * L_new * L_cached)
    return (full_attn - cached_attn) * n_layers


def generate_pairs(n_chunks, n_pairs, sim_matrix, seed, logger):
    """Matches Colab's pair generation exactly (structural offsets 1-3, etc.)"""
    rng    = np.random.default_rng(seed)
    pairs  = []
    safe_n = max(n_chunks - 3, 1)
    for _ in range(n_pairs):
        pri = int(rng.integers(0, safe_n))
        r   = rng.random()
        if r < 0.25:
            q_type = "semantic"
            sims = sim_matrix[pri].copy()
            sims[max(0, pri-2):min(n_chunks, pri+3)] = -1
            sec = int(np.argmax(sims))
        elif r < 0.50:
            q_type = "structural"
            offset = int(rng.choice([1, 2, 3]))
            sec = min(n_chunks - 1, pri + offset)
        else:
            q_type = "multi-hop"
            if rng.random() < 0.5:
                sec = min(n_chunks - 1, pri + 2)
            else:
                sims = sim_matrix[pri].copy()
                sims[max(0, pri-2):min(n_chunks, pri+3)] = -1
                sec = int(np.argmax(sims))
        pairs.append((pri, sec, q_type))

    counts = {t: sum(1 for _, _, qt in pairs if qt == t) for t in ["semantic", "structural", "multi-hop"]}
    logger.info(f"  Generated {n_pairs} pairs: {counts}")
    return pairs


# ── GPU Warmup ─────────────────────────────────────────────────────────────────
def warmup_gpu(tokenizer, model, device, logger, n_passes=N_WARMUP_PASSES):
    """
    Run several dummy forward+extend passes before real measurement begins.
    First CUDA calls pay one-time kernel JIT / cuDNN autotune cost that
    otherwise pollutes the first few real measurements.
    """
    logger.info(f"  Running {n_passes} GPU warmup passes...")
    dummy_primary = "The quick brown fox jumps over the lazy dog. " * 15
    dummy_extend  = "This is a warmup extension text for CUDA kernel initialization. " * 8

    for i in range(n_passes):
        try:
            kv, ids = compute_kv_cache(dummy_primary, tokenizer, model, device)
            _, _, ms = extend_with_cache(dummy_extend, kv, ids, tokenizer, model,
                                         device, measure=True)
            del kv, ids
            torch.cuda.empty_cache()
            logger.info(f"    Warmup pass {i+1}/{n_passes}: {ms:.1f}ms")
        except Exception as e:
            logger.warning(f"    Warmup pass {i+1} failed (non-fatal): {e}")
            cuda_reset(logger)
    logger.info("  Warmup complete.")


# ── Single trial with retry ────────────────────────────────────────────────────
def run_trial(pri, sec, q_type, chunk_texts, sim_matrix, adj_matrix, K,
              sim_lru, sim_cos, sim_grp, tokenizer, model, device,
              model_config, logger, retries=MAX_RETRIES, legacy_timing=False):
    p_text = chunk_texts[pri]
    s_text = chunk_texts[sec]

    for attempt in range(1, retries + 1):
        try:
            sim_lru.access(pri)
            lru_hit = int(sim_lru.access(sec))

            c_ids = get_prefetch_cosine(sim_matrix, pri, K)
            sim_cos.access(pri); sim_cos.prefetch(c_ids)
            cos_hit = int(sim_cos.access(sec))

            g_ids = get_prefetch_graph(adj_matrix, pri, K)
            sim_grp.access(pri); sim_grp.prefetch(g_ids)
            grp_hit = int(sim_grp.access(sec))

            # ── Cold — empty_cache before, matching Colab ──
            torch.cuda.empty_cache()
            kv, ids = compute_kv_cache(p_text, tokenizer, model, device)
            cold_n_tok = ids.shape[1]
            if lru_hit and not legacy_timing:
                # v4 FIX: a real cache would already hold sec's KV — no forward
                # pass needed. v3 called extend_with_cache here unconditionally,
                # which is the bug: hits never got cheaper, so cold (always the
                # shortest resident context) could never lose.
                cold_ms = HIT_LATENCY_MS
            else:
                _, _, cold_ms = extend_with_cache(s_text, kv, ids, tokenizer, model, device)

            # ── Cosine ──
            c_texts = [chunk_texts[i] for i in c_ids]
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

            # ── Graph ──
            g_texts = [chunk_texts[i] for i in g_ids]
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

            s_tok = tokenizer(s_text, return_tensors="pt", truncation=True,
                              max_length=EXTEND_MAX_LEN, add_special_tokens=False
                              )["input_ids"].shape[1]

            flops_cos = estimate_flops_saved(cos_n_tok, s_tok, model_config["hidden_dim"],
                                             model_config["n_heads"], model_config["n_layers"]) if cos_n_tok > 0 else 0.0
            flops_grp = estimate_flops_saved(grp_n_tok, s_tok, model_config["hidden_dim"],
                                             model_config["n_heads"], model_config["n_layers"]) if grp_n_tok > 0 else 0.0

            return {
                "cold_ms": round(cold_ms, 4), "cosine_ms": round(cosine_ms, 4), "graph_ms": round(graph_ms, 4),
                "lru_sim_hit": lru_hit, "cos_sim_hit": cos_hit, "grp_sim_hit": grp_hit,
                "cold_n_tokens": cold_n_tok, "cosine_n_tokens": cos_n_tok, "graph_n_tokens": grp_n_tok,
                "flops_saved_cosine": round(flops_cos, 0), "flops_saved_graph": round(flops_grp, 0),
            }

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.warning(f"  Trial attempt {attempt}/{retries} failed: {e}")
            cuda_reset(logger)
            if attempt == retries:
                logger.error(f"  Giving up on trial pri={pri} sec={sec}")
                return None
            time.sleep(2 * attempt)
    return None


# ── Main dataset loop ──────────────────────────────────────────────────────────
def run_dataset(dataset_key, model_name, tokenizer, model, device, output_dir, logger, model_config,
                 legacy_timing=False, graph_construction="threshold", top_m=TOP_M_SEMANTIC):
    cfg   = DATASET_CONFIGS[dataset_key]
    label = cfg["label"]
    logger.info("=" * 70)
    logger.info(f"DATASET: {label}  |  MODEL: {model_name}")
    logger.info("=" * 70)

    t0 = time.perf_counter()
    zip_path = hf_hub_download(repo_id=cfg["hf_name"], filename="data.zip", repo_type="dataset")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            target = f"data/{cfg['split']}.jsonl"
            z.extract(target, tmpdir)
            df = pd.read_json(os.path.join(tmpdir, target), lines=True)

    target_chars = 100_000
    contexts = []
    for ctx in df[cfg["text_key"]]:
        contexts.append(ctx)
        if sum(len(c) for c in contexts) >= target_chars:
            break
    raw = "\n\n".join(contexts)
    logger.info(f"  Raw text: {len(raw):,} chars  ({time.perf_counter()-t0:.1f}s)")

    splitter    = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)
    chunk_texts = splitter.split_text(raw)
    n           = len(chunk_texts)
    logger.info(f"  Chunks: {n}")

    logger.info("Computing embeddings ...")
    t0 = time.perf_counter()
    embedder   = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunk_texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    embed_time = time.perf_counter() - t0
    logger.info(f"  Embeddings: {embeddings.shape}  ({embed_time:.2f}s)")

    sim_matrix = cosine_similarity(embeddings)
    adj_matrix, build_time = build_graph(embeddings, sim_matrix, logger,
                                         graph_construction=graph_construction, top_m=top_m)

    logger.info("Generating query pairs ...")
    all_pairs = generate_pairs(n, N_PAIRS, sim_matrix, RANDOM_SEED, logger)

    safe_model = model_name.replace("/", "_").replace("-", "_")
    tag        = "legacy" if legacy_timing else "hitaware"
    graph_tag  = f"topm{top_m}" if graph_construction == "topm" else "thresh"
    csv_path   = output_dir / f"results_{safe_model}_{dataset_key}_{tag}_{graph_tag}.csv"
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
        "cold_ms", "cosine_ms", "graph_ms", "lru_sim_hit", "cos_sim_hit", "grp_sim_hit",
        "cold_n_tokens", "cosine_n_tokens", "graph_n_tokens",
        "flops_saved_cosine", "flops_saved_graph", "embed_time_s", "graph_build_time_s",
    ]
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    skipped_trials = 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        for ablation in ABLATION_CONFIGS:
            K, CAP = ablation["K"], ablation["CACHE_CAPACITY"]
            key = (K, CAP)
            start_i = completed.get(key, 0)
            if start_i >= N_PAIRS:
                logger.info(f"  ⏭  K={K} already complete ({start_i} rows)")
                continue

            logger.info(f"\n  ── K={K}, cap={CAP} | resuming from run {start_i} ──")

            # NEW: warmup before each K config, not just once at model load.
            # This matters because switching K changes prompt lengths, which
            # can trigger new kernel autotuning on some CUDA backends.
            warmup_gpu(tokenizer, model, device, logger, n_passes=1)

            sim_lru = KVCacheSimulator(CAP)
            sim_cos = KVCacheSimulator(CAP)
            sim_grp = KVCacheSimulator(CAP)
            config_start = time.perf_counter()

            for run_i, (pri, sec, q_type) in enumerate(all_pairs):
                if run_i < start_i:
                    continue

                result = run_trial(pri, sec, q_type, chunk_texts, sim_matrix, adj_matrix, K,
                                   sim_lru, sim_cos, sim_grp, tokenizer, model, device,
                                   model_config, logger, legacy_timing=legacy_timing)

                if result is None:
                    skipped_trials += 1
                    logger.warning(f"  Skipping run {run_i} (all retries failed)")
                    continue

                writer.writerow({
                    "model_name": model_name, "gpu_name": gpu_name, "seed": RANDOM_SEED,
                    "dataset": label, "config_k": K, "config_cap": CAP, "run": run_i, "type": q_type,
                    "embed_time_s": round(embed_time, 3), "graph_build_time_s": round(build_time, 3),
                    **result,
                })
                f.flush()
                os.fsync(f.fileno())

                if (run_i + 1) % 50 == 0 or run_i == N_PAIRS - 1:
                    elapsed = time.perf_counter() - config_start
                    done    = run_i - start_i + 1
                    eta_s   = (elapsed / done) * (N_PAIRS - run_i - 1) if done > 0 else 0
                    logger.info(f"     K={K} | {run_i+1:>3}/{N_PAIRS} | {elapsed/60:.1f}min elapsed | "
                               f"ETA {eta_s/60:.1f}min | cold={result['cold_ms']:.1f} "
                               f"cos={result['cosine_ms']:.1f} grp={result['graph_ms']:.1f} ms")

            logger.info(f"  ✓ K={K} done in {(time.perf_counter()-config_start)/60:.1f} min")

    if skipped_trials:
        logger.warning(f"  Total skipped trials: {skipped_trials}")
    logger.info(f"  CSV: {csv_path}")
    return csv_path


def print_summary(csv_path, logger):
    df = pd.read_csv(csv_path)
    logger.info("\n" + "=" * 70)
    logger.info(f"SUMMARY — {csv_path.name}")
    logger.info("=" * 70)
    for k in sorted(df["config_k"].unique()):
        sub = df[df["config_k"] == k]
        cos_m, grp_m = sub["cosine_ms"].mean(), sub["graph_ms"].mean()
        delta = cos_m - grp_m
        cos_hr, grp_hr = sub["cos_sim_hit"].mean()*100, sub["grp_sim_hit"].mean()*100
        _, p_val = stats.ttest_rel(sub["cosine_ms"], sub["graph_ms"])
        sig = "***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else "ns"
        logger.info(f"  K={k} | cos={cos_m:.2f}ms ({cos_hr:.1f}%HR) grp={grp_m:.2f}ms ({grp_hr:.1f}%HR) Δ={delta:+.2f}ms  {sig}")
        for qt in ["semantic", "structural", "multi-hop"]:
            sub_q = sub[sub["type"] == qt]
            if len(sub_q) < 2: continue
            d = sub_q["cosine_ms"].mean() - sub_q["graph_ms"].mean()
            _, p = stats.ttest_rel(sub_q["cosine_ms"], sub_q["graph_ms"])
            sq = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            logger.info(f"    {qt:>12}: Δ={d:+.3f}ms  {sq}  (n={len(sub_q)})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()), default=["hotpot"])
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16"])
    p.add_argument("--legacy_timing", action="store_true",
                    help="Reproduce exact v3 behavior (always re-encode sec, even on a hit). "
                         "Use this ONLY to regenerate the old numbers for an A/B table in the "
                         "paper — leave it off for the corrected, hit-aware run.")
    p.add_argument("--graph_construction", default="threshold", choices=["threshold", "topm"],
                    help="threshold: keep SEM_THRESHOLD-gated edges (original). "
                         "topm: connect each node to its top_m most similar chunks regardless "
                         "of absolute similarity — decouples graph density from the corpus's "
                         "raw similarity distribution.")
    p.add_argument("--top_m", type=int, default=TOP_M_SEMANTIC,
                    help="Neighbors per node when --graph_construction topm.")
    return p.parse_args()


# MODEL_CONFIGS = {
#     "Qwen2.5-7B": {"hidden_dim": 3584, "n_heads": 28, "n_layers": 28},
#     "Qwen2.5-14B": {"hidden_dim": 5120, "n_heads": 40, "n_layers": 48},
#     "Qwen2.5-1.5B": {"hidden_dim": 1536, "n_heads": 12, "n_layers": 28},
#     "Qwen2.5-3B": {"hidden_dim": 2048, "n_heads": 16, "n_layers": 36},
#     "default": {"hidden_dim": 4096, "n_heads": 32, "n_layers": 32},
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
        if key.lower() in model_name.lower():
            return cfg
    return MODEL_CONFIGS["default"]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1]
    logger = setup_logging(output_dir, model_tag)

    logger.info("=" * 70)
    logger.info("KV Cache Chunk Prefetching Experiment (v4 — hit-aware timing)")
    logger.info(f"Model:    {args.model}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Output:   {output_dir.resolve()}")
    if args.legacy_timing:
        logger.warning("  --legacy_timing is ON: reproducing v3 behavior "
                        "(hits are NOT cheaper). Only use this to regenerate old numbers.")
    else:
        logger.info("  Hit-aware timing ON (v4 default): a cache hit skips the real "
                     f"re-encode of sec and records {HIT_LATENCY_MS}ms.")
    logger.info(f"  Graph construction: {args.graph_construction}" +
                (f" (top_m={args.top_m})" if args.graph_construction == "topm" else ""))
    logger.info("=" * 70)

    if not torch.cuda.is_available():
        logger.error("No CUDA GPU found. Exiting.")
        sys.exit(1)

    device = "cuda:0"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name}  ({vram_gb:.1f} GB)")

    if args.torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.torch_dtype == "float16":
        dtype = torch.float16
    else:
        cc = torch.cuda.get_device_capability(0)
        dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
        logger.info(f"  Auto dtype: {'bfloat16' if dtype==torch.bfloat16 else 'float16'} (cc {cc[0]}.{cc[1]})")

    logger.info(f"Loading model {args.model} ...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
    )
    model.eval()

    load_time = time.perf_counter() - t0
    vram_used = torch.cuda.memory_allocated(0) / 1e9
    logger.info(f"Model loaded in {load_time:.1f}s  |  VRAM used: {vram_used:.2f} GB")

    model_config = get_model_config(args.model)
    logger.info(f"Architecture config: {model_config}")

    # ── One-time warmup at model load (in addition to per-K warmup) ──
    warmup_gpu(tokenizer, model, device, logger, n_passes=N_WARMUP_PASSES)

    all_csvs = []
    total_start = time.perf_counter()

    for ds_key in args.datasets:
        csv_path = run_dataset(ds_key, args.model, tokenizer, model, device,
                               output_dir, logger, model_config, legacy_timing=args.legacy_timing,
                               graph_construction=args.graph_construction, top_m=args.top_m)
        all_csvs.append(csv_path)
        print_summary(csv_path, logger)

    total_h = (time.perf_counter() - total_start) / 3600
    logger.info(f"\nAll datasets done in {total_h:.2f} hrs")
    logger.info(f"CSVs: {[str(p) for p in all_csvs]}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
