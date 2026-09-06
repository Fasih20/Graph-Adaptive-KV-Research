"""
benchmark_harness.py
=====================
Real-hardware replacement for the canonical script's run_trial/run_dataset.

CSV SCHEMA: the fieldnames list below is copied verbatim, same order, from
the canonical script's run_dataset() (its `fieldnames = [...]` literal —
this repo was not given a separate sample_results.csv, see the delivery
message; the fieldnames list embedded in the canonical .py IS the ground
truth schema, since it's literally the code that produced your existing
CSVs). Do not reorder, rename, or add columns to this list — anything new
(e.g. eager-mode calibration numbers) goes in a companion file instead, see
scripts/calibrate_eager.sh.

THREE DELIBERATE SEMANTIC DEVIATIONS from the simulated harness (each is
necessary given a real server replaces a local model call, not an oversight
— flagged in detail in the delivery message, summarized here):

  1. *_ms now measures one real end-to-end TTFT per policy per trial
     (warm [pri]+candidates off the critical path, THEN measure the
     request that appends the target chunk), rather than the simulated
     harness's "always locally recompute, unless simulator says hit, then
     charge exactly 0ms" hack. Real cache hits show up as genuinely lower
     TTFT because LMCache actually skips computation — we don't need to
     fake it, so v4's "hit-aware timing" skip-and-zero is intentionally
     NOT reproduced here.

  2. *_n_tokens / flops_saved_* still use the ORIGINAL blob-truncation
     formula (join candidates into one string, truncate the WHOLE blob to
     EXTEND_MAX_LEN=256) purely so these columns stay numerically
     comparable to your existing plots. The ACTUAL request sent to vLLM
     (via cache_manager.build_prompt_token_ids) truncates PER CHUNK
     instead, because blob-level truncation would arbitrarily slice through
     a chunk mid-token-stream, which is actively harmful to CacheBlend's
     chunk-hash matching. This means the CSV's n_tokens columns describe a
     slightly different quantity than what's literally on the wire for
     large K. Flagged in implementation notes; revisit if it matters for
     your analysis.

  3. No `legacy_timing` flag. It existed only to A/B the hit-aware-timing
     hack from (1); since there's nothing to fake here, there's nothing to
     toggle.

*_sim_hit columns are still produced by the untouched KVCacheSimulator, for
continuity with the simulated harness's hit-rate columns — but see
implementation_plan.md's "Semantic Mismatch" note: they're a simple LRU
prediction, not a report of what LMCache's own (byte/block-budgeted)
eviction actually did.
"""

import os
import csv
import time
import logging
import subprocess
from pathlib import Path

import graph_algorithms as ga
from prefetch_policies import CosinePrefetchPolicy, GraphPrefetchPolicy, AdaptivePrefetchPolicy
from metrics_utils import estimate_flops_saved
from exp_config import (RANDOM_SEED, N_PAIRS, ABLATION_CONFIGS, SEPARATOR,
                     PRIMARY_MAX_LEN, EXTEND_MAX_LEN, SEM_WEIGHT, STRUCT_WEIGHT,
                     HARDCODED_ADAPTIVE_WEIGHTS, RESULTS_DIR, get_model_config)

logger = logging.getLogger("benchmark_harness")

FIELDNAMES = [
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


def get_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=10)
        return out.strip().splitlines()[0]
    except Exception:
        return "unknown"


def _blob_token_count(tokenizer, text, max_length, add_special_tokens):
    if not text:
        return 0
    ids = tokenizer(text, truncation=True, max_length=max_length,
                    add_special_tokens=add_special_tokens)["input_ids"]
    return len(ids)


def _resident_and_target_counts(tokenizer, p_text, candidate_texts, s_text):
    """Reproduces the ORIGINAL script's blob-truncation token-count formula
    exactly (see module docstring, deviation #2) — used only for CSV
    n_tokens/flops columns, not for the actual wire request."""
    primary_n = _blob_token_count(tokenizer, p_text, PRIMARY_MAX_LEN, True)
    if candidate_texts:
        joined = SEPARATOR.join(candidate_texts)
        extra_n = _blob_token_count(tokenizer, joined, EXTEND_MAX_LEN, False)
    else:
        extra_n = 0
    n_resident = primary_n + extra_n
    s_tok = _blob_token_count(tokenizer, s_text, EXTEND_MAX_LEN, False)
    return n_resident, s_tok


def run_trial(pri, sec, q_type, chunk_store, sim_matrix, adj_matrix, raw_edges,
             K, sim_lru, sim_cos, sim_grp, sim_adp, adaptive_weights,
             cache_manager, model_config, logger, max_retries=3):
    p_text = chunk_store.text(pri)
    s_text = chunk_store.text(sec)

    cos_policy = CosinePrefetchPolicy(sim_matrix)
    grp_policy = GraphPrefetchPolicy(adj_matrix)
    adp_policy = AdaptivePrefetchPolicy(raw_edges, adaptive_weights)

    for attempt in range(1, max_retries + 1):
        try:
            sim_lru.access(pri)
            lru_hit = int(sim_lru.access(sec))

            c_ids = cos_policy.get_prefetch_ids(pri, K)
            sim_cos.access(pri); sim_cos.prefetch(c_ids)
            cos_hit = int(sim_cos.access(sec))

            g_ids = grp_policy.get_prefetch_ids(pri, K)
            sim_grp.access(pri); sim_grp.prefetch(g_ids)
            grp_hit = int(sim_grp.access(sec))

            a_ids = adp_policy.get_prefetch_ids(pri, K, q_type)
            sim_adp.access(pri); sim_adp.prefetch(a_ids)
            adp_hit = int(sim_adp.access(sec))

            result = {}
            # IMPORTANT: cold's context must be the SAME SIZE as the other
            # three arms (primary + K chunks), not just the primary chunk
            # alone. A RAG query that needs K retrieved chunks needs them
            # whether or not they were prefetched — "cold" should mean
            # "fetch them at query time instead of ahead of time", not
            # "don't use them at all". Using ids=[] here was a genuine
            # confound: cold's latency was flat across K only because its
            # context never grew with K, while cosine/graph/adaptive grew
            # linearly with K purely from added token count — that
            # difference in TOKEN COUNT, not caching, was what earlier
            # results were actually measuring. Reuse cosine's candidate set
            # here just to fix the context SIZE; cold ignores its identity
            # by design (no warm_prefix below), so which K ids it's set to
            # doesn't matter, only how many.
            policies = [
                ("cold",     c_ids, "cold_ms",     "cold_n_tokens"),
                ("cosine",   c_ids, "cosine_ms",   "cosine_n_tokens"),
                ("graph",    g_ids, "graph_ms",    "graph_n_tokens"),
                ("adaptive", a_ids, "adaptive_ms", "adaptive_n_tokens"),
            ]
            s_tok = None
            for name, ids, ms_key, ntok_key in policies:
                # Flush the REAL cache before every single policy arm, not
                # just once per (K, CAP) config. Without this, all 600 pairs
                # within a config shared one never-cleared LMCache instance:
                # later trials' "cold" arm could get free residual warmth
                # left behind by an earlier trial's cosine/graph/adaptive
                # call on the same primary chunk, and cold structurally never
                # pays CacheBlend's per-chunk-boundary recompute tax that
                # cosine/graph/adaptive always pay for their K extra chunks
                # — biasing every comparison toward cold regardless of
                # whether real prefetch reuse was happening. This makes each
                # arm start from a genuinely empty cache, at the cost of
                # much more wall-clock time (worth it for a small N / small
                # model run — not for the full 600-pair sweep).
                if not cache_manager.flush_cache():
                    logger.warning(f"  Cache flush failed before {name} arm — "
                                   f"this trial's result may be contaminated")
                candidate_texts = chunk_store.texts(ids) if ids else []
                if name != "cold":
                    # Off critical path: warm primary + candidates, block
                    # until done since we need them resident before the
                    # timed call. Cold deliberately skips this — it must
                    # hit true cold recompute for its full context (same
                    # size as the other arms now, see note above) to
                    # represent "no prefetching happened" rather than
                    # "prefetching happened but for a smaller context".
                    cache_manager.warm_prefix([pri] + list(ids), chunk_store, block=True)
                timing = cache_manager.query_with_chunks(
                    s_text, context_chunk_ids=[pri] + list(ids), chunk_store=chunk_store)
                result[ms_key] = round(timing["ttft_ms"], 4)

                n_resident, s_tok = _resident_and_target_counts(
                    cache_manager.tokenizer, p_text, candidate_texts, s_text)
                result[ntok_key] = n_resident

            def flops(n_cached):
                return estimate_flops_saved(n_cached, s_tok, model_config["hidden_dim"],
                                            model_config["n_heads"], model_config["n_layers"]
                                            ) if n_cached > 0 else 0.0

            result.update({
                "lru_sim_hit": lru_hit, "cos_sim_hit": cos_hit,
                "grp_sim_hit": grp_hit, "adp_sim_hit": adp_hit,
                "flops_saved_cosine":   round(flops(result["cosine_n_tokens"]), 0),
                "flops_saved_graph":    round(flops(result["graph_n_tokens"]), 0),
                "flops_saved_adaptive": round(flops(result["adaptive_n_tokens"]), 0),
            })
            return result

        except Exception as e:
            logger.warning(f"  Trial attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                logger.error(f"  Giving up on trial pri={pri} sec={sec}")
                return None
            time.sleep(2 * attempt)
    return None


def run_dataset(dataset_key, model_name, chunk_store, cache_manager, model_config,
                adaptive_weights=HARDCODED_ADAPTIVE_WEIGHTS, output_dir: Path = RESULTS_DIR,
                n_pairs=N_PAIRS, seed=RANDOM_SEED):
    gpu_name = get_gpu_name()
    safe_model = model_name.replace("/", "_").replace("-", "_")
    csv_path = output_dir / f"results_real_{safe_model}_{dataset_key}.csv"
    csv_exists = csv_path.exists()

    completed = {}
    if csv_exists:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (int(row["config_k"]), int(row["config_cap"]))
                completed[key] = completed.get(key, 0) + 1
        logger.info(f"  Resuming — completed rows: {completed}")

    all_pairs = chunk_store.generate_pairs(n_pairs, seed)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not csv_exists:
            writer.writeheader()

        for ablation in ABLATION_CONFIGS:
            K, CAP = ablation["K"], ablation["CACHE_CAPACITY"]
            key = (K, CAP)
            start_i = completed.get(key, 0)
            if start_i >= n_pairs:
                logger.info(f"  \u23ed  K={K} already complete ({start_i} rows)")
                continue

            logger.info(f"\n  \u2500\u2500 K={K}, cap={CAP} | resuming from run {start_i} \u2500\u2500")
            # Fresh simulators for this K. Real-cache isolation now happens
            # PER TRIAL, per policy arm (see run_trial) — this flush is just
            # a courtesy reset so the first trial of each K starts clean too.
            if not cache_manager.flush_cache():
                logger.warning("  Proceeding despite failed cache flush — "
                               "results for this K may be contaminated by the previous K's cache state")
            sim_lru = ga.KVCacheSimulator(CAP)
            sim_cos = ga.KVCacheSimulator(CAP)
            sim_grp = ga.KVCacheSimulator(CAP)
            sim_adp = ga.KVCacheSimulator(CAP)
            config_start = time.perf_counter()

            for run_i, (pri, sec, q_type) in enumerate(all_pairs):
                if run_i < start_i:
                    continue

                result = run_trial(
                    pri, sec, q_type, chunk_store, chunk_store.sim_matrix,
                    chunk_store.adj_matrix, chunk_store.raw_edges,
                    K, sim_lru, sim_cos, sim_grp, sim_adp,
                    adaptive_weights, cache_manager, model_config, logger,
                )
                if result is None:
                    logger.warning(f"  Skipping run {run_i} (all retries failed)")
                    continue

                writer.writerow({
                    "model_name": model_name, "gpu_name": gpu_name,
                    "seed": seed, "dataset": chunk_store.cfg["label"],
                    "config_k": K, "config_cap": CAP, "run": run_i, "type": q_type,
                    "embed_time_s": round(chunk_store.embed_time_s, 3),
                    "graph_build_time_s": round(chunk_store.graph_build_time_s, 3),
                    "adp_alpha_semantic":   adaptive_weights.get("semantic",   {}).get("alpha", SEM_WEIGHT),
                    "adp_beta_semantic":    adaptive_weights.get("semantic",   {}).get("beta",  STRUCT_WEIGHT),
                    "adp_alpha_structural": adaptive_weights.get("structural", {}).get("alpha", SEM_WEIGHT),
                    "adp_beta_structural":  adaptive_weights.get("structural", {}).get("beta",  STRUCT_WEIGHT),
                    "adp_alpha_multihop":   adaptive_weights.get("multi-hop",  {}).get("alpha", SEM_WEIGHT),
                    "adp_beta_multihop":    adaptive_weights.get("multi-hop",  {}).get("beta",  STRUCT_WEIGHT),
                    **result,
                })
                f.flush()
                os.fsync(f.fileno())

                if (run_i + 1) % 50 == 0 or run_i == n_pairs - 1:
                    elapsed = time.perf_counter() - config_start
                    logger.info(
                        f"     K={K} | {run_i+1:>3}/{n_pairs} | {elapsed/60:.1f}min | "
                        f"cold={result['cold_ms']:.1f} cos={result['cosine_ms']:.1f} "
                        f"grp={result['graph_ms']:.1f} adp={result['adaptive_ms']:.1f} ms")

            logger.info(f"  \u2713 K={K} in {(time.perf_counter()-config_start)/60:.1f} min")

    logger.info(f"  CSV: {csv_path}")
    return csv_path
