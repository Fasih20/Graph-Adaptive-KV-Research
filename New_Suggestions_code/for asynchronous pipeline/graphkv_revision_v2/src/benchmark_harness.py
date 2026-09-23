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

  1. *_ms now measures one real end-to-end TTFT per policy per trial (warm
     [pri]+candidates off the critical path, THEN measure the request that
     appends the target chunk), rather than the simulated harness's
     "always locally recompute, unless simulator says hit, then charge
     exactly 0ms" hack (kv_cache_experiment_adaptive_v2.py's "HIT-AWARE
     TIMING" section / finalForRest.py's v4 changelog). Dropping the
     hardcoded 0ms was the right call — it's not a real number on any
     hardware — but the hit-aware skip was ALSO the only thing keeping
     cold from structurally winning every trial: your own v3/Colab-era
     runs hit exactly this failure under the OLD name "cold always wins"
     (finalForRest.py's docstring: "cosine/graph could only ever look
     equal-or-worse than cold, never better... since no condition could
     ever skip work, cold (shortest context) wins cleanly every time").
     That happened there because every policy always fully re-encoded
     `sec`, so a longer resident context (cosine/graph's K candidates)
     was pure added cost with no offsetting saving anywhere. This file
     had reintroduced the identical structural problem through a
     different door: the timed call's context_chunk_ids used to be
     `[pri] + ids`, so cold's timed request was always smaller than
     cosine/graph/adaptive's by construction — same "shortest context
     always wins" mechanism, just moved from "sec always re-measured" to
     "prompt size always asymmetric". Fixed below: every policy's timed
     request is now the same `[pri] + s_text`, so a real cache hit on
     `sec` (from correctly-guessed prefetch candidates) is what has to
     show up as lower TTFT — nothing about the request shape does the
     job by default the way `cold_ms` used to. See run_trial()'s
     docstring for the mechanics.

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

BUG FIX (post-delivery): run_trial()'s timed request used to be built from
`[pri] + ids` (ids = the K prefetched candidates — empty for cold, K long
for cosine/graph/adaptive), so cold's timed prompt was structurally smaller
than the other three policies' on every single trial, independent of
whether any caching happened at all. That alone explains "cold beats
cosine/graph" regardless of K or hit rate. Fixed: the timed call is now
`context_chunk_ids=[pri]` for every policy — see run_trial()'s docstring
for the full explanation and for a second, related fix (simulator
bookkeeping no longer re-runs on HTTP retries).
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
             cache_manager, model_config, logger, max_retries=3,
             flush_between_arms=True):
    """
    BUG-FIX NOTE (see delivery message for the full writeup, and the module
    docstring's deviation #1 for how this connects to your own v3->v4
    "cold always wins" fix in finalForRest.py / kv_cache_experiment_adaptive_v2.py):

    The timed request used to be built from `[pri] + ids`, where `ids` is
    the K prefetched-candidate list -- empty for "cold", K chunks long for
    cosine/graph/adaptive. That meant cold's timed prompt was ALWAYS
    smaller than the other three policies' prompts, by construction, for
    every trial, regardless of whether caching worked at all: cold had
    nothing extra to pay for, cosine/graph/adaptive always did (more raw
    tokens to attend over / more KV lookup and block-table bookkeeping,
    scaling with K, even on a full cache hit). On top of that, `sec` was
    ALWAYS appended fresh as the trailing query text for every policy, so
    even a trial where the prefetch policy correctly guessed `sec` (a
    simulator "hit") got no benefit at all -- `sec` was never actually
    served from the warmed content, just recomputed again at the end
    regardless. Structurally, cosine/graph/adaptive could only ever look
    equal-or-worse than cold, never better -- the exact same shape as the
    "cold always wins" failure your own hit-aware-timing fix targeted, just
    arrived at through request-shape asymmetry instead of through
    unconditionally re-measuring `sec`.

    Fix: the TIMED call is now `context_chunk_ids=[pri]` for every policy
    (see the loop below) -- identical prompt, identical size, for cold,
    cosine, graph, and adaptive alike. `ids` (the K speculative
    candidates) still gets warmed off the critical path, exactly as
    before; the only thing that can now differ between policies is
    whether that warming happened to make `sec`'s own KV resident (i.e.
    whether `sec` was one of the K candidates guessed from `pri` -- the
    same condition sim_cos/sim_grp/sim_adp's hit bookkeeping below is
    already predicting). That's the actual thing this benchmark is
    supposed to measure. Concretely this also means `sec` gets warmed (if
    at all) at a different position than where it appears in the timed
    prompt -- relying on LMCache/CacheBlend's non-prefix chunk reuse,
    which is the specific feature this whole harness exists to exercise.

    Second fix: the KVCacheSimulator calls below (`sim_*.access` /
    `sim_*.prefetch`) mutate `sim_lru`/`sim_cos`/`sim_grp`/`sim_adp`, which
    are shared, persist across the whole K-sweep. They used to run INSIDE
    the retry loop, so a transient HTTP failure that triggered a retry
    would silently replay those mutations a second time (double-accessing
    pri/sec, double-prefetching ids), corrupting the simulator's state for
    every trial after it in this K value -- not just the one that failed.
    They now run exactly once, before the retry loop; only the actual
    network calls are retried.

    Third fix (flush_between_arms, default True): without this, all 4
    policy arms of a trial -- and every trial after it in the same K-sweep
    -- share one never-cleared cache. cosine's warm_prefix might leave
    chunk 47 resident; if graph's OWN candidate list also happens to
    include chunk 47 a few lines later, graph gets an unearned "free" hit
    it didn't actually predict, via cosine's leftover warmth rather than
    its own guess. flush_between_arms=True flushes before every single
    arm (cold included) so a hit can only come from THIS arm's own warm
    step. Caveat inherited from cache_manager.flush_cache()'s own
    docstring: that endpoint clears LMCache's L1/CPU store but does not
    necessarily evict vLLM's OWN GPU-resident blocks, so this reduces
    cross-arm contamination rather than provably eliminating it -- if
    sanity_check_caching.py's /metrics or /status output suggests
    otherwise on your version, that's the thing to dig into. This also
    roughly doubles HTTP round trips per trial (one extra flush per arm)
    -- meaningful if you're timeboxed (e.g. a Kaggle session), so it's a
    parameter, not hardcoded. Set it False for a faster, less isolated
    run; the wire-parity fix above holds either way.
    """
    p_text = chunk_store.text(pri)
    s_text = chunk_store.text(sec)

    cos_policy = CosinePrefetchPolicy(sim_matrix)
    grp_policy = GraphPrefetchPolicy(adj_matrix)
    adp_policy = AdaptivePrefetchPolicy(raw_edges, adaptive_weights)

    # ── Simulator bookkeeping: side-effecting, must run exactly once ───────
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

    policies = [
        ("cold",     [],    "cold_ms",     "cold_n_tokens"),
        ("cosine",   c_ids, "cosine_ms",   "cosine_n_tokens"),
        ("graph",    g_ids, "graph_ms",    "graph_n_tokens"),
        ("adaptive", a_ids, "adaptive_ms", "adaptive_n_tokens"),
    ]

    # ── Network calls: safe to retry, no Python-side mutation on replay ────
    for attempt in range(1, max_retries + 1):
        try:
            result = {}
            wire_tokens = {}
            s_tok = None
            for name, ids, ms_key, ntok_key in policies:
                if flush_between_arms:
                    if not cache_manager.flush_cache():
                        logger.warning(f"  cache flush failed before {name} arm "
                                       f"(pri={pri} sec={sec}) — this trial's "
                                       f"cross-arm isolation may be compromised")
                candidate_texts = chunk_store.texts(ids) if ids else []
                # Off critical path: warm primary + speculative candidates,
                # block until done since we need them resident (if the
                # prefetch policy guessed right) before the timed call.
                cache_manager.warm_prefix([pri] + list(ids), chunk_store, block=True)

                # Timed call: SAME [pri] + s_text request for every policy —
                # see the module/function docstring above for why this must
                # not include `ids`.
                timing = cache_manager.query_with_chunks(
                    s_text, context_chunk_ids=[pri], chunk_store=chunk_store)
                result[ms_key] = round(timing["ttft_ms"], 4)
                wire_tokens[name] = timing["n_prompt_tokens"]

                n_resident, s_tok = _resident_and_target_counts(
                    cache_manager.tokenizer, p_text, candidate_texts, s_text)
                result[ntok_key] = n_resident

            # Regression guard for the bug above: every policy's ACTUAL wire
            # request is now supposed to be identical. If this ever fires,
            # something has reintroduced a per-policy prompt-size asymmetry
            # and the ms columns are no longer a fair comparison.
            if len(set(wire_tokens.values())) != 1:
                logger.warning(f"  pri={pri} sec={sec}: policies sent "
                               f"differently-sized prompts {wire_tokens} — "
                               f"cross-policy *_ms comparison is not fair "
                               f"for this trial")

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


def _extract_hits_queries(counters):
    """Sums every matching key so this works whether your vLLM version
    exposes one counter or a per-engine breakdown of several."""
    if not counters:
        return 0.0, 0.0
    hits = sum(v for k, v in counters.items() if "hits" in k.lower())
    queries = sum(v for k, v in counters.items() if "queries" in k.lower())
    return hits, queries


def run_dataset(dataset_key, model_name, chunk_store, cache_manager, model_config,
                adaptive_weights=HARDCODED_ADAPTIVE_WEIGHTS, output_dir: Path = RESULTS_DIR,
                n_pairs=N_PAIRS, seed=RANDOM_SEED, flush_between_arms=True,
                track_real_hits=True):
    gpu_name = get_gpu_name()
    safe_model = model_name.replace("/", "_").replace("-", "_")
    csv_path = output_dir / f"results_real_{safe_model}_{dataset_key}.csv"
    csv_exists = csv_path.exists()

    # Companion file, NOT the main FIELDNAMES-constrained CSV (see module
    # docstring: "anything new... goes in a companion file instead"). Real,
    # vLLM-reported /metrics deltas per trial -- ground truth to set
    # against lru_sim_hit/cos_sim_hit/grp_sim_hit/adp_sim_hit in the main
    # CSV, which are a theoretical LRU prediction, not a measurement.
    metrics_csv_path = output_dir / f"results_real_{safe_model}_{dataset_key}_realmetrics.csv"
    metrics_fieldnames = ["config_k", "config_cap", "run",
                          "prefix_cache_hits_delta", "prefix_cache_queries_delta"]
    metrics_csv_exists = metrics_csv_path.exists()
    metrics_f = None
    metrics_writer = None
    if track_real_hits:
        metrics_f = open(metrics_csv_path, "a", newline="", encoding="utf-8")
        metrics_writer = csv.DictWriter(metrics_f, fieldnames=metrics_fieldnames)
        if not metrics_csv_exists:
            metrics_writer.writeheader()

    completed = {}
    if csv_exists:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (int(row["config_k"]), int(row["config_cap"]))
                completed[key] = completed.get(key, 0) + 1
        logger.info(f"  Resuming — completed rows: {completed}")

    all_pairs = chunk_store.generate_pairs(n_pairs, seed)

    try:
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
                # Fresh simulators AND a real cache flush, so this K's trials
                # start from the same "nothing resident yet" condition the
                # simulated harness's fresh KVCacheSimulator(CAP) implies.
                # (See benchmark_harness.py module docstring / delivery message
                # re: this being an addition beyond the plan's literal text.)
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

                    metrics_before = cache_manager.get_prefix_cache_counters() if track_real_hits else None

                    result = run_trial(
                        pri, sec, q_type, chunk_store, chunk_store.sim_matrix,
                        chunk_store.adj_matrix, chunk_store.raw_edges,
                        K, sim_lru, sim_cos, sim_grp, sim_adp,
                        adaptive_weights, cache_manager, model_config, logger,
                        flush_between_arms=flush_between_arms,
                    )
                    if result is None:
                        logger.warning(f"  Skipping run {run_i} (all retries failed)")
                        continue

                    if track_real_hits:
                        metrics_after = cache_manager.get_prefix_cache_counters()
                        h0, q0 = _extract_hits_queries(metrics_before)
                        h1, q1 = _extract_hits_queries(metrics_after)
                        metrics_writer.writerow({
                            "config_k": K, "config_cap": CAP, "run": run_i,
                            "prefix_cache_hits_delta": round(h1 - h0, 1),
                            "prefix_cache_queries_delta": round(q1 - q0, 1),
                        })
                        metrics_f.flush()

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
    finally:
        if metrics_f is not None:
            metrics_f.close()

    logger.info(f"  CSV: {csv_path}")
    if track_real_hits:
        logger.info(f"  Real /metrics deltas: {metrics_csv_path}")
    return csv_path
