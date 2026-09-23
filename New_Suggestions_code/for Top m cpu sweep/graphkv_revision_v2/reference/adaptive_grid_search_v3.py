"""
adaptive_grid_search_v2.py
===========================
Stage 1 of the adaptive-weight extension.

Runs entirely offline — no GPU needed (a tokenizer is loaded, but only
for counting tokens on CPU; no model weights are loaded).
Reads your existing hit-aware CSV files, tries every (alpha, beta) on a
grid, and finds the pair that minimizes EXPECTED latency for each
non-semantic query type.

CHANGES vs v1 of this file:
  - Objective is now hit-aware expected latency, not a token-count linear
    fit pooled across hit+miss rows. v1's comment explained it avoided
    scoring by hit rate because "the paper's own Hit-Rate Paradox disproves
    hit rate predicts latency" — but that conclusion was drawn from
    PRE-hit-aware-fix data, where a hit never got cheaper than a miss, so
    of course hit rate didn't predict latency: nothing did except token
    count. Now that hits really do cost ~0ms, hit rate is exactly the
    right thing to fold into the objective. We fit miss-only latency (NOT
    pooled with hit rows, which are ~0ms and would corrupt the fit) as a
    function of token count, then score each (alpha, beta) as
    expected_ms = hit_rate*HIT_LATENCY_MS + (1-hit_rate)*predicted_miss_ms.
  - "semantic" is skipped entirely (never grid-searched) — it's defined as
    argmax(cosine_similarity[pri]), i.e. exactly cosine's own ranking, so
    cosine wins it by construction and including it just biases the
    optimizer toward whatever superficially helps that unwinnable case.
  - Alpha grid clamped to [0.35, 0.80] (was [0.10, 0.90]) to discourage
    the optimizer from collapsing to a degenerate near-0/near-1 weight
    that overfits this one dataset.
  - ONE (alpha, beta) chosen per query type, aggregated (mean expected
    latency) ACROSS all K in the CSV — not a separate weight per K. More
    free parameters on a single dataset with no held-out split compounds
    the exact overfitting risk the alpha-clamp is guarding against. Per-K
    performance of the single chosen weight is still reported, as a
    validation table, so K-dependence is visible rather than hidden.
  - get_adaptive_prefetch is now 2-hop (was 1-hop), with 0.9*min+0.1*max
    blended scoring, matching kv_cache_experiment_adaptive_v2.py exactly —
    the grid search must score candidates the way they'll actually be
    selected at inference time or the tuned weights don't transfer.
  - build_raw_graph now supports --graph_construction topm (must match
    whatever the real run uses, or weights are tuned on the wrong graph).

Output:
  adaptive_weights_<model>_<dataset>.json — one (alpha, beta) per query
    type (semantic excluded), plus per-K validation numbers
  grid_search_<model>_<dataset>.csv — full grid, every (K, q_type, alpha)

Usage:
  python adaptive_grid_search_v2.py \
      --csv_dir ./results \
      --dataset_key hotpot --model_tag Qwen2.5-1.5B \
      --output_dir ./adaptive_results \
      --graph_construction topm --top_m 20
"""

import os
import sys
import json
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import zipfile, tempfile

# ── Shared constants (must match experiment script) ───────────────────────────
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
SEM_THRESHOLD   = 0.55
MAX_DEGREE      = 100   # was 10 — mismatched kv_cache_experiment_adaptive_v2.py/final_v2.py's 100
RANDOM_SEED     = 42
SEPARATOR       = "\n\n"
PRIMARY_MAX_LEN = 512   # must match kv_cache_experiment_adaptive.py
EXTEND_MAX_LEN  = 256   # must match kv_cache_experiment_adaptive.py

# Short model tag (as passed via --model_tag) -> HF repo id, for loading the
# tokenizer used to count tokens offline (CPU only, no GPU/model weights needed).
# MODEL_REPO_MAP = {
#     "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
#     "Qwen2.5-3B":   "Qwen/Qwen2.5-3B-Instruct",
#     "Qwen2.5-7B":   "Qwen/Qwen2.5-7B-Instruct",
#     "Qwen2.5-14B":  "Qwen/Qwen2.5-14B-Instruct",
# }

MODEL_REPO_MAP = {
    # Qwen
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B":   "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B":   "Qwen/Qwen2.5-7B-Instruct",

    # Llama
    "Llama3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",

    # Gemma
    "Gemma3-1B": "google/gemma-3-1b-it",
    "Gemma3-4B": "google/gemma-3-4b-it",

    # Phi
    "Phi3.5": "microsoft/Phi-3.5-mini-instruct",
}

DATASET_CONFIGS = {
    "hotpot":     {"hf_name": "THUDM/LongBench", "split": "hotpotqa",        "text_key": "context", "label": "HotpotQA"},
    "2wiki":      {"hf_name": "THUDM/LongBench", "split": "2wikimqa",        "text_key": "context", "label": "2WikiMultiHopQA"},
    "musique":    {"hf_name": "THUDM/LongBench", "split": "musique",         "text_key": "context", "label": "MuSiQue"},
    "multifield": {"hf_name": "THUDM/LongBench", "split": "multifieldqa_en", "text_key": "context", "label": "MultiFieldQA-en"},
}

Q_TYPES = ["semantic", "structural", "multi-hop"]


# ── Grid definition ───────────────────────────────────────────────────────────
# Alpha = semantic weight, Beta = structural weight.
# We constrain alpha + beta = 1 so only alpha varies.
# Clamped to [0.35, 0.80]: letting the search reach 0.95+/0.05- lets it
# collapse toward "ignore structural signal entirely" (or vice versa) to
# chase noise in a single dataset with no held-out split -- clamping is a
# cheap regularizer against exactly that.
ALPHA_GRID = np.round(np.arange(0.35, 0.85, 0.05), 2)


# ── Graph utilities ───────────────────────────────────────────────────────────
def build_raw_graph(sim_matrix: np.ndarray, graph_construction="topm", top_m=20) -> dict:
    """
    Build the raw edge store: for each (i,j) pair that qualifies,
    store (sem_sim, struct_score). Weights are NOT computed yet —
    we compute them on-the-fly during grid search so we can vary alpha.

    MUST match kv_cache_experiment_adaptive_v2.py's build_graph exactly
    (same graph_construction/top_m) or the weights are being tuned on a
    different graph than they'll actually run on.

    Returns: raw_edges dict  (i -> list of (j, sem_c, struct_score))
    """
    n = len(sim_matrix)
    raw_edges: dict[int, list] = {i: [] for i in range(n)}

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
            is_adj = (dist == 1)
            if graph_construction == "topm":
                is_sem = (j in topm_sets[i]) or (i in topm_sets[j])
            else:
                is_sem = (sem >= SEM_THRESHOLD)
            if is_sem or is_adj:
                sem_c = sem if is_sem else 0.0
                raw_edges[i].append((j, sem_c, ss))
                raw_edges[j].append((i, sem_c, ss))

    return raw_edges


def get_adaptive_prefetch(raw_edges: dict, pri: int, k: int,
                          alpha: float, beta: float) -> list:
    """
    Two-hop re-rank of neighbours of `pri` using (alpha, beta), matching
    get_prefetch_adaptive in kv_cache_experiment_adaptive_v2.py exactly
    (0.9*min(w1,w2) + 0.1*max(w1,w2) blended hop scoring) — the grid search
    must score candidates the same way they'll actually be selected at
    inference time, or the tuned weights don't transfer.

    No max_degree cap here (removed) — matches get_prefetch_adaptive's fix:
    naturally returns fewer than k if fewer than k candidates are reachable,
    same principle get_prefetch_graph already used. The old ranked[:MAX_DEGREE]
    cap here was silently evaluating K>10 grid points against a truncated
    pool during weight search.
    """
    neighbours = raw_edges.get(pri, [])
    if not neighbours:
        return []

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
    return [j for j, _ in ranked[:k]]


def prefetch_token_count(tokenizer, primary_text: str, prefetch_texts: list) -> int:
    """
    Token count of the KV context at the timed extend step (primary + prefetch),
    truncated exactly as kv_cache_experiment_adaptive.py truncates them. This is
    what real GPU latency actually scales with — not whether a specific target
    chunk happened to land in the top-K.
    """
    n = len(tokenizer(primary_text, truncation=True,
                      max_length=PRIMARY_MAX_LEN)["input_ids"])
    if prefetch_texts:
        n += len(tokenizer(SEPARATOR.join(prefetch_texts), truncation=True,
                           max_length=EXTEND_MAX_LEN, add_special_tokens=False)["input_ids"])
    return n


# ── Load corpus (same logic as experiment script) ─────────────────────────────
def load_corpus(dataset_key: str) -> tuple[list[str], np.ndarray]:
    cfg = DATASET_CONFIGS[dataset_key]
    print(f"  Loading corpus: {cfg['label']} ...")
    zip_path = hf_hub_download(repo_id=cfg["hf_name"], filename="data.zip",
                               repo_type="dataset")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            target = f"data/{cfg['split']}.jsonl"
            z.extract(target, tmpdir)
            df = pd.read_json(os.path.join(tmpdir, target), lines=True)

    contexts = []
    for ctx in df[cfg["text_key"]]:
        contexts.append(ctx)
        if sum(len(c) for c in contexts) >= 100_000:
            break
    raw = "\n\n".join(contexts)

    splitter    = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)
    chunk_texts = splitter.split_text(raw)
    print(f"  Chunks: {len(chunk_texts)}")

    print("  Computing embeddings ...")
    embedder   = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunk_texts, batch_size=32, convert_to_numpy=True,
                                 show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)
    print(f"  Sim matrix: {sim_matrix.shape}")
    return chunk_texts, sim_matrix


# ── Main grid search ──────────────────────────────────────────────────────────
HIT_LATENCY_MS = 0.0   # must match kv_cache_experiment_adaptive_v2.py


def run_grid_search(csv_path: Path, chunk_texts: list, sim_matrix: np.ndarray,
                    dataset_key: str, model_tag: str, output_dir: Path,
                    tokenizer, graph_construction="topm", top_m=20):

    print(f"\nLoading CSV: {csv_path.name}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}  |  K values: {sorted(df['config_k'].unique())}")

    # Build raw edge store once — reused for all (alpha, beta) combinations.
    # MUST match the graph_construction the real run will use (see build_raw_graph
    # docstring) — this is v2's fix for training/eval graph mismatch (v1 always
    # trained on threshold-gated edges regardless of what was actually run).
    print(f"  Building raw edge store (graph_construction={graph_construction}"
          f"{f', top_m={top_m}' if graph_construction=='topm' else ''}) ...")
    raw_edges = build_raw_graph(sim_matrix, graph_construction=graph_construction, top_m=top_m)
    n_chunks  = len(chunk_texts)

    # ── Simulation cache class (same LRU as experiment) ──────────────────────
    from collections import OrderedDict

    class LRU:
        def __init__(self, cap): self.cap = cap; self.c = OrderedDict()
        def access(self, x):
            hit = x in self.c
            if hit: self.c.move_to_end(x)
            else:
                if len(self.c) >= self.cap: self.c.popitem(last=False)
                self.c[x] = True
            return hit
        def prefetch(self, ids):
            for x in ids:
                if x not in self.c:
                    if len(self.c) >= self.cap: self.c.popitem(last=False)
                    self.c[x] = True

    all_records = []       # per (K, q_type, alpha) — for the full grid CSV
    per_type_by_k = {qt: {} for qt in Q_TYPES}   # qt -> {k: {alpha: expected_ms}}

    # Regenerate the exact deterministic pair sequence used by the real run
    # (shared across K, since generate_pairs doesn't depend on K).
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    safe_n = max(n_chunks - 3, 1)
    all_pairs_local = []
    for _ in range(600):
        pri = int(rng.integers(0, safe_n))
        r   = rng.random()
        if r < 0.25:
            qt = "semantic"
            sims = sim_matrix[pri].copy()
            sims[max(0, pri-2):min(n_chunks, pri+3)] = -1
            sec = int(np.argmax(sims))
        elif r < 0.50:
            qt = "structural"
            offset = int(rng.choice([1, 2, 3]))
            sec = min(n_chunks - 1, pri + offset)
        else:
            qt = "multi-hop"
            if rng.random() < 0.5:
                sec = min(n_chunks - 1, pri + 2)
            else:
                sims = sim_matrix[pri].copy()
                sims[max(0, pri-2):min(n_chunks, pri+3)] = -1
                sec = int(np.argmax(sims))
        all_pairs_local.append((pri, sec, qt))

    for k_val in sorted(df["config_k"].unique()):
        k_df = df[df["config_k"] == k_val].copy().reset_index(drop=True)
        CAP  = int(k_df["config_cap"].iloc[0])
        print(f"\n  K={k_val}  ({len(k_df)} rows) ...")

        # ── Fit MISS-ONLY latency(tokens), per condition, pooled ──────────────
        # v1 pooled hit+miss rows together into one linear fit. That was fine
        # before the hit-aware timing fix (latency really was ~continuous in
        # token count back then), but now a hit is ~0ms regardless of token
        # count and a miss is the real, token-scaling cost — pooling them
        # mixes two different regimes into one line and corrupts the fit.
        # Filtering to misses only (via each condition's own hit column) fixes
        # this; if the true relationship is closer to flat (as our diagnostic
        # on the 1.5B data suggested — miss cost dominated by a fixed forward-
        # pass cost, not the marginal prefetched tokens) the fitted slope will
        # simply come out near zero on its own rather than us assuming it.
        cold_miss = k_df[k_df["lru_sim_hit"] == 0]
        cos_miss  = k_df[k_df["cos_sim_hit"] == 0]
        grp_miss  = k_df[k_df["grp_sim_hit"] == 0]
        xs = np.concatenate([cold_miss["cold_n_tokens"], cos_miss["cosine_n_tokens"], grp_miss["graph_n_tokens"]]).astype(float)
        ys = np.concatenate([cold_miss["cold_ms"],       cos_miss["cosine_ms"],       grp_miss["graph_ms"]]).astype(float)
        if len(xs) < 2:
            print(f"    Not enough miss rows at K={k_val} to fit — skipping.")
            continue
        slope, intercept = np.polyfit(xs, ys, 1)
        print(f"    miss-only latency(tokens) fit: ms = {slope:.5f} * n_tokens + {intercept:.3f}  "
              f"(n={len(xs)} pooled MISS points; hit rows excluded)")

        for q_type in Q_TYPES:
            if q_type == "semantic":
                # Never learned — see FALLBACK_ADAPTIVE_WEIGHTS comment in the
                # experiment script. Skipping this keeps the optimizer from
                # being biased by a query type cosine wins by definition.
                continue

            qt_df = df[(df["config_k"] == k_val) & (df["type"] == q_type)].copy()
            if qt_df.empty or "run" not in qt_df.columns:
                continue

            for alpha in ALPHA_GRID:
                beta = round(1.0 - alpha, 2)

                sim_adp       = LRU(CAP)
                hit_by_run    = {}
                tokens_by_run = {}

                for run_idx, (pri, sec, qt) in enumerate(all_pairs_local):
                    # Cache state is shared across query types within a K config
                    # in the real experiment, so we must replay every pair in
                    # order, not just the ones matching q_type.
                    a_ids = get_adaptive_prefetch(raw_edges, pri, k_val, alpha, beta)
                    sim_adp.access(pri)
                    sim_adp.prefetch(a_ids)
                    hit = int(sim_adp.access(sec))
                    if qt == q_type:
                        hit_by_run[run_idx] = hit
                        if hit:
                            tokens_by_run[run_idx] = 0.0   # unused: hit -> HIT_LATENCY_MS regardless
                        else:
                            a_texts = [chunk_texts[j] for j in a_ids]
                            tokens_by_run[run_idx] = prefetch_token_count(
                                tokenizer, chunk_texts[pri], a_texts)

                # Align to the ACTUAL CSV rows for this query type via the "run"
                # column — robust to any trials the real GPU run skipped.
                matched = qt_df[qt_df["run"].isin(hit_by_run)]
                if matched.empty:
                    continue

                row_hits   = matched["run"].map(hit_by_run).to_numpy()
                row_tokens = matched["run"].map(tokens_by_run).to_numpy(dtype=float)
                cos_lats   = matched["cosine_ms"].to_numpy()
                grp_lats   = matched["graph_ms"].to_numpy()

                # THE FIX (suggestion #1): expected latency, hit-aware —
                # not a continuous token-count line applied uniformly.
                predicted_miss_ms = slope * row_tokens + intercept
                adp_lats          = np.where(row_hits == 1, HIT_LATENCY_MS, predicted_miss_ms)
                adp_hit_rate      = float(row_hits.mean())
                expected_ms       = float(np.mean(adp_lats))
                delta             = float(np.mean(cos_lats) - expected_ms)

                all_records.append({
                    "dataset": dataset_key, "model": model_tag,
                    "k": k_val, "q_type": q_type,
                    "alpha": alpha, "beta": beta,
                    "adp_hit_rate": round(adp_hit_rate, 4),
                    "adp_expected_ms": round(expected_ms, 4),
                    "cos_mean_ms": round(float(np.mean(cos_lats)), 4),
                    "grp_mean_ms": round(float(np.mean(grp_lats)), 4),
                    "delta_vs_cosine": round(delta, 4),
                    "fixed_delta": round(float(np.mean(cos_lats) - np.mean(grp_lats)), 4),
                })
                per_type_by_k[q_type].setdefault(k_val, {})[alpha] = expected_ms

    grid_df = pd.DataFrame(all_records)
    grid_csv = output_dir / f"grid_search_{model_tag}_{dataset_key}.csv"
    grid_df.to_csv(grid_csv, index=False)
    print(f"\n  Full grid saved: {grid_csv}")

    # ── Pick ONE (alpha, beta) per query type, aggregated ACROSS all K ────────
    # Not per-K (see kv_cache_experiment_adaptive_v2.py main() for the full
    # rationale): more free parameters on one dataset with no held-out split
    # is exactly the overfitting risk the alpha-clamp above is already
    # guarding against. We score each alpha by its MEAN expected latency
    # across every K present in the CSV, and report per-K performance of
    # that single choice as a validation table so any K-dependence is
    # visible rather than papered over by refitting a new weight at each K.
    best_weights = {}

    print(f"\n{'='*70}")
    print(f"  OPTIMAL ADAPTIVE WEIGHTS  (one per query type, aggregated across K)")
    print(f"{'='*70}")

    for q_type in ["structural", "multi-hop"]:
        by_k = per_type_by_k[q_type]
        if not by_k:
            continue
        ks_present = sorted(by_k.keys())
        # mean expected latency across K, for each alpha that has data at every K
        alphas_common = set.intersection(*(set(by_k[k].keys()) for k in ks_present))
        if not alphas_common:
            print(f"  {q_type}: no alpha had data at every K — skipping")
            continue
        agg = {a: np.mean([by_k[k][a] for k in ks_present]) for a in alphas_common}
        best_alpha = min(agg, key=agg.get)   # lower expected latency = better
        best_beta  = round(1.0 - best_alpha, 2)

        # Validation table: how does this single choice do at each K?
        per_k_perf = {}
        for k_val in ks_present:
            sub = grid_df[(grid_df["k"] == k_val) & (grid_df["q_type"] == q_type) &
                          (grid_df["alpha"] == best_alpha)]
            if sub.empty: continue
            row = sub.iloc[0]
            per_k_perf[str(k_val)] = {
                "expected_ms": float(row["adp_expected_ms"]),
                "delta_vs_cosine": float(row["delta_vs_cosine"]),
                "hit_rate": float(row["adp_hit_rate"]),
            }

        best_weights[q_type] = {
            "alpha": float(best_alpha),
            "beta":  float(best_beta),
            "mean_expected_ms_across_k": round(float(agg[best_alpha]), 3),
            "per_k_validation": per_k_perf,
        }

        print(f"\n  {q_type:>12}  alpha={best_alpha:.2f}  beta={best_beta:.2f}")
        for k_str, perf in per_k_perf.items():
            print(f"      K={k_str}: expected={perf['expected_ms']:.2f}ms  "
                  f"Δ_vs_cosine={perf['delta_vs_cosine']:+.3f}ms  "
                  f"hit_rate={perf['hit_rate']*100:.1f}%")

    print(f"\n  semantic: not learned (fixed default — see experiment script's "
          f"FALLBACK_ADAPTIVE_WEIGHTS comment)")

    # Save weights JSON — flat, per query type, no per-K nesting
    weights_path = output_dir / f"adaptive_weights_{model_tag}_{dataset_key}.json"
    full_output = {
        "model": model_tag, "dataset": dataset_key,
        "graph_construction": graph_construction,
        "top_m": top_m if graph_construction == "topm" else None,
        "note": ("One (alpha, beta) per query type, chosen to minimize MEAN "
                 "expected latency across all K in the source CSV (not per-K — "
                 "see rationale in kv_cache_experiment_adaptive_v2.py). "
                 "'semantic' intentionally excluded — see that file's "
                 "FALLBACK_ADAPTIVE_WEIGHTS comment. per_k_validation shows how "
                 "the single chosen weight performs at each K, for transparency."),
        "weights": best_weights,
    }
    with open(weights_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n  Weights saved: {weights_path}")

    return weights_path

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Grid search for adaptive alpha/beta weights")
    p.add_argument("--csv_dir",     required=True, help="Directory containing result CSVs")
    p.add_argument("--dataset_key", required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--model_tag",   required=True, help="Short model name, e.g. Qwen2.5-1.5B")
    p.add_argument("--csv_file",    default=None,
                   help="Explicit CSV path (overrides csv_dir + model_tag pattern)")
    p.add_argument("--output_dir",  default="./adaptive_results")
    p.add_argument("--model_repo",  default=None,
                   help="HF repo id for the tokenizer used to count prefetch "
                        "tokens (e.g. Qwen/Qwen2.5-1.5B-Instruct). Inferred "
                        "from --model_tag via MODEL_REPO_MAP if omitted.")
    p.add_argument("--graph_construction", default="topm", choices=["threshold", "topm"],
                   help="MUST match what kv_cache_experiment_adaptive_v2.py will "
                        "use — training weights on one graph and running them "
                        "on another silently invalidates the whole exercise.")
    p.add_argument("--top_m", type=int, default=20,
                   help="Neighbors per node when --graph_construction topm.")
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.csv_file:
        csv_path = Path(args.csv_file)
    else:
        # Try to find the CSV automatically
        csv_dir = Path(args.csv_dir)
        pattern = f"*{args.model_tag.replace('.', '_')}*{args.dataset_key}*"
        matches = list(csv_dir.glob(pattern))
        if not matches:
            # Broader search
            matches = list(csv_dir.glob(f"*{args.dataset_key}*.csv"))
        if not matches:
            print(f"ERROR: no CSV found in {csv_dir} matching dataset={args.dataset_key}")
            sys.exit(1)
        csv_path = matches[0]
        print(f"Auto-detected CSV: {csv_path}")

    print("=" * 60)
    print(f"Adaptive Weight Grid Search")
    print(f"  Dataset:  {args.dataset_key}")
    print(f"  Model:    {args.model_tag}")
    print(f"  CSV:      {csv_path}")
    print(f"  Output:   {output_dir}")
    print(f"  Graph:    {args.graph_construction}" +
          (f" (top_m={args.top_m})" if args.graph_construction == "topm" else ""))
    print(f"  Grid:     alpha in {ALPHA_GRID.tolist()}")
    print("=" * 60)

    repo_id = args.model_repo or MODEL_REPO_MAP.get(
        args.model_tag, f"Qwen/{args.model_tag}-Instruct")
    print(f"  Loading tokenizer for token-count latency proxy: {repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    chunk_texts, sim_matrix = load_corpus(args.dataset_key)
    run_grid_search(csv_path, chunk_texts, sim_matrix,
                    args.dataset_key, args.model_tag, output_dir, tokenizer,
                    graph_construction=args.graph_construction, top_m=args.top_m)

    print("\n✅ Grid search complete.")


if __name__ == "__main__":
    main()
