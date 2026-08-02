"""
qa_quality_eval.py
====================
Answers your supervisor's question directly: "Does graph-selected context
produce better answers than cosine-selected context?" — now extended to a
3-way comparison including adaptive, per your follow-up.

Design (reuses the SAME verbatim-ported retrieval functions from
graph_algorithms.py as the KV-cache-latency experiment — build_graph's
top-M + 2-hop construction, get_prefetch_cosine, get_prefetch_graph,
get_prefetch_adaptive):

  Each HotpotQA question gets its OWN private chunk pool and graph, built
  only from that question's own supporting context (NOT shared across
  questions). Earlier versions of this script built one graph across
  several concatenated questions' contexts, which let build_graph's
  structural-adjacency edges connect chunks from two totally unrelated
  questions just because they landed next to each other after
  concatenation — that's noise that degrades the graph condition for
  reasons having nothing to do with cosine-vs-graph quality. Per-question
  graphs avoid that entirely, and are also just the more realistic setup:
  HotpotQA's own "distractor setting" already gives each question its own
  mixed relevant+irrelevant multi-document context.

  Per question:
    1. Split its context into chunks (same splitter/CHUNK_SIZE as
       chunk_store.py), embed, build a private graph (ga.build_graph —
       topm + 2-hop, unmodified).
    2. pri = argmax cosine sim between the question embedding and this
       question's own chunks.
    3. Three context sets, all anchored on the same pri:
         cosine_ids   = [pri] + ga.get_prefetch_cosine(sim_matrix, pri, K)
         graph_ids    = [pri] + ga.get_prefetch_graph(adj_matrix, pri, K)
         adaptive_ids = [pri] + ga.get_prefetch_adaptive(raw_edges, pri, K,
                                  q_type, HARDCODED_ADAPTIVE_WEIGHTS)
       q_type defaults to "multi-hop" for every question — HotpotQA is a
       multi-hop-by-construction dataset, so that's the honest choice
       rather than guessing per-question. Override with --q_type if you
       want to test the other weight sets anyway.
    4. Ask the SAME LLM the SAME question three times, once per context
       set. Score all three against gold answers with standard EM/F1.
  Aggregate: mean EM/F1 per condition, 3-way per-question win/tie counts.

Note on small per-question pools: a HotpotQA row's own context is only
~10 short paragraphs (rarely more than 20-30 chunks after splitting), so
with the default --top_m 20, `build_graph`'s topm construction can end up
connecting nearly every chunk to every other one — there just aren't 20
"other" chunks to be selective about. If cosine/graph/adaptive look very
similar in your results, try a smaller --top_m (e.g. 5-8) so the graph
actually discriminates within these smaller per-question pools; the
shared-corpus benchmark's default of 20 was sized for a much bigger chunk
pool than a single question's own context.

--cheap (default) keeps this affordable on free tiers: few questions, short
answers, and EM/F1-only grading (no LLM-judge pass, which would double+ the
API calls). --full raises n_questions and answer length. K defaults to 12
(>8, per your supervisor's instruction) — override with --K, must be > 8.

IMPORTANT CAVEAT, distinct from the bug fixed above: this measures answer
QUALITY. Your earlier K-sweep (cache hit rate at K=12/14/16) measured
whether a prefetched set contained a specific target CHUNK. Those are
related but not the same metric — a policy can "hit" more often without
producing better LLM answers (redundant chunks, worse ordering, etc.), so
don't assume the cache-hit-rate winner is automatically the answer-quality
winner. This script is how you actually check that assumption.
"""

import os
import sys
import csv
import json
import time
import logging
import zipfile
import tempfile
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import graph_algorithms as ga
import qa_scoring as qs
from llm_providers import LLMClient, QuotaExhaustedError
from exp_config import (CHUNK_SIZE, CHUNK_OVERLAP, DATASET_CONFIGS, RANDOM_SEED,
                        GRAPH_CONSTRUCTION, TOP_M, RESULTS_DIR, HARDCODED_ADAPTIVE_WEIGHTS)

logger = logging.getLogger("qa_quality_eval")

PROMPT_TMPL = (
    "Answer the question using ONLY the context below. Be concise — a few "
    "words or a short phrase, not a full sentence, unless the question "
    "genuinely requires more.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)

CONDITIONS = ["cosine", "graph", "adaptive"]


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    return logging.getLogger("qa_quality_eval")


def build_tag(args) -> str:
    """Uniquely identifies a (provider, backend, dataset, K, top_m, q_type,
    graph_construction, n_questions, seed) config. MUST include every field
    that changes what's being measured — top_m in particular, so that a
    top_m sweep gets a separate CSV + checkpoint per value instead of
    everything overwriting one file."""
    provider_tag = f"{args.provider}{'-'+args.backend if args.backend else ''}"
    tag = (f"{provider_tag}_{args.dataset}_top{args.top_m}_K{args.K}_"
          f"{args.q_type}_{args.graph_construction}_n{args.n_questions}_seed{args.seed}")
    rep = getattr(args, "rep", None)
    if rep is not None:
        tag += f"_rep{rep}"
    return tag


def checkpoint_path(args) -> Path:
    return RESULTS_DIR / f"checkpoint_{build_tag(args)}.json"


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"out_rows": [], "done_idx": []}


def save_checkpoint(path: Path, out_rows: list, done_idx: list):
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"out_rows": out_rows, "done_idx": done_idx}, f)
    tmp.replace(path)  # atomic-ish, avoids a half-written checkpoint on interrupt


def load_hotpot_rows(dataset_key: str, n_rows: int, seed: int) -> pd.DataFrame:
    cfg = DATASET_CONFIGS[dataset_key]
    zip_path = hf_hub_download(repo_id=cfg["hf_name"], filename="data.zip", repo_type="dataset")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            target = f"data/{cfg['split']}.jsonl"
            z.extract(target, tmpdir)
            df = pd.read_json(os.path.join(tmpdir, target), lines=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n_rows, len(df)), replace=False)
    return df.iloc[idx].reset_index(drop=True)


def build_context_str(chunk_texts, ids):
    return "\n\n".join(chunk_texts[i] for i in ids)


def cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return (embeddings @ embeddings.T) / (norms @ norms.T)


def run(args) -> str:
    """Returns 'complete', 'partial' (stopped early on a persistent quota
    limit — safe to re-run the identical command later to resume), or
    'no_data' (nothing usable happened at all)."""
    logger = setup_logging()
    if args.K <= 8:
        raise ValueError("--K must be > 8 (per your supervisor's instruction)")

    args.n_questions = args.n_questions or (8 if args.cheap else 60)
    max_tokens = 24 if args.cheap else 64

    cp_path = checkpoint_path(args)
    if getattr(args, "fresh", False) and cp_path.exists():
        cp_path.unlink()

    tag      = build_tag(args)
    csv_path = RESULTS_DIR / f"qa_quality_{tag}.csv"
    if csv_path.exists() and not cp_path.exists() and not getattr(args, "fresh", False):
        # A completed config's checkpoint is deleted on success (see bottom of
        # this function) — so "CSV exists, no checkpoint" means this exact
        # config already finished. Without this check, re-running a sweep
        # after some top_m values already completed would re-download the
        # dataset, re-embed, and re-issue ALL n_questions worth of API calls
        # for those already-done values before ever reaching the one you
        # actually need to resume — burning quota on work that's already
        # sitting in a CSV. Use --fresh to force a genuine restart instead.
        logger.info(f"  {tag}: already complete ({csv_path.name} exists, no checkpoint) — "
                    f"skipping, 0 new API calls. Use --fresh to force a restart.")
        return "complete"

    logger.info("=" * 70)
    logger.info("Graph vs. cosine vs. adaptive context selection — answer quality")
    logger.info(f"provider={args.provider} backend={args.backend} cheap={args.cheap} "
                f"K={args.K} top_m={args.top_m} q_type={args.q_type} n_questions={args.n_questions}")
    logger.info("=" * 70)

    client = LLMClient(provider=args.provider, backend=args.backend, model=args.model,
                       cheap=args.cheap, api_key=args.api_key, rpm=args.rpm)

    rows = load_hotpot_rows(args.dataset, args.n_questions, args.seed)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)

    ckpt = load_checkpoint(cp_path)
    out_rows = ckpt["out_rows"]
    done_idx = set(ckpt["done_idx"])
    if done_idx:
        logger.info(f"  resuming from checkpoint: {len(done_idx)}/{len(rows)} questions already done "
                    f"({cp_path.name})")

    hit_quota_limit = False
    for i, row in rows.iterrows():
        if i in done_idx:
            continue

        question = row["input"]
        gold = list(row["answers"])

        # Private per-question chunk pool + graph — no cross-question contamination.
        chunk_texts = splitter.split_text(row["context"])
        if len(chunk_texts) < 3:
            logger.warning(f"  [{i}] context too short after splitting ({len(chunk_texts)} chunks) — skipping")
            done_idx.add(i)
            save_checkpoint(cp_path, out_rows, sorted(done_idx))
            continue
        embeddings = embedder.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)
        sim_matrix = cosine_sim_matrix(embeddings)
        adj_matrix, raw_edges, _ = ga.build_graph(
            embeddings, sim_matrix, logger, graph_construction=args.graph_construction, top_m=args.top_m)

        q_embedding = embedder.encode([question], convert_to_numpy=True)[0]
        q_sim = (embeddings @ q_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_embedding) + 1e-9)
        pri = int(np.argmax(q_sim))
        k = min(args.K, len(chunk_texts) - 1)

        ids = {
            "cosine": [pri] + [int(x) for x in ga.get_prefetch_cosine(sim_matrix, pri, k)],
            "graph": [pri] + [int(x) for x in ga.get_prefetch_graph(adj_matrix, pri, k)],
            "adaptive": [pri] + [int(x) for x in ga.get_prefetch_adaptive(
                raw_edges, pri, k, args.q_type, HARDCODED_ADAPTIVE_WEIGHTS)],
        }

        answers, ems, f1s = {}, {}, {}
        try:
            for cond in CONDITIONS:
                prompt = PROMPT_TMPL.format(context=build_context_str(chunk_texts, ids[cond]), question=question)
                answers[cond] = client.generate(prompt, max_tokens=max_tokens).text.strip()
                ems[cond] = qs.exact_match(answers[cond], gold)
                f1s[cond] = qs.max_f1(answers[cond], gold)
        except QuotaExhaustedError as e:
            logger.error(f"  [{i}] {e}")
            logger.error(f"  Stopping here — {len(done_idx)}/{len(rows)} questions done and saved to "
                        f"checkpoint. Re-run this EXACT SAME command later (once quota resets) to resume.")
            hit_quota_limit = True
            break
        except Exception as e:
            logger.error(f"  [{i}] API call failed, skipping: {e}")
            continue

        out_rows.append({
            "q_idx": int(i), "question": question, "gold_answers": " | ".join(gold),
            "pool_size": len(chunk_texts),
            **{f"n_{c}_chunks": len(ids[c]) for c in CONDITIONS},
            **{f"{c}_answer": answers[c] for c in CONDITIONS},
            **{f"{c}_em": ems[c] for c in CONDITIONS},
            **{f"{c}_f1": round(f1s[c], 4) for c in CONDITIONS},
        })
        done_idx.add(i)
        save_checkpoint(cp_path, out_rows, sorted(done_idx))
        logger.info(f"  [{i+1}/{len(rows)}] pool={len(chunk_texts)} "
                    f"cos_f1={f1s['cosine']:.2f} grp_f1={f1s['graph']:.2f} adp_f1={f1s['adaptive']:.2f}  "
                    f"Q: {question[:55]}")

    n = len(out_rows)
    if n == 0:
        logger.error("No questions completed successfully — nothing to report.")
        return "no_data"

    # Recomputed from the full cumulative out_rows (not tracked incrementally)
    # so resumed runs get correct aggregates over old + new rows together.
    sums = {c: {"em": 0.0, "f1": 0.0} for c in CONDITIONS}
    win_counts = {c: 0 for c in CONDITIONS}
    win_counts["tie"] = 0
    for r in out_rows:
        for c in CONDITIONS:
            sums[c]["em"] += r[f"{c}_em"]
            sums[c]["f1"] += r[f"{c}_f1"]
        f1s_r = {c: r[f"{c}_f1"] for c in CONDITIONS}
        best_f1 = max(f1s_r.values())
        winners = [c for c in CONDITIONS if f1s_r[c] == best_f1]
        if len(winners) == 1:
            win_counts[winners[0]] += 1
        else:
            win_counts["tie"] += 1

    tag = build_tag(args)
    csv_path = RESULTS_DIR / f"qa_quality_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    logger.info("\n" + "=" * 70)
    status_label = "PARTIAL (stopped on quota limit)" if hit_quota_limit else "RESULTS"
    logger.info(f"{status_label} over {n} questions (K={args.K}, top_m={args.top_m}, q_type={args.q_type}, {tag})")
    for cond in CONDITIONS:
        logger.info(f"  {cond:9s}: mean EM={sums[cond]['em']/n:.3f}  mean F1={sums[cond]['f1']/n:.3f}")
    logger.info(f"  per-question F1 winner: " +
                "  ".join(f"{c}={win_counts[c]}" for c in CONDITIONS) + f"  tie={win_counts['tie']}")
    if n < 30:
        logger.info(f"  NOTE: n={n} is a smoke test, not a reliable result — re-run with "
                    f"--n_questions 40+ (or --full) before drawing any real conclusion.")
    logger.info(f"  CSV: {csv_path}")
    logger.info("=" * 70)

    if hit_quota_limit:
        return "partial"

    if cp_path.exists():
        cp_path.unlink()  # config fully done — next identical run starts fresh, not "already complete, 0 to do"
    return "complete"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True, choices=["gemini", "openai", "anthropic"])
    p.add_argument("--backend", choices=["openai", "openrouter", "groq"],
                  help="Required when --provider openai. Picks base_url + api-key env var.")
    p.add_argument("--model", help="Override the provider's default model")
    p.add_argument("--api_key", help="Override the provider's default env var")
    p.add_argument("--cheap", action="store_true", default=True,
                  help="Small n_questions, short answers, free-tier-friendly models (default ON)")
    p.add_argument("--full", dest="cheap", action="store_false",
                  help="Larger n_questions, longer answers — costs real money/quota")
    p.add_argument("--rpm", type=int, default=5,
                  help="Max LLM calls per rolling 60s window (sliding-window limiter, not a "
                       "hard batch-then-sleep). Default 5 is safe under both Gemini's free-tier "
                       "10 RPM and Groq's free-tier 30 RPM. Pass 0 to disable.")
    p.add_argument("--K", type=int, default=12, help="Must be > 8")
    p.add_argument("--top_m", type=int, default=TOP_M,
                  help="Consider lowering (e.g. 5-8) — see module docstring re: small per-question pools")
    p.add_argument("--q_type", default="multi-hop", choices=["semantic", "structural", "multi-hop"],
                  help="Adaptive-policy weight set to use. Defaults to multi-hop since HotpotQA is multi-hop by construction.")
    p.add_argument("--n_questions", type=int, default=None)
    p.add_argument("--dataset", default="hotpot", choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--graph_construction", default=GRAPH_CONSTRUCTION, choices=["threshold", "topm"])
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--fresh", action="store_true",
                  help="Ignore/delete any existing checkpoint for this exact config and start over")
    args = p.parse_args()

    if args.provider == "openai" and not args.backend:
        p.error("--backend is required when --provider openai (openrouter | groq | openai)")

    status = run(args)
    if status == "partial":
        sys.exit(1)


if __name__ == "__main__":
    main()
