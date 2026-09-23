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
import time
import logging
import zipfile
import tempfile
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from text_chunking import get_recursive_character_text_splitter  # see that module's docstring -- chunk_store.py's earlier fix here didn't actually route around the problem
from sentence_transformers import SentenceTransformer

import graph_algorithms as ga
import qa_scoring as qs
from llm_providers import LLMClient
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


def run(args):
    logger = setup_logging()
    if args.K <= 8:
        raise ValueError("--K must be > 8 (per your supervisor's instruction)")

    n_questions = args.n_questions or (8 if args.cheap else 60)
    max_tokens = 24 if args.cheap else 64

    logger.info("=" * 70)
    logger.info("Graph vs. cosine vs. adaptive context selection — answer quality")
    logger.info(f"provider={args.provider} backend={args.backend} cheap={args.cheap} "
                f"K={args.K} top_m={args.top_m} q_type={args.q_type} n_questions={n_questions}")
    logger.info("=" * 70)

    client = LLMClient(provider=args.provider, backend=args.backend, model=args.model,
                       cheap=args.cheap, api_key=args.api_key)

    rows = load_hotpot_rows(args.dataset, n_questions, args.seed)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    splitter = get_recursive_character_text_splitter()(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)

    out_rows = []
    sums = {c: {"em": 0.0, "f1": 0.0} for c in CONDITIONS}
    win_counts = {c: 0 for c in CONDITIONS}
    win_counts["tie"] = 0

    for i, row in rows.iterrows():
        question = row["input"]
        gold = list(row["answers"])

        # Private per-question chunk pool + graph — no cross-question contamination.
        chunk_texts = splitter.split_text(row["context"])
        if len(chunk_texts) < 3:
            logger.warning(f"  [{i}] context too short after splitting ({len(chunk_texts)} chunks) — skipping")
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
        except Exception as e:
            logger.error(f"  [{i}] API call failed, skipping: {e}")
            continue

        for cond in CONDITIONS:
            sums[cond]["em"] += ems[cond]
            sums[cond]["f1"] += f1s[cond]
        best_f1 = max(f1s.values())
        winners = [c for c in CONDITIONS if f1s[c] == best_f1]
        if len(winners) == 1:
            win_counts[winners[0]] += 1
        else:
            win_counts["tie"] += 1

        out_rows.append({
            "q_idx": i, "question": question, "gold_answers": " | ".join(gold),
            "pool_size": len(chunk_texts),
            **{f"n_{c}_chunks": len(ids[c]) for c in CONDITIONS},
            **{f"{c}_answer": answers[c] for c in CONDITIONS},
            **{f"{c}_em": ems[c] for c in CONDITIONS},
            **{f"{c}_f1": round(f1s[c], 4) for c in CONDITIONS},
        })
        logger.info(f"  [{i+1}/{len(rows)}] pool={len(chunk_texts)} "
                    f"cos_f1={f1s['cosine']:.2f} grp_f1={f1s['graph']:.2f} adp_f1={f1s['adaptive']:.2f}  "
                    f"Q: {question[:55]}")

    n = len(out_rows)
    if n == 0:
        logger.error("No questions completed successfully — nothing to report.")
        return

    tag = f"{args.provider}{'-'+args.backend if args.backend else ''}"
    csv_path = RESULTS_DIR / f"qa_quality_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    logger.info("\n" + "=" * 70)
    logger.info(f"RESULTS over {n} questions (K={args.K}, top_m={args.top_m}, q_type={args.q_type}, {tag})")
    for cond in CONDITIONS:
        logger.info(f"  {cond:9s}: mean EM={sums[cond]['em']/n:.3f}  mean F1={sums[cond]['f1']/n:.3f}")
    logger.info(f"  per-question F1 winner: " +
                "  ".join(f"{c}={win_counts[c]}" for c in CONDITIONS) + f"  tie={win_counts['tie']}")
    if n < 30:
        logger.info(f"  NOTE: n={n} is a smoke test, not a reliable result — re-run with "
                    f"--n_questions 40+ (or --full) before drawing any real conclusion.")
    logger.info(f"  CSV: {csv_path}")
    logger.info("=" * 70)


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
    p.add_argument("--K", type=int, default=12, help="Must be > 8")
    p.add_argument("--top_m", type=int, default=TOP_M,
                  help="Consider lowering (e.g. 5-8) — see module docstring re: small per-question pools")
    p.add_argument("--q_type", default="multi-hop", choices=["semantic", "structural", "multi-hop"],
                  help="Adaptive-policy weight set to use. Defaults to multi-hop since HotpotQA is multi-hop by construction.")
    p.add_argument("--n_questions", type=int, default=None)
    p.add_argument("--dataset", default="hotpot", choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--graph_construction", default=GRAPH_CONSTRUCTION, choices=["threshold", "topm"])
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = p.parse_args()

    if args.provider == "openai" and not args.backend:
        p.error("--backend is required when --provider openai (openrouter | groq | openai)")

    run(args)


if __name__ == "__main__":
    main()
