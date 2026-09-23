"""
correctness_check.py
=====================
Implementation of implementation_plan.md Phase 3's correctness_check.py:
N=20 sample trials comparing "fresh" (no cache reuse — everything
recomputed) vs. "blended" (CacheBlend reuse + ~15% boundary recompute)
generations for the SAME prompt, to quantify CacheBlend's approximation
error against the reference (fresh) output.

Method per trial:
  1. flush_cache()
  2. Send the full prompt (primary + a few context chunks + query) with
     nothing warmed first -> "fresh" generation (full compute baseline).
  3. flush_cache() again, then warm_chunks() each context chunk
     individually so LMCache genuinely stores their KV.
  4. Send the identical prompt again -> "blended" generation (this time
     LMCache reuses the warmed chunks' KV + CacheBlend's recompute-ratio
     correction at boundaries).
  5. Compare fresh vs. blended:
       - token exact-match rate (position-wise, up to min length)
       - output embedding cosine similarity (all-MiniLM-L6-v2, same
         embedder chunk_store already uses)
       - perplexity delta: mean per-token -logprob of the BLENDED
         sequence's own tokens (self-perplexity under greedy decode)
         minus the FRESH sequence's self-perplexity. A growing delta
         across trials is evidence CacheBlend's recompute ratio is too
         aggressive for this recompute_ratio/chunk_size combination.

Requires the completions endpoint to return logprobs (`logprobs: 1` in the
request) — verify your vLLM version's OpenAI-compatible server exposes this
for the /v1/completions route (it does as of the versions the plan pins to,
but confirm since this is exactly the kind of API-surface detail that
changes across vLLM releases).
"""

import csv
import logging
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from exp_config import RESULTS_DIR, EXTEND_MAX_LEN

logger = logging.getLogger("correctness_check")

FIELDNAMES = [
    "trial", "pri", "sec", "n_context_chunks",
    "token_exact_match_rate", "output_cosine_sim",
    "fresh_perplexity", "blended_perplexity", "perplexity_delta",
    "fresh_text_preview", "blended_text_preview",
]


def _generate_with_logprobs(cache_manager, token_ids: List[int], max_tokens: int = 50):
    resp = cache_manager._session.post(
        f"{cache_manager.base_url}/completions",
        json={"model": cache_manager.model_name, "prompt": token_ids,
              "max_tokens": max_tokens, "temperature": 0, "logprobs": 1},
        timeout=cache_manager.request_timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    text = choice["text"]
    logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
    out_ids = cache_manager.tokenizer(text, add_special_tokens=False)["input_ids"]
    return text, out_ids, logprobs


def _self_perplexity(logprobs: List[float]) -> float:
    vals = [lp for lp in logprobs if lp is not None]
    if not vals:
        return float("nan")
    return float(np.exp(-np.mean(vals)))


def run_correctness_check(chunk_store, cache_manager, n_trials: int = 20,
                          n_context_chunks: int = 4, max_tokens: int = 50,
                          seed: int = 42, output_dir: Path = RESULTS_DIR) -> Path:
    rng = np.random.default_rng(seed)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    csv_path = output_dir / "correctness_report.csv"

    rows = []
    for trial in range(n_trials):
        pri = int(rng.integers(0, chunk_store.n_chunks))
        candidate_pool = [i for i in range(chunk_store.n_chunks) if i != pri]
        n_ctx = min(n_context_chunks, len(candidate_pool))
        context_ids = rng.choice(candidate_pool, size=n_ctx, replace=False).tolist()
        sec_pool = [i for i in candidate_pool if i not in context_ids]
        sec = int(rng.choice(sec_pool)) if sec_pool else int(rng.choice(candidate_pool))

        all_ids = [pri] + context_ids
        query_text = chunk_store.text(sec)

        try:
            # ── Fresh (no prior warming) ──
            cache_manager.flush_cache()
            token_ids = cache_manager.build_prompt_token_ids(chunk_store.texts(all_ids) + [query_text])
            fresh_text, fresh_ids, fresh_lp = _generate_with_logprobs(cache_manager, token_ids, max_tokens)

            # ── Blended (warm each context chunk individually, then re-ask) ──
            # Deliberately warm_chunks(), not warm_prefix(): warm_prefix
            # would send [pri]+context_ids as ONE combined request at the
            # exact same token positions used below, which is just ordinary
            # prefix-cache reuse and wouldn't require any CacheBlend
            # boundary correction at all (near-perfect match, always) --
            # that would tell us nothing about CacheBlend's approximation
            # error, which is the whole point of this script. warm_chunks
            # caches each chunk independently (each at its own position 0),
            # so the combined request below can only be served by actually
            # exercising CacheBlend's non-prefix reuse + recompute-ratio
            # stitching, which is what "blended" is supposed to mean here.
            cache_manager.flush_cache()
            cache_manager.warm_chunks(all_ids, chunk_store, block=True)
            blended_text, blended_ids, blended_lp = _generate_with_logprobs(cache_manager, token_ids, max_tokens)
        except Exception as e:
            logger.error(f"  Trial {trial} failed: {e}")
            continue

        min_len = min(len(fresh_ids), len(blended_ids))
        matches = sum(1 for i in range(min_len) if fresh_ids[i] == blended_ids[i])
        exact_match_rate = matches / min_len if min_len else float("nan")

        emb = embedder.encode([fresh_text, blended_text], convert_to_numpy=True)
        denom = (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])) or 1e-9
        cos_sim = float(np.dot(emb[0], emb[1]) / denom)

        fresh_ppl = _self_perplexity(fresh_lp)
        blended_ppl = _self_perplexity(blended_lp)

        row = {
            "trial": trial, "pri": pri, "sec": sec, "n_context_chunks": n_ctx,
            "token_exact_match_rate": round(exact_match_rate, 4),
            "output_cosine_sim": round(cos_sim, 4),
            "fresh_perplexity": round(fresh_ppl, 4) if fresh_ppl == fresh_ppl else "",
            "blended_perplexity": round(blended_ppl, 4) if blended_ppl == blended_ppl else "",
            "perplexity_delta": round(blended_ppl - fresh_ppl, 4) if (fresh_ppl == fresh_ppl and blended_ppl == blended_ppl) else "",
            "fresh_text_preview": fresh_text[:80].replace("\n", " "),
            "blended_text_preview": blended_text[:80].replace("\n", " "),
        }
        rows.append(row)
        logger.info(f"  Trial {trial}: exact_match={row['token_exact_match_rate']:.2%} "
                    f"cos_sim={row['output_cosine_sim']:.4f} "
                    f"ppl_delta={row['perplexity_delta']}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        mean_match = np.mean([r["token_exact_match_rate"] for r in rows])
        mean_cos = np.mean([r["output_cosine_sim"] for r in rows])
        logger.info(f"\nCorrectness summary over {len(rows)} trials: "
                    f"mean token exact-match={mean_match:.2%}, mean output cosine sim={mean_cos:.4f}")
        logger.info(f"Verification-plan threshold: exact-match mismatch should be < 5% "
                    f"({'PASS' if (1 - mean_match) < 0.05 else 'FAIL'})")
    logger.info(f"  CSV: {csv_path}")
    return csv_path
