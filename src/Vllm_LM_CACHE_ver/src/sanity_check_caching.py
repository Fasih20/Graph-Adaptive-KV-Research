"""
sanity_check_caching.py
=========================
The most basic possible test of whether ANY KV reuse is happening at all,
decoupled from the K-sweep/policy machinery: send the exact same combined
prompt twice in a row and compare TTFT. If the second call isn't
meaningfully faster than the first, nothing downstream (cosine/graph/
adaptive vs. cold) can show a caching benefit either, no matter how correct
the retrieval-policy code is — this isolates "is the plumbing working" from
"which retrieval policy is better", which is what benchmark_harness.py
actually measures.

This also runs a SECOND test using warm_prefix() + a different trailing
query (rather than resending the literal same prompt) — that's closer to
the real benchmark's actual usage pattern (same context prefix, different
final question each time) and is the one that actually needs to show a
speedup for the K-sweep results to be meaningful.

Run this BEFORE trusting run_cosine_baseline.sh's numbers.
"""

import sys
import logging
import argparse

from transformers import AutoTokenizer

from exp_config import MODEL_NAME, INTEGRATION_PATH
from vllm_launcher import VLLMLauncher
from chunk_store import ChunkStore
from cache_manager import CacheManager

logger = logging.getLogger("sanity_check_caching")


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    return logging.getLogger("sanity_check_caching")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hotpot")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--integration_path", default=INTEGRATION_PATH, choices=["mp", "inprocess"])
    p.add_argument("--n_context_chunks", type=int, default=6)
    args = p.parse_args()

    logger = setup_logging()

    with VLLMLauncher(model_name=args.model, integration_path=args.integration_path) as launcher:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        cache_manager = CacheManager(tokenizer=tokenizer, model_name=args.model)
        try:
            chunk_store = ChunkStore(args.dataset).prepare()
            context_ids = list(range(args.n_context_chunks))
            query_a = chunk_store.text(args.n_context_chunks)
            query_b = chunk_store.text(args.n_context_chunks + 1)

            logger.info("=" * 70)
            logger.info("TEST 1: literal identical prompt sent twice")
            logger.info("=" * 70)
            cache_manager.flush_cache()
            r1 = cache_manager.query_with_chunks(query_a, context_ids, chunk_store)
            r2 = cache_manager.query_with_chunks(query_a, context_ids, chunk_store)
            logger.info(f"  1st call (cold):    {r1['ttft_ms']:.1f} ms  ({r1['n_prompt_tokens']} tokens)")
            logger.info(f"  2nd call (repeat):  {r2['ttft_ms']:.1f} ms  ({r2['n_prompt_tokens']} tokens)")
            speedup1 = (r1["ttft_ms"] - r2["ttft_ms"]) / r1["ttft_ms"] * 100
            logger.info(f"  speedup: {speedup1:+.1f}%  "
                        f"{'(looks like SOME caching is happening)' if speedup1 > 15 else '(NO meaningful caching detected — check LMCache/vLLM setup before trusting the K-sweep)'}")

            logger.info("=" * 70)
            logger.info("TEST 2: warm_prefix(context) once, then two DIFFERENT queries "
                        "against that same context — this is what benchmark_harness.py actually does")
            logger.info("=" * 70)
            cache_manager.flush_cache()
            cache_manager.warm_prefix(context_ids, chunk_store, block=True)
            r3 = cache_manager.query_with_chunks(query_a, context_ids, chunk_store)
            r4 = cache_manager.query_with_chunks(query_b, context_ids, chunk_store)
            logger.info(f"  query A (after warm_prefix): {r3['ttft_ms']:.1f} ms")
            logger.info(f"  query B (same prefix, different query): {r4['ttft_ms']:.1f} ms")

            logger.info("=" * 70)
            logger.info("TEST 3: cold baseline for comparison — flush, then query WITHOUT "
                        "any warm step at all, same context+query as query A above")
            logger.info("=" * 70)
            cache_manager.flush_cache()
            r5 = cache_manager.query_with_chunks(query_a, context_ids, chunk_store)
            logger.info(f"  query A, no warm (true cold): {r5['ttft_ms']:.1f} ms")
            speedup2 = (r5["ttft_ms"] - r3["ttft_ms"]) / r5["ttft_ms"] * 100
            logger.info(f"  warm_prefix speedup vs true cold: {speedup2:+.1f}%  "
                        f"{'(caching IS helping)' if speedup2 > 15 else '(still no meaningful benefit — see notes below)'}")

            logger.info("=" * 70)
            logger.info("If TEST 1/3 show no speedup: check the LMCache server log/metrics "
                        "directly (results/lmcache-server.log, or its /status endpoint) for "
                        "actual hit/miss counts rather than inferring purely from timing — "
                        "confirm the CacheBlend flags in vllm_launcher.py "
                        "(--kv-transfer-config, --block-size, --attention-backend) actually "
                        "match what your installed LMCache version expects.")
        finally:
            cache_manager.close()


if __name__ == "__main__":
    main()
