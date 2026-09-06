"""
run_experiment.py
==================
Top-level entrypoint: launches vLLM+LMCache, prepares the corpus, and runs
the benchmark harness for one dataset. See scripts/run_full_benchmark.sh and
scripts/run_cosine_baseline.sh for typical invocations.
"""

import sys
import logging
import argparse
from pathlib import Path

from transformers import AutoTokenizer

from exp_config import (MODEL_NAME, INTEGRATION_PATH, RESULTS_DIR, N_PAIRS,
                     HARDCODED_ADAPTIVE_WEIGHTS, get_model_config,
                     GRAPH_CONSTRUCTION, TOP_M)
from vllm_launcher import VLLMLauncher
from chunk_store import ChunkStore
from cache_manager import CacheManager
import benchmark_harness


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                 logging.FileHandler(RESULTS_DIR / "run_experiment.log")],
    )
    return logging.getLogger("run_experiment")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hotpot", choices=["hotpot", "2wiki", "musique", "multifield"])
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--integration_path", default=INTEGRATION_PATH, choices=["mp", "inprocess"])
    p.add_argument("--n_pairs", type=int, default=N_PAIRS)
    p.add_argument("--graph_construction", default=GRAPH_CONSTRUCTION, choices=["threshold", "topm"])
    p.add_argument("--top_m", type=int, default=TOP_M)
    p.add_argument("--cosine_only", action="store_true",
                   help="Phase 1 smoke run: skip graph/adaptive policies entirely "
                        "(not implemented as a separate code path here — the harness "
                        "always computes all four; this flag is a placeholder for "
                        "scripts/run_cosine_baseline.sh's intent. Wire it up if you "
                        "want a genuinely cheaper Phase-1-only run.)")
    args = p.parse_args()

    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("Real KV Cache Prefetching — vLLM + LMCache CacheBlend")
    logger.info(f"Model: {args.model} | Dataset: {args.dataset} | Path: {args.integration_path}")
    logger.info("=" * 70)

    model_config = get_model_config(args.model)

    with VLLMLauncher(model_name=args.model, integration_path=args.integration_path) as launcher:
        if not launcher.health_check():
            logger.warning("Initial health_check() failed — proceeding anyway; "
                           "first request will surface the real error if the server isn't ready.")

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        cache_manager = CacheManager(tokenizer=tokenizer, model_name=args.model)

        try:
            logger.info("Preparing corpus (load -> embed -> build_graph) ...")
            chunk_store = ChunkStore(args.dataset, graph_construction=args.graph_construction,
                                     top_m=args.top_m).prepare()

            logger.info("Running benchmark harness ...")
            csv_path = benchmark_harness.run_dataset(
                args.dataset, args.model, chunk_store, cache_manager, model_config,
                adaptive_weights=HARDCODED_ADAPTIVE_WEIGHTS, n_pairs=args.n_pairs,
            )
            logger.info(f"Done. Results: {csv_path}")
        finally:
            cache_manager.close()


if __name__ == "__main__":
    main()
