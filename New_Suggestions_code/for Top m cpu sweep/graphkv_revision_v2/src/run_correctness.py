"""
run_correctness.py
===================
CLI entrypoint for correctness_check.py (implementation_plan.md Phase 3).
"""

import sys
import logging
import argparse

from transformers import AutoTokenizer

from exp_config import MODEL_NAME, INTEGRATION_PATH, RESULTS_DIR
from vllm_launcher import VLLMLauncher
from chunk_store import ChunkStore
from cache_manager import CacheManager
from correctness_check import run_correctness_check


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                 logging.FileHandler(RESULTS_DIR / "run_correctness.log")],
    )
    return logging.getLogger("run_correctness")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hotpot", choices=["hotpot", "2wiki", "musique", "multifield"])
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--integration_path", default=INTEGRATION_PATH, choices=["mp", "inprocess"])
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--n_context_chunks", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=50)
    args = p.parse_args()

    logger = setup_logging()
    logger.info("Correctness check (Phase 3): fresh vs. CacheBlend-blended generation")

    with VLLMLauncher(model_name=args.model, integration_path=args.integration_path) as launcher:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        cache_manager = CacheManager(tokenizer=tokenizer, model_name=args.model)
        try:
            chunk_store = ChunkStore(args.dataset).prepare()
            run_correctness_check(chunk_store, cache_manager, n_trials=args.n_trials,
                                  n_context_chunks=args.n_context_chunks,
                                  max_tokens=args.max_tokens)
        finally:
            cache_manager.close()


if __name__ == "__main__":
    main()
