"""
determinism_check.py
======================
Runs the SAME exact config (same seed -> same questions, same top_m, same K)
through qa_quality_eval.run() N times in a row, then diffs the per-question
F1 scores across repeats.

Why this exists: cosine's selected chunks (get_prefetch_cosine) depend only
on sim_matrix and pri — NOT on top_m or anything build_graph touches. So for
a fixed seed, the cosine prompt sent to the LLM is byte-identical across
repeats. If cosine_f1 for the same question still changes between repeats,
that's not retrieval-policy signal — it's the LLM API not being perfectly
deterministic at temperature=0.0 (common with hosted APIs; batching/routing
on the serving side breaks bit-exact reproducibility even at temp 0).

Run this BEFORE spending --full budget on a top_m sweep: if cosine (or any
condition) swings by more than the differences you're trying to interpret
between conditions, you don't have a reliable signal yet at that n, full
stop — no amount of additional top_m values fixes that.

Each repeat gets its own tag (via args.rep) so nothing overwrites: results/
qa_quality_<tag>_rep0.csv, _rep1.csv, etc. This script does the join/diff
after they're all done.

Usage:
  python determinism_check.py --provider gemini --top_m 20 --n_reps 2
"""

import sys
import logging
import argparse

import pandas as pd

import qa_quality_eval as qqe
from exp_config import DATASET_CONFIGS, RANDOM_SEED, GRAPH_CONSTRUCTION, TOP_M

logger = logging.getLogger("determinism_check")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True, choices=["gemini", "openai", "anthropic"])
    p.add_argument("--backend", choices=["openai", "openrouter", "groq"])
    p.add_argument("--model")
    p.add_argument("--api_key")
    p.add_argument("--cheap", action="store_true", default=True)
    p.add_argument("--full", dest="cheap", action="store_false")
    p.add_argument("--rpm", type=int, default=5)
    p.add_argument("--K", type=int, default=12, help="Must be > 8")
    p.add_argument("--top_m", type=int, default=TOP_M)
    p.add_argument("--q_type", default="multi-hop", choices=["semantic", "structural", "multi-hop"])
    p.add_argument("--n_questions", type=int, default=None)
    p.add_argument("--dataset", default="hotpot", choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--graph_construction", default=GRAPH_CONSTRUCTION, choices=["threshold", "topm"])
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                  help="Held FIXED across repeats on purpose — same questions every time, "
                       "we're isolating LLM sampling noise, not question-selection variance")
    p.add_argument("--n_reps", type=int, default=2, help="How many independent passes over the same config")
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()

    if args.provider == "openai" and not args.backend:
        p.error("--backend is required when --provider openai (openrouter | groq | openai)")

    qqe.setup_logging()
    logger.info(f"determinism check: {args.n_reps} reps at fixed top_m={args.top_m}, seed={args.seed} "
                f"(provider={args.provider} backend={args.backend})")

    csv_paths = []
    for rep in range(args.n_reps):
        sub_args = argparse.Namespace(**vars(args))
        sub_args.rep = rep
        sub_args.n_questions = args.n_questions

        logger.info(f"\n----- rep {rep} -----")
        status = qqe.run(sub_args)
        if status != "complete":
            logger.error(f"rep {rep} did not complete cleanly (status={status}) — "
                        f"re-run this same command to resume, then re-run determinism_check "
                        f"once all reps have a completed checkpoint-free CSV.")
            sys.exit(1)
        csv_paths.append(qqe.RESULTS_DIR / f"qa_quality_{qqe.build_tag(sub_args)}.csv")

    # Join all reps on q_idx, compare per-condition F1 across reps.
    dfs = [pd.read_csv(p_).set_index("q_idx") for p_ in csv_paths]
    conditions = qqe.CONDITIONS

    logger.info("\n" + "=" * 70)
    logger.info(f"DETERMINISM CHECK over {len(dfs[0])} questions x {args.n_reps} reps "
                f"(top_m={args.top_m}, seed={args.seed})")
    any_flips = False
    for cond in conditions:
        col = f"{cond}_f1"
        merged = pd.concat([df[col] for df in dfs], axis=1)
        merged.columns = [f"rep{i}" for i in range(args.n_reps)]
        std_per_q = merged.std(axis=1)
        n_changed = int((std_per_q > 1e-9).sum())
        if n_changed:
            any_flips = True
        logger.info(f"  {cond:9s}: {n_changed}/{len(merged)} questions had a DIFFERENT F1 across "
                    f"reps  |  mean per-question std={std_per_q.mean():.4f}  max={std_per_q.max():.4f}")
        if n_changed:
            changed_qs = std_per_q[std_per_q > 1e-9].index.tolist()
            logger.info(f"    changed q_idx: {changed_qs}")

    logger.info("=" * 70)
    if any_flips:
        logger.warning("Some questions produced DIFFERENT F1 scores across repeats with an IDENTICAL "
                      "config — this is LLM API non-determinism (temperature=0 is not a hard "
                      "guarantee on most hosted APIs), not retrieval-policy signal. Treat any "
                      "top_m/condition difference smaller than what you saw here as noise, not "
                      "a real effect, especially at small n.")
    else:
        logger.info(f"No F1 differences across {args.n_reps} reps for any condition — this specific "
                    f"config looks reproducible at this n. (Doesn't rule out noise at other top_m "
                    f"values or larger n — it's a spot check, not a guarantee.)")


if __name__ == "__main__":
    main()
