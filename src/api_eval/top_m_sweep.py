"""
top_m_sweep.py
================
Runs qa_quality_eval.py across a sweep of --top_m values, so you can see
whether the graph/adaptive conditions actually pull ahead of cosine once
top_m is small enough to matter for HotpotQA's small per-question chunk
pools (see qa_quality_eval.py's own module docstring on this).

Resumability, built on top of qa_quality_eval's own per-question checkpoint
(results/checkpoint_<tag>.json, tag includes top_m so values never collide):

  - Each top_m value is its own qa_quality_eval run() call under the hood,
    with its own checkpoint file and its own results CSV.
  - If a run hits a persistent quota limit (QuotaExhaustedError bubbling up
    as run() returning "partial"), THIS SCRIPT STOPS THE WHOLE SWEEP right
    there — it does NOT skip ahead to the next top_m value, so you don't
    end up with an incomplete middle value and a complete later one.
  - Re-run the EXACT SAME command later (once the provider's quota window
    has reset) and it picks up mid-top_m-value from that value's own
    checkpoint, finishes it, then continues on to any top_m values not
    started yet.
  - A top_m value that finished completely has its checkpoint file deleted
    by qa_quality_eval.run() on success, so future re-runs of the sweep
    skip it immediately (no completed API calls repeated).

Sizing vs. free tier:
  --cheap (default) means n_questions=8, so each top_m value costs
  8 questions x 3 calls = 24 calls. Four default top_m values (5/8/12/20)
  = 96 calls total for the whole sweep — comfortably inside any provider's
  free daily cap in a single sitting.

  --full means n_questions=60 -> 180 calls per top_m value -> 720 calls
  across four values. That may or may not fit a given day's quota (Gemini's
  exact free daily cap isn't confirmed here, so don't assume it fits) —
  which is exactly why the checkpoint/resume above exists: if you hit the
  wall partway through, just re-run tomorrow and it continues, no manual
  bookkeeping needed.

Usage:
  python top_m_sweep.py --provider gemini --cheap
  python top_m_sweep.py --provider openai --backend groq --full --top_m_values 5,8,12,20
"""

import sys
import logging
import argparse

import qa_quality_eval as qqe
from exp_config import DATASET_CONFIGS, RANDOM_SEED, GRAPH_CONSTRUCTION

logger = logging.getLogger("top_m_sweep")

DEFAULT_TOP_M_VALUES = [5, 8, 12, 20]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True, choices=["gemini", "openai", "anthropic"])
    p.add_argument("--backend", choices=["openai", "openrouter", "groq"],
                  help="Required when --provider openai.")
    p.add_argument("--model", help="Override the provider's default model")
    p.add_argument("--api_key", help="Override the provider's default env var")
    p.add_argument("--cheap", action="store_true", default=True,
                  help="Small n_questions, short answers, free-tier-friendly (default ON)")
    p.add_argument("--full", dest="cheap", action="store_false",
                  help="Larger n_questions, longer answers — costs real quota, see module docstring")
    p.add_argument("--rpm", type=int, default=5,
                  help="Max LLM calls per rolling 60s window, applied within each top_m run")
    p.add_argument("--K", type=int, default=12, help="Must be > 8. Held fixed across the sweep.")
    p.add_argument("--top_m_values", default=",".join(str(v) for v in DEFAULT_TOP_M_VALUES),
                  help="Comma-separated top_m values to sweep, e.g. '5,8,12,20'")
    p.add_argument("--q_type", default="multi-hop", choices=["semantic", "structural", "multi-hop"])
    p.add_argument("--n_questions", type=int, default=None)
    p.add_argument("--dataset", default="hotpot", choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--graph_construction", default=GRAPH_CONSTRUCTION, choices=["threshold", "topm"])
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--fresh", action="store_true",
                  help="Ignore/delete existing checkpoints for EVERY top_m value and start the whole sweep over")
    args = p.parse_args()

    if args.provider == "openai" and not args.backend:
        p.error("--backend is required when --provider openai (openrouter | groq | openai)")

    try:
        top_m_values = [int(v.strip()) for v in args.top_m_values.split(",") if v.strip()]
    except ValueError:
        p.error(f"--top_m_values must be a comma-separated list of ints, got {args.top_m_values!r}")

    qqe.setup_logging()
    logger.info(f"top_m sweep: {top_m_values}  "
                f"(provider={args.provider} backend={args.backend} cheap={args.cheap})")

    for tm in top_m_values:
        sub_args = argparse.Namespace(**vars(args))
        sub_args.top_m = tm
        # qqe.run() fills this in from --cheap/--full if None; keep that logic
        # in one place rather than duplicating it here.
        sub_args.n_questions = args.n_questions

        logger.info(f"\n----- top_m={tm} -----")
        status = qqe.run(sub_args)

        if status == "partial":
            remaining = [v for v in top_m_values if v > tm] if tm in top_m_values else []
            later = top_m_values[top_m_values.index(tm) + 1:]
            logger.warning(f"Sweep stopped mid-run at top_m={tm} (hit a persistent quota limit). "
                          f"Re-run this EXACT SAME command later to resume top_m={tm} from its "
                          f"checkpoint, then continue on to: {later}")
            sys.exit(1)
        elif status == "no_data":
            logger.error(f"top_m={tm} produced no usable rows at all — stopping sweep to investigate "
                        f"rather than continuing on with more likely-broken values.")
            sys.exit(1)
        logger.info(f"top_m={tm}: complete.")

    logger.info("\n" + "=" * 70)
    logger.info("Sweep complete for all top_m values: " + ", ".join(str(v) for v in top_m_values))
    logger.info("Per-value results: results/qa_quality_<provider..>_top<N>_....csv")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
