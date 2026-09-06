#!/usr/bin/env bash
# Phase 1 smoke run: small n_pairs, single dataset, to validate the vLLM +
# LMCache MP-mode pipeline end-to-end before committing to a full 600-pair
# x 10-K-value run. Still computes all four policies per trial (cosine,
# graph, adaptive alongside cold) — see run_experiment.py's --cosine_only
# docstring if you want a genuinely cosine-only code path instead.
#
# Usage: scripts/run_cosine_baseline.sh [--dataset hotpot] [--model <id>]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 src/run_experiment.py --n_pairs 30 "$@"
