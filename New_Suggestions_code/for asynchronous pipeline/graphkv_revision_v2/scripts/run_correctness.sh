#!/usr/bin/env bash
# Phase 3 correctness check: N=20 fresh-vs-blended generation comparison.
# Usage: scripts/run_correctness.sh [--dataset hotpot] [--model <id>] [--n_trials 20]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 src/run_correctness.py "$@"
