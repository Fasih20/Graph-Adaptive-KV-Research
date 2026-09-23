#!/usr/bin/env bash
# Hit-conditional latency breakdown for an existing results_real_*.csv —
# run this after run_full_benchmark.sh / run_cosine_baseline.sh to see
# whether hits are actually faster than misses (the real test of whether
# caching is helping), not just the raw cross-policy averages.
#
# Usage: scripts/analyze_results.sh [--csv results/results_real_hotpot.csv] [--k 12]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 src/analyze_results.py "$@"
