#!/usr/bin/env bash
# Full benchmark: all ABLATION_CONFIGS (K=3..16), N_PAIRS=600 pairs each,
# for one dataset. Resumable — re-running skips (config_k, config_cap)
# combinations already present in the output CSV.
#
# Usage: scripts/run_full_benchmark.sh [--dataset hotpot] [--model <id>] \
#          [--integration_path mp|inprocess]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 src/run_experiment.py "$@"
