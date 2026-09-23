#!/usr/bin/env bash
# Eager-mode TTFT/decode calibration (implementation_plan.md §3).
# Launches a PLAIN vLLM (no LMCache) twice — CUDA graphs vs --enforce-eager —
# and measures the delta. Run this BEFORE run_full_benchmark.sh so you have
# a measured overhead number to report alongside the main results, not an
# assumed-zero one.
#
# Usage: scripts/calibrate_eager.sh [--model <hf_model_id>]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 src/calibrate_eager.py "$@"

echo ""
echo "Wrote results/eager_calibration.csv and results/eager_calibration_summary.json."
echo "Check the summary's eager_ttft_delta_ms per bucket before trusting the main"
echo "benchmark's TTFT comparisons — if it's > 5% of a bucket's cold TTFT, flag it"
echo "as a systematic bias per implementation_plan.md."
