#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-gemini-2.5-flash-lite}"
OUTPUT="${2:-outputs/gemini_hotpot_smoke}"

python3 run_quality.py \
  --model "$MODEL" \
  --n-train 30 \
  --n-dev 10 \
  --n-test 3 \
  --k-values 10 \
  --context-token-budget 768 \
  --output-dir "$OUTPUT"
