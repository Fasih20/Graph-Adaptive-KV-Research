#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-gemini-2.5-flash-lite}"
OUTPUT="${2:-outputs/gemini_hotpot60_K10}"

python3 run_quality.py \
  --model "$MODEL" \
  --n-train 120 \
  --n-dev 40 \
  --n-test 60 \
  --k-values 10 \
  --context-token-budget 768 \
  --output-dir "$OUTPUT"
