#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_gemini_dataset60.sh DATASET MODEL OUTPUT_DIR}"
MODEL="${2:-gemini-2.5-flash-lite}"
OUTPUT="${3:-outputs/gemini_${DATASET}60_K10}"

python3 run_quality.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --n-train 120 \
  --n-dev 40 \
  --n-test 60 \
  --k-values 10 \
  --context-token-budget 768 \
  --output-dir "$OUTPUT"
