#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 MODEL_ID OUTPUT_DIR [DATASET] [GPU]" >&2
  exit 2
fi

model_id="$1"
output_dir="$2"
dataset="${3:-hotpot}"
gpu="${4:-0}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python run_repaired.py \
  --dataset "$dataset" \
  --model "$model_id" \
  --gpu "$gpu" \
  --events 200 \
  --train-events 300 \
  --dev-events 150 \
  --max-documents 60 \
  --max-chars-per-document 12000 \
  --document-chunk-tokens 384 \
  --document-chunk-overlap-tokens 64 \
  --l1-size-gb 0.5 \
  --k-values 3,6,10,16 \
  --repetitions 2 \
  --output-dir "$output_dir"
