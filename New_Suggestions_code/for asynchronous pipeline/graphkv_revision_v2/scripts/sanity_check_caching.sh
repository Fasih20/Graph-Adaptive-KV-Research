#!/usr/bin/env bash
# Run this BEFORE trusting run_cosine_baseline.sh's numbers. Isolates
# "is any KV reuse happening at all" from "which retrieval policy is better".
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 src/sanity_check_caching.py "$@"
