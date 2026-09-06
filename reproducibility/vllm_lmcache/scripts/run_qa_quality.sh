#!/usr/bin/env bash
# Graph-selected vs. cosine-selected context — answer quality eval.
# No GPU / vLLM / LMCache needed — pure API calls, safe to run while your
# local GPU stack is still being sorted out.
#
# Examples:
#   scripts/run_qa_quality.sh --provider gemini
#   scripts/run_qa_quality.sh --provider openai --backend groq
#   scripts/run_qa_quality.sh --provider openai --backend openrouter
#   scripts/run_qa_quality.sh --provider anthropic          # uses your $5 trial credit
#   scripts/run_qa_quality.sh --provider gemini --full --K 16 --n_questions 60
#
# API keys read from env vars: GEMINI_API_KEY, OPENROUTER_API_KEY,
# GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY (or pass --api_key).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 src/qa_quality_eval.py "$@"
