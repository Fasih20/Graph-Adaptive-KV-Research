#!/usr/bin/env python3
"""CLI for the repaired GraphKV Gemini answer-quality experiment."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from graphkv_quality.pipeline import run
from graphkv_quality.provider import list_gemini_models


def comma_ints(value: str) -> list[int]:
    try:
        result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("K values must be positive")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Leakage-safe GraphKV answer-quality evaluation through the Gemini REST API."
    )
    result.add_argument("--model", default="gemini-2.5-flash-lite", help="Pin a stable Gemini model ID")
    result.add_argument(
        "--dataset",
        default="hotpot",
        choices=["hotpot", "2wiki"],
        help="Structured QA dataset with sentence-level supporting-fact labels",
    )
    result.add_argument("--api-key", help="Prefer the GEMINI_API_KEY environment variable")
    result.add_argument("--list-models", action="store_true", help="List accessible generateContent models, then exit")
    result.add_argument("--preflight-only", action="store_true", help="Validate the model/key without loading data")
    result.add_argument("--output-dir", default="outputs/gemini_hotpot_quality")

    result.add_argument("--n-train", type=int, default=120, help="Supporting-label fitting rows; no API calls")
    result.add_argument("--n-dev", type=int, default=40, help="Online-rate tuning rows; no API calls")
    result.add_argument("--n-test", type=int, default=60, help="Held-out validation questions sent to Gemini")
    result.add_argument("--k-values", type=comma_ints, default=[10], help="Comma-separated K values (default: 10)")
    result.add_argument("--top-m", type=int, default=5, help="Semantic neighbours per node")
    result.add_argument("--max-degree", type=int, default=100)
    result.add_argument("--weight-step", type=float, default=0.05)
    result.add_argument("--context-token-budget", type=int, default=768)
    result.add_argument("--max-output-tokens", type=int, default=64)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--shuffle-buffer", type=int, default=10_000)

    result.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    result.add_argument("--embedding-device", default="cpu", choices=["cpu", "cuda"])
    result.add_argument("--budget-tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")

    result.add_argument(
        "--rpm",
        type=int,
        default=0,
        help="Optional proactive requests/minute cap. 0 disables fixed waits; 429 Retry-After is still honoured.",
    )
    result.add_argument("--max-retries", type=int, default=4)
    result.add_argument("--connect-timeout", type=float, default=10.0)
    result.add_argument("--read-timeout", type=float, default=90.0)
    result.add_argument("--max-retry-wait", type=float, default=90.0)
    return result


def validate(args, command: argparse.ArgumentParser) -> None:
    positive = {
        "--n-train": args.n_train,
        "--n-dev": args.n_dev,
        "--n-test": args.n_test,
        "--top-m": args.top_m,
        "--max-degree": args.max_degree,
        "--context-token-budget": args.context_token_budget,
        "--max-output-tokens": args.max_output_tokens,
        "--max-retries": args.max_retries,
    }
    for name, value in positive.items():
        if value <= 0:
            command.error(f"{name} must be positive")
    if args.rpm < 0:
        command.error("--rpm cannot be negative")
    if not (0 < args.weight_step <= 1):
        command.error("--weight-step must be in (0, 1]")


def main() -> int:
    command = parser()
    args = command.parse_args()
    validate(args, command)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    if args.list_models:
        print(json.dumps(list_gemini_models(args.api_key), indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
