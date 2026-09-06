"""
calibrate_eager.py
===================
Implements implementation_plan.md §3's "Eager-Mode Calibration Protocol":
quantifies --enforce-eager's TTFT/decode overhead so the main benchmark's
numbers can be reported alongside a known, measured bias rather than an
assumed-zero one.

This launches a PLAIN vLLM server — no LMCache, no CacheBlend — twice:
  Run A: CUDA graphs enabled (vLLM default, enforce_eager=False)
  Run B: eager mode (enforce_eager=True)
Same 10 prompts x 3 context-length buckets (~512 / ~2048 / ~4096 tokens) in
both runs, measuring both TTFT (first-token latency) and per-token decode
latency.

Output: results/eager_calibration.csv (prompt_id, context_tokens, mode,
ttft_ms, decode_per_token_ms) plus results/eager_calibration_summary.json
with per-bucket mean deltas and 95% CIs. This is a companion artifact, kept
deliberately OUT of the main results CSV schema (see benchmark_harness.py's
module docstring) so the main CSV's columns stay exactly what your existing
analysis scripts expect.
"""

import json
import time
import logging
import argparse
import subprocess
from pathlib import Path

import numpy as np
import requests
from scipy import stats
from transformers import AutoTokenizer

from exp_config import MODEL_NAME, VLLM_HOST, VLLM_PORT, MAX_MODEL_LEN, RESULTS_DIR
from vllm_launcher import VLLMLauncher, ProcessHandle

logger = logging.getLogger("calibrate_eager")

CONTEXT_BUCKETS = {"short": 512, "medium": 2048, "long": 4096}
N_PROMPTS_PER_BUCKET = 10
DECODE_TOKENS = 32
FILLER = ("The quick brown fox jumps over the lazy dog while researchers "
          "study long-context attention mechanisms and cache reuse. ")


def _build_prompt_token_ids(tokenizer, target_len: int, prompt_id: int):
    rng_text = (FILLER * ((target_len // 12) + 4))
    ids = tokenizer(rng_text, add_special_tokens=True)["input_ids"][:target_len]
    return ids


def _measure_one(base_url: str, model_name: str, token_ids: list, max_tokens: int = DECODE_TOKENS):
    t0 = time.perf_counter()
    ttft_ms = None
    n_tokens_seen = 0
    with requests.post(
        f"{base_url}/completions",
        json={"model": model_name, "prompt": token_ids, "max_tokens": max_tokens,
              "temperature": 0, "stream": True},
        stream=True, timeout=180,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000
            n_tokens_seen += 1
    total_ms = (time.perf_counter() - t0) * 1000
    decode_per_token_ms = ((total_ms - ttft_ms) / max(n_tokens_seen - 1, 1)) if ttft_ms is not None else None
    return ttft_ms, decode_per_token_ms


def run_mode(mode: str, tokenizer, model_name: str = MODEL_NAME) -> list:
    enforce_eager = (mode == "eager")
    logger.info(f"Launching plain vLLM (no LMCache) — mode={mode} enforce_eager={enforce_eager}")

    args = ["vllm", "serve", model_name, "--host", VLLM_HOST, "--port", str(VLLM_PORT),
            "--max-model-len", str(MAX_MODEL_LEN)]
    if enforce_eager:
        args.append("--enforce-eager")
    log_path = RESULTS_DIR / f"vllm-calibrate-{mode}.log"
    log_f = open(log_path, "w")
    logger.info(f"  Logs: tail -f {log_path}  (first launch: several minutes — "
                f"model load + {'no ' if enforce_eager else ''}CUDA graph capture)")
    proc = subprocess.Popen(args, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    handle = ProcessHandle(f"vllm-calibrate-{mode}", proc)
    try:
        VLLMLauncher._wait_for_port(VLLM_HOST, VLLM_PORT, f"vLLM ({mode})", timeout_s=900,
                                    proc=proc, log_path=log_path)
        base_url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
        rows = []
        for bucket, target_len in CONTEXT_BUCKETS.items():
            for pid in range(N_PROMPTS_PER_BUCKET):
                token_ids = _build_prompt_token_ids(tokenizer, target_len, pid)
                ttft_ms, decode_ms = _measure_one(base_url, model_name, token_ids)
                rows.append({"prompt_id": f"{bucket}_{pid}", "context_tokens": len(token_ids),
                            "mode": mode, "ttft_ms": round(ttft_ms, 4) if ttft_ms else None,
                            "decode_per_token_ms": round(decode_ms, 4) if decode_ms else None})
                logger.info(f"  [{mode}] {bucket} p{pid}: ttft={ttft_ms:.1f}ms decode/tok={decode_ms:.2f}ms")
        return rows
    finally:
        handle.terminate()
        log_f.close()


def summarize(rows: list) -> dict:
    import pandas as pd
    df = pd.DataFrame(rows)
    df["bucket"] = df["prompt_id"].str.rsplit("_", n=1).str[0]
    summary = {}
    for bucket in CONTEXT_BUCKETS:
        sub = df[df["bucket"] == bucket]
        graphs = sub[sub["mode"] == "graphs"]
        eager = sub[sub["mode"] == "eager"]
        if len(graphs) < 2 or len(eager) < 2:
            continue

        def delta_ci(col):
            g, e = graphs[col].dropna().values, eager[col].dropna().values
            n = min(len(g), len(e))
            diffs = e[:n] - g[:n]
            mean = float(np.mean(diffs))
            sem = stats.sem(diffs) if n > 1 else 0.0
            ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if n > 1 and sem > 0 else (mean, mean)
            return mean, [float(ci[0]), float(ci[1])]

        ttft_mean, ttft_ci = delta_ci("ttft_ms")
        decode_mean, decode_ci = delta_ci("decode_per_token_ms")
        graphs_ttft_mean = float(graphs["ttft_ms"].dropna().mean())
        summary[bucket] = {
            "eager_ttft_delta_ms": round(ttft_mean, 4), "eager_ttft_delta_95ci": ttft_ci,
            "eager_decode_delta_ms": round(decode_mean, 4), "eager_decode_delta_95ci": decode_ci,
            "graphs_ttft_mean_ms": round(graphs_ttft_mean, 4),
        }
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_NAME)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    all_rows = []
    all_rows += run_mode("graphs", tokenizer, args.model)
    all_rows += run_mode("eager", tokenizer, args.model)

    import csv
    csv_path = RESULTS_DIR / "eager_calibration.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "context_tokens", "mode", "ttft_ms", "decode_per_token_ms"])
        w.writeheader()
        w.writerows(all_rows)
    logger.info(f"Wrote {csv_path}")

    summary = summarize(all_rows)
    summary_path = RESULTS_DIR / "eager_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote {summary_path}")
    for bucket, s in summary.items():
        baseline = s.get("graphs_ttft_mean_ms") or 0.0
        pct = (abs(s["eager_ttft_delta_ms"]) / baseline * 100) if baseline else float("inf")
        flag = f"FLAG (>5%: {pct:.1f}% of baseline)" if baseline and pct > 5 else ""
        logger.info(f"  {bucket}: eager_ttft_delta={s['eager_ttft_delta_ms']:.2f}ms "
                    f"({pct:.1f}% of {baseline:.1f}ms baseline)  CI={s['eager_ttft_delta_95ci']}  "
                    f"eager_decode_delta={s['eager_decode_delta_ms']:.3f}ms/tok  {flag}")


if __name__ == "__main__":
    main()
