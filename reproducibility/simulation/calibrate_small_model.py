#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def timed(callable_, synchronize, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        value = callable_()
        synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
        del value
    return float(np.median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate simulator proxies on one GPU/model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lengths", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--transfer-mib", nargs="+", type=int, default=[1, 4, 16, 32])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", default="calibration.json")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU calibration")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    ).eval()
    device = next(model.parameters()).device
    synchronize = torch.cuda.synchronize

    token = tokenizer.encode(" calibration", add_special_tokens=False)[0]
    prefill_rows = []
    with torch.inference_mode():
        warmup = torch.full((1, max(args.lengths)), token, dtype=torch.long, device=device)
        model(input_ids=warmup, use_cache=True)
        synchronize()
        for length in args.lengths:
            ids = torch.full((1, length), token, dtype=torch.long, device=device)
            milliseconds = timed(
                lambda: model(input_ids=ids, use_cache=True).past_key_values,
                synchronize,
                args.repeats,
            )
            prefill_rows.append({"tokens": length, "median_ms": milliseconds})

    token_x = np.asarray([row["tokens"] for row in prefill_rows], dtype=float)
    token_y = np.asarray([row["median_ms"] for row in prefill_rows], dtype=float)
    token_slope, token_intercept = np.polyfit(token_x, token_y, 1)

    transfer_rows = []
    for mib in args.transfer_mib:
        elements = mib * 1024 * 1024 // 2
        source = torch.empty(elements, dtype=torch.float16, device=device)
        milliseconds = timed(lambda: source.clone(), synchronize, args.repeats)
        transfer_rows.append({"mib": mib, "median_ms": milliseconds})
    mib_x = np.asarray([row["mib"] for row in transfer_rows], dtype=float)
    mib_y = np.asarray([row["median_ms"] for row in transfer_rows], dtype=float)
    mib_slope, mib_intercept = np.polyfit(mib_x, mib_y, 1)

    payload = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(),
        "warning": "Hardware-specific median microbenchmark; not an end-to-end latency claim.",
        "prefill_measurements": prefill_rows,
        "device_copy_measurements": transfer_rows,
        "cost_model": {
            "cache_lookup_ms": 0.01,
            "miss_intercept_ms": max(0.0, float(token_intercept)),
            "miss_ms_per_token": max(0.0, float(token_slope)),
            "prefetch_intercept_ms": max(0.0, float(mib_intercept)),
            "prefetch_ms_per_mb": max(0.0, float(mib_slope)),
            "overlap_window_ms": 0.0,
            "prefetch_mode": "blocking",
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

