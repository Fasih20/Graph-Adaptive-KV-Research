#!/usr/bin/env python3
"""Strict structural validator for a completed repaired benchmark."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "run_id", "repetition", "dataset", "event_id", "document_id", "query_type",
    "policy", "k", "current_chunk_id", "target_chunk_id", "predicted_ids",
    "prediction_hit", "prompt_token_hash", "prompt_tokens", "policy_ms",
    "speculative_population_ms", "ready_cache_ttft_ms", "request_e2e_ms",
    "end_to_end_ttft_ms", "end_to_end_request_ms", "prefetch_bytes_scheduled",
    "prefetch_bytes_completed", "lmcache_log_evidence", "request_success",
    "measurement_semantics",
    "cache_semantics",
    "trace_semantics",
}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    for required in (
        "environment_manifest.json", "run_manifest.json", "fitted_policy.json",
        "workload_stats.json", "run_scale_estimate.json", "summary_by_query_type.csv",
        "pooled_results_by_query_type.csv", "headline_mean_results.csv",
        "postrun_protocol_report.json",
    ):
        if not (root / required).exists():
            fail(errors, f"missing {required}")
    manifest = json.loads((root / "run_manifest.json").read_text()) if (root / "run_manifest.json").exists() else {}
    config = manifest.get("config", {})
    policies = list(config.get("policies", []))
    k_values = [int(value) for value in config.get("k_values", [])]
    repetitions = int(config.get("repetitions", 0))
    expected_events = int(config.get("events", 0))
    if not policies or not k_values or repetitions < 1 or expected_events < 1:
        fail(errors, "run_manifest config is incomplete")

    if not config.get("skip_sanity", False):
        sanity_path = root / "sanity" / "sanity_report.json"
        if not sanity_path.exists():
            fail(errors, "missing sanity/sanity_report.json")
        else:
            sanity = json.loads(sanity_path.read_text())
            if not sanity.get("critical_passed"):
                fail(errors, "sanity critical checks did not pass")
            if sanity.get("checks", {}).get("V_G_capacity_eviction", {}).get("status") != "pass":
                fail(errors, "mandatory operational capacity-eviction control did not pass")

    parity: dict[tuple[int, int], set[tuple[str, int]]] = defaultdict(set)
    completed = 0
    for policy in policies:
        for k in k_values:
            for repetition in range(repetitions):
                group = root / "arms" / f"policy={policy}" / f"k={k}" / f"rep={repetition}"
                marker = group / "COMPLETE.json"
                if not marker.exists():
                    fail(errors, f"missing completion marker for {policy}, K={k}, rep={repetition}")
                    continue
                info = json.loads(marker.read_text())
                events_path = root / info["events_path"]
                if not events_path.exists():
                    fail(errors, f"completion marker points to missing file: {events_path}")
                    continue
                rows = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
                if len(rows) != expected_events:
                    fail(errors, f"{policy}, K={k}, rep={repetition}: expected {expected_events} rows, found {len(rows)}")
                ids = [int(row["event_id"]) for row in rows]
                if len(ids) != len(set(ids)):
                    fail(errors, f"duplicate event ids in {events_path}")
                for row in rows:
                    missing = REQUIRED_FIELDS - set(row)
                    if missing:
                        fail(errors, f"{events_path}: missing fields {sorted(missing)}")
                        break
                    if row["policy"] != policy or int(row["k"]) != k or int(row["repetition"]) != repetition:
                        fail(errors, f"arm labels disagree with path in {events_path}")
                        break
                    if row["measurement_semantics"] != "exact_chunk_access_prefetch_microbenchmark":
                        fail(errors, f"unexpected measurement semantics in {events_path}")
                    if row["cache_semantics"] != "stock_lmcache_exact_prefix_per_chunk_not_cacheblend":
                        fail(errors, f"unexpected cache semantics in {events_path}")
                    if row["trace_semantics"] != "controlled_synthetic_document_split":
                        fail(errors, f"unexpected trace semantics in {events_path}")
                    if float(row["end_to_end_ttft_ms"]) + 1e-9 < float(row["ready_cache_ttft_ms"]):
                        fail(errors, f"end-to-end TTFT below ready-cache TTFT in {events_path}")
                    parity[(repetition, int(row["event_id"]))].add((row["prompt_token_hash"], int(row["prompt_tokens"])))
                completed += 1

    for key, prompts in parity.items():
        if len(prompts) != 1:
            fail(errors, f"prompt parity failed for repetition/event {key}: {sorted(prompts)}")

    expected_arms = len(policies) * len(k_values) * repetitions
    if completed != expected_arms:
        fail(errors, f"validated {completed}/{expected_arms} expected arms")

    protocol_path = root / "postrun_protocol_report.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if repetitions >= 2 and protocol.get("order_independence_status") != "pass":
            fail(errors, "randomized repeated-block order validation did not pass")
        if repetitions < 2:
            warnings.append(
                "only one randomized block was run; order independence is not established"
            )
        if not protocol.get("trace_capacity_pressure_observed"):
            warnings.append(
                "no shadow-LRU eviction occurred during the trace; operational LMCache "
                "capacity is still checked by the mandatory pre-run control"
            )

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"VALIDATION PASSED: {completed} isolated arms, {len(parity)} paired events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
