#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def require(condition, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GraphKV simulator output invariants")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    output = Path(args.output_dir)
    paths = sorted(output.glob("results_*.csv"))
    require(paths, "no result CSVs found")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))

    keys = ["policy", "k", "event_id"]
    require(not frame.duplicated(keys).any(), "duplicate policy/K/event result rows")
    require(frame[keys].notna().all().all(), "missing result identity")
    for column in ("prediction_hit", "residency_hit", "current_prefetch_hit"):
        require(frame[column].isin([True, False]).all(), f"invalid boolean in {column}")

    current = frame.current_prefetch_hit.astype(bool)
    require((~current | frame.prediction_hit.astype(bool)).all(), "prefetch hit without prediction hit")
    require((~current | frame.residency_hit.astype(bool)).all(), "prefetch hit without residency")
    require(
        (~current | frame.target_source.fillna("").str.startswith("prefetch:")).all(),
        "current-prefetch hit has wrong provenance",
    )
    require(
        (frame.miss_reason.ne("late_prefetch") | frame.prediction_hit.astype(bool)).all(),
        "late-prefetch row was not predicted",
    )
    require(
        (frame.miss_reason.ne("prior_residency") | frame.residency_hit.astype(bool)).all(),
        "prior-residency row is not resident",
    )
    universe = frame.candidate_universe_size.dropna()
    require((universe >= 0).all(), "negative candidate-universe size")

    online = frame[frame.policy.str.contains("online")]
    if not online.empty:
        updated = online.online_updated.astype(bool)
        require(
            (~updated | online.online_update_reason.eq("ranking_miss_update")).all(),
            "online update occurred for a non-ranking-miss reason",
        )
        require((~updated | ~online.prediction_hit.astype(bool)).all(), "online update occurred on a hit")
        require(
            (~updated | online.candidate_coverage.astype(bool)).all(),
            "online update occurred on a coverage miss",
        )

    split_sets = {
        name: set(documents) for name, documents in manifest["split_documents"].items()
    }
    require(not split_sets["train"] & split_sets["dev"], "train/dev document leakage")
    require(not split_sets["train"] & split_sets["test"], "train/test document leakage")
    require(not split_sets["dev"] & split_sets["test"], "dev/test document leakage")
    evaluated_documents = set(frame.document_id.unique())
    require(
        evaluated_documents == set(manifest.get("evaluated_test_documents", evaluated_documents)),
        "result documents do not match manifest",
    )

    oracle = frame[frame.policy.eq("one_step_target_oracle")]
    if not oracle.empty and (oracle.k > 0).all():
        require(oracle.prediction_hit.astype(bool).all(), "oracle failed to predict its target")

    if manifest["trace_kind"] == "heldout_document_stratified_synthetic_600_compatibility":
        expected_k = [3, 4, 5, 6, 8, 10, 12, 14, 16]
        require(sorted(map(int, frame.k.unique())) == expected_k, "full comparison K set changed")
        expected_events = int(manifest["evaluated_test_events"])
        counts = frame.groupby(["policy", "k"]).size()
        require((counts == expected_events).all(), "incomplete policy/K block")
        trace = pd.read_csv(output / "trace_test.csv")
        proportions = manifest["comparison_query_proportions"]
        expected_counts = {
            query_type: round(expected_events * proportion)
            for query_type, proportion in proportions.items()
        }
        require(
            trace.query_type.value_counts().to_dict() == expected_counts,
            f"full comparison query counts are not exact: expected {expected_counts}",
        )

    if manifest.get("run_mode") == "stress_smoke":
        require(frame.miss_reason.eq("late_prefetch").any(), "stress smoke produced no late prefetch")
        eviction_columns = [
            "current_insert_evictions", "prefetch_evictions", "target_insert_evictions"
        ]
        require(
            any(frame[column].fillna("[]").ne("[]").any() for column in eviction_columns),
            "stress smoke produced no eviction",
        )

    print(f"PASS: {len(frame)} rows, {frame.policy.nunique()} policies, K={sorted(frame.k.unique())}")
    print(f"Trace kind: {manifest['trace_kind']}")
    if manifest["trace_kind"] == "grouped_synthetic_diagnostic":
        print("CAUTION: mechanics diagnostic only; not independent paper evidence.")
    if manifest["trace_kind"] == "heldout_document_stratified_synthetic_600_compatibility":
        print("CAUTION: old-run compatibility ablation; targets are still synthetic.")


if __name__ == "__main__":
    main()
