#!/usr/bin/env python3
"""Reanalyse old question-level local-model quality CSVs without pandas."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


POLICIES = ("cosine", "graph", "adaptive")


def bootstrap(values, seed: int, resamples: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    estimates = values[draws].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return mean, float(low), float(high)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Extracted result root containing qa_quality_*.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("legacy_reanalysis"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()

    files = sorted(args.root.rglob("qa_quality_*.csv"))
    if not files:
        parser.error(f"No qa_quality_*.csv files found below {args.root}")
    summaries = []
    deltas = []
    for file_index, path in enumerate(files):
        with path.open(newline="", encoding="utf-8") as handle:
            raw = list(csv.DictReader(handle))
        grouped = defaultdict(list)
        for row in raw:
            grouped[int(row["config_k"])].append(row)
        for k, rows in sorted(grouped.items()):
            for policy in POLICIES:
                summaries.append(
                    {
                        "source_file": path.name,
                        "k": k,
                        "policy": policy,
                        "questions": len(rows),
                        "mean_em": float(np.mean([float(row[f"{policy}_em"]) for row in rows])),
                        "mean_f1": float(np.mean([float(row[f"{policy}_f1"]) for row in rows])),
                    }
                )
            for policy in ("graph", "adaptive"):
                for metric in ("em", "f1"):
                    values = [
                        float(row[f"{policy}_{metric}"]) - float(row[f"cosine_{metric}"])
                        for row in rows
                    ]
                    mean, low, high = bootstrap(
                        values,
                        args.seed + file_index * 1009 + k + sum(map(ord, policy + metric)),
                        args.resamples,
                    )
                    deltas.append(
                        {
                            "source_file": path.name,
                            "k": k,
                            "policy": policy,
                            "baseline": "cosine",
                            "metric": metric,
                            "questions": len(rows),
                            "mean_delta": mean,
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
    write_csv(args.output_dir / "legacy_local_quality_summary.csv", summaries)
    write_csv(args.output_dir / "legacy_local_quality_paired_deltas.csv", deltas)
    print(f"Analysed {len(files)} files; outputs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
