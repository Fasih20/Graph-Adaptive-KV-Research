"""
analyze_results.py
===================
Companion analysis script for results_real_<dataset>.csv (benchmark_harness.py's
output). Not part of the original canonical script — added because raw
policy-vs-policy mean *_ms, averaged across ALL trials, is easy to
misread: cosine/graph/adaptive's mean blends together "trials where the
prefetch policy correctly guessed `sec`" (should be fast, if caching is
genuinely working) with "trials where it guessed wrong" (should look
about the same as cold, not better, not worse). Diluting those together
makes a real effect look small, and makes "no effect at all" hard to
distinguish from "a real effect on a modest hit rate".

This prints, per policy, per K:
  - hit rate (from the existing *_sim_hit columns — lru<->cold,
    cos<->cosine, grp<->graph, adp<->adaptive)
  - mean ms overall, and split by hit==1 vs hit==0
  - paired % speedup vs cold_ms (computed per-row, then averaged — NOT
    computed from the difference of two aggregated means, which would be
    a biased estimator here since K/dataset/run composition can differ
    trial to trial)

Also prints an ALL-K rollup, and flags (doesn't fail) any policy whose
hit-only mean ms is NOT meaningfully lower than its own miss-only mean —
that specific comparison is the real test of "is caching actually
helping", independent of whatever cold is doing.

Usage:
  python3 src/analyze_results.py [--csv path/to/results_real_hotpot.csv] [--k 12]
  (with no --csv, uses the newest results_real_*.csv under RESULTS_DIR)
"""

import sys
import glob
import argparse
from pathlib import Path

import pandas as pd

from exp_config import RESULTS_DIR

POLICIES = [
    ("cold",     None,           "cold_ms"),      # no prefetch policy of its own
    ("cosine",   "cos_sim_hit",  "cosine_ms"),
    ("graph",    "grp_sim_hit",  "graph_ms"),
    ("adaptive", "adp_sim_hit",  "adaptive_ms"),
]


def _find_latest_csv() -> Path:
    candidates = sorted(glob.glob(str(RESULTS_DIR / "results_real_*.csv")),
                        key=lambda p: Path(p).stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No results_real_*.csv found under {RESULTS_DIR} — pass --csv explicitly")
    return Path(candidates[0])


def _paired_speedup_pct(df: pd.DataFrame, policy_col: str) -> float:
    """Mean of per-row (cold_ms - policy_ms) / cold_ms, in percent. Paired
    (same-row) comparison, not a diff of two aggregate means, since each
    row already has both cold_ms and policy_ms measured on the identical
    request under the identical conditions."""
    sub = df[["cold_ms", policy_col]].dropna()
    sub = sub[sub["cold_ms"] > 0]
    if sub.empty:
        return float("nan")
    return float(((sub["cold_ms"] - sub[policy_col]) / sub["cold_ms"]).mean() * 100)


def _report_slice(df: pd.DataFrame, label: str):
    n = len(df)
    print(f"\n{label}  (n={n})")
    if n == 0:
        return
    print(f"  {'policy':<10} {'hit%':>6} {'mean_ms':>10} {'hit_ms':>10} {'miss_ms':>10} {'spd_vs_cold%':>13}")
    for name, hit_col, ms_col in POLICIES:
        if ms_col not in df.columns:
            continue
        overall_mean = df[ms_col].mean()
        if hit_col and hit_col in df.columns:
            hit_rate = df[hit_col].mean() * 100
            hit_ms = df.loc[df[hit_col] == 1, ms_col].mean()
            miss_ms = df.loc[df[hit_col] == 0, ms_col].mean()
        else:
            hit_rate, hit_ms, miss_ms = float("nan"), float("nan"), float("nan")
        spd = _paired_speedup_pct(df, ms_col) if ms_col != "cold_ms" else 0.0
        print(f"  {name:<10} {hit_rate:>5.1f}% {overall_mean:>10.2f} "
              f"{hit_ms:>10.2f} {miss_ms:>10.2f} {spd:>12.1f}%")

    print("  caching sanity check (hit_ms should be well below miss_ms; if it isn't,")
    print("  see sanity_check_caching.py's LMCache /status + log output before trusting")
    print("  the vs-cold speedup numbers above):")
    for name, hit_col, ms_col in POLICIES:
        if not hit_col or hit_col not in df.columns or ms_col not in df.columns:
            continue
        hit_ms = df.loc[df[hit_col] == 1, ms_col].mean()
        miss_ms = df.loc[df[hit_col] == 0, ms_col].mean()
        n_hit = int(df[hit_col].sum())
        if n_hit < 5:
            print(f"    {name:<10} too few hits (n={n_hit}) to say anything meaningful yet")
        elif pd.isna(hit_ms) or pd.isna(miss_ms) or miss_ms <= 0:
            print(f"    {name:<10} not enough data")
        elif hit_ms < miss_ms * 0.9:
            print(f"    {name:<10} OK — hits are {((miss_ms - hit_ms) / miss_ms * 100):.0f}% faster than misses (n_hit={n_hit})")
        else:
            print(f"    {name:<10} ⚠ hits are NOT meaningfully faster than misses "
                  f"(hit={hit_ms:.1f}ms vs miss={miss_ms:.1f}ms, n_hit={n_hit}) — "
                  f"caching may not be engaging for this policy even though the "
                  f"request shape is now fair")


def _report_real_metrics(csv_path: Path, df: pd.DataFrame):
    """Cross-checks the simulated hit rates against benchmark_harness.py's
    companion *_realmetrics.csv (vLLM's own /metrics deltas, real not
    simulated) if one exists next to the main CSV. Not a strict
    per-trial join -- the companion file is one row per trial with
    aggregate hits/queries across all 4 policy arms combined -- but the
    aggregate real hit rate is still the single most important sanity
    check available: if it's near zero while the simulated hit rates
    above are 30-80%, the *_ms speedups are not coming from correctly-
    predicted prefetches, whatever the simulated numbers suggest."""
    metrics_path = csv_path.parent / (csv_path.stem + "_realmetrics.csv")
    if not metrics_path.exists():
        print(f"\n(no {metrics_path.name} found next to this CSV -- re-run with "
              f"track_real_hits=True / without --no-track-real-hits to get this)")
        return
    mdf = pd.read_csv(metrics_path)
    if mdf.empty:
        print(f"\n{metrics_path.name} exists but is empty.")
        return
    print(f"\nREAL /metrics CROSS-CHECK  (from {metrics_path.name}, vLLM's own counters, "
          f"aggregated across all 4 policy arms per trial -- not simulated)")
    for k in sorted(mdf["config_k"].unique()):
        sub = mdf[mdf["config_k"] == k]
        total_hits = sub["prefix_cache_hits_delta"].sum()
        total_queries = sub["prefix_cache_queries_delta"].sum()
        real_rate = (total_hits / total_queries * 100) if total_queries > 0 else float("nan")
        sim_rate = float("nan")
        if "cos_sim_hit" in df.columns:
            sim_rate = df.loc[df["config_k"] == k, "cos_sim_hit"].mean() * 100
        print(f"  K={k:<3} real_hit_rate={real_rate:5.1f}%  "
              f"(hits={total_hits:.0f}/queries={total_queries:.0f})   "
              f"vs. simulated cos_sim_hit={sim_rate:5.1f}%")
    if (mdf["prefix_cache_hits_delta"].sum() == 0 and mdf["prefix_cache_queries_delta"].sum() > 0):
        print("  ⚠ real hits are ZERO across the entire run despite nonzero queries -- "
              "vLLM's own counters say nothing was ever served from cache, regardless of "
              "what the simulated hit rate or the *_ms numbers above suggest. Run "
              "sanity_check_caching.py before trusting anything else in this file.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--k", type=int, default=None, help="Restrict to one config_k value")
    args = p.parse_args()

    csv_path = args.csv or _find_latest_csv()
    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV is empty — nothing to analyze.")
        return

    _report_real_metrics(csv_path, df)

    if args.k is not None:
        df = df[df["config_k"] == args.k]

    _report_slice(df, "ALL K COMBINED")
    for k in sorted(df["config_k"].unique()):
        _report_slice(df[df["config_k"] == k], f"K = {k}")


if __name__ == "__main__":
    main()
