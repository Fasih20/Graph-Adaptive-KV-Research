# Real KV Cache Prefetching — vLLM + LMCache CacheBlend

Implements `implementation_plan.md` Phases 1–4 (Phase 0 / environment setup
skipped — already done on the university machine).

## What's here vs. what was ported verbatim

`src/graph_algorithms.py` and `src/metrics_utils.py` are **byte-for-byte
ports** of `kv_cache_experiment_adaptive_v2(2).py`'s `build_graph`,
`get_prefetch_cosine`, `get_prefetch_graph`, `get_prefetch_adaptive`,
`generate_pairs`, `KVCacheSimulator`, and `estimate_flops_saved`. This is
checked, not just asserted: `tests/test_policy_fidelity.py` AST-extracts the
real functions straight out of the original `.py` file and diffs their
output against the ported module across 7 synthetic corpora (including
near-duplicate-embedding cases specifically designed to stress tie-breaking
and insertion order). Run it yourself:

```bash
python3 tests/test_policy_fidelity.py
```

It currently passes on every case. Do not "clean up" `graph_algorithms.py`
without re-running this — see its module docstring.

Everything else (`chunk_store.py`, `cache_manager.py`, `vllm_launcher.py`,
`benchmark_harness.py`, `correctness_check.py`) is new: the plumbing that
replaces the canonical script's local-transformers calls with a real
vLLM + LMCache CacheBlend server. Read `benchmark_harness.py`'s module
docstring before trusting its numbers against old plots — it documents
three deliberate semantic deviations from the simulated harness (real TTFT
instead of the hit-aware-timing 0ms hack; blob- vs per-chunk truncation for
token-count columns; no `legacy_timing` flag).

## Project structure

```
src/
  config.py              constants (ported + new real-hardware config)
  graph_algorithms.py    PORTED VERBATIM — build_graph, get_prefetch_*, generate_pairs, KVCacheSimulator
  metrics_utils.py        PORTED VERBATIM — estimate_flops_saved
  prefetch_policies.py   class interface wrapping the above (Phase 1)
  chunk_store.py         dataset loading, chunking, embedding, graph build (Phase 1)
  cache_manager.py       vLLM/LMCache bridge: warm_chunks, query_with_chunks, flush_cache (Phase 2)
  vllm_launcher.py       process management — MP mode (default) or in-process patch (Phase 2)
  benchmark_harness.py   main measurement loop, exact CSV schema (Phase 3)
  correctness_check.py   fresh-vs-blended output comparison (Phase 3)
  calibrate_eager.py     --enforce-eager overhead calibration
  run_experiment.py      CLI entrypoint for the main benchmark
  run_correctness.py     CLI entrypoint for the correctness check
scripts/
  calibrate_eager.sh
  run_cosine_baseline.sh   small smoke run
  run_full_benchmark.sh    full K=3..16 x 600-pair run
  run_correctness.sh
tests/
  test_policy_fidelity.py
results/                  CSV outputs land here
```

## Before running on the university machine

1. **`scripts/calibrate_eager.sh`** — run this first. It launches plain
   vLLM (no LMCache) twice, CUDA graphs vs. `--enforce-eager`, and writes
   `results/eager_calibration.csv` / `eager_calibration_summary.json`.
   Check the TTFT delta before trusting the main run's numbers.

2. **`scripts/run_cosine_baseline.sh`** — a 30-pair smoke run to confirm
   the vLLM + LMCache MP-mode pipeline actually comes up and produces
   sane-looking rows before committing to the full 6000-trial run.

3. **`scripts/run_full_benchmark.sh`** — the real thing. Resumable (skips
   `(config_k, config_cap)` pairs already in the output CSV).

4. **`scripts/run_correctness.sh`** — N=20 fresh-vs-blended comparison,
   independent of the above.

## CLI-flag / endpoint disclaimer

`vllm_launcher.py` and `cache_manager.py` assemble vLLM/LMCache CLI flags
and HTTP endpoints (`--kv-transfer-config`, `--block-size`,
`POST /cache/clear`, etc.) from `implementation_plan.md` plus current
`docs.lmcache.ai` / `docs.vllm.ai` documentation as of this delivery. This
API surface moves fast on both projects — if `run_cosine_baseline.sh` fails
at server startup, `_build_lmcache_server_args` / `_build_vllm_mp_args` in
`vllm_launcher.py` is the one place to check flag names against your
actually-installed versions.

## Adaptive weights actually used (Phase 4)

`config.HARDCODED_ADAPTIVE_WEIGHTS` — semantic 0.90/0.10, structural
0.35/0.65, multi-hop 0.35/0.65, all hardcoded (no re-run of the grid search
against real-cache data yet). Note these differ from
`FALLBACK_ADAPTIVE_WEIGHTS` (the canonical script's own built-in default of
0.40/0.60 structural, 0.70/0.30 multi-hop) — intentional, see
`config.py`'s comment.
