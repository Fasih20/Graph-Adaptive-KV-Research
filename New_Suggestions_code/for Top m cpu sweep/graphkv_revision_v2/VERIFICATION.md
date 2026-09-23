# Verification status

Verified locally on 2026-09-04:

- Python byte-compilation passed for the complete bundle.
- The preceding v2.1 bundle passed eight repaired-core tests:
  - Prometheus parsing and counter deltas;
  - byte-capacity LRU eviction;
  - post-observation online updates;
  - single-GPU launch arguments and connector configuration;
  - deterministic segmented prompt construction;
  - current vLLM paged-help discovery via `--help=all`;
  - fallback to complete plain help on older vLLM CLIs;
  - LMCache retrieved/stored-token log fallback parsing.
- This capacity-pressure revision passes Python byte-compilation and all ten
  repaired-core test functions through a direct fixture-compatible harness,
  including the two new tokenizer-window cases for exact overlap, configured
  token ceiling, and invalid overlap. The delivery workspace does not contain
  pytest itself, so the standard `pytest` command must still be rerun after
  notebook installation.
- The v2.3 reporting revision passes two additional direct tests: pooled means
  are calculated from raw events rather than means-of-means, and the predicted
  real-request count matches the exact policy/K/event/repetition formula.
- Composite model KV sizing is covered directly so Gemma 3-style
  `text_config` metadata is handled without breaking text-only configs.
- v2.3.1 passes a fourteenth direct test proving that exact Prometheus metric
  selection excludes similarly named histogram buckets and counters.
- v2.3.2 adds a fifteenth direct regression test for a successful vLLM stream
  whose one generated token has empty decoded text, including the EOS/special-
  token request controls sent to vLLM.
- All seven original graph-policy fidelity cases passed against the canonical
  source, including duplicate-embedding tie and `MAX_DEGREE` stress cases.
- A synthetic completed experiment passed `validate_repaired.py`.
- The same synthetic output successfully produced flattened events, every
  metric summary, paired deltas, clustered bootstrap intervals, diagnostics,
  and the headline dashboard.
- Shell scripts passed `bash -n`.

Not executable in the delivery workspace:

- Live vLLM, LMCache, CUDA, and T4/A100 controls. This workspace has no NVIDIA
  runtime. The bundle therefore refuses a full run until its on-machine
  preflight and fresh-process cold/wrong/oracle/capacity controls pass.

The first real run must be the documented five-event T4 smoke. Its logs and
`sanity_report.json` are part of the scientific evidence, not optional setup
noise.
