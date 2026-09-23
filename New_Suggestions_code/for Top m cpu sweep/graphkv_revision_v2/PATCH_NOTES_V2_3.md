# GraphKV portable runner v2.3.2

## v2.3.2 empty-decoded-token fix

- Completion requests now use vLLM's supported `min_tokens`, `ignore_eos`, and
  `skip_special_tokens` controls. A one-token prefill probe therefore cannot
  terminate immediately at EOS and disappear during special-token decoding.
- A successful streamed completion choice is accepted even if its decoded text
  is empty. The request is still rejected if the stream contains no completion
  choice at all.
- Existing `COMPLETE.json` arms remain resumable; only a failed partial attempt
  is repeated under fresh vLLM and LMCache processes.

## Reporting

- Prints a pooled, event-weighted headline table at the end of every completed
  benchmark. It includes prediction recall, ready-cache TTFT, service duration,
  speculative population cost, conservative end-to-end TTFT/request duration,
  completed traffic, audit-cache residency, and LMCache retrieved tokens.
- Writes `headline_mean_results.csv` for the compact table and
  `pooled_results_by_query_type.csv` for every mean and median, pooled directly
  from raw events across repetitions.
- Preserves the existing per-repetition `summary_by_query_type.csv` for paired
  and repeated-block analysis.
- The complete summary now also aggregates target rank, candidate-universe
  size, prompt/capacity values, overlap, pre-prefetch residency, cache
  occupancy, and invariant completion/success diagnostics that were already
  present in raw event records.

## Workload safety

- Prints the main-arm process count and real vLLM request upper bound before
  preflight.
- Saves the calculation to `run_scale_estimate.json` after configuration
  validation, with candidate-population and current/target requests separated.
- Documents a four-K single-T4 Kaggle screening protocol before any 600-event
  confirmation.
- Adds `scripts/run_kaggle_screen.sh`, a quoted-argument wrapper for the same
  portable single-model screen.
- Rejects empty or duplicate policy/K lists so accidental duplicate arms cannot
  consume notebook quota.

## Model portability

- KV-byte sizing now reads decoder dimensions from `text_config` when a model
  uses a composite multimodal configuration, covering Gemma 3 4B-style configs
  without changing text-only model behavior.

## v2.3.1 capacity-control race fix

- Waits for LMCache's asynchronous store queue and capacity eviction to settle
  before probing recent and old entries.
- Refreshes the newest sentinel after pressure settles, then proves that it is
  cached while the oldest entry is less cached.
- Uses two independent evidence paths: LMCache retrieval/eviction logs and
  vLLM's exact `request_prefill_kv_computed_tokens_sum` counter. This prevents
  a false failure when an LMCache log line is flushed just after the request
  returns.
- Automatically reruns a matching failed sanity report with the current
  checker instead of repeatedly reusing the known failure.

## Claim clarity

- Documents every timing, prediction, capacity, traffic, telemetry, and audit
  field.
- Explicitly states that `run_repaired.py` does not measure FLOPs. Historical
  analytical FLOPs code is not used by the repaired benchmark.
- Documents that fixed graph alone uses `0.7/0.3`; offline adaptive weights are
  freshly fitted on train documents, and online warm starts from those weights.
