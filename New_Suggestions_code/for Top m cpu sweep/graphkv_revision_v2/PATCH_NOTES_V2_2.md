# GraphKV portable runner v2.2

This revision responds to the first 50-event T4 preliminary run.

## Scientific changes

- Replaces 500-character graph nodes (about 80 tokens in the observed run)
  with tokenizer-bounded document chunks. Defaults are 384 tokens and 64
  tokens of overlap.
- Separates `--document-chunk-tokens` from LMCache's internal `--chunk-size`;
  these are different concepts and are now named independently.
- Reduces default LMCache L1 capacity from 4.0 GiB to 0.5 GiB. The former was
  larger than the prepared corpus and produced zero evictions.
- Adds an operational capacity-pressure control. It independently populates
  chunks totaling at least 1.25 times L1, checks that the newest chunk
  retrieves, and checks that the oldest retrieves fewer tokens. This is a
  mandatory critical check.
- Makes two randomized repetitions the default. Execution order for every
  complete `(policy, K)` arm is saved for each repetition.
- Writes `postrun_protocol_report.json`, separating operational LMCache
  eviction evidence from shadow-LRU trace evictions and labeling whether
  order independence is established.

## Reliability and reporting changes

- A sanity report can be reused only when runtime configuration, prepared
  workload fingerprint, and repetition count all match.
- Workload statistics now report token-length distribution, corpus KV bytes,
  L1 capacity, corpus-to-capacity ratio, and approximate chunk capacity.
- The strict validator now checks the exact-chunk measurement/cache semantics
  actually written by the runner and validates repeated-block order status.
- The run manifest uses the same exact-chunk semantics as event records.
- Changes the default LMCache HTTP port from 8080 to 18080 because Colab
  commonly has an unrelated service bound to 8080; preflight still refuses
  any occupied required port.
- Added direct tests for token ceilings, exact overlap, and invalid overlap.
- Defines `shadow_current_prefetch_hit` strictly: the current prefetch must
  introduce the target and the target must still be resident after all K
  candidate insertions. This distinction matters once evictions are enabled.

## Claim boundary

The runner remains a stock-LMCache exact-chunk access/prepopulation
microbenchmark. It is not CacheBlend, non-prefix context fusion, or a
production asynchronous prefetch coordinator.
