# Simulator implementation status

## Implemented and tested

- frozen legacy graph behavior plus document-aware structural boundaries;
- target-free policy API;
- entry/byte constrained LRU and provenance;
- blocking/deadline prefetch event semantics;
- separate prediction and residency outcomes;
- offline train-only global weights and query-type oracle;
- online-from-scratch and offline-warm-start projected mistake updates;
- document-level train/dev/test splits;
- atomic exact-state resume;
- model-config KV byte accounting;
- honest legacy and componentized theoretical FLOP helpers;
- hardware micro-calibration utility;
- aggregate analysis and paired cluster bootstrap helper;
- balanced document/query-type smoke sampling;
- transferable relative-offset transition baseline;
- candidate-universe size/coverage reporting;
- normalized projected online updates and expanded eta tuning;
- exact held-out 600-query, nine-K comparison mode;
- legacy-versus-repaired comparison exporter;
- 16 unit/integration tests, including uninterrupted-versus-resumed equality.

## Deliberately not claimed as complete evidence

- The grouped-synthetic trace is not independent and therefore cannot establish
  real-world prediction superiority.
- Proxy latency cannot establish vLLM/LMCache speedup.
- The query-type oracle is not deployable without a measured router.
- The calibration microbenchmark is hardware-specific and is not end-to-end
  serving telemetry.
- Large-model/A100 experiments should wait for manual smoke-output review.

## Next implementation phase after approval

Repair and isolate the vLLM/LMCache implementation using this simulator's trace,
policy, provenance, and metric contracts. Do not merge simulator and vLLM
results into one claim.
