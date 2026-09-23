# Portable exact-chunk mode

The public `LMCacheMPConnector` supports verifiable prefix reuse but does not
invoke CacheBlend V3's non-prefix `CB_*` protocol. Therefore this bundle uses a
portable exact-chunk experiment:

1. Observe the current chunk.
2. Predict K possible next chunks.
3. Compute and store each predicted chunk independently.
4. Request the actual next target chunk independently.
5. Record whether LMCache retrieved it, its TTFT, population cost, traffic,
   shadow-cache residency/evictions, policy time, and end-to-end cost.

The mandatory oracle must retrieve more target tokens than both cold and wrong
candidate controls. A separate capacity-pressure control uses independently
addressed chunks totaling at least 1.25 times the configured L1 byte budget;
the newest chunk must retrieve while the oldest retrieves fewer tokens. Either
failure blocks the benchmark.

Graph/document chunks are token-controlled (384 tokens with 64-token overlap
by default). LMCache's `--chunk-size` remains a separate 16-token transfer
granularity. The default 0.5 GiB L1 budget is intentionally smaller than the
prepared corpus, unlike the earlier 4 GiB setting that produced no evictions.

Every repetition randomizes the order of complete `(policy, K)` arms. At least
two repetitions with different orders are required before the post-run report
labels order independence as established.

This mode answers whether graph/adaptive selection improves real next-chunk
cache access relative to cosine under equal K. It must not be described as
CacheBlend or multi-document context fusion.
