# Verification record — 2026-09-19

Executed locally:

- 12 V2 CPU tests passed, including blocked-background/foreground ordering,
  synchronous exposure, online update eligibility, one-hop/two-hop isolation,
  cross-document structural exclusion, equal-weight policy equivalence,
  missing/numerically different log-probability rejection, configuration guard,
  snapshot checksum restore, backup failure preservation, and interruption
  restart/complete-block skip integration.
- All Python files compiled.
- All seven notebook code cells parsed.
- Both revised CLI help paths loaded.
- Supplied final Qwen archive: 18,000 complete event rows, 1,200 paired groups,
  zero target-prompt hash mismatches, zero one-token output hash mismatches,
  13,155 requests with recorded retrieval evidence.

NOT executed here:

- GPU numerical reuse gate, vLLM/LMCache startup, single-T4 asynchronous
  performance, complete local generation, live dataset downloads, Google OAuth
  authentication and a real Drive upload/restore.
- F1 drift with old/new generated answers, statistically independent final
  cross-model results, production RAG timing, or any CacheBlend operation.

The cache gate is mandatory in the new systems CLI. Logs missing retrieval
evidence produce an inconclusive/failing gate, not a performance-only pass.
The tests use fake inference clients for scheduling/recovery and do not stand
in for an actual GPU run.

The final archive checks validate saved parity only. They cannot prove that
every cache tensor is correct, and one-token equality can miss numerical
differences. Independent-chunk concatenation is not used in the audited final
Qwen path; the new QA path uses full-prompt inference without LMCache.
