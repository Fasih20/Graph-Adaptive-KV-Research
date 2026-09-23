# Start with START_HERE.md and GraphKV_Revision_Kaggle.ipynb

This package contains the V2 revision plus the frozen predecessor for reference. The instructions below are historical.

# GraphKV isolated vLLM + LMCache chunk-access benchmark (v2.3.2)

This is the repaired real-cache runner for GraphKV. It is designed for one
NVIDIA GPU on Kaggle, Colab, or RunPod. Its portable public-package mode is an
exact chunk-access prefetch benchmark: every predicted chunk is populated
independently, and the next target chunk is requested independently so stock
LMCache can provide verifiable exact-prefix retrieval.

This mode compares graph/adaptive/cosine next-chunk selection using real cache
traffic and TTFT. It does **not** claim non-prefix CacheBlend fusion inside a
multi-document RAG prompt; that requires the separate `CBKVConnector` plugin,
which is not included in the public LMCache 0.5.4 wheel.

## What is fixed

- Every `(policy, K, repetition)` runs in a new vLLM + LMCache process pair.
- `CUDA_VISIBLE_DEVICES` exposes only the selected physical GPU and vLLM uses
  tensor/pipeline parallel size 1. Kaggle's second T4 is never combined.
- Native vLLM automatic prefix caching is disabled; exact chunk reuse comes
  from LMCache and is verified by cold/wrong/oracle plus capacity controls.
- MiniLM embeddings and graph construction finish before vLLM starts, on CPU
  by default, so the embedder does not consume T4 model memory.
- Every policy sends identical timed token IDs for an event; validation checks
  prompt hashes across policies and K values.
- Prediction hits, byte-LRU shadow residency, and real LMCache/vLLM telemetry
  are separate. A shadow hit is never called a real hardware hit.
- Graph nodes are tokenizer-bounded document chunks (384 tokens with 64-token
  overlap by default). This is independent from LMCache's internal 16-token
  transfer chunk, eliminating the earlier character/token ambiguity.
- The default 0.5 GiB L1 budget intentionally creates cache pressure. A fourth
  fresh-process control overfills L1 and must demonstrate operational eviction
  before policy arms are allowed to start.
- TTFT, complete request time, policy time, speculative population cost,
  exposed cost after overlap, and end-to-end time are reported.
- Candidate KV is currently computed with real inference. That cost is measured
  and included in end-to-end columns; it is not hidden as a free prefetch.
- Offline weights use training documents, eta uses dev documents, and online
  updates occur only after observing the target.
- Partial attempts are retained but never resumed in-place. A retry gets fresh
  processes and a new attempt directory.
- Raw logs, commands, manifests, prompt/output hashes, Prometheus deltas, GPU
  snapshots, JSONL, CSV summaries, paired deltas, confidence intervals, and
  charts are preserved.
- Before preflight, the runner prints an upper bound on isolated arms and real
  vLLM requests; after configuration validation it saves the same calculation.
  At completion, it prints exact event-weighted means
  across repetitions and writes both a compact headline CSV and a complete
  pooled query-type CSV.

The connector follows the official [LMCache quickstart](https://docs.lmcache.ai/getting_started/quickstart.html),
[MP configuration](https://docs.lmcache.ai/mp/configuration.html), and
[HTTP API](https://docs.lmcache.ai/mp/http_api.html). Installed CLI features
are checked before launch because vLLM and LMCache change quickly.

## Measurement names

All event metrics below are aggregated as both means and medians. The compact
`headline_mean_results.csv` is pooled directly from every raw event across
repetitions, so it does not make the statistical mistake of averaging means
from unequal groups.

### Latency and compute time

| Column | Meaning |
|---|---|
| `current_access_ttft_ms` | TTFT of the real request that makes the current chunk resident |
| `current_access_request_ms` | Complete duration of that current-chunk request |
| `ready_cache_ttft_ms` | TTFT after candidate population has completed |
| `request_e2e_ms` | Complete timed service-request duration |
| `policy_ms` | Candidate-selection CPU time |
| `speculative_population_ms` | Real inference used to compute/store candidate KV |
| `available_overlap_ms` | Declared interval available to hide population work |
| `exposed_population_ms` | `max(0, population - overlap)` |
| `end_to_end_ttft_ms` | `policy + exposed population + ready-cache TTFT` |
| `end_to_end_request_ms` | `policy + exposed population + complete request` |

`ready_cache_ttft_ms` answers, "If speculative work was already completed,
how quickly did demand begin producing output?" `end_to_end_ttft_ms` is the
conservative comparison because it charges the policy for speculative work
that could not be hidden. With the default `--overlap-ms 0`, all population
cost is exposed.

### Prediction, residency, and traffic

| Column | Meaning |
|---|---|
| `prediction_hit` | Target appeared in top-K; prediction recall, not a hardware hit |
| `target_rank` | One-based rank of the target, or null when missed; its aggregate is conditional on prediction hits |
| `candidate_universe_size` | Number of graph candidates that could be ranked |
| `predicted_count` | Number of candidates selected for the event |
| `completed_prefetch_count` | Candidate-population requests that completed |
| `prefetch_completion_fraction` | Completed count divided by selected count |
| `prefetch_bytes_scheduled` | Model-aware theoretical KV bytes selected |
| `prefetch_bytes_completed` | Model-aware theoretical KV bytes whose real population request completed |
| `population_request_count` | Real one-token vLLM calls used to populate candidates |
| `population_prompt_tokens` | Tokens processed by those population calls |
| `shadow_residency_before_prefetch` | Target present in the auditable byte-LRU model before this event's prefetch |
| `shadow_residency_hit` | Target present in the byte-LRU model after all K candidate insertions |
| `shadow_current_prefetch_hit` | This event introduced the target and it survived until demand |
| `shadow_cache_bytes_after` | Bytes occupied in the audit cache after the event |
| `*_insert_eviction_count` | Audit-cache evictions caused by current, prefetch, or target insertion |

The byte-LRU fields test policy behavior under a fixed capacity, but they are
deliberately named `shadow_*`: they are not passed off as LMCache telemetry.

### Real cache evidence and audit fields

| Column | Meaning |
|---|---|
| `lmcache_retrieved_tokens_delta` | Service-request tokens reported retrieved by LMCache metrics/logs |
| `lmcache_stored_tokens_delta` | Service-request tokens reported stored by LMCache metrics/logs |
| `vllm_prefix_cache_hit_tokens_delta` | Native vLLM prefix-hit telemetry; native prefix caching is disabled for the experiment |
| `population_lmcache_metrics_delta` | Complete LMCache metric delta during candidate population |
| `service_lmcache_metrics_delta` | Complete LMCache metric delta during demand |
| `service_vllm_metrics_delta` | Complete vLLM metric delta during demand |
| `prompt_tokens`, `prompt_token_hash` | Proves identical timed demand prompts across policies and K |
| `weights_before`, `weights_after`, `online_updated`, `online_update_reason` | Audits every online learning decision |
| timestamps, log evidence, GPU snapshots, response/output hashes | Failure diagnosis and reproducibility evidence |

### FLOPs claim

The repaired real-cache runner does **not** report measured FLOPs. Historical
reference code in this package contains an analytical FLOPs-saved equation,
but that is not executed by `run_repaired.py` and is not a hardware
measurement. This runner instead reports measured wall-clock timings, real
prompt/population token counts, LMCache retrieval telemetry, and model-aware
KV-byte estimates. Hardware FLOPs would require a separate profiler experiment
and should not be mixed into the latency run because profiling changes timing.

## Adaptive-weight provenance

- `graph_fixed` always uses semantic/structural weights `0.7/0.3`.
- `adaptive_offline_global` freshly searches semantic weight 0.00 through 1.00
  in steps of 0.05 on the training-document trace; structural weight is
  `1 - semantic`. It selects the pair with best prediction recall over the
  configured K values. A tie prefers the pair closest to `0.7/0.3`.
- No vLLM baseline run is required to fit these weights. Fitting uses graph
  predictions and known training targets on CPU, before any test arm runs.
- `adaptive_online_warm` starts each isolated test arm at that freshly fitted
  pair. Its learning rate is selected on dev documents at K=6, then it updates
  only after an observable ranking miss. It never sees a test target before
  predicting that event.
- Exact learned weights, every grid-search result, the chosen online learning
  rate, and train/dev/test document IDs are saved in `fitted_policy.json`.

Changing `--k-values` changes the offline fitting objective, so weights are
relearned for the selected K set rather than copied from an earlier run.

## Install on Kaggle or Colab

Enable a GPU runtime, upload/extract this folder, then run:

```bash
bash scripts/install_notebook.sh
```

Restart the runtime after installing vLLM, LMCache, PyTorch, or CUDA
extensions. The requirements pin the intended vLLM/LMCache candidate pair but
allow their resolver to select ABI-compatible PyTorch and transitive packages.
The resolved environment is saved in `environment_manifest.json`.

GraphKV does not import or require `torchaudio`. If a notebook image contains a
TorchAudio wheel built for a different PyTorch/CUDA combination, leave
TorchAudio uninstalled instead of mixing a nightly TorchAudio wheel into the
resolved vLLM/LMCache environment:

```bash
uv pip uninstall --system torchaudio
```

## 1. Preflight

On Kaggle with two T4s, `--gpu 0` exposes only the first T4 to child processes:

```bash
python run_repaired.py \
  --dataset hotpot \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --gpu 0 \
  --preflight-only \
  --output-dir outputs/hotpot_qwen15b_smoke
```

Preflight stops for missing Linux/GPU/commands/packages/ports or required CLI
flags. New vLLM releases use paged help, so the runner checks both
`vllm serve --help=all` and the older plain-help form. It saves the inspected
help in `vllm_cli_help.txt` and `lmcache_cli_help.txt`; a genuine flag mismatch
also writes `preflight_cli_failure.json` with executable paths and versions.

## 2. Mandatory fresh-process controls

This launches separate cold, wrong-candidate, oracle, and capacity-pressure
process pairs. It checks prompt parity, output agreement, telemetry, exact
target retrieval for the oracle, and real eviction after overfilling L1.

```bash
python run_repaired.py \
  --dataset hotpot \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --gpu 0 \
  --events 5 --train-events 80 --dev-events 40 \
  --max-documents 30 \
  --document-chunk-tokens 384 \
  --document-chunk-overlap-tokens 64 \
  --l1-size-gb 0.5 \
  --k-values 5 \
  --repetitions 1 \
  --sanity-only \
  --output-dir outputs/hotpot_qwen15b_smoke
```

Do not run a large experiment if this exits with code 2. Inspect
`sanity/sanity_report.json`, `sanity/*/lmcache.log`, and `sanity/*/vllm.log`.
A passing report is reused when the runtime configuration is unchanged.
An existing failed report is automatically rerun with the current checker;
you do not need to delete the output directory.

## 3. Five-event T4 smoke

```bash
python run_repaired.py \
  --dataset hotpot \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --gpu 0 \
  --events 5 --train-events 80 --dev-events 40 \
  --max-documents 30 \
  --document-chunk-tokens 384 \
  --document-chunk-overlap-tokens 64 \
  --l1-size-gb 0.5 \
  --k-values 5 \
  --repetitions 1 \
  --policies no_prefetch,cosine,graph_fixed,adaptive_offline_global,adaptive_online_warm \
  --output-dir outputs/hotpot_qwen15b_smoke
```

Use a new output directory whenever scientific settings change. The program
refuses to combine different configurations in one folder.

## 4. T4 validation before 600 events

Two randomized repetitions around the strongest K region are more useful than
immediately sweeping every K:

```bash
python run_repaired.py \
  --dataset hotpot \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --gpu 0 \
  --events 100 --train-events 300 --dev-events 150 \
  --max-documents 60 --max-chars-per-document 12000 \
  --document-chunk-tokens 384 \
  --document-chunk-overlap-tokens 64 \
  --l1-size-gb 0.5 \
  --k-values 5,6 \
  --repetitions 2 \
  --output-dir outputs/hotpot_qwen15b_validation
```

This means 20 isolated main arms plus four controls. Fresh startup per arm is
deliberate causal isolation; it is outside event latency.

## 5. Kaggle single-T4 screen

Kaggle's two T4s must not be tensor-parallelized for the latency comparison.
Use `--gpu 0`; the runner exposes only that physical GPU and forces tensor and
pipeline parallel size 1. Do not run a second concurrent benchmark on GPU 1:
shared CPU, RAM, storage, and model-download contention would contaminate the
latency comparison.

Start with four K values representing low budget, the earlier knee, medium,
and high budget:

```bash
python run_repaired.py \
  --dataset hotpot \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --gpu 0 \
  --events 200 --train-events 300 --dev-events 150 \
  --max-documents 60 --max-chars-per-document 12000 \
  --document-chunk-tokens 384 \
  --document-chunk-overlap-tokens 64 \
  --l1-size-gb 0.5 \
  --k-values 3,6,10,16 \
  --repetitions 2 \
  --output-dir outputs/hotpot_qwen15b_kaggle_screen
```

This is 40 isolated main arms and at most 72,000 vLLM requests, plus four
sanity-control process pairs. Your proposed `3,6,8,10,12,14,16` set at 600
events and two repetitions is at most 415,200 requests. The original nine K
values reach 482,400. The four-K set retains the useful curve shape while
cutting both the candidate-population and total-request upper bounds by about
55% relative to all nine K values.

If the screen is stable and positive, a 600-event single-model confirmation
with the same four K values has an upper bound of 216,000 requests. Completed
arms are safely reusable after interruption; an incomplete arm is intentionally
restarted in fresh processes.

The same screen is also available as a wrapper, for example:

```bash
bash scripts/run_kaggle_screen.sh \
  Qwen/Qwen2.5-1.5B-Instruct \
  outputs/hotpot_qwen15b_kaggle_screen
```

## 6. RunPod publication run

After the T4 controls and 100-event validation pass:

```bash
python run_repaired.py \
  --dataset hotpot \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --gpu 0 \
  --events 600 --train-events 600 --dev-events 300 \
  --max-documents 100 --max-chars-per-document 16000 \
  --document-chunk-tokens 384 \
  --document-chunk-overlap-tokens 64 \
  --l1-size-gb 0.5 \
  --k-values 3,4,5,6,8,10,12,14,16 \
  --repetitions 3 \
  --output-dir outputs/hotpot_qwen15b_full
```

Confirm whether RunPod provides an A100 40 GB or 80 GB before increasing model
size or GPU memory use. Never pool raw milliseconds from different GPU models.

For 2Wiki and MuSiQue, use new output directories and change `--dataset`.
Those add independent workload evidence. Merely changing the causal-model name
does not create a new access trace.

## Validate, analyze, package

```bash
python validate_repaired.py outputs/hotpot_qwen15b_validation
python analyze_repaired.py outputs/hotpot_qwen15b_validation
bash scripts/package_results.sh outputs/hotpot_qwen15b_validation
```

Packaging validates first and refuses incomplete or prompt-mismatched output.
It creates a ZIP and SHA-256 checksum.

## Output layout

```text
output/
  environment_manifest.json
  run_manifest.json
  fitted_policy.json
  workload/chunks.jsonl
  workload/trace_{train,dev,test}.csv
  sanity/sanity_report.json
  arms/policy=.../k=.../rep=.../
    COMPLETE.json
    attempt_001/
      launch_commands.json
      vllm.log
      lmcache.log
      events.jsonl
  summary_by_query_type.csv
  pooled_results_by_query_type.csv
  headline_mean_results.csv
  run_scale_estimate.json
  postrun_protocol_report.json
  analysis/
    all_events_flat.csv
    summary_all_metrics.csv
    paired_policy_deltas.csv
    diagnostics.json
    charts/00_headline_dashboard.png
```

## Claim discipline

This mode is a **real vLLM + LMCache exact-chunk cache-selection / speculative
prepopulation microbenchmark**. It tests target-access TTFT and whether the
benefit survives measured speculative-compute cost. It does not test
non-prefix CacheBlend fusion, answer quality, or a production coordinator-driven
L2-to-L1/GPU prefetch operation. Those require separately labeled experiments.
The bundled LongBench event targets are a controlled synthetic diagnostic,
split by document; they are not presented as logged production access traces.
