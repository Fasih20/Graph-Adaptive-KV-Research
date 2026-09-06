# GraphKV repaired simulator

This package is a clean simulator implementation for checking GraphKV policy
mechanics before spending A100 time. It does **not** overwrite or reinterpret
the original CSVs.

## What is repaired

- The existing Top-M graph construction and two-hop scoring are frozen in
  `graphkv_sim/legacy_graph.py`; the boundary-safe constructor is separate.
- Structural edges never cross document boundaries.
- `PolicyContext` contains no future target. The target is revealed only after
  the policy returns its top-K prediction.
- Prediction recall, candidate coverage, cache residency, current-prefetch
  hits, prior residency, late prefetch, and eviction/rejection are separate.
- LRU capacity can be constrained by entries, model-aware KV bytes, or both.
- Cache lookup, demand miss, prefetch cost, and available overlap are explicit.
- The existing adaptive idea is split into offline-global, query-type oracle,
  online-from-scratch, and offline-warm-start online variants.
- Online learning uses a projected mistake-driven ranking update. It updates
  only on a reachable prediction miss—not on coverage misses or cache failures.
- Offline weights use training documents only; learning rate uses dev
  documents only; policies are reported on test documents only.
- Atomic checkpoints contain all result rows, exact LRU order/provenance, and
  online policy weights. Resume is tested against uninterrupted execution.
- A no-prefetch baseline, random, sequential, transferable relative-offset
  Markov, cosine, fixed graph, adaptive variants, and a clearly marked
  one-step target oracle are included.

## Important interpretation

The default grouped-synthetic trace is a **diagnostic**. Its targets still use
semantic/structural signals, so it is suitable for finding implementation bugs
and sensitivity patterns, not for an unbiased headline accuracy claim. For a
paper result, pass an independently produced sequential access trace with
`--trace-csv`.

The query-type adaptive policy is explicitly named an oracle because it assumes
the query type is known at prediction time. The deployable adaptive comparisons
are the global offline policy and the two online policies.

## Colab quick check

Use a fresh GPU runtime, upload and unzip the package, then:

```bash
%cd /content/graphkv_sim_repaired
!pip install -q -e ".[colab]"
!pytest -q
!python run_simulation.py --smoke --output-dir outputs/colab_smoke
!python validate_outputs.py outputs/colab_smoke
```

The smoke run uses `Qwen/Qwen2.5-0.5B-Instruct` configuration/tokenization, six
to eight LongBench documents, `K=3`, and at most 20 test events. It does not load
the causal-LM weights, so 16 GB VRAM is more than sufficient; the embedding
model is the main compute component.

Review:

```python
import pandas as pd
pd.read_csv("outputs/colab_smoke/summary.csv")
```

Run the cache/deadline stress check as a separate output:

```bash
!python run_simulation.py --stress-smoke --output-dir outputs/colab_stress
!python validate_outputs.py outputs/colab_stress
```

## Repaired 600-query comparison

The historical scripts used **nine**, not eight, K values and capacity 16. The
new compatibility command runs 600 held-out test events at
`K={3,4,5,6,8,10,12,14,16}` with an exact query mixture of 150 semantic, 150
structural, and 300 multi-hop events:

```bash
!python run_simulation.py \
  --full-comparison-600 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset hotpot \
  --output-dir outputs/full_600_qwen1_5b_hotpot
!python validate_outputs.py outputs/full_600_qwen1_5b_hotpot
```

This does not load the Qwen causal-model weights. The model name supplies the
tokenizer and architecture for token/KV-byte accounting, so it is suitable for
Colab even when the corresponding causal LM would not fit comfortably.

The command also creates 600 training and 200 development events on different
documents. Offline weights are fit on train, online learning rate on dev, and
all reported policies see the same 600 test events. The approximately
100k-character corpus scale and capacity 16 mirror the old experiment, but
document boundaries and held-out evaluation are repaired. Therefore compare
trends and rankings with the historical run; do not call the numerical values
an exact reproduction of the old invalid protocol.

If you have the old simulator CSV, align the closest comparable fields with:

```bash
!python compare_legacy_results.py \
  /content/old_results.csv \
  outputs/full_600_qwen1_5b_hotpot \
  --output outputs/legacy_vs_repaired_comparison.csv
```

The script maps old simulated cache hits to new residency rates. It deliberately
does not equate the old query-type adaptive condition with the new deployable
online learner; the closest old mapping is the explicitly labelled query-type
oracle baseline.

The two primary graph conditions are:

- `graph_fixed`: frozen `alpha=0.70`, `beta=0.30`, with no learning.
- `adaptive_online_warm`: starts from a train-only global weight vector and
  changes it after reachable top-K mistakes during the test stream.

`adaptive_offline_global` is retained as the necessary ablation: it starts from
the same learned weights but freezes them, isolating whether online updates add
anything beyond offline tuning. The query-type version remains labelled an
oracle diagnostic.

Only after tests and smoke results look structurally correct:

```bash
!python run_simulation.py \
  --max-documents 30 \
  --max-test-events 0 \
  --k 3 8 16 \
  --capacity-entries 32 \
  --output-dir outputs/pilot_30docs
```

Do not call that a real workload evaluation unless the trace is independent.

## Independent trace contract

CSV columns:

```text
event_id,session_id,step_id,document_id,current_chunk_id,target_chunk_id,query_type
```

`query_type` is optional and defaults to `unknown`. IDs must refer to the saved
`chunks.jsonl` for the same corpus. Train/dev/test splits are by whole document.

## Byte capacity and model awareness

Per-chunk KV size is:

```text
2 × layers × KV_heads × head_dim × tokens × bytes_per_element
```

The architecture is read from Hugging Face `AutoConfig`; it is not guessed from
the model name. Use `--capacity-gib` for byte-aware experiments. Supplying both
entry and byte capacity enforces both constraints.

## Latency and FLOPs

Default latency values are visible proxy assumptions, not measurements. For a
hardware-specific pilot, run:

```bash
!python calibrate_small_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output calibration.json
!python run_simulation.py --smoke \
  --cost-model-json calibration.json \
  --output-dir outputs/calibrated_smoke
```

The supplied code's old FLOP formula is preserved as
`legacy_attention_flops_saved`, but accurately labelled as an analytical
attention-only proxy. It algebraically cancels the new-token length and does
not use the old `n_heads` argument. A second function reports explicit
projection/MLP and attention components, still as theory—not profiler output.

## Prefetch timing modes

- `blocking`: every requested prefetch completes; any amount beyond the overlap
  window contributes exposed latency.
- `deadline`: only a prefix of requested transfers that fits fully inside the
  overlap window becomes resident. Correct predictions arriving later are
  logged as `late_prefetch`.

These are controlled simulator assumptions. Real vLLM/LMCache telemetry remains
necessary for a systems claim.

## Output contract

Every result row includes both policy and systems outcomes. Key fields are:

- `prediction_hit`, `candidate_coverage`, `candidate_universe_size`;
- `residency_hit`, `current_prefetch_hit`, `target_source`, `miss_reason`;
- completed IDs, evictions, oversize rejections, KV bytes;
- lookup/prefetch/demand/total proxy latency components;
- online update reason and weights before/after.

`run_manifest.json` records model shape, trace kind, split documents, fitted
weights, tuned learning rate, graph parameters, and seed. `summary.csv` contains
aggregate metrics. Checkpoints can safely resume an interrupted run.

`validate_outputs.py` audits row uniqueness, prediction/residency implications,
cache provenance, legal online updates, oracle behavior, and split isolation.

## Validation order

1. `pytest -q` (currently 16 tests).
2. `--smoke`, run `validate_outputs.py`, and inspect update rows.
3. `--stress-smoke` to exercise deadline and small-cache behavior.
4. `--full-comparison-600` for historical-shape comparison.
5. An independent trace pilot for actual paper evidence.
6. Only then run vLLM/LMCache or A100 jobs.
