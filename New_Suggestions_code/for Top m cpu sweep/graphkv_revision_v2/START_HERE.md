# GraphKV revision v2 — correctness first

This is a new experiment version. Do not merge its rows into archived result
tables. The old `run_repaired.py` is retained for reference; use the new entry
points below. Single GPU only. No CacheBlend, cross-GPU transfer, or lost-in-the-middle experiment is enabled.

## What the early audit actually found

The supplied final Qwen manifest calls the experiment an exact-chunk-access
microbenchmark. The runner populates `[candidate.text]` independently and later
serves `[target.text]` independently with the same prefix construction. It does
not splice these caches into a combined RAG context. Prediction recall depends
on selected IDs, not numerical KV correctness.

`audit/archived_qwen_check.json` contains a check of the supplied ZIP:
18,000 rows, 1,200 paired event groups, zero prompt-hash mismatches and zero
one-token output-hash mismatches across policies/K within paired groups;
13,155 requests report retrieval evidence. This is useful evidence, NOT proof
of full numerical cache correctness or concatenated-context reuse.

The final packaged vLLM `PolicyAdapter` already uses `OfflineAdaptivePolicy`
for both fixed and offline policies. The earlier statement that these two
final real-run policies necessarily use different pruning was incorrect.
Historical graph and API fixed policies do differ. We retain those functions
and provide drift comparisons instead of overwriting their interpretation.

## Mandatory order

1. Prepare the workload on CPU.
2. Run CPU drift analysis (before expensive experiment expansion).
3. Run the separate GPU reuse gate. Stop if it is failed OR inconclusive.
4. Only after inspecting the report, run a small execution smoke/pilot.
5. Expand a frozen experiment matrix after observing measured runtime.

The gate has six isolated runtime launches (cold/warm for three cases). It
compares generated tokens and their log-probabilities, requires positive
retrieval evidence for the repeated prefix, and requires a shifted prefix to
miss except for any genuinely shared initial tokens. Numerical tolerance is
0.001 absolute token log-probability. A failed gate may reflect missing log
telemetry or numerical variability as well as invalid reuse: inspect the report;
do not increase tolerance or bypass the gate to obtain a pass.

## Environment

Use the known working Kaggle environment if possible. On a new GPU session:

```bash
python -m pip install -r requirements-revision.txt
python -m unittest discover -s tests -p test_revision.py -v
```

The retained candidate profile pins vLLM 0.27.0 / LMCache 0.5.4, matching the
previous run. Do not independently upgrade torch/torchaudio or install nightly
CUDA wheels. GPU/CLI compatibility still needs validation in Kaggle. This
package was tested on CPU here; no real GPU execution is claimed.

The notebook `GraphKV_Revision_Kaggle.ipynb` covers setup, backup, preparation,
drift, gate, and a guarded smoke command. Internet and model access are needed.

```bash
python run_revision.py prepare --output-dir /kaggle/working/graphkv_v2_hotpot \
  --dataset hotpot --model Qwen/Qwen2.5-1.5B-Instruct --events 60
python run_revision.py drift --output-dir /kaggle/working/graphkv_v2_hotpot \
  --top-m 5 10 20 --k 3 6 10
python run_revision.py gate --output-dir /kaggle/working/graphkv_v2_hotpot
```

Read `gate/reuse_gate.json` before running inference experiments. Changing the
model, runtime settings, installed dependency versions, or gate implementation
invalidates its environment binding. No bypass switch is provided.

### The requested ablations are implemented

| Policy | Semantic edges | Structure | Hops | Weights |
|---|---|---|---|---|
| cosine | Full cosine ranking | No | Direct | None |
| semantic_only | Symmetric Top-M | No | 1 | 1/0 |
| structure | Symmetric Top-M | Yes | 1 | 0.7/0.3 |
| two_hop | Symmetric Top-M | Yes | 2 | 0.7/0.3 |
| offline | Same | Same | 2 | Fitted on train |
| online | Same | Same | 2 | Train initialization; dev-selected step |

No second degree-pruning rule is introduced in V2. Top-M ties use chunk ID.
All two-hop variants share the same feature-wise path blend and ranking.
Offline selection searches 21 weight pairs. Online step selection uses only
development events. M/K grids are explicit CLI lists, not automatically tuned
on test performance. `drift_report.json` compares V2 with final-run scoring
at equal weights and also the historical fixed scorer. It does NOT assert
submitted headline recall is unchanged: exact replay also needs the original
embeddings/trace/version. No F1 is inferred from recall.

### Execution smoke (run only after the gate)

```bash
python run_revision.py run --output-dir /kaggle/working/graphkv_v2_hotpot \
  --top-m 20 --k 6 --policies no_prefetch cosine two_hop \
  --modes async --lead-ms 0 --block-events 30 --repetitions 1
```

This is six fresh isolated blocks for 60 events. It is not the full pilot
matrix. The default maximum is 64 isolated blocks; excessive requests fail
before server startup. Changing the run matrix requires a NEW output folder.
Prepare that folder and run its gate; do not edit manifest hashes.

### Timing and workload scope

Prediction occurs once the current chunk ID is available. One background
request at a time can overlap the current foreground request. Target arrival
is scheduled after the current request finishes plus `--lead-ms`; that delay
is a controlled variable, not a measured production arrival distribution.
The same rule applies to every policy. The target submits without waiting
for the entire preparation queue. Unstarted candidates are cancelled at
demand; an already-running request drains after target timing ends. Its cost
is retained in cycle time and throughput. No GPU preemption is claimed.

`arrival_to_first_token_ms` includes target submission delay/queueing;
`application_with_policy_ms` includes policy, current request, actual lead
time and target completion. `cycle_including_drain_ms` additionally includes
leftover background work. `target_service_ttft_ms` starts at target submission.
Metric collection and backup happen between measured cycles. This is a
closed-loop controlled trace benchmark, not an offered-load production server
benchmark. Synthetic trace generation and LongBench context-record boundaries
are retained and explicitly labeled; they must not be described as true
within-article structure or observed user retrieval sequences.

Readiness is recorded as preparation completion before demand, not as a
guarantee of cache residence. Global backend counters cannot attribute hits
to overlapping requests, so `backend_cache_hit` is null. KV bytes are a model
dimension estimate, not measured PCIe traffic. FLOPs remain theoretical; no
hardware FLOP measurement has been added.

## Full-prompt local answer quality

```bash
python run_quality_managed.py --dataset hotpot \
  --model Qwen/Qwen2.5-1.5B-Instruct --gpu 0 \
  --output-dir /kaggle/working/quality_v2_hotpot \
  --questions 60 --top-m 5 --k 10 --include-legacy-fixed
```

Use `--dataset 2wiki` or `--dataset musique` and a new output directory.
This wrapper prepares inputs before starting the model, then uses a plain
vLLM server with no KV-transfer connector and native prefix caching disabled.
It evaluates answer quality, NOT cache acceleration. Every selected context
is fully prefilled by normal model computation; independent chunk KV states
are never substituted into the full prompt. No CacheBlend support is implied.

The evaluator uses the same V2 policy code, equalized reference-token budgets,
chat-template application, saved token IDs, EM, normalized answer-token F1,
fully delivered support recall and target hits. Hotpot/2Wiki nodes are labeled
sentences; MuSiQue nodes are labeled paragraphs. Consequently MuSiQue structure
may be absent when each article contributes one paragraph. Support metrics
are not directly interchangeable across these annotation granularities.
F1 follows the existing API evaluator normalization (not a claim of byte-for-byte
official benchmark scoring). Compare matched questions/budgets when comparing
local and Gemini policy effects. The API used a different model/tokenizer and
possibly a different prompt wrapper; absolute scores are not model-controlled.

Online quality replay uses gold supporting-evidence feedback AFTER each
prediction. It is explicitly oracle-supervised evaluation, not deployable
feedback from an unanswered live question. Generation is independent of future
feedback; precomputed selections reproduce that ordered replay. Each answer
is atomically saved and skipped on resume. Failed/unfinished answers never
receive fabricated scores or a complete marker. `--include-legacy-fixed`
adds real generations for the old fixed ranking, allowing direct F1 drift
measurement. Without it, only selection/support drift can be reported.

## Automatic Google Drive backups

Install rclone from its official distribution. On your own laptop, create a
Google Drive remote named `gdrive` with `rclone config`, authorizing in your
browser. Prefer your own Google OAuth client as recommended by rclone, and
the `drive.file` scope if backing up only files created by this tool.
Official instructions: https://rclone.org/drive/

Put the CONTENTS of your rclone config into a PRIVATE Kaggle Secret named
`RCLONE_CONFIG_TEXT`. Do not upload it as a public dataset or include it in
outputs. The notebook writes it outside the output tree with permission 0600.
Enable that secret for the notebook. Set:

```python
import os
os.environ['GRAPHKV_BACKUP_REMOTE'] = 'gdrive:GraphKV_Backups/v2_hotpot_run01'
```

Use a separate remote folder for every experiment. The runner snapshots after
preparation, gate completion, every five events by default, every completed
block, and on a handled interruption. Change `--backup-every` as needed.
Each archive has per-file SHA-256 hashes. Upload is followed by a downloaded
content check; failure stops further work at the saved local checkpoint.
No Drive files are deleted (`rclone copyto`, never `sync`). OAuth credentials
stay outside the output directory and snapshot. Uploads can be expensive;
their time is excluded from inference metrics and occurs between events.

### Restore

```bash
rclone copyto gdrive:GraphKV_Backups/v2_hotpot_run01/CHOSEN-SNAPSHOT.zip /kaggle/working/recovered.zip
python restore_snapshot.py /kaggle/working/recovered.zip /kaggle/working/graphkv_v2_hotpot
```

The destination must be empty. Reuse this exact code package and rerun the
same command/configuration. Completed independent blocks are verified and
skipped. An interrupted systems block starts again cold, with its online
weights reset to its documented initial state. Its partial events remain in
the diagnostic snapshot but are not mixed with the restarted block's rows.
Cache contents are not assumed to survive a VM shutdown. If the environment
changed, rerun the gate; scientific runtime changes require a new experiment.

No system can promise zero loss after an abrupt shutdown: work since the last
verified remote snapshot may need repeating. With defaults, a running block
may repeat at most 30 events; reduce `--block-events` to 10 for cheaper recovery
at the cost of more model startups and a different block-reset protocol.

## Analysis and acceptance

`run_revision.py summary --output-dir ...` reports means, p95, throughput and
paired deltas from complete blocks. It averages repeated executions before
block resampling. Confidence intervals are descriptive if blocks share source
records; they are not 1,200 independent-question intervals. Local quality
reports question-paired intervals for fixed policies and withholds an online
CI for a single dependent replay sequence. Stronger population claims need
independent sessions/documents and repeated sequences.

First inspect: `reuse_gate.json`, `drift_report.json`, `backup_status.json`.
Then inspect: `headline_results.csv`, `paired_deltas.csv`, and quality outputs.
No pass, speedup, significance, or cross-platform guarantee is pre-recorded.
