# GraphKV repaired Gemini answer-quality evaluator

This is a CPU/API experiment. It does **not** launch vLLM or LMCache and it is
safe to run on a laptop, in another Colab CPU runtime, or while a separate
Kaggle GPU job is running.

It answers a different question from the systems benchmark:

- vLLM + LMCache: does a policy make the next chunk reusable and reduce
  ready-cache TTFT on real hardware?
- This evaluator: does the context selected by that policy improve the final
  answer's exact match (EM) or token F1?

Do not merge API response time with vLLM TTFT. The script records Gemini
network latency only as a diagnostic.

## What was repaired

- Supports structured HotpotQA and 2WikiMultiHopQA rows while preserving
  document, sentence, and supporting-fact structure.
- Structural edges connect sentences only inside the same document.
- Compares `cosine`, fixed graph (0.7/0.3), leakage-safe fitted offline graph,
  and prequential online graph.
- Fits offline weights on a disjoint training sample using supporting-fact
  recall. It does not use API-generated answers or test labels.
- Tunes online learning rate on a disjoint development sample. On test data,
  the online policy selects context first, then sees the supporting target and
  updates for the next question.
- Enforces equal context budgets using one pinned reference tokenizer and
  records both reference and provider token counts.
- Randomises condition order deterministically within each question/K.
- Checkpoints after every successful policy call. Rerunning the identical
  command resumes without repeating completed calls.
- Caches identical model+prompt responses, which can save calls when two
  policies select identical context.
- Removes the old compulsory 5-RPM limiter. `--rpm 0` makes calls immediately
  and still honours Gemini's HTTP 429 `Retry-After`. Set `--rpm` to the active
  limit displayed for your project in AI Studio if you want proactive pacing.
- Pins a stable model ID by default instead of a moving `-latest` alias.

## Install

```bash
python -m pip install -r requirements.txt
export GEMINI_API_KEY='your-key'
```

Never paste the key into a notebook output or commit it to a ZIP.

List the models accessible to this key:

```bash
python run_quality.py --list-models
```

Model availability and free-tier limits are project-specific. Check the
active RPM, input-TPM, and requests-per-day values in Google AI Studio before
a long run. Use a stable model ID returned by the command; avoid `-latest` and
preview aliases for a paper experiment.

## Run in this order

Preflight (zero generation calls):

```bash
python run_quality.py \
  --model gemini-2.5-flash-lite \
  --preflight-only \
  --output-dir outputs/gemini_hotpot60_K10
```

Three-question smoke test (12 calls before identical-prompt cache savings):

```bash
bash scripts/run_gemini_smoke.sh \
  gemini-2.5-flash-lite \
  outputs/gemini_hotpot_smoke
```

Held-out 60-question K=10 run (240 maximum calls):

```bash
bash scripts/run_gemini_hotpot60.sh \
  gemini-2.5-flash-lite \
  outputs/gemini_hotpot60_K10
```

Equivalent 2WikiMultiHopQA run with the same scientific settings:

```bash
bash scripts/run_gemini_dataset60.sh \
  2wiki \
  gemini-2.5-flash-lite \
  outputs/gemini_2wiki60_K10
```

The generic wrapper accepts `hotpot` or `2wiki`. MuSiQue and MultiFieldQA are
not accepted because their available schemas do not provide the same
sentence-level supporting-fact structure required by this evaluator's offline
and online learning protocol.

To use a newer stable Flash-Lite model that your key can access, replace the
model string everywhere and use a new output directory. Never mix two model
IDs in one output directory.

If AI Studio says the model permits 15 RPM, add `--rpm 15`. At 5 RPM, 240
calls have a hard lower bound of 48 minutes; the old evaluator's fixed 5-RPM
default is why a 60-question/three-policy run could take more than 36 minutes
before normal response and retry time. With `--rpm 0`, this evaluator avoids
that artificial delay and slows only when the API tells it to.

## Outputs

- `question_policy_results.csv`: one complete row per question, K, and policy.
- `summary.csv`: exact mean EM, F1, evidence recall, target hit rate, token
  counts, diagnostic API time, candidate coverage, and online update rate.
- `paired_deltas_vs_cosine.csv`: paired question-level mean differences and
  95% bootstrap confidence intervals.
- `fitted_policy.json`: fitted offline weights and online learning rate with
  their complete train/dev search tables.
- `run_manifest.json`: immutable scientific configuration and hash.
- `checkpoint.json`: atomic resume state after every successful call.
- `response_cache.json`: prompt-hash cache; no API key is stored.
- `COMPLETE.json`: written only when every expected event exists.

Exit code 2 means the run stopped safely because of quota or interruption.
Rerun the identical command. A changed scientific argument requires a new
output directory, preventing accidental mixing of conditions.

## Metrics

- **EM:** 1 only when the normalised generated answer exactly matches a gold
  answer; otherwise 0.
- **Token F1:** overlap between normalised answer tokens and the best gold
  answer, balancing precision and recall.
- **Supporting-fact recall:** fraction of labelled evidence sentences fully
  delivered after the equal-token truncation. The pre-budget selection recall
  is also retained, separating retrieval quality from budget truncation.
- **Feedback-target hit:** whether the deterministic supporting sentence used
  for offline/online policy feedback is included.
- **Reference context tokens:** the equal budget enforced by the pinned Qwen
  tokenizer. Provider input tokens are separately reported because Gemini uses
  a different tokenizer.
- **API latency:** wall time of fresh Gemini HTTP generation calls. It is not
  a cache-speed measurement and should not be used as the systems headline.

Run `pytest -q` before collection. For a paper claim, report paired confidence
intervals, not only mean EM/F1.
