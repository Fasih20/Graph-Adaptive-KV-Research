# Colab/local quick start

Use a separate CPU runtime or your laptop. Do not install this inside the
Kaggle notebook that is currently running vLLM.

## Colab cells

Upload and unzip `graphkv_api_quality_repaired_v1_0_0.zip`, then:

```python
%cd /content/graphkv_api_quality_repaired
!python -m pip install -q -r requirements.txt
```

Keep the API key out of notebook source and output:

```python
import getpass
import os

os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API key: ")
```

Check which model IDs this key can call:

```python
!python run_quality.py --list-models
```

Preflight one pinned stable model:

```python
!python run_quality.py \
  --model gemini-2.5-flash-lite \
  --preflight-only \
  --output-dir /content/outputs/gemini_hotpot60_K10
```

Run the smoke test:

```python
!bash scripts/run_gemini_smoke.sh \
  gemini-2.5-flash-lite \
  /content/outputs/gemini_hotpot_smoke
```

Only if `COMPLETE.json` appears in the smoke output, start the 60-question
run with a **new** output directory:

```python
!bash scripts/run_gemini_hotpot60.sh \
  gemini-2.5-flash-lite \
  /content/outputs/gemini_hotpot60_K10
```

If your AI Studio project shows a known RPM limit, replace the last command
with the direct command and add that value, for example `--rpm 15`. Otherwise
leave the default at zero and let HTTP 429 `Retry-After` control pacing.

On interruption or exit code 2, rerun the exact same command. Do not delete
`checkpoint.json` or `response_cache.json`; they prevent paid/quota-consuming
calls from being repeated.

Download the output only after `COMPLETE.json` exists:

```python
!zip -qr /content/gemini_hotpot60_K10.zip /content/outputs/gemini_hotpot60_K10

from google.colab import files
files.download("/content/gemini_hotpot60_K10.zip")
```

## Call count and expected time

The full default has 60 questions × 1 K × 4 policies = 240 maximum generation
calls. Identical-prompt caching can reduce this. A project capped at 5 RPM
cannot finish those calls in under 48 minutes; 15 RPM has a 16-minute lower
bound. Network time and throttling add to that. The train/dev fitting phase is
local and uses no Gemini calls.
