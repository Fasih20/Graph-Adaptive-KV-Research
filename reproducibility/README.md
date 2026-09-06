# GraphKV reproducibility package

Code for **Graph-Guided and Adaptive KV-Cache Prefetching for Retrieval-Augmented Generation**.

## Start here

The package contains three current implementations and three matching notebooks:

| Experiment | Code | Notebook | What it measures |
|---|---|---|---|
| Simulated cache | `simulation/` | `notebooks/01_simulated_cache.ipynb` | Prediction, simulated residency/traffic, and analytical latency proxies |
| Gemini answer quality | `api_quality/` | `notebooks/02_gemini_api_quality.ipynb` | EM, token F1, support recall, and feedback-target hit |
| Real vLLM + LMCache | `vllm_lmcache/` | `notebooks/03_vllm_lmcache.ipynb` | Real ready-cache TTFT, request time, population cost, telemetry, and cache controls |

`legacy_previous_version/` preserves the historical scripts that produced the archived six-model simulated results. They are clearly separated from the repaired runners used for the current experiments.

## Recommended order

1. Open the relevant notebook from `notebooks/`.
2. Run its installation and test cells.
3. Run the smoke or preflight cells.
4. Start the full experiment only after validation passes.
5. Keep the output directory unchanged when resuming an interrupted run.

Each code directory has a detailed README describing the method, metrics, commands, output files, resumability, and interpretation. API keys are read from `GEMINI_API_KEY`; no key is included in this package.

## Versions

- Repaired simulator: 0.2.0
- Repaired Gemini evaluator: 1.1.0
- Repaired vLLM + LMCache runner: 2.3.2
- Package assembled: 2026-09-06

## Paper configurations captured in the notebooks

- Gemini: HotpotQA and 2WikiMultiHopQA, 120 train / 40 development / 60 held-out questions, `K=10`, `M=5`, 768 reference-token context budget.
- Real cache: Qwen2.5-1.5B-Instruct on one T4, 600 train / 600 development / 600 test accesses, `K={6,10,16}`, two repetitions, seed 43, Top-M 20, and a 0.5 GiB LMCache L1 capacity.
- Simulation: repaired 600-event comparison over `K={3,4,5,6,8,10,12,14,16}`.

The notebooks are reconstructed reproducibility notebooks matching the recorded commands and manifests. The original historical pipeline notebook remains under `legacy_previous_version/`.
