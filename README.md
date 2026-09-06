# GraphKV: current paper reproduction

The current repaired implementations and three runnable notebooks are in
[`reproducibility/`](reproducibility/README.md). That directory contains the
simulated-cache, Gemini answer-quality, and real vLLM + LMCache experiments.
The older `src/` directory is retained because it produced the archived
six-model results reported in the paper.

---

# KV-Cache Graph Prefetching

Code accompanying the paper **"KC_CACHE_GRAPH_PREFETCHING"** — a graph-guided, adaptive KV-cache chunk-prefetching strategy for multi-hop QA, evaluated both via a controlled simulated-timing harness (local GPU, open-weight models) and via API-served models (Gemini / OpenAI / Groq / OpenRouter).

## Overview

Given a long multi-hop context split into chunks, the system builds a similarity graph over chunks and prefetches likely-needed chunks into a simulated KV cache ahead of time, instead of encoding everything cold. Two prefetch policies are compared against a cold/no-cache baseline:

- **Fixed-graph (cosine/threshold or top-m)** — prefetch neighbors above a similarity threshold or within each chunk's top-m nearest neighbors.
- **Adaptive** — same graph, but edge weights are a learned combination `alpha * semantic_similarity + beta * structural_score`, with `alpha`/`beta` tuned per question type (`semantic`, `structural`, `multi-hop`) via grid search.

Datasets: **HotpotQA, 2WikiMultiHopQA, MuSiQue, MultiFieldQA-en** (all pulled from `THUDM/LongBench` on the Hugging Face Hub).
Models evaluated: **Qwen2.5 (1.5B/3B/7B), Llama-3.2 (1B/3B), Gemma-3 (1B)** locally, plus **Gemini, OpenAI/Groq/OpenRouter-served models** via API.

> The v4 fix in `kv_cache_experiment_baseline.py`: earlier versions unconditionally re-ran `extend_with_cache()` even on a cache hit, so cosine/graph conditions could never look better than cold. v4 skips the real forward pass on a hit and records near-zero latency instead — this is the change that makes the local-GPU numbers trustworthy. Worth mentioning explicitly if Sir asks why numbers changed between draft versions.

## Repository Structure

```
src/
├── adaptive_grid_search.py            # Grid search over (alpha, beta) per question type -> adaptive_weights_<tag>_<dataset>.json
├── kv_cache_experiment_baseline.py     # Local-GPU baseline: cold vs cosine/threshold vs top-m graph prefetch (no adaptive weights)
├── kv_cache_experiment_adaptive.py     # Local-GPU adaptive run: same as baseline + learned (alpha, beta) weights per q-type
└── api_eval/
    ├── exp_config.py                  # Shared constants (chunk size, seeds, dataset configs, adaptive weight fallbacks, vLLM/LMCache env config)
    ├── graph_algorithms.py            # KVCacheSimulator + graph/prefetch policy functions (ported verbatim — do not edit, see its docstring)
    ├── llm_providers.py               # Unified client for Gemini / OpenAI-compatible (openai, groq, openrouter) / Anthropic
    ├── qa_scoring.py                  # SQuAD-style normalization, Exact Match, token F1
    ├── qa_quality_eval.py             # Runs one (provider, dataset, top_m) config end-to-end, scores answers, checkpoints per-question
    ├── top_m_sweep.py                 # Runs qa_quality_eval across multiple --top_m values, one at a time, resumable
    └── determinism_check.py           # Repeats the same config --n_reps times to check API-response determinism

Results/    # Generated CSVs, checkpoints, plots, and full per-model text reports
Paper and Latex Project files/   # Paper PDF + LaTeX source
```

## Setup

```bash
pip install torch transformers huggingface_hub sentence-transformers \
            langchain-text-splitters scikit-learn scipy numpy pandas requests
```

(Python 3.10+ recommended. GPU + CUDA build of `torch` needed for the local-model scripts.)

### API keys (for everything under `src/api_eval/`)

Set whichever provider(s) you're using as environment variables:

```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."        # when --provider openai --backend openai
export GROQ_API_KEY="..."          # when --provider openai --backend groq
export OPENROUTER_API_KEY="..."    # when --provider openai --backend openrouter
export ANTHROPIC_API_KEY="..."     # when --provider anthropic
```

Datasets are fetched automatically on first run via `huggingface_hub.hf_hub_download` from `THUDM/LongBench` — no manual download needed, and no HF token required for that public repo.

## Usage

All commands below are run from inside `src/` (or `src/api_eval/` for the API scripts), matching how they were actually run.

### 1. Local-GPU baseline (cold vs. cosine/graph prefetch, no learned weights)

```bash
python kv_cache_experiment_baseline.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --datasets hotpot multifield musique 2wiki \
    --graph_construction topm \
    --top_m 20 \
    --output_dir ./results_baseline \
    --torch_dtype float16
```
- `--graph_construction`: `threshold` (default) or `topm`
- `--torch_dtype`: `auto` / `float16` / `bfloat16`
- `--legacy_timing`: reverts to the old (buggy) always-re-encode timing, for comparison only — don't use for real numbers.

### 2. Local-GPU adaptive run (full 2-step pipeline: grid search → weighted run)

```bash
DATASETS=("hotpot" "multifield" "musique" "2wiki")
MODEL_REPO="meta-llama/Llama-3.2-3B-Instruct"
MODEL_TAG="Llama3.2-3B"

# Step 1: learn (alpha, beta) per question type from an existing baseline results CSV
for ds in "${DATASETS[@]}"; do
    python adaptive_grid_search.py \
        --csv_dir results \
        --csv_file "./results/results_meta_llama_Llama_3.2_3B_Instruct_${ds}_hitaware_topm20.csv" \
        --dataset_key "$ds" \
        --model_tag "$MODEL_TAG" \
        --model_repo "$MODEL_REPO" \
        --graph_construction topm \
        --top_m 20
done

# Step 2: re-run with the learned adaptive weights
for ds in "${DATASETS[@]}"; do
    python kv_cache_experiment_adaptive.py \
        --model "$MODEL_REPO" \
        --datasets "$ds" \
        --graph_construction topm \
        --top_m 20 \
        --adaptive_weights "./adaptive_results/adaptive_weights_${MODEL_TAG}_${ds}.json" \
        --output_dir ./results_v4_adaptive_llama3b \
        --torch_dtype float16
done
```
`adaptive_grid_search.py` requires an **existing** baseline/hit-aware results CSV to grid-search over (that's what `--csv_file`/`--csv_dir` point at) — run the baseline script for that model/dataset first if it doesn't exist yet. `kv_cache_experiment_adaptive.py` also supports `--load_in_8bit` / `--load_in_4bit` for larger models.

### 3. API-based evaluation (Gemini / OpenAI / Groq / OpenRouter)

Set the relevant API key first, then, from `src/api_eval/`:

```bash
# Single top_m, single provider
python qa_quality_eval.py --provider gemini --dataset hotpot --top_m 12 --K 12

# Sweep across top_m values (recommended — this is what produced Results/API_ver)
python top_m_sweep.py --provider gemini --model gemini-3.5-flash-lite --full
python top_m_sweep.py --provider openai --backend groq --full --top_m_values 12,20

# Determinism check: repeat the same config n_reps times
python determinism_check.py --provider gemini --n_reps 2
```
- `--provider openai` **requires** `--backend openai|groq|openrouter`.
- `--cheap` (default) = 8 questions/top_m (~24 calls); `--full` = 60 questions/top_m (~180 calls) — mind provider free-tier quotas.
- Runs are **resumable**: each `(provider, dataset, top_m)` config checkpoints per-question progress to `results/checkpoint_<tag>.json`. If a run stops on a quota error, re-run the *exact same command* later — the sweep won't skip ahead to the next `top_m`, it resumes the interrupted one first. On successful completion the checkpoint for that config is deleted automatically.
- `--K` must be > 8 (fixed at 12 across the reported results).

## Reproducing the results in `Results/`

- `Results/all_csvs/` — one CSV per `(model, dataset)` from the local-GPU adaptive/hit-aware runs, named `results_[adaptive_]<model>_<dataset>_hitaware_topm20.csv`.
- `Results/{Gemma,Llama,Qwen}/` — full per-model text + plot reports from those local runs.
- `Results/API_ver/details/` — per-question checkpoints and `qa_quality_<provider>_<dataset>_top<N>_...csv` outputs from the `top_m_sweep.py` runs above; `summary_table.*` and the `.png` plots are aggregated from those.

## Notes for review

- `src/api_eval/graph_algorithms.py` is **ported verbatim** from the original simulated-harness script on purpose, so its prefetch-policy behavior stays byte-identical between the simulated and real-cache code paths — don't refactor it without re-validating against `tests/test_policy_fidelity.py` if that exists in your copy.
- `exp_config.py`'s `HARDCODED_ADAPTIVE_WEIGHTS` (structural 0.35/0.65, multi-hop 0.35/0.65) intentionally differ from the script's own `FALLBACK_ADAPTIVE_WEIGHTS` (structural 0.40/0.60, multi-hop 0.70/0.30) — the hardcoded values are the actual grid-search output used for the Phase-4 results, not a bug.
- `exp_config.py` also carries vLLM/LMCache environment config (`LMCACHE_*`, `VLLM_*`) for a real-hardware CacheBlend integration path — this isn't exercised by any script above; it's scaffolding for a further real-serving experiment, not part of the reported results.

## Citation

```bibtex
@article{TODO_citation_key,
  title   = {KC Cache Graph Prefetching},
  author  = {TODO},
  year    = {2026},
  note    = {Preprint}
}
```
