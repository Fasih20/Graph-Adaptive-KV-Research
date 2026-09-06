"""
config.py
=========
Single source of truth for constants used across the real-cache experiment.

Everything in the "PORTED — DO NOT CHANGE" block below is copied verbatim
from kv_cache_experiment_adaptive_v2(2).py so that build_graph / generate_pairs /
the prefetch policies see the exact same numbers they were validated against
in the simulated harness. If you need to change one of these for a real
experiment, do it consciously and re-read graph_algorithms.py's docstrings
first — several of these constants are load-bearing for tie-breaking and
graph topology, not just tunable knobs.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# PORTED — DO NOT CHANGE (identical to kv_cache_experiment_adaptive_v2(2).py)
# ─────────────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
SEM_THRESHOLD   = 0.55
SEM_WEIGHT      = 0.7          # fixed-graph default alpha
STRUCT_WEIGHT   = 0.3          # fixed-graph default beta
MAX_DEGREE      = 100
RANDOM_SEED     = 42
N_PAIRS         = 600
SEPARATOR       = "\n\n"       # used only for the *simulated-timing* text join;
                                # NOT the same as LMCACHE_BLEND_SPECIAL_STR below
PRIMARY_MAX_LEN = 512
EXTEND_MAX_LEN  = 256
GRAPH_CONSTRUCTION = "topm"    # matches final_v2.py / the canonical script's default
TOP_M           = 20

DATASET_CONFIGS = {
    "hotpot":     {"hf_name": "THUDM/LongBench", "split": "hotpotqa",        "text_key": "context", "label": "HotpotQA"},
    "2wiki":      {"hf_name": "THUDM/LongBench", "split": "2wikimqa",        "text_key": "context", "label": "2WikiMultiHopQA"},
    "musique":    {"hf_name": "THUDM/LongBench", "split": "musique",         "text_key": "context", "label": "MuSiQue"},
    "multifield": {"hf_name": "THUDM/LongBench", "split": "multifieldqa_en", "text_key": "context", "label": "MultiFieldQA-en"},
}

ABLATION_CONFIGS = [
    {"K": 3,  "CACHE_CAPACITY": 16},
    {"K": 6,  "CACHE_CAPACITY": 16},
    {"K": 8,  "CACHE_CAPACITY": 16},
    {"K": 12, "CACHE_CAPACITY": 16},
    {"K": 16, "CACHE_CAPACITY": 16},
]

# The script's own built-in fallback (used only when no grid-search JSON is
# supplied). Kept here ONLY for parity/regression tests against the
# canonical file — the real Phase-4 run does NOT use this dict, see
# HARDCODED_ADAPTIVE_WEIGHTS below.
FALLBACK_ADAPTIVE_WEIGHTS = {
    "semantic":   {"alpha": 0.90, "beta": 0.10},
    "structural": {"alpha": 0.40, "beta": 0.60},
    "multi-hop":  {"alpha": 0.70, "beta": 0.30},
}

# ─────────────────────────────────────────────────────────────────────────
# Phase-4 weights actually used, as supplied by the user (from a completed
# grid search against real-cache-comparable prior results). These are NOT
# the same as FALLBACK_ADAPTIVE_WEIGHTS above:
#   - semantic   0.90/0.10  (identical — semantic is never learned in either case)
#   - structural 0.35/0.65  (differs from the script's 0.40/0.60 fallback)
#   - multi-hop  0.35/0.65  (differs from the script's 0.70/0.30 fallback)
# See the chat message accompanying this delivery for the explicit
# call-out of this difference — it is intentional, not a bug.
# ─────────────────────────────────────────────────────────────────────────
HARDCODED_ADAPTIVE_WEIGHTS = {
    "semantic":   {"alpha": 0.90, "beta": 0.10},
    "structural": {"alpha": 0.35, "beta": 0.65},
    "multi-hop":  {"alpha": 0.35, "beta": 0.65},
}

MODEL_CONFIGS = {
    "Qwen2.5-7B":   {"hidden_dim": 3584, "n_heads": 28, "n_layers": 28},
    "Qwen2.5-14B":  {"hidden_dim": 5120, "n_heads": 40, "n_layers": 48},
    "Qwen2.5-1.5B": {"hidden_dim": 1536, "n_heads": 12, "n_layers": 28},
    "Qwen2.5-3B":   {"hidden_dim": 2048, "n_heads": 16, "n_layers": 36},
    "default":      {"hidden_dim": 4096, "n_heads": 32, "n_layers": 32},
}

def get_model_config(model_name: str) -> dict:
    for key, cfg in MODEL_CONFIGS.items():
        if key.lower() in model_name.lower():
            return cfg
    return MODEL_CONFIGS["default"]


# ─────────────────────────────────────────────────────────────────────────
# Real-hardware run configuration (new for this phase — not present in the
# simulated harness, since it never talked to a real vLLM/LMCache process)
# ─────────────────────────────────────────────────────────────────────────
MODEL_NAME       = os.environ.get("EXP_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_MODEL_LEN    = int(os.environ.get("EXP_MAX_MODEL_LEN", "8192"))
VLLM_HOST        = os.environ.get("VLLM_HOST", "localhost")
VLLM_PORT        = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_BASE_URL    = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

LMCACHE_SERVER_HOST = os.environ.get("LMCACHE_SERVER_HOST", "localhost")
LMCACHE_SERVER_PORT = int(os.environ.get("LMCACHE_SERVER_PORT", "5555"))
# The lmcache server's FastAPI http_server (health/status/cache-clear) —
# a SEPARATE port from the ZMQ mp-server port above. Confirm this against
# your installed LMCache version; --http-port is the flag as of the
# docs.lmcache.ai/mp/index.html http_server.py module.
LMCACHE_HTTP_PORT   = int(os.environ.get("LMCACHE_HTTP_PORT", "9000"))
LMCACHE_HTTP_BASE_URL = f"http://{LMCACHE_SERVER_HOST}:{LMCACHE_HTTP_PORT}"

# CacheBlend chunk_size MUST equal vLLM --block-size * 4 (per LMCache's
# CacheBlendEngineSpec validation). 256 / 4 = 64.
VLLM_BLOCK_SIZE = 64

# Integration path: "mp" (default, no vLLM patch — production recommended)
# or "inprocess" (5-line vLLM patch, pinned commit, single process).
INTEGRATION_PATH = os.environ.get("EXP_INTEGRATION_PATH", "mp")

# enforce_eager is mandatory for CacheBlend regardless of path (see
# implementation_plan.md §3 / LMCache issue #2639). Never set this False
# for a CacheBlend-enabled run.
ENFORCE_EAGER = True

# CacheBlend env vars (from implementation_plan.md §5 / blend_kv_v1 example).
# NOTE: one community walkthrough of this same example renders the
# separator with surrounding spaces (" # # ") in its lmcache_config.yaml
# rather than "# #". We use the plan's literal value below since that's
# what was pulled directly from the example source, but this is worth a
# byte-for-byte check against your installed LMCache version's example
# before a real run — a mismatched separator string silently changes
# chunk-boundary tokenization and would quietly invalidate blend results.
LMCACHE_ENV = {
    "LMCACHE_ENABLE_BLENDING":       "True",
    "LMCACHE_USE_LAYERWISE":         "True",
    "LMCACHE_BLEND_SPECIAL_STR":     "# #",
    "LMCACHE_BLEND_CHECK_LAYERS":    "1",
    "LMCACHE_BLEND_RECOMPUTE_RATIOS": "0.15",
    "LMCACHE_CHUNK_SIZE":            "256",
}

RESULTS_DIR = Path(os.environ.get("EXP_RESULTS_DIR", "./results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
