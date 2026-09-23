"""
test_policy_fidelity.py
========================
Proves graph_algorithms.py is behaviorally IDENTICAL to the canonical
kv_cache_experiment_adaptive_v2(2).py, by extracting the original functions
straight out of that file's AST (so we don't need torch/transformers/etc.
installed just to prove the pure-Python graph/prefetch logic matches) and
diffing outputs against our ported module across many random corpora,
including adversarial cases designed to expose tie-breaking / ordering bugs
(near-duplicate embeddings, so cosine/graph scores collide and insertion
order actually matters).

Run: python3 tests/test_policy_fidelity.py
"""
import ast
import sys
import time
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ORIGINAL_SRC_CANDIDATES = [
    Path("/mnt/user-data/uploads/kv_cache_experiment_adaptive_v2.py"),
    Path("/mnt/user-data/uploads/kv_cache_experiment_adaptive_v2_2_.py"),
    Path(__file__).resolve().parent.parent / "reference" / "kv_cache_experiment_adaptive_v2.py",
]
ORIGINAL_SRC = next((p for p in ORIGINAL_SRC_CANDIDATES if p.exists()), ORIGINAL_SRC_CANDIDATES[-1])
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import graph_algorithms as ported  # our ported module

logger = logging.getLogger("fidelity_test")
logger.addHandler(logging.NullHandler())


def load_original_functions():
    """AST-extract the pure-python functions/classes from the canonical
    file, without importing torch/transformers/etc."""
    src = ORIGINAL_SRC.read_text()
    tree = ast.parse(src)
    wanted_funcs = {"build_graph", "get_prefetch_cosine", "get_prefetch_graph",
                    "get_prefetch_adaptive", "generate_pairs"}
    wanted_classes = {"KVCacheSimulator"}

    ns = {
        "np": np, "sp": sp, "time": time, "logging": logging,
        "OrderedDict": OrderedDict,
        "SEM_THRESHOLD": 0.55, "SEM_WEIGHT": 0.7, "STRUCT_WEIGHT": 0.3,
        "MAX_DEGREE": 100,
    }
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in wanted_funcs)
            or (isinstance(n, ast.ClassDef) and n.name in wanted_classes)]
    found = {n.name for n in body}
    missing = (wanted_funcs | wanted_classes) - found
    assert not missing, f"Could not find in original source: {missing}"
    mod = ast.Module(body=body, type_ignores=[])
    exec(compile(mod, "<original>", "exec"), ns)
    return ns


def make_corpus(n, dim, seed, near_duplicate_frac=0.0):
    """Synthetic embeddings. near_duplicate_frac controls how many chunks
    are near-exact duplicates of another chunk's embedding, to force score
    ties and stress-test tie-break/insertion-order fidelity."""
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, dim)).astype(np.float64)
    n_dupe = int(n * near_duplicate_frac)
    for _ in range(n_dupe):
        src_i = rng.integers(0, n)
        dst_i = rng.integers(0, n)
        emb[dst_i] = emb[src_i] + rng.normal(scale=1e-9, size=dim)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def cosine_sim(emb):
    return emb @ emb.T


def compare_lists(a, b, ctx):
    assert a == b, f"MISMATCH [{ctx}]: ported={a}  original={b}"


def run_case(n, dim, seed, dupe_frac, graph_construction, top_m):
    ctx0 = f"n={n} dim={dim} seed={seed} dupe={dupe_frac} gc={graph_construction} top_m={top_m}"
    emb = make_corpus(n, dim, seed, dupe_frac)
    sim = cosine_sim(emb)

    orig = load_original_functions()

    adj_p, raw_p, _ = ported.build_graph(emb, sim, logger, graph_construction, top_m)
    adj_o, raw_o, _ = orig["build_graph"](emb, sim, logger, graph_construction, top_m)

    assert (adj_p.toarray() == adj_o.toarray()).all(), f"adj_matrix MISMATCH [{ctx0}]"
    assert raw_p == raw_o, f"raw_edges MISMATCH [{ctx0}]"

    rng = np.random.default_rng(seed + 999)
    weights_variants = [
        {"semantic": {"alpha": 0.9, "beta": 0.1},
         "structural": {"alpha": 0.35, "beta": 0.65},
         "multi-hop": {"alpha": 0.35, "beta": 0.65}},
        {"semantic": {"alpha": 0.9, "beta": 0.1},
         "structural": {"alpha": 0.4, "beta": 0.6},
         "multi-hop": {"alpha": 0.7, "beta": 0.3}},
    ]

    for trial in range(40):
        pri = int(rng.integers(0, n))
        k = int(rng.integers(1, min(n, 20)))
        qtype = rng.choice(["semantic", "structural", "multi-hop"])
        aw = weights_variants[trial % len(weights_variants)]

        c_p = ported.get_prefetch_cosine(sim, pri, k)
        c_o = orig["get_prefetch_cosine"](sim, pri, k)
        compare_lists(c_p, c_o, f"cosine pri={pri} k={k} [{ctx0}]")

        g_p = ported.get_prefetch_graph(adj_p, pri, k)
        g_o = orig["get_prefetch_graph"](adj_o, pri, k)
        compare_lists(g_p, g_o, f"graph pri={pri} k={k} [{ctx0}]")

        a_p = ported.get_prefetch_adaptive(raw_p, pri, k, qtype, aw)
        a_o = orig["get_prefetch_adaptive"](raw_o, pri, k, qtype, aw)
        compare_lists(a_p, a_o, f"adaptive pri={pri} k={k} qtype={qtype} [{ctx0}]")

    # generate_pairs — full sequence must match exactly (shared RNG stream)
    pairs_p = ported.generate_pairs(n, 200, sim, 42, logger)
    pairs_o = orig["generate_pairs"](n, 200, sim, 42, logger)
    assert pairs_p == pairs_o, f"generate_pairs MISMATCH [{ctx0}]"

    # KVCacheSimulator — behavioral check (LRU + prefetch semantics)
    sim_p = ported.KVCacheSimulator(8)
    sim_o = orig["KVCacheSimulator"](8)
    seq = np.random.default_rng(seed + 1).integers(0, n, size=50)
    for cid in seq:
        hp = sim_p.access(int(cid))
        ho = sim_o.access(int(cid))
        assert hp == ho, f"KVCacheSimulator.access MISMATCH [{ctx0}]"
    assert list(sim_p.cache.items()) == list(sim_o.cache.items())

    print(f"  OK  {ctx0}")


def main():
    cases = [
        dict(n=40,  dim=16, seed=1,  dupe_frac=0.0,  graph_construction="topm",      top_m=20),
        dict(n=40,  dim=16, seed=2,  dupe_frac=0.3,  graph_construction="topm",      top_m=20),  # ties
        dict(n=60,  dim=8,  seed=3,  dupe_frac=0.5,  graph_construction="topm",      top_m=10),  # heavy ties, small top_m
        dict(n=25,  dim=32, seed=4,  dupe_frac=0.0,  graph_construction="threshold", top_m=20),
        dict(n=25,  dim=32, seed=5,  dupe_frac=0.4,  graph_construction="threshold", top_m=20),
        dict(n=120, dim=16, seed=6,  dupe_frac=0.0,  graph_construction="topm",      top_m=20),  # MAX_DEGREE pruning kicks in
        dict(n=8,   dim=4,  seed=7,  dupe_frac=0.0,  graph_construction="topm",      top_m=20),  # tiny corpus edge case
    ]
    print("Running fidelity checks against original canonical source ...")
    for case in cases:
        run_case(**case)
    print("\nALL FIDELITY CHECKS PASSED — graph_algorithms.py matches the canonical "
          "kv_cache_experiment_adaptive_v2(2).py byte-for-byte on every tested case, "
          "including near-duplicate-embedding tie-break stress cases and MAX_DEGREE pruning.")


if __name__ == "__main__":
    main()
