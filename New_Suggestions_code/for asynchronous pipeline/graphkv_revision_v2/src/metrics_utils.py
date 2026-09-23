"""
metrics_utils.py
=================
PORTED VERBATIM from kv_cache_experiment_adaptive_v2(2).py — the theoretical
FLOPs-saved estimator used to populate flops_saved_cosine/graph/adaptive in
the CSV. Not explicitly named in the user's fidelity call-out list, but it
directly determines CSV column *values*, so it gets the same verbatim
treatment as the named functions.
"""


def estimate_flops_saved(n_cached, n_new, hidden_dim, n_heads, n_layers):
    L_full  = n_cached + n_new
    full    = 4 * hidden_dim * (L_full ** 2)
    cached  = 4 * hidden_dim * (n_new ** 2 + 2 * n_new * n_cached)
    return (full - cached) * n_layers
