from __future__ import annotations


def legacy_attention_flops_saved(
    n_cached_tokens: int, n_new_tokens: int, hidden_size: int, num_layers: int
) -> float:
    """Exact formula used by the supplied simulator, with an honest name.

    It is an analytical attention-only proxy, not a hardware FLOP counter. The
    old ``n_heads`` argument was unused. Algebraically, the expression measures
    attention work avoided by not recomputing the cached prefix.
    """

    cached, new = int(n_cached_tokens), int(n_new_tokens)
    full_length = cached + new
    full_attention = 4 * int(hidden_size) * full_length**2
    reused_attention = 4 * int(hidden_size) * (new**2 + 2 * new * cached)
    return float((full_attention - reused_attention) * int(num_layers))


def approximate_prefix_reuse_flops_saved(
    n_cached_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    vocab_size: int = 0,
) -> dict[str, float]:
    """Approximate dense-transformer prefix work avoided by KV reuse.

    Reports terms separately so papers cannot silently present a proxy as a
    profiler measurement. The linear term covers Q/K/V/O and gated-MLP matrix
    multiplies for cached tokens; attention is the cached-prefix self-attention
    term. Embedding and normalization costs are intentionally omitted.
    """

    l = int(n_cached_tokens)
    d = int(hidden_size)
    m = int(intermediate_size)
    layers = int(num_layers)
    projections = layers * l * (8 * d * d + 6 * d * m)
    attention = layers * 2 * d * l * l
    logits = 2 * l * d * int(vocab_size) if vocab_size else 0
    return {
        "projection_mlp_flops_saved": float(projections),
        "attention_flops_saved": float(attention),
        "logit_flops_saved_if_all_prefix_logits_computed": float(logits),
        "total_without_logits": float(projections + attention),
    }

