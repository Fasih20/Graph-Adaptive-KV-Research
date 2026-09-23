"""Tokenizer-bounded document chunking, independent of cache block sizes."""

from __future__ import annotations


def split_text_by_tokens(tokenizer, text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """Split one document into reproducible tokenizer-bounded windows."""
    chunk_tokens = int(chunk_tokens)
    overlap_tokens = int(overlap_tokens)
    if chunk_tokens < 1:
        raise ValueError("document_chunk_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError(
            "document_chunk_overlap_tokens must be non-negative and smaller "
            "than document_chunk_tokens"
        )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    step = chunk_tokens - overlap_tokens
    chunks: list[str] = []
    for start in range(0, len(token_ids), step):
        window = token_ids[start:start + chunk_tokens]
        if not window:
            break
        decoded = tokenizer.decode(
            window,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if decoded:
            # Decoding then encoding can very rarely change token count at a
            # boundary. Truncate once more so the upper bound is true for the
            # exact text later sent to vLLM.
            exact_ids = tokenizer.encode(
                decoded,
                add_special_tokens=False,
                truncation=True,
                max_length=chunk_tokens,
            )
            decoded = tokenizer.decode(
                exact_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if decoded:
                chunks.append(decoded)
        if start + chunk_tokens >= len(token_ids):
            break
    return chunks
