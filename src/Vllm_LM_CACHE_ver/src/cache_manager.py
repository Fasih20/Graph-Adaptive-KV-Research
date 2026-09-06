"""
cache_manager.py
=================
Bridge between prefetch policies and the running vLLM + LMCache (CacheBlend)
server. This is the piece that has no equivalent in the simulated harness —
the simulated harness never talked to a real inference server, it just
tracked hits/misses in KVCacheSimulator and charged a locally-measured
transformers forward pass for the latency number. Here, the *_sim_hit
bookkeeping is still done with the untouched KVCacheSimulator (so the CSV's
hit columns stay comparable to the simulated harness — see
implementation_plan.md's "Semantic Mismatch" note), but the *_ms latency
numbers come from real HTTP calls to vLLM.

Token-level chunk assembly follows the blend_kv_v1 pattern described in
implementation_plan.md §5: each chunk is tokenized independently (no special
tokens) and spliced with LMCACHE_BLEND_SPECIAL_STR separator tokens, so
LMCache's chunk-hash matching sees the same per-chunk token boundaries no
matter what order the chunks are concatenated in.
"""

import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import requests

from exp_config import (VLLM_BASE_URL, LMCACHE_HTTP_BASE_URL, LMCACHE_ENV,
                     MODEL_NAME, EXTEND_MAX_LEN)

logger = logging.getLogger("cache_manager")

BLEND_SEPARATOR_STR = LMCACHE_ENV["LMCACHE_BLEND_SPECIAL_STR"]


class CacheManager:
    def __init__(self, tokenizer, model_name: str = MODEL_NAME,
                 base_url: str = VLLM_BASE_URL,
                 lmcache_http_url: str = LMCACHE_HTTP_BASE_URL,
                 warm_pool_size: int = 8, request_timeout_s: float = 120.0):
        """
        tokenizer: a transformers.PreTrainedTokenizer for `model_name`,
                   loaded by the caller (vllm_launcher.py / benchmark_harness.py)
                   so this class doesn't own model-loading concerns.
        """
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.lmcache_http_url = lmcache_http_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        self._sep_ids = self.tokenizer(
            BLEND_SEPARATOR_STR, add_special_tokens=False)["input_ids"]
        self._warm_pool = ThreadPoolExecutor(max_workers=warm_pool_size,
                                              thread_name_prefix="cache-warm")
        self._session = requests.Session()

    # ── Token-level chunk assembly (blend_kv_v1 pattern) ───────────────────
    def build_prompt_token_ids(self, chunk_texts: List[str],
                                add_bos: bool = True) -> List[int]:
        """Tokenize each chunk independently, splice with separator token
        ids between every pair of adjacent chunks. Mirrors blend.py's
        build_prompt_token_ids so LMCache's chunk-hash lookup sees exactly
        the same per-chunk boundaries regardless of chunk order."""
        ids: List[int] = []
        bos = self.tokenizer.bos_token_id
        if add_bos and bos is not None:
            ids.append(bos)
        for i, text in enumerate(chunk_texts):
            if i > 0:
                ids.extend(self._sep_ids)
            ids.extend(self.tokenizer(text, add_special_tokens=False,
                                      truncation=True, max_length=EXTEND_MAX_LEN)["input_ids"])
        return ids

    # ── warm_chunks: off-critical-path cache population ────────────────────
    def warm_chunks(self, chunk_ids: List[int], chunk_store, block: bool = False):
        """Fires one tiny completion (max_tokens=1) PER chunk id, each chunk
        alone at token position 0 in its own request.

        CAUTION: this puts every chunk at position 0 when warmed, but a
        later query_with_chunks() call places all-but-the-first chunk at a
        DIFFERENT absolute token position (after the preceding chunks +
        separators). KV is position-dependent (rotary embeddings etc.), so
        that positional mismatch can silently prevent reuse even if the
        content is identical — this is the likely explanation if you're
        seeing cosine/graph/adaptive consistently SLOWER than cold with no
        improving trend as K grows (no reuse ever happening, just a bigger
        prompt being fully recomputed every time). Prefer warm_prefix()
        below, which doesn't have this problem. Kept here only in case you
        specifically want to test CacheBlend's position-INDEPENDENT
        reordering-reuse claim in isolation."""
        futures = []
        for cid in chunk_ids:
            text = chunk_store.text(cid)
            token_ids = self.build_prompt_token_ids([text])
            futures.append(self._warm_pool.submit(self._fire_and_forget, token_ids))
        if block:
            for f in futures:
                f.result()
        return futures

    def warm_prefix(self, chunk_ids: List[int], chunk_store, block: bool = True):
        """Sends the WHOLE context as ONE combined request, in the exact
        same chunk order and separator placement that query_with_chunks()
        will use for the same chunk_ids — so the prefix is byte-for-byte
        identical (same absolute token positions) between this warm call
        and the later timed query. This is standard prefix caching (vLLM
        supports this natively, independent of LMCache/CacheBlend), so it's
        the safer bet for actually observing a real speedup, and a good
        first checkpoint before trusting CacheBlend's fancier
        position-independent chunk-reordering reuse.

        block=True by default (unlike warm_chunks) because the whole point
        here is that the SAME exact prefix must already be resident before
        the timed call — "off critical path" only matters for the timed
        call itself, not for this one-time setup step."""
        if not chunk_ids:
            return None
        texts = chunk_store.texts(chunk_ids)
        token_ids = self.build_prompt_token_ids(texts)
        future = self._warm_pool.submit(self._fire_and_forget, token_ids)
        if block:
            future.result()
        return future

    def _fire_and_forget(self, token_ids: List[int]):
        try:
            self._session.post(
                f"{self.base_url}/completions",
                json={"model": self.model_name, "prompt": token_ids,
                      "max_tokens": 1, "temperature": 0},
                timeout=self.request_timeout_s,
            )
        except requests.RequestException as e:
            logger.warning(f"warm_chunks request failed (non-fatal): {e}")

    # ── query_with_chunks: the timed request ───────────────────────────────
    def query_with_chunks(self, query_text: str, context_chunk_ids: List[int],
                          chunk_store, max_tokens: int = 1):
        """
        Assembles [context chunks...] + [query_text] into one blend-tagged
        token sequence and measures TTFT (time to first streamed token) —
        the real-hardware analog of the simulated harness's extend_with_cache
        timing. max_tokens=1 by default: we care about prefill/TTFT, not
        generation, matching the canonical script's forward-pass-only timing.

        Returns dict: {ttft_ms, n_prompt_tokens}
        """
        context_texts = chunk_store.texts(context_chunk_ids) if context_chunk_ids else []
        all_texts = context_texts + [query_text]
        token_ids = self.build_prompt_token_ids(all_texts)

        t0 = time.perf_counter()
        ttft_ms = None
        try:
            with self._session.post(
                f"{self.base_url}/completions",
                json={"model": self.model_name, "prompt": token_ids,
                      "max_tokens": max_tokens, "temperature": 0, "stream": True},
                stream=True, timeout=self.request_timeout_s,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    # First non-empty SSE chunk marks first token received.
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    break
        except requests.RequestException as e:
            logger.error(f"query_with_chunks request failed: {e}")
            raise

        return {"ttft_ms": ttft_ms, "n_prompt_tokens": len(token_ids)}

    # ── flush_cache: reset LMCache state between ablation configs ──────────
    def flush_cache(self) -> bool:
        """POSTs to LMCache's http_server /cache/clear endpoint (clears all
        L1/CPU-resident KV). Confirm this endpoint name/port against your
        installed LMCache version — it comes from docs.lmcache.ai/mp/index.html
        (http_server.py: 'POST /cache/clear for clearing all KV cache data
        in L1 (CPU) memory'). This does not necessarily evict GPU-resident
        blocks in vLLM's own allocator; if you see cross-config contamination
        in results, that's the first thing to check."""
        try:
            resp = self._session.post(f"{self.lmcache_http_url}/cache/clear",
                                      timeout=self.request_timeout_s)
            resp.raise_for_status()
            logger.debug("  LMCache cache/clear OK")
            return True
        except requests.RequestException as e:
            logger.error(f"  flush_cache failed: {e} — results after this point "
                        f"may be contaminated by stale cache state")
            return False

    def health_check(self) -> bool:
        try:
            r1 = self._session.get(f"{self.base_url.rsplit('/v1', 1)[0]}/health", timeout=10)
            r2 = self._session.get(f"{self.lmcache_http_url}/healthcheck", timeout=10)
            return r1.ok and r2.ok
        except requests.RequestException:
            return False

    def close(self):
        self._warm_pool.shutdown(wait=True)
        self._session.close()
