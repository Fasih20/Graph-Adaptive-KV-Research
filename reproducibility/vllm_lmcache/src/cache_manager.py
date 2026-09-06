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
        content is identical UNLESS CacheBlend's non-prefix reuse is
        actually engaging. That position-independence is exactly the
        property correctness_check.py needs (it wants to force a blend
        across chunk boundaries, not a trivial same-sequence prefix hit —
        see that module's docstring), which is what it uses this for.
        benchmark_harness.py, by contrast, uses warm_prefix() below for its
        off-critical-path warming, so this function does NOT explain
        cosine/graph/adaptive vs. cold behavior in the main K-sweep — see
        benchmark_harness.py's module docstring for that bug instead."""
        futures = []
        for cid in chunk_ids:
            text = chunk_store.text(cid)
            token_ids = self.build_prompt_token_ids([text])
            futures.append(self._warm_pool.submit(self._fire_and_forget, token_ids))
        if block:
            n_failed = sum(1 for f in futures if not f.result())
            if n_failed:
                logger.warning(f"warm_chunks: {n_failed}/{len(futures)} chunk warm-ups "
                               f"failed — those chunks are NOT actually resident, silently")
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
            ok = future.result()
            if not ok:
                # block=True means the caller is about to time a request
                # assuming this content is already resident. Silently
                # returning here would make a FAILED warm-up look
                # indistinguishable from a successful one, and the timed
                # call downstream would just measure a cold/partial-cold
                # prompt while being recorded as if prefetching had a fair
                # chance to help. Raise so the caller's retry logic (if any)
                # can see this instead of it being swallowed.
                raise RuntimeError(
                    f"warm_prefix: failed to warm {len(chunk_ids)} chunk(s) "
                    f"(ids={chunk_ids}) — see the preceding warning for the "
                    f"underlying HTTP error")
        return future

    def _fire_and_forget(self, token_ids: List[int]) -> bool:
        """Returns True on success. This is the only thing standing between
        "we intended to pre-populate the cache" and "the timed call quietly
        recomputes everything" — a warm-up that fails here must not look the
        same as one that succeeded, so (unlike a plain try/except swallow)
        this checks the HTTP status and reports it back to the caller."""
        try:
            resp = self._session.post(
                f"{self.base_url}/completions",
                json={"model": self.model_name, "prompt": token_ids,
                      "max_tokens": 1, "temperature": 0},
                timeout=self.request_timeout_s,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning(f"warm request failed: {e}")
            return False

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
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    # OpenAI-compatible SSE payloads are "data: {...}".
                    # Some proxies/servers also emit blank-looking keep-alive
                    # or ":"-prefixed comment lines to hold the connection
                    # open on a slow-to-first-token request; those are
                    # non-empty but carry no token, so timing off "first
                    # non-blank line" (the old check) can under-report TTFT
                    # if any ever show up. Require an actual data line.
                    # Compared as bytes (not decode_unicode=True) so this
                    # doesn't depend on requests' encoding auto-detection,
                    # which falls back to guessing for text/* responses
                    # without an explicit charset — the "data:"/"[DONE]"
                    # markers themselves are always plain ASCII regardless.
                    if not raw_line.startswith(b"data:"):
                        continue
                    payload = raw_line[len(b"data:"):].strip()
                    if payload == b"[DONE]":
                        break
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    break
        except requests.RequestException as e:
            logger.error(f"query_with_chunks request failed: {e}")
            raise

        if ttft_ms is None:
            # Stream ended (or hit [DONE]) without ever producing a token —
            # e.g. the prompt silently exceeded --max-model-len, or the
            # server returned an error body over a 200 stream. Raise rather
            # than hand back {"ttft_ms": None, ...}, which would previously
            # crash later with an opaque `TypeError` inside round(None, 4)
            # instead of a message that says what actually happened.
            raise RuntimeError(
                f"query_with_chunks: no token received for a "
                f"{len(token_ids)}-token prompt — check the vLLM server log "
                f"(possible causes: prompt longer than --max-model-len, or "
                f"a server-side error returned inside the SSE stream)")

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
            logger.info("  LMCache cache/clear OK")
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

    # ── Real (not simulated) prefix-cache counters ──────────────────────────
    def get_prefix_cache_counters(self) -> Optional[dict]:
        """Scrapes vLLM's own /metrics (Prometheus text format) for its
        native prefix-cache hit/query counters -- e.g. vllm:prefix_cache_hits
        and vllm:prefix_cache_queries, both TOKEN-granularity cumulative
        counters (confirmed current names per docs.vllm.ai/en/stable/design/
        metrics.html; exact suffix/labels can vary by version, e.g. a
        _total suffix or a per-engine label). This is real, vLLM-reported
        ground truth -- NOT the KVCacheSimulator's theoretical prediction
        that populates *_sim_hit / flops_saved_* in the main CSV.

        Returns a dict of {full_metric_line_name: value} for every metric
        line whose name contains "prefix_cache", or None if /metrics
        couldn't be reached. Deliberately returns whatever it finds rather
        than assuming one exact name -- take a reading before and after the
        window you care about and diff matching keys; don't read these as
        an absolute hit rate on their own since they're cumulative since
        server start.

        Known gap: these track vLLM's OWN internal GPU block-hash cache.
        Content resolved via LMCache's external connector (e.g. served
        from L1/CPU tier after missing vLLM's local pool) may or may not
        be reflected in these same counters depending on your LMCache
        version's integration -- cross-check against LMCache's own
        /status and server log (see sanity_check_caching.py) rather than
        treating this alone as the final word either.
        """
        metrics_url = f"{self.base_url.rsplit('/v1', 1)[0]}/metrics"
        try:
            resp = self._session.get(metrics_url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"  could not reach {metrics_url}: {e}")
            return None

        out = {}
        for line in resp.text.splitlines():
            if not line or line.startswith("#"):
                continue
            # Prometheus text format: `metric_name{labels} value` or
            # `metric_name value` -- split off the value (last token) and
            # keep everything before it (name + labels) as the key.
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            name_and_labels, value_str = parts
            if "prefix_cache" not in name_and_labels.lower():
                continue
            try:
                out[name_and_labels] = float(value_str)
            except ValueError:
                continue
        return out

    def close(self):
        self._warm_pool.shutdown(wait=True)
        self._session.close()
