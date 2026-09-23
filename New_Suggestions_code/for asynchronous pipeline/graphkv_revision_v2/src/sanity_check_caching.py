"""
sanity_check_caching.py
=========================
The most basic possible test of whether ANY KV reuse is happening at all,
decoupled from the K-sweep/policy machinery: send the exact same combined
prompt twice in a row and compare TTFT. If the second call isn't
meaningfully faster than the first, nothing downstream (cosine/graph/
adaptive vs. cold) can show a caching benefit either, no matter how correct
the retrieval-policy code is — this isolates "is the plumbing working" from
"which retrieval policy is better", which is what benchmark_harness.py
actually measures.

This also runs a SECOND test using warm_prefix() + a different trailing
query (rather than resending the literal same prompt) — that's closer to
the real benchmark's actual usage pattern (same context prefix, different
final question each time) and is the one that actually needs to show a
speedup for the K-sweep results to be meaningful.

TIMING ALONE IS NOT PROOF: a documented, real-world failure mode for
LMCache's blend/CacheBlend path is a request that succeeds, returns a
normal-looking response, and shows NO speedup because the server only ever
stored KV and never actually reused it — see e.g.
github.com/LMCache/LMCache/issues/1936, where exactly this was reported
against the /v1/completions + input_ids pattern used here. So don't just
read the percentages below: this script also diffs vLLM's own real
/metrics prefix-cache counters (ground truth from vLLM's scheduler, not
a simulation), pulls LMCache's /status endpoint, and greps its server log
for store/lookup/hit activity around each test — you get direct evidence
either way instead of inferring purely from a millisecond delta.

Run this BEFORE trusting run_cosine_baseline.sh's numbers.
"""

import sys
import logging
import argparse
from typing import Optional

import requests
from transformers import AutoTokenizer

from exp_config import MODEL_NAME, INTEGRATION_PATH, RESULTS_DIR
from vllm_launcher import VLLMLauncher
from chunk_store import ChunkStore
from cache_manager import CacheManager

logger = logging.getLogger("sanity_check_caching")


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    return logging.getLogger("sanity_check_caching")


def _print_lmcache_status(cache_manager: CacheManager, label: str):
    """Best-effort dump of LMCache's own /status endpoint. Schema isn't
    guaranteed stable across versions (it's documented as an
    operator/debug endpoint, not a monitoring API), so this just prints the
    raw JSON for a human to read rather than trying to parse specific
    fields out of it."""
    try:
        r = cache_manager._session.get(f"{cache_manager.lmcache_http_url}/status", timeout=10)
        r.raise_for_status()
        logger.info(f"  [{label}] LMCache /status:")
        body = r.text
        logger.info(f"    {body[:1500]}{' ...(truncated)' if len(body) > 1500 else ''}")
    except requests.RequestException as e:
        logger.warning(f"  [{label}] could not reach LMCache /status ({e}) — "
                       f"if this keeps failing, http_server.py may not be up "
                       f"on {cache_manager.lmcache_http_url}")


def _tail_lmcache_log(label: str, n: int = 15):
    """Best-effort tail of the LMCache server's own log for lines that look
    cache-related (store/hit/miss/lookup/blend), so you can see what the
    server itself claims happened rather than only inferring it from
    latency. Only exists in MP mode — in-process mode logs into
    vllm-inprocess.log instead, mixed in with vLLM's own output."""
    log_path = RESULTS_DIR / "lmcache-server.log"
    if not log_path.exists():
        logger.info(f"  [{label}] {log_path} not found — normal for "
                    f"--integration_path inprocess (check vllm-inprocess.log instead)")
        return
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        keywords = ("hit", "miss", "store", "lookup", "blend", "cache")
        interesting = [ln for ln in lines if any(kw in ln.lower() for kw in keywords)]
        tail = (interesting or lines)[-n:]
        logger.info(f"  [{label}] last {len(tail)} cache-related line(s) from {log_path}:")
        for ln in tail:
            logger.info(f"    {ln}")
        if not interesting:
            logger.warning(f"  [{label}] no store/hit/lookup/blend-looking lines found "
                           f"anywhere in the log yet — that itself is worth investigating")
    except OSError as e:
        logger.warning(f"  [{label}] could not read {log_path}: {e}")


def _print_real_cache_delta(cache_manager: CacheManager, label: str, before: Optional[dict], after: Optional[dict]):
    """Diffs two get_prefix_cache_counters() snapshots and prints what
    actually changed, in vLLM's own words — this is the real, ground-truth
    counterpart to the speedup-% numbers above, which are inferred from
    timing alone. If before/after show a hits delta > 0, vLLM's own
    scheduler is confirming reuse happened; if hits stay at 0 while queries
    go up, nothing was served from cache no matter what the timing looked
    like."""
    if before is None or after is None:
        logger.warning(f"  [{label}] couldn't get a real /metrics reading — "
                       f"see the warning above")
        return
    keys = sorted(set(before) | set(after))
    if not keys:
        logger.warning(f"  [{label}] /metrics reachable but no prefix_cache_* "
                       f"lines found at all — check the metric names for your "
                       f"vLLM version against docs.vllm.ai/en/stable/design/metrics.html")
        return
    logger.info(f"  [{label}] real vLLM /metrics delta (cumulative counters, this window only):")
    for k in keys:
        b, a = before.get(k, 0.0), after.get(k, 0.0)
        if a != b:
            logger.info(f"    {k}: {b:.0f} -> {a:.0f}  (+{a - b:.0f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hotpot")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--integration_path", default=INTEGRATION_PATH, choices=["mp", "inprocess"])
    p.add_argument("--n_context_chunks", type=int, default=6)
    args = p.parse_args()

    logger = setup_logging()

    with VLLMLauncher(model_name=args.model, integration_path=args.integration_path) as launcher:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        cache_manager = CacheManager(tokenizer=tokenizer, model_name=args.model)
        try:
            chunk_store = ChunkStore(args.dataset).prepare()
            context_ids = list(range(args.n_context_chunks))
            query_a = chunk_store.text(args.n_context_chunks)
            query_b = chunk_store.text(args.n_context_chunks + 1)

            logger.info("=" * 70)
            logger.info("TEST 1: literal identical prompt sent twice")
            logger.info("=" * 70)
            cache_manager.flush_cache()
            before = cache_manager.get_prefix_cache_counters()
            r1 = cache_manager.query_with_chunks(query_a, context_ids, chunk_store)
            r2 = cache_manager.query_with_chunks(query_a, context_ids, chunk_store)
            after = cache_manager.get_prefix_cache_counters()
            logger.info(f"  1st call (cold):    {r1['ttft_ms']:.1f} ms  ({r1['n_prompt_tokens']} tokens)")
            logger.info(f"  2nd call (repeat):  {r2['ttft_ms']:.1f} ms  ({r2['n_prompt_tokens']} tokens)")
            speedup1 = (r1["ttft_ms"] - r2["ttft_ms"]) / r1["ttft_ms"] * 100
            logger.info(f"  speedup: {speedup1:+.1f}%  "
                        f"{'(looks like SOME caching is happening)' if speedup1 > 15 else '(NO meaningful caching detected — check LMCache/vLLM setup before trusting the K-sweep)'}")
            _print_real_cache_delta(cache_manager, "TEST 1", before, after)
            _print_lmcache_status(cache_manager, "after TEST 1")
            _tail_lmcache_log("after TEST 1")

            logger.info("=" * 70)
            logger.info("TEST 2: warm_prefix(context) once, then two DIFFERENT queries "
                        "against that same context — this is what benchmark_harness.py actually does")
            logger.info("=" * 70)
            cache_manager.flush_cache()
            before = cache_manager.get_prefix_cache_counters()
            cache_manager.warm_prefix(context_ids, chunk_store, block=True)
            r3 = cache_manager.query_with_chunks(query_a, context_ids, chunk_store)
            r4 = cache_manager.query_with_chunks(query_b, context_ids, chunk_store)
            after = cache_manager.get_prefix_cache_counters()
            logger.info(f"  query A (after warm_prefix): {r3['ttft_ms']:.1f} ms")
            logger.info(f"  query B (same prefix, different query): {r4['ttft_ms']:.1f} ms")
            _print_real_cache_delta(cache_manager, "TEST 2", before, after)
            _print_lmcache_status(cache_manager, "after TEST 2")
            _tail_lmcache_log("after TEST 2")

            logger.info("=" * 70)
            logger.info("TEST 3: cold baseline for comparison — flush, then query WITHOUT "
                        "any warm step at all, same context+query as query A above")
            logger.info("=" * 70)
            cache_manager.flush_cache()
            before = cache_manager.get_prefix_cache_counters()
            r5 = cache_manager.query_with_chunks(query_a, context_ids, chunk_store)
            after = cache_manager.get_prefix_cache_counters()
            logger.info(f"  query A, no warm (true cold): {r5['ttft_ms']:.1f} ms")
            speedup2 = (r5["ttft_ms"] - r3["ttft_ms"]) / r5["ttft_ms"] * 100
            logger.info(f"  warm_prefix speedup vs true cold: {speedup2:+.1f}%  "
                        f"{'(caching IS helping)' if speedup2 > 15 else '(still no meaningful benefit — see notes below)'}")
            _print_real_cache_delta(cache_manager, "TEST 3", before, after)
            _print_lmcache_status(cache_manager, "after TEST 3")
            _tail_lmcache_log("after TEST 3")

            logger.info("=" * 70)
            if speedup1 <= 15 and speedup2 <= 15:
                logger.warning(
                    "Neither test showed a meaningful speedup. Read the /metrics deltas, "
                    "/status dumps, and log tails printed above FIRST — if they show no "
                    "hits/store/lookup activity at all, this is a caching-plumbing problem "
                    "(check the CacheBlend flags in vllm_launcher.py: --kv-transfer-config, "
                    "--block-size against your installed LMCache version's current examples), "
                    "not something benchmark_harness.py's policy code can fix. If they DO show "
                    "store/hit activity but timing still doesn't reflect it, that's worth "
                    "reporting upstream — see github.com/LMCache/LMCache/issues/1936 for a "
                    "similar report against this same /v1/completions + input_ids pattern.")
            else:
                logger.info("Both tests show a real speedup — the plumbing looks alive. Run "
                           "the full K-sweep and then scripts/analyze_results.sh to check "
                           "hit-vs-miss latency within each policy.")
        finally:
            cache_manager.close()


if __name__ == "__main__":
    main()
