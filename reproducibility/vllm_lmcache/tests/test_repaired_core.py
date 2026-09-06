from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_policies import OnlineAdaptivePolicy
from chunk_store import model_kv_bytes
from token_chunking import split_text_by_tokens
from isolated_runtime import IsolatedRuntime, RuntimeConfig
from real_cache_client import RealCacheClient, metric_delta, parse_prometheus
from repaired_benchmark import (
    ByteLRU,
    SUMMARY_NUMERIC_FIELDS,
    _format_headline_table,
    _headline_rows,
    _log_token_total,
    _pooled_summary_rows,
)
from sanity_controls import _named_metric_sum
from run_repaired import estimate_main_workload
import runtime_env
from runtime_env import child_environment, collect_cli_help


class FakeTokenizer:
    bos_token_id = 1

    def encode(self, text, **kwargs):
        limit = kwargs.get("max_length")
        values = [2 + (ord(character) % 61) for character in text]
        if kwargs.get("add_special_tokens", True):
            values = [self.bos_token_id, *values]
        return values[:limit] if limit else values


class WindowTokenizer:
    def encode(self, text, **kwargs):
        values = [int(value) for value in text.split()] if text.strip() else []
        limit = kwargs.get("max_length")
        return values[:limit] if kwargs.get("truncation") and limit else values

    def decode(self, values, **kwargs):
        return " ".join(map(str, values))


def runtime_config():
    return RuntimeConfig(
        model="example/model",
        model_revision="abc123",
        gpu_id="1",
        vllm_host="127.0.0.1",
        vllm_port=8000,
        lmcache_host="127.0.0.1",
        lmcache_port=5555,
        lmcache_http_port=8080,
        lmcache_prometheus_port=9090,
        max_model_len=4096,
        block_size=16,
        chunk_size=16,
        gpu_memory_utilization=0.8,
        l1_size_gb=1,
        startup_timeout_s=10,
        seed=42,
    )


def test_prometheus_parser_and_delta():
    before = parse_prometheus("# HELP x\na_total 2\nb_total{kind=\"x\"} 4\n")
    after = parse_prometheus("a_total 5\nb_total{kind=\"x\"} 4\n")
    assert before == {"a_total": 2.0, 'b_total{kind="x"}': 4.0}
    assert metric_delta(before, after) == {"a_total": 3.0}


def test_named_metric_sum_excludes_histogram_buckets():
    metrics = {
        'vllm:request_prefill_kv_computed_tokens_sum{engine="0"}': 384.0,
        'vllm:request_prefill_kv_computed_tokens_count{engine="0"}': 1.0,
        'vllm:request_prefill_kv_computed_tokens_bucket{le="500"}': 1.0,
    }
    assert _named_metric_sum(
        metrics, "vllm:request_prefill_kv_computed_tokens_sum"
    ) == 384.0


def test_lmcache_log_token_fallback_counts_only_requested_operation():
    lines = [
        "Stored 96 tokens in 0.004 seconds",
        "Retrieved 32 tokens in 0.001 seconds",
        "Retrieved 64 tokens in 0.002 seconds",
    ]
    assert _log_token_total(lines, "retriev") == 96
    assert _log_token_total(lines, "stor") == 96


def test_byte_lru_is_byte_bounded():
    cache = ByteLRU(10)
    assert cache.insert(1, 6) == []
    assert cache.insert(2, 6) == [1]
    assert cache.bytes == 6
    assert cache.contains(2)
    assert cache.insert(3, 11) == []
    assert not cache.contains(3)


def test_pooled_means_are_event_weighted_across_repetitions():
    template = {name: 0.0 for name in SUMMARY_NUMERIC_FIELDS}
    rows = []
    for repetition, values in ((0, [10.0]), (1, [20.0, 30.0, 40.0])):
        for index, latency in enumerate(values):
            rows.append(
                {
                    **template,
                    "policy": "cosine",
                    "k": 3,
                    "repetition": repetition,
                    "query_type": "semantic",
                    "ready_cache_ttft_ms": latency,
                    "prediction_hit": 1.0,
                    "prefetch_bytes_completed": 1024 ** 2,
                }
            )
    pooled = _pooled_summary_rows(rows)
    overall = next(row for row in pooled if row["query_type"] == "all")
    assert overall["events"] == 4
    assert overall["repetitions"] == 2
    assert overall["mean_ready_cache_ttft_ms"] == 25.0
    headline = _headline_rows(pooled)
    assert headline[0]["mean_completed_prefetch_mib"] == 1.0
    table = _format_headline_table(headline)
    assert "ready_TTFT_ms" in table
    assert "25.0000" in table


def test_workload_estimate_counts_real_candidate_requests():
    estimate = estimate_main_workload(
        ("no_prefetch", "cosine", "adaptive_offline_global"),
        (3, 6),
        events=100,
        repetitions=2,
    )
    assert estimate["main_isolated_process_pairs"] == 12
    assert estimate["current_plus_target_requests"] == 2400
    assert estimate["candidate_population_request_upper_bound"] == 3600
    assert estimate["total_inference_request_upper_bound"] == 6000


def test_model_kv_bytes_supports_composite_text_config():
    text = type(
        "TextConfig",
        (),
        {
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "hidden_size": 1024,
            "head_dim": 128,
            "num_hidden_layers": 12,
        },
    )()
    composite = type("CompositeConfig", (), {"text_config": text})()
    assert model_kv_bytes(composite, token_count=10) == 2 * 12 * 2 * 128 * 10 * 2


def test_document_chunking_is_token_bounded_with_exact_overlap():
    tokenizer = WindowTokenizer()
    chunks = split_text_by_tokens(
        tokenizer,
        " ".join(map(str, range(10))),
        chunk_tokens=4,
        overlap_tokens=1,
    )
    assert chunks == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]
    assert all(len(tokenizer.encode(chunk)) <= 4 for chunk in chunks)


def test_document_chunking_rejects_invalid_overlap():
    tokenizer = WindowTokenizer()
    try:
        split_text_by_tokens(tokenizer, "1 2 3", chunk_tokens=3, overlap_tokens=3)
    except ValueError as exc:
        assert "smaller" in str(exc)
    else:
        raise AssertionError("invalid overlap was accepted")


def test_online_policy_updates_only_after_miss():
    edges = {
        0: [(1, 0.9, 0.0), (2, 0.1, 1.0)],
        1: [(0, 0.9, 0.0)],
        2: [(0, 0.1, 1.0)],
    }
    policy = OnlineAdaptivePolicy(edges, (0.9, 0.1), eta=0.5)
    prediction = policy.predict(0, 1)
    assert prediction.ids == [1]
    update = policy.observe(2, prediction)
    assert update["updated"]
    assert update["weights_after"][1] > update["weights_before"][1]


def test_single_gpu_launch_contract(tmp_path):
    cfg = runtime_config()
    runtime = IsolatedRuntime(cfg, tmp_path)
    lm_help = " ".join([
        "--host", "--port", "--http-port", "--chunk-size", "--l1-size-gb",
        "--eviction-policy", "--engine-type", "--supported-transfer-mode",
        "--instance-id", "--prometheus-port", "--metrics-sample-rate",
    ])
    vllm_help = " ".join([
        "--kv-transfer-config", "--tensor-parallel-size", "--pipeline-parallel-size",
        "--gpu-memory-utilization", "--no-enable-prefix-caching", "--block-size",
        "--seed", "--revision", "--enforce-eager",
    ])
    lm = runtime._lmcache_command(lm_help, "run-1")
    vllm = runtime._vllm_command(vllm_help)
    assert lm[lm.index("--chunk-size") + 1] == "16"
    assert lm[lm.index("--engine-type") + 1] == "default"
    assert vllm[vllm.index("--tensor-parallel-size") + 1] == "1"
    assert "--no-enable-prefix-caching" in vllm
    connector = json.loads(vllm[vllm.index("--kv-transfer-config") + 1])
    assert connector["kv_connector_module_path"] == "lmcache.integration.vllm.lmcache_mp_connector"
    assert child_environment("1")["CUDA_VISIBLE_DEVICES"] == "1"


def test_vllm_paged_help_is_collected(monkeypatch):
    calls = []

    def fake_capture(args, timeout_s=30.0):
        calls.append(args)
        if args[-1] == "--help=all":
            output = " ".join([
                "--kv-transfer-config", "--tensor-parallel-size",
                "--pipeline-parallel-size", "--gpu-memory-utilization",
                "--no-enable-prefix-caching",
            ])
            return {"command": args, "returncode": 0, "output": output}
        return {"command": args, "returncode": 0, "output": "basic serve help"}

    monkeypatch.setattr(runtime_env, "run_capture", fake_capture)
    help_text = collect_cli_help(["vllm", "serve"], exhaustive=True)
    assert calls == [
        ["vllm", "serve", "--help=all"],
        ["vllm", "serve", "--help"],
    ]
    assert runtime_env._missing_flags(help_text, runtime_env.REQUIRED_VLLM_FLAGS) == []


def test_vllm_old_help_fallback(monkeypatch):
    def fake_capture(args, timeout_s=30.0):
        if args[-1] == "--help=all":
            return {
                "command": args,
                "returncode": 2,
                "output": "error: unrecognized arguments: --help=all",
            }
        return {"command": args, "returncode": 0, "output": "legacy complete help"}

    monkeypatch.setattr(runtime_env, "run_capture", fake_capture)
    assert collect_cli_help(["vllm", "serve"], exhaustive=True) == "legacy complete help"


def test_prompt_ids_are_stable_and_segmented():
    client = RealCacheClient(
        tokenizer=FakeTokenizer(),
        model="m",
        vllm_base_url="http://localhost:8000",
        lmcache_http_url="http://localhost:8080",
        lmcache_metrics_url="http://localhost:9090/metrics",
        blend_separator=" # # ",
        gpu_id="0",
    )
    try:
        first = client.prompt_ids(["alpha", "beta"])
        second = client.prompt_ids(["alpha", "beta"])
        assert first == second
        assert first[0] == FakeTokenizer.bos_token_id
        assert client.separator_ids == first[1 + len("alpha"):1 + len("alpha") + len(client.separator_ids)]
    finally:
        client.close()


def test_successful_empty_stream_choice_is_valid_prefill_completion():
    class FakeResponse:
        ok = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield b'data: {"id":"cmpl-empty","choices":[{"text":"","finish_reason":"length"}]}'
            yield b"data: [DONE]"

    class FakeSession:
        def __init__(self):
            self.body = None

        def post(self, url, *, json, stream, timeout):
            self.body = json
            return FakeResponse()

        def close(self):
            return None

    client = RealCacheClient(
        tokenizer=FakeTokenizer(),
        model="m",
        vllm_base_url="http://localhost:8000",
        lmcache_http_url="http://localhost:8080",
        lmcache_metrics_url="http://localhost:9090/metrics",
        blend_separator=" # # ",
        gpu_id="0",
    )
    client.session.close()
    client.session = FakeSession()
    timing = client.completion([1, 2, 3], max_tokens=1)
    assert timing.response_id == "cmpl-empty"
    assert timing.ttft_ms >= 0
    assert client.session.body["min_tokens"] == 1
    assert client.session.body["ignore_eos"] is True
    assert client.session.body["skip_special_tokens"] is False
