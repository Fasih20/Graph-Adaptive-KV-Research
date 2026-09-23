"""Isolated GraphKV real-cache benchmark.

The causal unit is one complete policy/K/repetition trace in a fresh
vLLM+LMCache process pair.  Partial attempts are never resumed in-place;
rerunning creates a new attempt so cache and online-policy state cannot be
silently reconstructed incorrectly.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from adaptive_policies import (
    OfflineAdaptivePolicy,
    OnlineAdaptivePolicy,
    Prediction,
    fit_offline_weights,
    tune_eta,
)
from isolated_runtime import IsolatedRuntime, RuntimeConfig
from real_cache_client import RealCacheClient, metric_delta, select_metric_sum


_TOKEN_LOG_PATTERN = re.compile(
    r"(?:retrieved|retrieve|stored|store)\D+(\d+)\s+tokens?",
    re.IGNORECASE,
)


SUMMARY_NUMERIC_FIELDS = (
    "prediction_hit",
    "target_rank",
    "candidate_universe_size",
    "prompt_tokens",
    "capacity_bytes",
    "ready_cache_ttft_ms",
    "request_e2e_ms",
    "policy_ms",
    "current_access_ttft_ms",
    "current_access_request_ms",
    "speculative_population_ms",
    "available_overlap_ms",
    "exposed_population_ms",
    "end_to_end_ttft_ms",
    "end_to_end_request_ms",
    "prefetch_bytes_scheduled",
    "prefetch_bytes_completed",
    "prefetch_completion_fraction",
    "shadow_residency_before_prefetch",
    "shadow_residency_hit",
    "shadow_current_prefetch_hit",
    "shadow_cache_bytes_after",
    "lmcache_retrieved_tokens_delta",
    "lmcache_stored_tokens_delta",
    "vllm_prefix_cache_hit_tokens_delta",
    "population_request_count",
    "population_prompt_tokens",
    "predicted_count",
    "completed_prefetch_count",
    "current_insert_eviction_count",
    "prefetch_eviction_count",
    "target_insert_eviction_count",
    "online_updated",
    "retry_count",
    "prefetch_completed_before_demand",
    "request_success",
)


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


class ByteLRU:
    """Auditable shadow cache; never mislabeled as LMCache ground truth."""

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = int(capacity_bytes)
        self.entries: OrderedDict[int, int] = OrderedDict()
        self.bytes = 0

    def contains(self, chunk_id: int) -> bool:
        return int(chunk_id) in self.entries

    def insert(self, chunk_id: int, size: int) -> list[int]:
        chunk_id, size = int(chunk_id), int(size)
        evicted: list[int] = []
        if size > self.capacity_bytes:
            return evicted
        if chunk_id in self.entries:
            self.bytes -= self.entries.pop(chunk_id)
        while self.entries and self.bytes + size > self.capacity_bytes:
            victim, victim_size = self.entries.popitem(last=False)
            self.bytes -= victim_size
            evicted.append(victim)
        self.entries[chunk_id] = size
        self.bytes += size
        return evicted


class PolicyAdapter:
    def __init__(self, name: str, store, offline_weights, eta: float):
        self.name = name
        self.store = store
        self.impl = None
        if name == "graph_fixed":
            self.impl = OfflineAdaptivePolicy(store.raw_edges, (0.7, 0.3))
        elif name == "adaptive_offline_global":
            self.impl = OfflineAdaptivePolicy(store.raw_edges, offline_weights)
        elif name == "adaptive_online_warm":
            self.impl = OnlineAdaptivePolicy(store.raw_edges, offline_weights, eta)
        elif name not in {"no_prefetch", "cosine"}:
            raise ValueError(f"unknown policy {name!r}")

    def predict(self, current: int, k: int) -> Prediction:
        if self.name == "no_prefetch":
            return Prediction([])
        if self.name == "cosine":
            similarities = self.store.sim_matrix[int(current)].copy()
            similarities[int(current)] = -np.inf
            ids = np.argsort(similarities)[::-1][: int(k)].astype(int).tolist()
            scores = {candidate: float(similarities[candidate]) for candidate in ids}
            return Prediction(ids, scores=scores, candidate_universe=set(range(len(similarities))) - {int(current)})
        return self.impl.predict(int(current), int(k))

    def observe(self, target: int, prediction: Prediction) -> dict:
        if self.impl is None:
            return {
                "updated": False,
                "update_reason": "non_adaptive_policy",
                "weights_before": None,
                "weights_after": None,
                "update_l2": 0.0,
            }
        return self.impl.observe(int(target), prediction)


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: Path
    policies: tuple[str, ...]
    k_values: tuple[int, ...]
    repetitions: int
    seed: int
    overlap_ms: float
    max_events: int
    max_tokens_per_chunk: int
    runtime: RuntimeConfig


def _attempt_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"attempt_{index:03d}").exists():
        index += 1
    path = root / f"attempt_{index:03d}"
    path.mkdir()
    return path


def _log_evidence(path: Path, offset: int) -> tuple[int, list[str]]:
    if not path.exists():
        return offset, []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        text = handle.read()
        new_offset = handle.tell()
    evidence = [
        line for line in text.splitlines()
        if any(word in line.lower() for word in ("retriev", "store", "lookup", "evict", "blend"))
    ]
    return new_offset, evidence[-30:]


def _log_token_total(lines: list[str], operation: str) -> int:
    operation = operation.lower()
    total = 0
    for line in lines:
        lowered = line.lower()
        if operation not in lowered:
            continue
        match = _TOKEN_LOG_PATTERN.search(line)
        if match:
            total += int(match.group(1))
    return total


def _append_jsonl(handle, row: dict) -> None:
    handle.write(_json(row) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _candidate_batches(client, store, ids: list[int], max_model_len: int, max_tokens_per_chunk: int):
    """Pack candidate segments without ever exceeding the vLLM prompt limit."""
    batches: list[list[int]] = []
    current: list[int] = []
    for candidate in ids:
        proposed = current + [candidate]
        token_ids = client.prompt_ids(
            store.texts(proposed), max_tokens_per_chunk=max_tokens_per_chunk
        )
        if len(token_ids) + 1 <= max_model_len:
            current = proposed
            continue
        if not current:
            raise RuntimeError(
                f"candidate chunk {candidate} cannot fit max_model_len={max_model_len}"
            )
        batches.append(current)
        current = [candidate]
    if current:
        batches.append(current)
    return batches


def _summary_rows(events_path: Path) -> list[dict]:
    rows = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    if not rows:
        return []
    result = []
    query_types = sorted({row["query_type"] for row in rows})
    for query_type in ["all", *query_types]:
        selected = rows if query_type == "all" else [row for row in rows if row["query_type"] == query_type]
        aggregate = {
            "policy": selected[0]["policy"],
            "k": selected[0]["k"],
            "repetition": selected[0]["repetition"],
            "query_type": query_type,
            "events": len(selected),
        }
        for name in SUMMARY_NUMERIC_FIELDS:
            values = [float(row[name]) for row in selected if row.get(name) is not None]
            aggregate[f"mean_{name}"] = float(np.mean(values)) if values else None
            aggregate[f"median_{name}"] = float(np.median(values)) if values else None
        result.append(aggregate)
    return result


def _pooled_summary_rows(rows: list[dict]) -> list[dict]:
    """Pool raw events across repetitions without averaging averages."""
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), int(row["k"]))].append(row)

    result: list[dict] = []
    for (policy, k), block in sorted(grouped.items()):
        query_types = sorted({str(row["query_type"]) for row in block})
        for query_type in ["all", *query_types]:
            selected = (
                block
                if query_type == "all"
                else [row for row in block if str(row["query_type"]) == query_type]
            )
            aggregate = {
                "policy": policy,
                "k": k,
                "query_type": query_type,
                "repetitions": len({int(row["repetition"]) for row in selected}),
                "events": len(selected),
            }
            for name in SUMMARY_NUMERIC_FIELDS:
                values = [
                    float(row[name])
                    for row in selected
                    if row.get(name) is not None
                ]
                aggregate[f"mean_{name}"] = float(np.mean(values)) if values else None
                aggregate[f"median_{name}"] = float(np.median(values)) if values else None
            result.append(aggregate)
    return result


def _headline_rows(pooled_rows: list[dict]) -> list[dict]:
    result = []
    for row in pooled_rows:
        if row["query_type"] != "all":
            continue
        result.append(
            {
                "policy": row["policy"],
                "k": row["k"],
                "repetitions": row["repetitions"],
                "events": row["events"],
                "prediction_recall": row["mean_prediction_hit"],
                "mean_ready_cache_ttft_ms": row["mean_ready_cache_ttft_ms"],
                "mean_service_request_ms": row["mean_request_e2e_ms"],
                "mean_policy_ms": row["mean_policy_ms"],
                "mean_speculative_population_ms": row["mean_speculative_population_ms"],
                "mean_exposed_population_ms": row["mean_exposed_population_ms"],
                "mean_end_to_end_ttft_ms": row["mean_end_to_end_ttft_ms"],
                "mean_end_to_end_request_ms": row["mean_end_to_end_request_ms"],
                "mean_completed_prefetch_mib": (
                    row["mean_prefetch_bytes_completed"] / (1024 ** 2)
                    if row["mean_prefetch_bytes_completed"] is not None
                    else None
                ),
                "shadow_residency_rate": row["mean_shadow_residency_hit"],
                "current_prefetch_hit_rate": row["mean_shadow_current_prefetch_hit"],
                "mean_lmcache_retrieved_tokens": row["mean_lmcache_retrieved_tokens_delta"],
                "mean_prefetch_evictions": row["mean_prefetch_eviction_count"],
                "online_update_rate": row["mean_online_updated"],
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_headline_table(rows: list[dict]) -> str:
    """Human-readable exact pooled means for notebook and driver.log output."""
    columns = (
        ("policy", "policy", "{}"),
        ("k", "K", "{:d}"),
        ("events", "N", "{:d}"),
        ("prediction_recall", "pred_%", "{:.2f}"),
        ("mean_ready_cache_ttft_ms", "ready_TTFT_ms", "{:.4f}"),
        ("mean_service_request_ms", "service_ms", "{:.4f}"),
        ("mean_speculative_population_ms", "population_ms", "{:.4f}"),
        ("mean_end_to_end_ttft_ms", "E2E_TTFT_ms", "{:.4f}"),
        ("mean_end_to_end_request_ms", "E2E_request_ms", "{:.4f}"),
        ("mean_completed_prefetch_mib", "prefetch_MiB", "{:.4f}"),
        ("shadow_residency_rate", "resident_%", "{:.2f}"),
        ("mean_lmcache_retrieved_tokens", "LMCache_tokens", "{:.2f}"),
    )
    rendered = []
    for row in rows:
        values = []
        for key, _, pattern in columns:
            value = row.get(key)
            if key in {"prediction_recall", "shadow_residency_rate"} and value is not None:
                value *= 100.0
            values.append("NA" if value is None else pattern.format(value))
        rendered.append(values)
    widths = [
        max([len(header), *(len(values[index]) for values in rendered)])
        for index, (_, header, _) in enumerate(columns)
    ]
    header = "  ".join(
        label.ljust(widths[index])
        for index, (_, label, _) in enumerate(columns)
    )
    separator = "  ".join("-" * width for width in widths)
    lines = [header, separator]
    for values in rendered:
        lines.append(
            "  ".join(
                value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
                for index, value in enumerate(values)
            )
        )
    return "\n".join(lines)


def run_arm(
    *,
    config: BenchmarkConfig,
    store,
    events,
    policy_name: str,
    k: int,
    repetition: int,
    offline_weights,
    eta: float,
) -> tuple[Path, list[dict]]:
    group = config.output_dir / "arms" / f"policy={policy_name}" / f"k={k}" / f"rep={repetition}"
    if (group / "COMPLETE.json").exists():
        complete = json.loads((group / "COMPLETE.json").read_text())
        existing = config.output_dir / complete["events_path"]
        return existing, _summary_rows(existing)
    attempt = _attempt_dir(group)
    run_id = f"{policy_name}-k{k}-r{repetition}-{uuid.uuid4().hex[:8]}"
    runtime_cfg = RuntimeConfig(**{**asdict(config.runtime), "seed": config.seed + repetition})
    policy = PolicyAdapter(policy_name, store, offline_weights, eta)
    shadow = ByteLRU(int(runtime_cfg.l1_size_gb * (1024 ** 3)))
    partial_path = attempt / "events.partial.jsonl"
    final_path = attempt / "events.jsonl"
    lmcache_log_offset = 0

    with IsolatedRuntime(runtime_cfg, attempt) as runtime:
        runtime.start(instance_id=run_id)
        client = RealCacheClient(
            tokenizer=store.tokenizer,
            model=runtime_cfg.model,
            vllm_base_url=f"http://{runtime_cfg.vllm_host}:{runtime_cfg.vllm_port}",
            lmcache_http_url=f"http://{runtime_cfg.lmcache_host}:{runtime_cfg.lmcache_http_port}",
            lmcache_metrics_url=f"http://{runtime_cfg.lmcache_host}:{runtime_cfg.lmcache_prometheus_port}/metrics",
            blend_separator=runtime_cfg.blend_separator,
            gpu_id=runtime_cfg.gpu_id,
        )
        try:
            with partial_path.open("w", encoding="utf-8") as handle:
                for index, event in enumerate(events[: config.max_events]):
                    event_started_unix_ns = time.time_ns()
                    current = store.chunk(event.current_chunk_id)
                    target = store.chunk(event.target_chunk_id)

                    current_timing = client.populate([current.text])
                    lmcache_log_offset, current_evidence = _log_evidence(
                        attempt / "lmcache.log", lmcache_log_offset
                    )
                    current_evictions = shadow.insert(current.chunk_id, current.kv_bytes)

                    policy_started = time.perf_counter_ns()
                    prediction = policy.predict(current.chunk_id, k)
                    policy_ms = (time.perf_counter_ns() - policy_started) / 1e6
                    predicted = [candidate for candidate in prediction.ids if candidate != current.chunk_id]
                    prediction_hit = target.chunk_id in predicted
                    target_rank = predicted.index(target.chunk_id) + 1 if prediction_hit else None
                    shadow_before = shadow.contains(target.chunk_id)

                    before_population = client.metrics()
                    population_start_unix_ns = time.time_ns()
                    population_started = time.perf_counter_ns()
                    population_timings = []
                    # Populate each prediction independently.  Stock LMCache
                    # can then retrieve an exact target chunk by its own token
                    # prefix; batching candidates would make later candidates
                    # non-prefix segments and would require CacheBlend.
                    candidate_batches = [[candidate] for candidate in predicted]
                    for batch in candidate_batches:
                        population_timings.append(client.populate(store.texts(batch)))
                    population_ms = (time.perf_counter_ns() - population_started) / 1e6 if predicted else 0.0
                    population_end_unix_ns = time.time_ns()
                    completed = list(predicted) if len(population_timings) == len(candidate_batches) else []
                    prefetch_evictions: list[int] = []
                    for candidate in completed:
                        chunk = store.chunk(candidate)
                        prefetch_evictions.extend(shadow.insert(candidate, chunk.kv_bytes))
                    shadow_after_prefetch = shadow.contains(target.chunk_id)
                    after_population = client.metrics()
                    lmcache_log_offset, population_evidence = _log_evidence(
                        attempt / "lmcache.log", lmcache_log_offset
                    )

                    before_service = after_population
                    service_ids = client.prompt_ids(
                        [target.text],
                        max_tokens_per_chunk=config.max_tokens_per_chunk,
                    )
                    if len(service_ids) + 1 > runtime_cfg.max_model_len:
                        raise RuntimeError(
                            f"service prompt has {len(service_ids)} tokens but max_model_len="
                            f"{runtime_cfg.max_model_len}"
                        )
                    service_start_unix_ns = time.time_ns()
                    service = client.completion(service_ids, max_tokens=1)
                    service_end_unix_ns = time.time_ns()
                    after_service = client.metrics()
                    target_evictions = shadow.insert(target.chunk_id, target.kv_bytes)
                    update = policy.observe(target.chunk_id, prediction)
                    lmcache_log_offset, service_evidence = _log_evidence(
                        attempt / "lmcache.log", lmcache_log_offset
                    )

                    population_lm = metric_delta(before_population["lmcache"], after_population["lmcache"])
                    service_lm = metric_delta(before_service["lmcache"], after_service["lmcache"])
                    service_vllm = metric_delta(before_service["vllm"], after_service["vllm"])
                    retrieved_metric = select_metric_sum(service_lm, ("retriev", "token"))
                    stored_metric = select_metric_sum(service_lm, ("stor", "token"))
                    prefix_metric = select_metric_sum(service_vllm, ("prefix", "hit"))
                    retrieved_tokens = (
                        retrieved_metric
                        if retrieved_metric is not None
                        else float(_log_token_total(service_evidence, "retriev"))
                    )
                    stored_tokens = (
                        stored_metric
                        if stored_metric is not None
                        else float(_log_token_total(service_evidence, "stor"))
                    )
                    exposed = max(0.0, population_ms - config.overlap_ms)
                    row = {
                        "experiment_id": config.output_dir.name,
                        "run_id": run_id,
                        "repetition": repetition,
                        "seed": config.seed + repetition,
                        "dataset": store.cfg["label"],
                        "event_id": event.event_id,
                        "session_id": event.session_id,
                        "step_id": event.step_id,
                        "document_id": event.document_id,
                        "query_type": event.query_type,
                        "policy": policy_name,
                        "k": int(k),
                        "capacity_bytes": shadow.capacity_bytes,
                        "current_chunk_id": current.chunk_id,
                        "target_chunk_id": target.chunk_id,
                        "predicted_ids": predicted,
                        "target_rank": target_rank,
                        "prediction_hit": int(prediction_hit),
                        "candidate_universe_size": len(prediction.candidate_universe),
                        "prompt_token_hash": service.prompt_hash,
                        "prompt_tokens": service.prompt_tokens,
                        "event_started_unix_ns": event_started_unix_ns,
                        "population_start_unix_ns": population_start_unix_ns,
                        "population_end_unix_ns": population_end_unix_ns,
                        "service_start_unix_ns": service_start_unix_ns,
                        "service_end_unix_ns": service_end_unix_ns,
                        "current_access_ttft_ms": current_timing.ttft_ms,
                        "current_access_request_ms": current_timing.request_e2e_ms,
                        "policy_ms": policy_ms,
                        "speculative_population_ms": population_ms,
                        "available_overlap_ms": config.overlap_ms,
                        "exposed_population_ms": exposed,
                        "ready_cache_ttft_ms": service.ttft_ms,
                        "request_e2e_ms": service.request_e2e_ms,
                        "end_to_end_ttft_ms": policy_ms + exposed + service.ttft_ms,
                        "end_to_end_request_ms": policy_ms + exposed + service.request_e2e_ms,
                        "prefetch_bytes_scheduled": sum(store.chunk(cid).kv_bytes for cid in predicted),
                        "prefetch_bytes_completed": sum(store.chunk(cid).kv_bytes for cid in completed),
                        "predicted_count": len(predicted),
                        "completed_prefetch_count": len(completed),
                        "prefetch_completion_fraction": len(completed) / len(predicted) if predicted else 1.0,
                        "population_request_count": len(candidate_batches),
                        "population_prompt_tokens": sum(
                            timing.prompt_tokens for timing in population_timings if timing is not None
                        ),
                        "shadow_residency_before_prefetch": int(shadow_before),
                        "shadow_residency_hit": int(shadow_after_prefetch),
                        "shadow_current_prefetch_hit": int(
                            target.chunk_id in completed
                            and not shadow_before
                            and shadow_after_prefetch
                        ),
                        "shadow_cache_bytes_after": shadow.bytes,
                        "current_insert_evictions": current_evictions,
                        "current_insert_eviction_count": len(current_evictions),
                        "prefetch_evictions": prefetch_evictions,
                        "prefetch_eviction_count": len(prefetch_evictions),
                        "target_insert_evictions": target_evictions,
                        "target_insert_eviction_count": len(target_evictions),
                        "lmcache_retrieved_tokens_delta": retrieved_tokens,
                        "lmcache_stored_tokens_delta": stored_tokens,
                        "vllm_prefix_cache_hit_tokens_delta": prefix_metric or 0.0,
                        "population_lmcache_metrics_delta": population_lm,
                        "service_lmcache_metrics_delta": service_lm,
                        "service_vllm_metrics_delta": service_vllm,
                        "lmcache_status_after": client.status() if index % 25 == 0 else None,
                        "gpu_after": client.gpu_snapshot() if index % 25 == 0 else None,
                        "current_lmcache_log_evidence": current_evidence,
                        "population_lmcache_log_evidence": population_evidence,
                        "lmcache_log_evidence": service_evidence,
                        "weights_before": update["weights_before"],
                        "online_updated": int(update["updated"]),
                        "online_update_reason": update["update_reason"],
                        "weights_after": update["weights_after"],
                        "output_text_hash": service.output_text_hash,
                        "response_id": service.response_id,
                        "retry_count": 0,
                        "prefetch_completed_before_demand": 1,
                        "request_success": 1,
                        "measurement_semantics": "exact_chunk_access_prefetch_microbenchmark",
                        "cache_semantics": "stock_lmcache_exact_prefix_per_chunk_not_cacheblend",
                        "trace_semantics": "controlled_synthetic_document_split",
                    }
                    _append_jsonl(handle, row)
        finally:
            client.close()
    os.replace(partial_path, final_path)
    complete = {
        "run_id": run_id,
        "events_path": str(final_path.relative_to(config.output_dir)),
        "events": min(len(events), config.max_events),
        "completed_at": time.time(),
    }
    _atomic_json(group / "COMPLETE.json", complete)
    return final_path, _summary_rows(final_path)


def run_benchmark(config: BenchmarkConfig, store, traces) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    offline_weights, fit_rows = fit_offline_weights(
        traces["train"], store.raw_edges, config.k_values
    )
    eta, eta_rows = tune_eta(traces["dev"], store.raw_edges, offline_weights, k=6)
    _atomic_json(
        config.output_dir / "fitted_policy.json",
        {
            "offline_weights": list(offline_weights),
            "eta": eta,
            "fit_rows": fit_rows,
            "eta_rows": eta_rows,
            "train_documents": sorted(store.document_splits["train"]),
            "dev_documents": sorted(store.document_splits["dev"]),
            "test_documents": sorted(store.document_splits["test"]),
        },
    )
    all_summaries: list[dict] = []
    execution_orders: list[list[tuple[str, int]]] = []
    event_paths: list[Path] = []
    for repetition in range(config.repetitions):
        order_rng = np.random.default_rng(config.seed + repetition)
        combinations = [(policy, k) for policy in config.policies for k in config.k_values]
        order_rng.shuffle(combinations)
        execution_orders.append(list(combinations))
        _atomic_json(
            config.output_dir / f"execution_order_rep_{repetition}.json",
            [{"policy": policy, "k": k} for policy, k in combinations],
        )
        for policy, k in combinations:
            events_path, summaries = run_arm(
                config=config,
                store=store,
                events=traces["test"],
                policy_name=policy,
                k=k,
                repetition=repetition,
                offline_weights=offline_weights,
                eta=eta,
            )
            event_paths.append(events_path)
            all_summaries.extend(summaries)
    summary_path = config.output_dir / "summary_by_query_type.csv"
    _write_csv(summary_path, all_summaries)

    raw_rows: list[dict] = []
    trace_evictions = 0
    for events_path in event_paths:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            raw_rows.append(row)
            trace_evictions += int(row.get("current_insert_eviction_count", 0))
            trace_evictions += int(row.get("prefetch_eviction_count", 0))
            trace_evictions += int(row.get("target_insert_eviction_count", 0))

    pooled_rows = _pooled_summary_rows(raw_rows)
    headline_rows = _headline_rows(pooled_rows)
    _write_csv(config.output_dir / "pooled_results_by_query_type.csv", pooled_rows)
    _write_csv(config.output_dir / "headline_mean_results.csv", headline_rows)
    print("\nPOOLED EVENT-WEIGHTED MEANS ACROSS REPETITIONS")
    print(_format_headline_table(headline_rows))
    print(
        "\nLatency interpretation: ready_TTFT/service_ms begin after candidate "
        "population; E2E columns include policy time plus exposed population cost."
    )
    unique_orders = len({tuple(order) for order in execution_orders})
    order_status = (
        "pass"
        if config.repetitions >= 2 and (len(execution_orders[0]) <= 1 or unique_orders >= 2)
        else "not_established"
    )
    protocol = {
        "repetitions": config.repetitions,
        "randomized_block_orders": [
            [{"policy": policy, "k": k} for policy, k in order]
            for order in execution_orders
        ],
        "unique_execution_orders": unique_orders,
        "order_independence_status": order_status,
        "trace_shadow_eviction_count": trace_evictions,
        "trace_capacity_pressure_observed": trace_evictions > 0,
        "publication_controls_passed": (
            order_status == "pass" and trace_evictions > 0
        ),
        "note": (
            "The pre-run sanity report validates operational LMCache eviction. "
            "Trace eviction counts come from the byte-LRU audit model and are "
            "kept separate from LMCache ground truth."
        ),
    }
    _atomic_json(config.output_dir / "postrun_protocol_report.json", protocol)
    return summary_path
