"""Mandatory fresh-process controls for the real-cache protocol."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from isolated_runtime import IsolatedRuntime, RuntimeConfig
from real_cache_client import RealCacheClient, metric_delta, select_metric_sum


_TOKEN_PATTERN = re.compile(r"(?:retrieved|retrieve)\D+(\d+)\s+tokens?", re.IGNORECASE)


def _evidence(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if any(
            word in line.lower()
            for word in ("retriev", "store", "lookup", "blend", "evict")
        )
    ][-200:]


def _evidence_since(path: Path, offset: int) -> tuple[int, list[str]]:
    if not path.exists():
        return offset, []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        text = handle.read()
        new_offset = handle.tell()
    lines = [
        line for line in text.splitlines()
        if any(word in line.lower() for word in ("retriev", "store", "lookup", "evict"))
    ]
    return new_offset, lines[-200:]


def _log_retrieved_tokens(lines: list[str]) -> int:
    total = 0
    for line in lines:
        match = _TOKEN_PATTERN.search(line)
        if match:
            total += int(match.group(1))
    return total


def _named_metric_sum(metrics: dict[str, float], metric_name: str) -> float | None:
    """Sum one Prometheus metric without accidentally including its buckets."""
    values = [
        float(value)
        for key, value in metrics.items()
        if key.split("{", 1)[0] == metric_name
    ]
    return sum(values) if values else None


def _wait_lmcache_quiescent(
    client,
    *,
    timeout_s: float = 60.0,
    stable_samples: int = 3,
) -> dict:
    """Wait until asynchronous L1 writes and capacity eviction have settled."""
    deadline = time.monotonic() + float(timeout_s)
    stable = 0
    last_snapshot: dict = {}
    while time.monotonic() < deadline:
        status = client.status()
        if not status:
            # Older servers may not expose queue state. A short grace period is
            # still safer than probing immediately after the population loop.
            time.sleep(1.0)
            return {"status_endpoint": "unavailable_grace_wait"}
        storage = status.get("storage_manager", {})
        l1 = storage.get("l1_manager", {})
        controller = storage.get("store_controller", {})
        eviction = storage.get("l1_eviction_controller", {})
        pending = int(controller.get("pending_keys_count", 0))
        in_flight = int(controller.get("in_flight_task_count", 0))
        write_locked = int(l1.get("write_locked_count", 0))
        temporary = int(l1.get("temporary_count", 0))
        usage = float(l1.get("memory_usage_ratio", 0.0))
        watermark = float(eviction.get("trigger_watermark", 1.0))
        queues_empty = pending == in_flight == write_locked == temporary == 0
        eviction_settled = usage <= watermark + 0.02
        last_snapshot = {
            "pending_keys_count": pending,
            "in_flight_task_count": in_flight,
            "write_locked_count": write_locked,
            "temporary_count": temporary,
            "memory_usage_ratio": usage,
            "trigger_watermark": watermark,
        }
        if queues_empty and eviction_settled:
            stable += 1
            if stable >= int(stable_samples):
                return {**last_snapshot, "stable_samples": stable}
        else:
            stable = 0
        time.sleep(0.2)
    raise RuntimeError(
        "LMCache did not reach a stable post-population state within "
        f"{timeout_s:.0f}s; last snapshot={last_snapshot}"
    )


def _condition(
    *,
    name: str,
    runtime_config: RuntimeConfig,
    output_dir: Path,
    store,
    current_id: int,
    target_id: int,
    population_ids: list[int],
) -> dict:
    run_dir = output_dir / name
    run_id = f"sanity-{name}-{uuid.uuid4().hex[:8]}"
    current = store.chunk(current_id)
    target = store.chunk(target_id)
    with IsolatedRuntime(runtime_config, run_dir) as runtime:
        runtime.start(run_id)
        client = RealCacheClient(
            tokenizer=store.tokenizer,
            model=runtime_config.model,
            vllm_base_url=f"http://{runtime_config.vllm_host}:{runtime_config.vllm_port}",
            lmcache_http_url=f"http://{runtime_config.lmcache_host}:{runtime_config.lmcache_http_port}",
            lmcache_metrics_url=(
                f"http://{runtime_config.lmcache_host}:"
                f"{runtime_config.lmcache_prometheus_port}/metrics"
            ),
            blend_separator=runtime_config.blend_separator,
            gpu_id=runtime_config.gpu_id,
        )
        try:
            client.populate([current.text])
            population_started = time.perf_counter_ns()
            population = client.populate(store.texts(population_ids)) if population_ids else None
            population_ms = (time.perf_counter_ns() - population_started) / 1e6
            before = client.metrics()
            # Chunk-access mode addresses the predicted target independently.
            # This exercises stock LMCache exact-prefix retrieval and does not
            # claim non-prefix CacheBlend context fusion.
            prompt = client.prompt_ids([target.text])
            service = client.completion(prompt, max_tokens=1)
            after = client.metrics()
            status = client.status()
        finally:
            client.close()
    log_lines = _evidence(run_dir / "lmcache.log")
    lm_delta = metric_delta(before["lmcache"], after["lmcache"])
    vllm_delta = metric_delta(before["vllm"], after["vllm"])
    return {
        "condition": name,
        "population_ids": population_ids,
        "population_ms": population_ms,
        "prompt_hash": service.prompt_hash,
        "prompt_tokens": service.prompt_tokens,
        "ttft_ms": service.ttft_ms,
        "request_e2e_ms": service.request_e2e_ms,
        "output_text_hash": service.output_text_hash,
        "lmcache_retrieved_tokens_metric": select_metric_sum(lm_delta, ("retriev", "token")),
        "lmcache_stored_tokens_metric": select_metric_sum(lm_delta, ("stor", "token")),
        "vllm_prefix_hit_metric": select_metric_sum(vllm_delta, ("prefix", "hit")),
        "lmcache_retrieved_tokens_log": _log_retrieved_tokens(log_lines),
        "lmcache_metrics_delta": lm_delta,
        "vllm_metrics_delta": vllm_delta,
        "lmcache_status": status,
        "log_evidence": log_lines,
    }


def _capacity_condition(
    *,
    runtime_config: RuntimeConfig,
    output_dir: Path,
    store,
    pressure_factor: float = 1.25,
) -> dict:
    """Prove that the configured cache cannot retain the full pressure set.

    The oldest chunk is populated first, followed by enough independently
    addressed chunks to exceed the estimated L1 byte budget.  The newest
    chunk must still retrieve while the oldest must retrieve fewer tokens.
    This is an operational eviction check, not merely a shadow-LRU estimate.
    """
    capacity_bytes = int(runtime_config.l1_size_gb * 1024 ** 3)
    minimum_control_tokens = max(64, 2 * int(runtime_config.chunk_size))
    ordered = sorted(
        (
            chunk for chunk in store.chunks
            if int(chunk.token_count) >= minimum_control_tokens
        ),
        key=lambda chunk: chunk.chunk_id,
    )
    if len(ordered) < 3:
        raise RuntimeError("capacity control requires at least three chunks")
    selected = []
    planned_bytes = 0
    required_bytes = int(capacity_bytes * float(pressure_factor))
    for chunk in ordered:
        selected.append(chunk)
        planned_bytes += int(chunk.kv_bytes)
        if planned_bytes >= required_bytes:
            break
    if planned_bytes < required_bytes:
        return {
            "status": "fail",
            "detail": (
                f"corpus KV estimate {planned_bytes} bytes cannot exceed "
                f"{pressure_factor:.2f}x L1 capacity {capacity_bytes} bytes"
            ),
            "capacity_bytes": capacity_bytes,
            "planned_population_bytes": planned_bytes,
            "population_chunks": len(selected),
        }

    run_dir = output_dir / "capacity"
    run_id = f"sanity-capacity-{uuid.uuid4().hex[:8]}"
    with IsolatedRuntime(runtime_config, run_dir) as runtime:
        runtime.start(run_id)
        client = RealCacheClient(
            tokenizer=store.tokenizer,
            model=runtime_config.model,
            vllm_base_url=f"http://{runtime_config.vllm_host}:{runtime_config.vllm_port}",
            lmcache_http_url=f"http://{runtime_config.lmcache_host}:{runtime_config.lmcache_http_port}",
            lmcache_metrics_url=(
                f"http://{runtime_config.lmcache_host}:"
                f"{runtime_config.lmcache_prometheus_port}/metrics"
            ),
            blend_separator=runtime_config.blend_separator,
            gpu_id=runtime_config.gpu_id,
        )
        try:
            for chunk in selected:
                client.populate([chunk.text])
            settled_after_pressure = _wait_lmcache_quiescent(client)

            # Refresh the most recently selected chunk after eviction has
            # settled. LMCache stores asynchronously, so probing immediately
            # after the loop can race the final store and produce a false
            # newest=0, oldest=0 failure even when eviction is operational.
            newest = selected[-1]
            client.populate([newest.text])
            settled_after_refresh = _wait_lmcache_quiescent(client)
            log_path = run_dir / "lmcache.log"
            offset = log_path.stat().st_size if log_path.exists() else 0

            newest_before = client.metrics()
            newest_prompt = client.prompt_ids([newest.text])
            newest_service = client.completion(newest_prompt, max_tokens=1)
            newest_after = client.metrics()
            time.sleep(0.5)
            offset, newest_evidence = _evidence_since(log_path, offset)

            oldest = selected[0]
            oldest_before = client.metrics()
            oldest_prompt = client.prompt_ids([oldest.text])
            oldest_service = client.completion(oldest_prompt, max_tokens=1)
            oldest_after = client.metrics()
            time.sleep(0.5)
            _, oldest_evidence = _evidence_since(log_path, offset)
            status = client.status()
        finally:
            client.close()

    newest_retrieved = _log_retrieved_tokens(newest_evidence)
    oldest_retrieved = _log_retrieved_tokens(oldest_evidence)
    newest_vllm_delta = metric_delta(newest_before["vllm"], newest_after["vllm"])
    oldest_vllm_delta = metric_delta(oldest_before["vllm"], oldest_after["vllm"])
    computed_metric = "vllm:request_prefill_kv_computed_tokens_sum"
    newest_computed = _named_metric_sum(newest_vllm_delta, computed_metric)
    oldest_computed = _named_metric_sum(oldest_vllm_delta, computed_metric)
    newest_cached = newest_retrieved > 0 or (
        newest_computed is not None
        and newest_computed < newest_service.prompt_tokens
    )
    oldest_less_cached = (
        oldest_retrieved < newest_retrieved
        if newest_retrieved > 0
        else (
            newest_computed is not None
            and oldest_computed is not None
            and oldest_computed > newest_computed
        )
    )
    all_evidence = _evidence(run_dir / "lmcache.log")
    eviction_logged = any("evict" in line.lower() for line in all_evidence)
    passed = eviction_logged and newest_cached and oldest_less_cached
    return {
        "status": "pass" if passed else "fail",
        "capacity_bytes": capacity_bytes,
        "pressure_factor": pressure_factor,
        "planned_population_bytes": planned_bytes,
        "population_chunks": len(selected),
        "oldest_chunk_id": oldest.chunk_id,
        "newest_chunk_id": newest.chunk_id,
        "oldest_prompt_tokens": oldest_service.prompt_tokens,
        "newest_prompt_tokens": newest_service.prompt_tokens,
        "oldest_retrieved_tokens": oldest_retrieved,
        "newest_retrieved_tokens": newest_retrieved,
        "oldest_vllm_computed_prefill_tokens": oldest_computed,
        "newest_vllm_computed_prefill_tokens": newest_computed,
        "newest_cached": newest_cached,
        "oldest_less_cached": oldest_less_cached,
        "eviction_logged": eviction_logged,
        "settled_after_pressure": settled_after_pressure,
        "settled_after_refresh": settled_after_refresh,
        "oldest_log_evidence": oldest_evidence,
        "newest_log_evidence": newest_evidence,
        "lmcache_status": status,
    }


def run_sanity_controls(
    output_dir: Path,
    runtime_config: RuntimeConfig,
    store,
    event,
    *,
    repetitions: int = 1,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    current_id, target_id = event.current_chunk_id, event.target_chunk_id
    wrong_id = next(
        chunk.chunk_id
        for chunk in store.chunks
        if chunk.chunk_id not in {current_id, target_id}
        and chunk.document_id != store.chunk(target_id).document_id
    )
    controls = {
        "cold": _condition(
            name="cold", runtime_config=runtime_config, output_dir=output_dir, store=store,
            current_id=current_id, target_id=target_id, population_ids=[]
        ),
        "wrong": _condition(
            name="wrong", runtime_config=runtime_config, output_dir=output_dir, store=store,
            current_id=current_id, target_id=target_id, population_ids=[wrong_id]
        ),
        "oracle": _condition(
            name="oracle", runtime_config=runtime_config, output_dir=output_dir, store=store,
            current_id=current_id, target_id=target_id, population_ids=[target_id]
        ),
    }
    capacity = _capacity_condition(
        runtime_config=runtime_config,
        output_dir=output_dir,
        store=store,
    )
    controls["capacity"] = capacity
    retrieval_controls = [controls[name] for name in ("cold", "wrong", "oracle")]
    prompt_parity = len({row["prompt_hash"] for row in retrieval_controls}) == 1
    output_agreement = len({row["output_text_hash"] for row in retrieval_controls}) == 1

    def retrieved(row: dict) -> float:
        metric = row["lmcache_retrieved_tokens_metric"]
        return float(metric) if metric is not None else float(row["lmcache_retrieved_tokens_log"])

    cold_retrieved = retrieved(controls["cold"])
    wrong_retrieved = retrieved(controls["wrong"])
    oracle_retrieved = retrieved(controls["oracle"])
    telemetry_present = any(
        bool(row["lmcache_metrics_delta"] or row["log_evidence"])
        for row in retrieval_controls
    )
    oracle_extra_retrieval = oracle_retrieved > max(cold_retrieved, wrong_retrieved)
    wrong_not_oracle = wrong_retrieved < oracle_retrieved
    checks = {
        "V_A_cold_fresh_process": {"status": "pass", "detail": "cold condition used a new process pair"},
        "V_B_native_apc": {"status": "not_run", "detail": "native vLLM APC is disabled; LMCache provides exact-chunk reuse"},
        "V_C_exact_chunk_reuse": {
            "status": "pass" if oracle_extra_retrieval else "fail",
            "detail": {"cold": cold_retrieved, "wrong": wrong_retrieved, "oracle": oracle_retrieved},
        },
        "V_D_wrong_candidate": {"status": "pass" if wrong_not_oracle else "fail"},
        "V_E_oracle_target": {"status": "pass" if oracle_extra_retrieval else "fail"},
        "V_F_prompt_parity": {"status": "pass" if prompt_parity else "fail"},
        "V_G_capacity_eviction": {
            "status": capacity["status"],
            "detail": {
                key: capacity.get(key)
                for key in (
                    "capacity_bytes", "planned_population_bytes", "population_chunks",
                    "oldest_retrieved_tokens", "newest_retrieved_tokens",
                    "oldest_vllm_computed_prefill_tokens",
                    "newest_vllm_computed_prefill_tokens", "newest_cached",
                    "oldest_less_cached", "eviction_logged", "detail",
                )
                if key in capacity
            },
        },
        "V_H_output_correctness": {"status": "pass" if output_agreement else "fail"},
        "V_I_order_independence": {
            "status": "planned" if repetitions >= 2 else "not_established",
            "detail": (
                f"{repetitions} randomized benchmark block(s) configured; "
                "post-run order validation is written after all arms complete"
            ),
        },
        "telemetry_available": {"status": "pass" if telemetry_present else "fail"},
    }
    critical = [
        "V_C_exact_chunk_reuse", "V_D_wrong_candidate", "V_E_oracle_target",
        "V_F_prompt_parity", "V_G_capacity_eviction", "V_H_output_correctness",
        "telemetry_available",
    ]
    report = {
        "created_at": time.time(),
        "event": asdict(event),
        "runtime": asdict(runtime_config),
        "workload_fingerprint": store.workload_fingerprint(),
        "configured_repetitions": int(repetitions),
        "controls": controls,
        "checks": checks,
        "critical_passed": all(checks[name]["status"] == "pass" for name in critical),
        "critical_checks": critical,
        "note": "Exact-chunk LMCache control only; this does not validate non-prefix CacheBlend context fusion. A one-event control validates wiring, not a stable latency effect size.",
    }
    (output_dir / "sanity_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
