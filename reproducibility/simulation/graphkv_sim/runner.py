from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .cache import LRUCache
from .cost import CostModel
from .models import Chunk, PolicyContext, Prediction, TraceEvent
from .policies import BasePolicy


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    fieldnames = sorted({key for row in rows for key in row})
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict, tuple))
                    else value
                    for key, value in row.items()
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _configuration_hash(metadata: dict) -> str:
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sanitize_prediction(
    prediction: Prediction, current_chunk_id: int, valid_ids: set[int], k: int
) -> Prediction:
    cleaned: list[int] = []
    seen: set[int] = set()
    for value in prediction.ids:
        candidate = int(value)
        if candidate == current_chunk_id or candidate not in valid_ids or candidate in seen:
            continue
        cleaned.append(candidate)
        seen.add(candidate)
        if len(cleaned) == k:
            break
    return Prediction(
        cleaned,
        {int(i): float(v) for i, v in prediction.scores.items() if int(i) in valid_ids},
        {int(i): tuple(map(float, v)) for i, v in prediction.features.items() if int(i) in valid_ids},
        (
            {int(i) for i in prediction.candidate_universe_ids if int(i) in valid_ids}
            if prediction.candidate_universe_ids is not None
            else None
        ),
        bool(prediction.uses_future_target),
    )


def run_policy_trace(
    *,
    policy: BasePolicy,
    events: Iterable[TraceEvent],
    chunks: Iterable[Chunk],
    k: int,
    cost_model: CostModel,
    capacity_entries: int | None,
    capacity_bytes: int | None,
    output_csv: str | Path,
    checkpoint_json: str | Path,
    run_metadata: dict[str, Any] | None = None,
    stop_after: int | None = None,
    checkpoint_every: int = 1,
) -> list[dict]:
    """Run one policy with reproducible cache events and exact JSON resume."""

    events = list(events)
    chunk_by_id = {int(chunk.chunk_id): chunk for chunk in chunks}
    valid_ids = set(chunk_by_id)
    if not events:
        raise ValueError("events cannot be empty")
    if k < 0:
        raise ValueError("k cannot be negative")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    for event in events:
        if event.current_chunk_id not in valid_ids or event.target_chunk_id not in valid_ids:
            raise ValueError(f"event {event.event_id} references an unknown chunk")

    metadata = {
        "schema_version": 1,
        "policy": policy.name,
        "k": int(k),
        "capacity_entries": capacity_entries,
        "capacity_bytes": capacity_bytes,
        "cost_model": cost_model.to_dict(),
        "event_ids": [event.event_id for event in events],
        "event_digest": hashlib.sha256(
            json.dumps([event.to_dict() for event in events], sort_keys=True).encode()
        ).hexdigest(),
        **(run_metadata or {}),
    }
    config_hash = _configuration_hash(metadata)
    checkpoint_path = Path(checkpoint_json)
    output_path = Path(output_csv)

    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if state["configuration_hash"] != config_hash:
            raise ValueError("checkpoint belongs to a different configuration")
        cache = LRUCache.from_state_dict(state["cache"])
        policy.load_state_dict(state.get("policy_state", {}))
        rows = list(state["rows"])
        start_index = int(state["next_index"])
    else:
        cache = LRUCache(capacity_entries, capacity_bytes)
        rows = []
        start_index = 0

    processed_this_call = 0
    for index in range(start_index, len(events)):
        if stop_after is not None and processed_this_call >= stop_after:
            break
        event = events[index]
        current = chunk_by_id[event.current_chunk_id]
        target = chunk_by_id[event.target_chunk_id]

        current_access = cache.access(current.chunk_id)
        current_insert = None
        demand_current_ms = 0.0
        if not current_access.hit:
            demand_current_ms = cost_model.demand_miss_ms(current.token_count)
            current_insert = cache.put(
                current.chunk_id, current.kv_bytes, "demand_current", event.event_id
            )

        context = PolicyContext(
            event.event_id,
            event.session_id,
            event.step_id,
            event.document_id,
            event.current_chunk_id,
            event.query_type,
        )
        prediction_start = time.perf_counter()
        if policy.is_oracle:
            prediction = policy.predict_with_target(context, event.target_chunk_id, k)
        else:
            prediction = policy.predict(context, k)
        policy_cpu_ms = (time.perf_counter() - prediction_start) * 1000.0
        prediction = _sanitize_prediction(prediction, event.current_chunk_id, valid_ids, k)

        target_before = cache.peek(target.chunk_id)
        already_resident_ids = [candidate for candidate in prediction.ids if cache.contains(candidate)]
        transfer_ids = [candidate for candidate in prediction.ids if not cache.contains(candidate)]
        transfer_sizes = [chunk_by_id[candidate].kv_bytes for candidate in transfer_ids]
        schedule = cost_model.schedule(transfer_sizes)
        completed_ids = transfer_ids[: schedule.completed_count]
        prefetch_evictions: list[int] = []
        oversize_prefetches: list[int] = []
        for candidate in completed_ids:
            insert = cache.put(
                candidate,
                chunk_by_id[candidate].kv_bytes,
                f"prefetch:{event.event_id}",
                event.event_id,
            )
            prefetch_evictions.extend(insert.evicted_ids)
            if insert.rejected_oversize:
                oversize_prefetches.append(candidate)

        target_access = cache.access(target.chunk_id)
        target_entry = cache.peek(target.chunk_id) if target_access.hit else None
        target_insert = None
        demand_target_ms = 0.0
        if not target_access.hit:
            demand_target_ms = cost_model.demand_miss_ms(target.token_count)
            target_insert = cache.put(
                target.chunk_id, target.kv_bytes, "demand_target", event.event_id
            )

        prediction_hit = target.chunk_id in prediction.ids
        target_in_completed = target.chunk_id in completed_ids
        current_prefetch_hit = bool(
            target_access.hit
            and target_entry is not None
            and target_entry.source == f"prefetch:{event.event_id}"
        )
        if current_prefetch_hit:
            miss_reason = "current_prefetch_hit"
        elif target_before is not None and target_access.hit:
            miss_reason = "prior_residency"
        elif prediction_hit and not target_in_completed:
            miss_reason = "late_prefetch"
        elif target_in_completed and not target_access.hit:
            miss_reason = "evicted_or_rejected_prefetch"
        else:
            miss_reason = "true_miss"

        update = policy.observe(context, target.chunk_id, prediction)
        prediction_coverage = (
            target.chunk_id in prediction.candidate_universe_ids
            if prediction.candidate_universe_ids is not None
            else None
        )
        candidate_universe_size = (
            len(prediction.candidate_universe_ids)
            if prediction.candidate_universe_ids is not None
            else None
        )
        lookup_proxy_ms = 2.0 * cost_model.cache_lookup_ms
        total_proxy_ms = (
            lookup_proxy_ms
            + schedule.exposed_prefetch_ms
            + demand_current_ms
            + demand_target_ms
        )
        rows.append(
            {
                "policy": policy.name,
                "oracle": bool(policy.is_oracle),
                "event_id": event.event_id,
                "session_id": event.session_id,
                "step_id": event.step_id,
                "document_id": event.document_id,
                "query_type": event.query_type,
                "k": int(k),
                "current_chunk_id": current.chunk_id,
                "target_chunk_id": target.chunk_id,
                "predicted_ids": prediction.ids,
                "completed_prefetch_ids": completed_ids,
                "prefetch_already_resident_ids": already_resident_ids,
                "prediction_hit": bool(prediction_hit),
                "candidate_coverage": prediction_coverage,
                "candidate_universe_size": candidate_universe_size,
                "residency_hit": bool(target_access.hit),
                "current_prefetch_hit": current_prefetch_hit,
                "target_source": target_access.source,
                "miss_reason": miss_reason,
                "current_demand_hit": bool(current_access.hit),
                "current_demand_source": current_access.source,
                "current_insert_evictions": list(current_insert.evicted_ids) if current_insert else [],
                "prefetch_evictions": prefetch_evictions,
                "target_insert_evictions": list(target_insert.evicted_ids) if target_insert else [],
                "oversize_prefetch_ids": oversize_prefetches,
                "immediate_unused_prefetch_ids": [i for i in completed_ids if i != target.chunk_id],
                "prefetch_bytes_scheduled": int(sum(transfer_sizes)),
                "prefetch_bytes_completed": int(sum(chunk_by_id[i].kv_bytes for i in completed_ids)),
                "cache_entries_after": len(cache),
                "cache_bytes_after": cache.current_bytes,
                "cache_lookup_ms": lookup_proxy_ms,
                "prefetch_proxy_ms": schedule.prefetch_ms,
                "exposed_prefetch_proxy_ms": schedule.exposed_prefetch_ms,
                "current_demand_proxy_ms": demand_current_ms,
                "target_demand_proxy_ms": demand_target_ms,
                "total_proxy_ms": total_proxy_ms,
                "policy_cpu_ms": policy_cpu_ms,
                "online_updated": bool(update.get("updated", False)),
                "online_update_reason": update.get("update_reason"),
                "weights_before": update.get("weights_before"),
                "weights_after": update.get("weights_after"),
                "online_update_l2": update.get("update_l2"),
            }
        )
        processed_this_call += 1
        next_index = index + 1
        if next_index % checkpoint_every == 0 or next_index == len(events):
            _atomic_json(
                checkpoint_path,
                {
                    "configuration_hash": config_hash,
                    "metadata": metadata,
                    "next_index": next_index,
                    "cache": cache.state_dict(),
                    "policy_state": policy.state_dict(),
                    "rows": rows,
                    "complete": next_index == len(events),
                },
            )

    if start_index + processed_this_call == len(events):
        _atomic_csv(output_path, rows)
    elif processed_this_call and (start_index + processed_this_call) % checkpoint_every:
        _atomic_json(
            checkpoint_path,
            {
                "configuration_hash": config_hash,
                "metadata": metadata,
                "next_index": start_index + processed_this_call,
                "cache": cache.state_dict(),
                "policy_state": policy.state_dict(),
                "rows": rows,
                "complete": False,
            },
        )
    return rows
