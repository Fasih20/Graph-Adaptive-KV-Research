from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .models import Chunk, TraceEvent


@dataclass(frozen=True)
class ModelShape:
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    bytes_per_element: int = 2

    def kv_bytes(self, token_count: int) -> int:
        return int(
            2
            * self.num_hidden_layers
            * self.num_key_value_heads
            * self.head_dim
            * int(token_count)
            * self.bytes_per_element
        )

    @classmethod
    def from_hf_config(cls, config, bytes_per_element: int = 2) -> "ModelShape":
        heads = int(config.num_attention_heads)
        hidden = int(config.hidden_size)
        return cls(
            num_hidden_layers=int(config.num_hidden_layers),
            num_key_value_heads=int(getattr(config, "num_key_value_heads", heads)),
            head_dim=int(getattr(config, "head_dim", hidden // heads)),
            bytes_per_element=int(bytes_per_element),
        )


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def build_chunks(
    documents: Mapping[str, str], tokenizer, splitter, model_shape: ModelShape
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document_id, text in documents.items():
        for position, chunk_text in enumerate(splitter.split_text(str(text))):
            token_count = len(tokenizer.encode(chunk_text, add_special_tokens=False))
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    document_id=str(document_id),
                    position=position,
                    text=chunk_text,
                    token_count=token_count,
                    kv_bytes=model_shape.kv_bytes(token_count),
                )
            )
    return chunks


def save_chunks_jsonl(chunks: Iterable[Chunk], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks_jsonl(path: str | Path) -> list[Chunk]:
    chunks = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                chunks.append(Chunk(**value))
    return chunks


def grouped_synthetic_events(
    chunks: list[Chunk], similarity_matrix: np.ndarray, seed: int = 42
) -> list[TraceEvent]:
    """Diagnostic synthetic workload with targets generated inside each document.

    This is deliberately labelled synthetic: semantic targets are derived from
    the same embedding geometry used by similarity policies. It is useful for
    regression and mechanics, not an unbiased headline quality evaluation.
    """

    rng = np.random.default_rng(seed)
    by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_document[chunk.document_id].append(chunk)
    events: list[TraceEvent] = []
    for document_id in sorted(by_document):
        ordered = sorted(by_document[document_id], key=lambda c: c.position)
        if len(ordered) < 4:
            continue
        ids = [c.chunk_id for c in ordered]
        positions = {c.chunk_id: i for i, c in enumerate(ordered)}
        for step, current in enumerate(ordered[:-1]):
            query_type = ("semantic", "structural", "multi-hop")[step % 3]
            if query_type == "structural":
                target = ordered[min(step + 1 + int(rng.integers(0, 2)), len(ordered) - 1)].chunk_id
            else:
                eligible = [
                    cid for cid in ids
                    if cid != current.chunk_id
                    and (query_type == "semantic" or abs(positions[cid] - step) >= 2)
                ]
                if not eligible:
                    eligible = [ids[min(step + 1, len(ids) - 1)]]
                semantic = max(eligible, key=lambda cid: (similarity_matrix[current.chunk_id, cid], -cid))
                if query_type == "multi-hop" and step + 2 < len(ordered):
                    target = ordered[step + 2].chunk_id if step % 2 else semantic
                else:
                    target = semantic
            events.append(
                TraceEvent(
                    event_id=len(events),
                    session_id=f"synthetic:{document_id}",
                    step_id=step,
                    document_id=document_id,
                    current_chunk_id=current.chunk_id,
                    target_chunk_id=int(target),
                    query_type=query_type,
                )
            )
    return events


def split_document_ids(
    document_ids: Iterable[str], seed: int = 42, fractions=(0.6, 0.2, 0.2)
) -> dict[str, set[str]]:
    """Split unique document IDs before trace generation."""

    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to one")
    documents = sorted(set(map(str, document_ids)))
    if len(documents) < 3:
        raise ValueError("at least three documents are required for train/dev/test")
    rng = np.random.default_rng(seed)
    documents = [documents[i] for i in rng.permutation(len(documents))]
    n = len(documents)
    n_train = max(1, int(np.floor(fractions[0] * n)))
    n_dev = max(1, int(np.floor(fractions[1] * n)))
    if n_train + n_dev >= n:
        n_train = max(1, n - 2)
        n_dev = 1
    result = {
        "train": set(documents[:n_train]),
        "dev": set(documents[n_train : n_train + n_dev]),
        "test": set(documents[n_train + n_dev :]),
    }
    if set.union(*result.values()) != set(documents):
        raise AssertionError("split lost a document")
    if any(
        result[a] & result[b]
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise AssertionError("document leakage between splits")
    return result


def _exact_type_labels(n_events: int, proportions: Mapping[str, float]) -> list[str]:
    if n_events < 1:
        raise ValueError("n_events must be positive")
    if not np.isclose(sum(proportions.values()), 1.0):
        raise ValueError("query-type proportions must sum to one")
    raw = {name: n_events * float(value) for name, value in proportions.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remaining = n_events - sum(counts.values())
    order = sorted(raw, key=lambda name: (-(raw[name] - counts[name]), name))
    for name in order[:remaining]:
        counts[name] += 1
    return [name for name in proportions for _ in range(counts[name])]


def stratified_ablation_events(
    chunks: list[Chunk],
    similarity_matrix: np.ndarray,
    n_events: int,
    allowed_document_ids: Iterable[str],
    seed: int = 42,
    proportions: Mapping[str, float] | None = None,
    event_id_start: int = 0,
    session_prefix: str = "ablation",
) -> list[TraceEvent]:
    """Generate an exact, document-safe version of the old synthetic ablation.

    Default counts are 25% semantic, 25% structural and 50% multi-hop. Targets
    intentionally use cosine/position signals for historical comparison, so
    this remains a controlled synthetic diagnostic rather than independent
    workload evidence.
    """

    proportions = proportions or {
        "semantic": 0.25,
        "structural": 0.25,
        "multi-hop": 0.50,
    }
    allowed = set(map(str, allowed_document_ids))
    by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.document_id in allowed:
            by_document[chunk.document_id].append(chunk)
    by_document = {
        document: sorted(values, key=lambda chunk: chunk.position)
        for document, values in by_document.items()
        if len(values) >= 4
    }
    if not by_document:
        raise ValueError("no allowed document has at least four chunks")

    rng = np.random.default_rng(seed)
    labels = _exact_type_labels(n_events, proportions)
    rng.shuffle(labels)
    documents = sorted(by_document)
    document_cycle = [documents[i % len(documents)] for i in range(n_events)]
    rng.shuffle(document_cycle)
    step_by_document: dict[str, int] = defaultdict(int)
    events: list[TraceEvent] = []
    for offset, (query_type, document_id) in enumerate(zip(labels, document_cycle)):
        ordered = by_document[document_id]
        safe_current_index = int(rng.integers(0, max(1, len(ordered) - 3)))
        current = ordered[safe_current_index]
        if query_type == "structural":
            target_index = min(
                len(ordered) - 1,
                safe_current_index + int(rng.choice([1, 2, 3])),
            )
            target = ordered[target_index]
        else:
            eligible = [
                candidate
                for local_index, candidate in enumerate(ordered)
                if abs(local_index - safe_current_index) > 2
            ]
            if not eligible:
                eligible = [ordered[min(len(ordered) - 1, safe_current_index + 1)]]
            semantic_target = max(
                eligible,
                key=lambda candidate: (
                    similarity_matrix[current.chunk_id, candidate.chunk_id],
                    -candidate.chunk_id,
                ),
            )
            if query_type == "multi-hop" and rng.random() < 0.5:
                target = ordered[min(len(ordered) - 1, safe_current_index + 2)]
            else:
                target = semantic_target
        step = step_by_document[document_id]
        step_by_document[document_id] += 1
        events.append(
            TraceEvent(
                event_id=event_id_start + offset,
                session_id=f"{session_prefix}:{document_id}",
                step_id=step,
                document_id=document_id,
                current_chunk_id=current.chunk_id,
                target_chunk_id=target.chunk_id,
                query_type=query_type,
            )
        )
    return events


def balanced_event_sample(
    events: Iterable[TraceEvent], limit: int, seed: int = 42
) -> list[TraceEvent]:
    """Cap a smoke trace without silently dropping later documents/types."""

    events = list(events)
    if limit <= 0 or limit >= len(events):
        return events
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[str, str], list[TraceEvent]] = defaultdict(list)
    for event in events:
        buckets[(event.document_id, event.query_type)].append(event)
    keys = sorted(buckets)
    for key in keys:
        rng.shuffle(buckets[key])
    selected: list[TraceEvent] = []
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    return [
        TraceEvent(
            event_id=index,
            session_id=event.session_id,
            step_id=event.step_id,
            document_id=event.document_id,
            current_chunk_id=event.current_chunk_id,
            target_chunk_id=event.target_chunk_id,
            query_type=event.query_type,
        )
        for index, event in enumerate(selected)
    ]


def load_trace_csv(path: str | Path) -> list[TraceEvent]:
    """Load an independent trace using the documented TraceEvent columns."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "event_id", "session_id", "step_id", "document_id",
            "current_chunk_id", "target_chunk_id",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"trace CSV is missing columns: {sorted(missing)}")
        return [TraceEvent.from_dict(row) for row in reader]


def save_trace_csv(events: Iterable[TraceEvent], path: str | Path) -> None:
    fields = list(TraceEvent.__dataclass_fields__)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(event) for event in events)


def split_by_document(
    events: list[TraceEvent], seed: int = 42, fractions=(0.6, 0.2, 0.2)
) -> dict[str, list[TraceEvent]]:
    """Deterministically split whole documents; never split rows/events."""

    sets = split_document_ids(
        {event.document_id for event in events}, seed=seed, fractions=fractions
    )
    return {
        name: [event for event in events if event.document_id in doc_ids]
        for name, doc_ids in sets.items()
    }


def document_orders(chunks: Iterable[Chunk]) -> dict[str, list[int]]:
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id].append(chunk)
    return {
        document: [c.chunk_id for c in sorted(values, key=lambda c: c.position)]
        for document, values in grouped.items()
    }
