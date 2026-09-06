from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    document_id: str
    position: int
    text: str
    token_count: int
    kv_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceEvent:
    event_id: int
    session_id: str
    step_id: int
    document_id: str
    current_chunk_id: int
    target_chunk_id: int
    query_type: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TraceEvent":
        return cls(
            event_id=int(value["event_id"]),
            session_id=str(value["session_id"]),
            step_id=int(value["step_id"]),
            document_id=str(value["document_id"]),
            current_chunk_id=int(value["current_chunk_id"]),
            target_chunk_id=int(value["target_chunk_id"]),
            query_type=str(value.get("query_type", "unknown")),
        )


@dataclass(frozen=True)
class PolicyContext:
    """Information available before the next target is revealed.

    Deliberately contains no target chunk id. The runner constructs this object
    before calling a non-oracle policy, making future-target leakage harder to
    introduce accidentally.
    """

    event_id: int
    session_id: str
    step_id: int
    document_id: str
    current_chunk_id: int
    query_type: str


@dataclass
class Prediction:
    ids: list[int]
    scores: dict[int, float] = field(default_factory=dict)
    features: dict[int, tuple[float, float]] = field(default_factory=dict)
    candidate_universe_ids: set[int] | None = None
    uses_future_target: bool = False
