from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass
class CacheEntry:
    chunk_id: int
    size_bytes: int
    source: str
    inserted_event_id: int


@dataclass(frozen=True)
class AccessResult:
    hit: bool
    source: str | None = None


@dataclass(frozen=True)
class InsertResult:
    inserted: bool
    evicted_ids: tuple[int, ...] = ()
    rejected_oversize: bool = False


class LRUCache:
    """LRU cache constrained by entries, bytes, or both.

    Values are stored from least recently used to most recently used. A miss
    never inserts automatically: demand loads and prefetches are explicit in
    the runner, so their provenance cannot be conflated.
    """

    def __init__(
        self,
        capacity_entries: int | None = None,
        capacity_bytes: int | None = None,
    ) -> None:
        if capacity_entries is None and capacity_bytes is None:
            raise ValueError("at least one cache capacity must be supplied")
        if capacity_entries is not None and capacity_entries < 1:
            raise ValueError("capacity_entries must be positive")
        if capacity_bytes is not None and capacity_bytes < 1:
            raise ValueError("capacity_bytes must be positive")
        self.capacity_entries = capacity_entries
        self.capacity_bytes = capacity_bytes
        self._entries: OrderedDict[int, CacheEntry] = OrderedDict()
        self.current_bytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    def contains(self, chunk_id: int) -> bool:
        return int(chunk_id) in self._entries

    def peek(self, chunk_id: int) -> CacheEntry | None:
        return self._entries.get(int(chunk_id))

    def access(self, chunk_id: int) -> AccessResult:
        chunk_id = int(chunk_id)
        entry = self._entries.get(chunk_id)
        if entry is None:
            return AccessResult(False, None)
        self._entries.move_to_end(chunk_id)
        return AccessResult(True, entry.source)

    def put(
        self, chunk_id: int, size_bytes: int, source: str, event_id: int
    ) -> InsertResult:
        chunk_id, size_bytes = int(chunk_id), int(size_bytes)
        if size_bytes < 0:
            raise ValueError("size_bytes cannot be negative")
        if self.capacity_bytes is not None and size_bytes > self.capacity_bytes:
            return InsertResult(False, rejected_oversize=True)

        old = self._entries.pop(chunk_id, None)
        if old is not None:
            self.current_bytes -= old.size_bytes

        entry = CacheEntry(chunk_id, size_bytes, str(source), int(event_id))
        self._entries[chunk_id] = entry
        self.current_bytes += size_bytes
        evicted: list[int] = []
        while self._over_capacity():
            evicted_id, evicted_entry = self._entries.popitem(last=False)
            self.current_bytes -= evicted_entry.size_bytes
            evicted.append(evicted_id)
        return InsertResult(chunk_id in self._entries, tuple(evicted), False)

    def _over_capacity(self) -> bool:
        too_many = (
            self.capacity_entries is not None
            and len(self._entries) > self.capacity_entries
        )
        too_large = (
            self.capacity_bytes is not None
            and self.current_bytes > self.capacity_bytes
        )
        return bool(too_many or too_large)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity_entries": self.capacity_entries,
            "capacity_bytes": self.capacity_bytes,
            "current_bytes": self.current_bytes,
            "entries_lru_to_mru": [asdict(v) for v in self._entries.values()],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "LRUCache":
        cache = cls(
            capacity_entries=state.get("capacity_entries"),
            capacity_bytes=state.get("capacity_bytes"),
        )
        for value in state.get("entries_lru_to_mru", []):
            entry = CacheEntry(
                chunk_id=int(value["chunk_id"]),
                size_bytes=int(value["size_bytes"]),
                source=str(value["source"]),
                inserted_event_id=int(value["inserted_event_id"]),
            )
            cache._entries[entry.chunk_id] = entry
            cache.current_bytes += entry.size_bytes
        if cache.current_bytes != int(state.get("current_bytes", cache.current_bytes)):
            raise ValueError("corrupt cache checkpoint: byte total does not match")
        if cache._over_capacity():
            raise ValueError("corrupt cache checkpoint: state exceeds capacity")
        return cache

