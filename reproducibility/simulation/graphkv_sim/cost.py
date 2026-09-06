from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor


@dataclass(frozen=True)
class PrefetchSchedule:
    completed_count: int
    lookup_ms: float
    prefetch_ms: float
    exposed_prefetch_ms: float


@dataclass(frozen=True)
class CostModel:
    """Transparent latency proxy; parameters may be calibrated empirically."""

    cache_lookup_ms: float = 0.01
    miss_intercept_ms: float = 0.05
    miss_ms_per_token: float = 0.002
    prefetch_intercept_ms: float = 0.01
    prefetch_ms_per_mb: float = 0.02
    overlap_window_ms: float = 0.0
    prefetch_mode: str = "blocking"  # blocking or deadline

    def __post_init__(self) -> None:
        if self.prefetch_mode not in {"blocking", "deadline"}:
            raise ValueError("prefetch_mode must be 'blocking' or 'deadline'")
        for name, value in asdict(self).items():
            if name != "prefetch_mode" and float(value) < 0:
                raise ValueError(f"{name} cannot be negative")

    def demand_miss_ms(self, token_count: int) -> float:
        return self.miss_intercept_ms + self.miss_ms_per_token * int(token_count)

    def one_prefetch_ms(self, size_bytes: int) -> float:
        mb = int(size_bytes) / (1024.0 * 1024.0)
        return self.prefetch_intercept_ms + self.prefetch_ms_per_mb * mb

    def schedule(self, sizes_bytes: list[int]) -> PrefetchSchedule:
        costs = [self.one_prefetch_ms(size) for size in sizes_bytes]
        total = float(sum(costs))
        if self.prefetch_mode == "blocking":
            completed = len(costs)
            exposed = max(0.0, total - self.overlap_window_ms)
        else:
            elapsed = 0.0
            completed = 0
            for cost in costs:
                if elapsed + cost > self.overlap_window_ms + 1e-12:
                    break
                elapsed += cost
                completed += 1
            exposed = 0.0
        return PrefetchSchedule(
            completed_count=completed,
            lookup_ms=self.cache_lookup_ms,
            prefetch_ms=total,
            exposed_prefetch_ms=exposed,
        )

    def to_dict(self) -> dict:
        return asdict(self)

