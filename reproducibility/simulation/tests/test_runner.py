from pathlib import Path

from graphkv_sim.cost import CostModel
from graphkv_sim.models import Chunk, PolicyContext, Prediction, TraceEvent
from graphkv_sim.policies import BasePolicy, OnlineAdaptivePolicy
from graphkv_sim.runner import run_policy_trace


class ListPolicy(BasePolicy):
    name = "list_policy"

    def __init__(self, ids):
        self.ids = ids

    def predict(self, context: PolicyContext, k: int) -> Prediction:
        return Prediction(self.ids[:k])


def fixtures():
    chunks = [Chunk(i, "d", i, str(i), 10, 4) for i in range(4)]
    events = [TraceEvent(0, "s", 0, "d", 0, 1, "semantic")]
    return chunks, events


def run_once(tmp_path: Path, policy, cost, capacity=4, suffix="x"):
    chunks, events = fixtures()
    return run_policy_trace(
        policy=policy, events=events, chunks=chunks, k=3, cost_model=cost,
        capacity_entries=capacity, capacity_bytes=None,
        output_csv=tmp_path / f"{suffix}.csv",
        checkpoint_json=tmp_path / f"{suffix}.json",
    )[0]


def test_prediction_and_residency_are_separate(tmp_path):
    row = run_once(tmp_path, ListPolicy([1]), CostModel(), suffix="hit")
    assert row["prediction_hit"] and row["residency_hit"]
    assert row["current_prefetch_hit"] and row["miss_reason"] == "current_prefetch_hit"

    deadline = CostModel(prefetch_mode="deadline", overlap_window_ms=0)
    row = run_once(tmp_path, ListPolicy([1]), deadline, suffix="late")
    assert row["prediction_hit"] and not row["residency_hit"]
    assert row["miss_reason"] == "late_prefetch"


def test_completed_target_can_be_evicted_before_demand(tmp_path):
    row = run_once(tmp_path, ListPolicy([1, 2]), CostModel(), capacity=1, suffix="evict")
    assert row["prediction_hit"] and not row["residency_hit"]
    assert row["miss_reason"] == "evicted_or_rejected_prefetch"


def test_prefetch_does_not_relabel_prior_residency(tmp_path):
    chunks = [Chunk(i, "d", i, str(i), 10, 4) for i in range(3)]
    events = [
        TraceEvent(0, "s", 0, "d", 0, 1, "x"),
        TraceEvent(1, "s", 1, "d", 2, 1, "x"),
    ]
    rows = run_policy_trace(
        policy=ListPolicy([1]), events=events, chunks=chunks, k=1,
        cost_model=CostModel(), capacity_entries=4, capacity_bytes=None,
        output_csv=tmp_path / "prior.csv", checkpoint_json=tmp_path / "prior.json",
    )
    assert rows[1]["miss_reason"] == "prior_residency"
    assert not rows[1]["current_prefetch_hit"]
    assert rows[1]["completed_prefetch_ids"] == []
    assert rows[1]["prefetch_already_resident_ids"] == [1]


def test_resume_restores_cache_policy_and_rows(tmp_path):
    chunks = [Chunk(i, "d", i, str(i), 10, 4) for i in range(4)]
    events = [
        TraceEvent(0, "s", 0, "d", 0, 1, "x"),
        TraceEvent(1, "s", 1, "d", 1, 2, "x"),
        TraceEvent(2, "s", 2, "d", 2, 3, "x"),
    ]
    edges = {
        0: [(1, .2, 1), (2, 1, 0)],
        1: [(0, .2, 1), (2, .2, 1), (3, 1, 0)],
        2: [(1, .2, 1), (3, .2, 1)],
        3: [(2, .2, 1)],
    }
    common = dict(
        events=events, chunks=chunks, k=1, cost_model=CostModel(),
        capacity_entries=2, capacity_bytes=None,
    )
    checkpoint = tmp_path / "resume.json"
    run_policy_trace(
        policy=OnlineAdaptivePolicy(edges, (.8, .2), .2),
        output_csv=tmp_path / "resume.csv", checkpoint_json=checkpoint,
        stop_after=1, **common,
    )
    resumed = run_policy_trace(
        policy=OnlineAdaptivePolicy(edges, (.8, .2), .2),
        output_csv=tmp_path / "resume.csv", checkpoint_json=checkpoint, **common,
    )
    full = run_policy_trace(
        policy=OnlineAdaptivePolicy(edges, (.8, .2), .2),
        output_csv=tmp_path / "full.csv", checkpoint_json=tmp_path / "full.json", **common,
    )
    for rows in (resumed, full):
        for row in rows:
            row.pop("policy_cpu_ms")
    assert resumed == full
