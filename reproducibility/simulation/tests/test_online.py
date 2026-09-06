import dataclasses

from graphkv_sim.models import PolicyContext
from graphkv_sim.policies import OnlineAdaptivePolicy


def test_policy_context_has_no_target_field():
    assert "target_chunk_id" not in {field.name for field in dataclasses.fields(PolicyContext)}


def test_online_updates_on_reachable_ranking_miss_only():
    raw_edges = {
        0: [(1, 0.1, 1.0), (2, 1.0, 0.0)],
        1: [(0, 0.1, 1.0)],
        2: [(0, 1.0, 0.0)],
    }
    context = PolicyContext(0, "s", 0, "d", 0, "unknown")
    policy = OnlineAdaptivePolicy(raw_edges, (0.9, 0.1), eta=0.5)
    prediction = policy.predict(context, 1)
    assert prediction.ids == [2]
    result = policy.observe(context, 1, prediction)
    assert result["updated"] and result["update_reason"] == "ranking_miss_update"
    assert policy.weights[1] > 0.1

    before = policy.weights.copy()
    prediction = policy.predict(context, 1)
    result = policy.observe(context, 99, prediction)
    assert result["update_reason"] == "coverage_miss"
    assert (policy.weights == before).all()

