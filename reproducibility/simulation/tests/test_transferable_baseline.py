from graphkv_sim.models import PolicyContext, TraceEvent
from graphkv_sim.policies import RelativeOffsetTransitionPolicy


def test_relative_offset_transition_generalizes_to_unseen_chunk_ids():
    orders = {"train": [0, 1, 2, 3], "test": [10, 11, 12, 13]}
    train = [
        TraceEvent(0, "s", 0, "train", 0, 2, "x"),
        TraceEvent(1, "s", 1, "train", 1, 3, "x"),
    ]
    policy = RelativeOffsetTransitionPolicy.fit(train, orders)
    context = PolicyContext(2, "t", 0, "test", 10, "unknown")
    prediction = policy.predict(context, 1)
    assert prediction.ids == [12]
    assert prediction.candidate_universe_ids == {11, 12, 13}
