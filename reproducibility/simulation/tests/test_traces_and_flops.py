from graphkv_sim.flops import legacy_attention_flops_saved
import numpy as np

from graphkv_sim.models import Chunk, TraceEvent
from graphkv_sim.traces import (
    balanced_event_sample,
    split_by_document,
    stratified_ablation_events,
)


def test_document_split_has_no_leakage_and_is_deterministic():
    events = [TraceEvent(i, f"s{i}", 0, f"d{i}", i, i, "x") for i in range(9)]
    first = split_by_document(events, 7)
    second = split_by_document(events, 7)
    assert first == second
    documents = [{event.document_id for event in first[name]} for name in ("train", "dev", "test")]
    assert not (documents[0] & documents[1] or documents[0] & documents[2] or documents[1] & documents[2])


def test_legacy_flop_formula_matches_supplied_code_and_ignores_new_length_after_cancellation():
    first = legacy_attention_flops_saved(100, 10, 896, 24)
    second = legacy_attention_flops_saved(100, 50, 896, 24)
    assert first == second == 4 * 896 * 100**2 * 24


def test_balanced_smoke_sample_covers_documents_and_types():
    events = []
    for document in range(3):
        for step in range(9):
            events.append(
                TraceEvent(
                    len(events), f"s{document}", step, f"d{document}", 0, 1,
                    ("semantic", "structural", "multi-hop")[step % 3],
                )
            )
    sampled = balanced_event_sample(events, 9, seed=3)
    assert len({event.document_id for event in sampled}) == 3
    assert {event.query_type for event in sampled} == {"semantic", "structural", "multi-hop"}


def test_stratified_ablation_has_exact_old_mix_and_heldout_documents():
    chunks = [
        Chunk(document * 6 + position, f"d{document}", position, "x", 1, 1)
        for document in range(3)
        for position in range(6)
    ]
    similarity = np.eye(len(chunks))
    events = stratified_ablation_events(
        chunks, similarity, 60, {"d1", "d2"}, seed=9
    )
    counts = {name: sum(event.query_type == name for event in events) for name in ("semantic", "structural", "multi-hop")}
    assert counts == {"semantic": 15, "structural": 15, "multi-hop": 30}
    assert {event.document_id for event in events} == {"d1", "d2"}
    assert all(chunks[event.current_chunk_id].document_id == chunks[event.target_chunk_id].document_id for event in events)
