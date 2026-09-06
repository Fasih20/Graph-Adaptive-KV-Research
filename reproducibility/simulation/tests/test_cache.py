from graphkv_sim.cache import LRUCache


def test_entry_and_byte_capacity_and_state_round_trip():
    cache = LRUCache(capacity_entries=2, capacity_bytes=10)
    cache.put(1, 4, "demand", 0)
    cache.put(2, 4, "prefetch:0", 0)
    cache.access(1)
    result = cache.put(3, 4, "prefetch:1", 1)
    assert result.evicted_ids == (2,)
    assert cache.contains(1) and cache.contains(3) and not cache.contains(2)
    restored = LRUCache.from_state_dict(cache.state_dict())
    assert restored.state_dict() == cache.state_dict()


def test_oversize_insert_does_not_destroy_resident_cache():
    cache = LRUCache(capacity_bytes=8)
    cache.put(1, 4, "demand", 0)
    result = cache.put(2, 9, "prefetch:0", 0)
    assert result.rejected_oversize and not result.inserted
    assert cache.contains(1) and not cache.contains(2)

