import pytest

from vision_track.reid.embedding_cache import FrameEmbeddingCache


def test_miss_then_hit():
    c = FrameEmbeddingCache(max_entries=8)
    assert c.get(track_id=3, frame_seq=10) is None
    c.put(track_id=3, frame_seq=10, features={"reid": [1.0]})
    assert c.get(track_id=3, frame_seq=10) == {"reid": [1.0]}


def test_new_frame_seq_invalidates_old_frame():
    c = FrameEmbeddingCache(max_entries=8)
    c.put(track_id=3, frame_seq=10, features={"reid": [1.0]})
    # Touching frame 11 drops everything from frame 10.
    c.begin_frame(11)
    assert c.get(track_id=3, frame_seq=10) is None
    assert c.get(track_id=3, frame_seq=11) is None
    c.put(track_id=3, frame_seq=11, features={"reid": [2.0]})
    assert c.get(track_id=3, frame_seq=11) == {"reid": [2.0]}


def test_get_with_stale_frame_seq_returns_none():
    c = FrameEmbeddingCache(max_entries=8)
    c.begin_frame(5)
    c.put(track_id=1, frame_seq=5, features={"reid": [9.0]})
    # A read tagged with a different (older) frame_seq must miss, not return stale.
    assert c.get(track_id=1, frame_seq=4) is None


def test_bounded_lru_eviction_within_frame():
    c = FrameEmbeddingCache(max_entries=2)
    c.begin_frame(7)
    c.put(track_id=1, frame_seq=7, features={"a": 1})
    c.put(track_id=2, frame_seq=7, features={"b": 2})
    c.get(track_id=1, frame_seq=7)            # touch 1 -> 2 is now LRU
    c.put(track_id=3, frame_seq=7, features={"c": 3})  # evicts track 2
    assert c.get(track_id=2, frame_seq=7) is None
    assert c.get(track_id=1, frame_seq=7) == {"a": 1}
    assert c.get(track_id=3, frame_seq=7) == {"c": 3}


def test_clear():
    c = FrameEmbeddingCache(max_entries=4)
    c.put(track_id=1, frame_seq=1, features={"x": 0})
    c.clear()
    assert c.get(track_id=1, frame_seq=1) is None
