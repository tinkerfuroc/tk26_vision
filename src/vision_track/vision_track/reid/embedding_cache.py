"""ROS-free, torch-free per-frame embedding cache.

Eliminates the up-to-4x/frame re-embedding of the same person crop across
_score_candidates / _verify_person_candidate / periodic_reid_validation /
_confirm_reid_candidate. Scoped to a single frame_seq: when the tracker advances
to a new frame, the previous frame's entries are dropped (appearances are not
reused across frames — only within the one update() call). Bounded LRU so a
crowd of stale track_ids cannot grow it without limit.
"""
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class FrameEmbeddingCache:
    def __init__(self, max_entries: int = 32):
        self._max = max(1, int(max_entries))
        self._store: "OrderedDict[Tuple[int, int], Dict[str, Any]]" = OrderedDict()
        self._frame_seq: Optional[int] = None

    def begin_frame(self, frame_seq: int) -> None:
        """Mark the start of processing for frame_seq; drop prior-frame entries."""
        if frame_seq != self._frame_seq:
            self._store.clear()
            self._frame_seq = frame_seq

    def get(self, track_id: int, frame_seq: int) -> Optional[Dict[str, Any]]:
        if frame_seq != self._frame_seq:
            return None
        key = (track_id, frame_seq)
        val = self._store.get(key)
        if val is not None:
            self._store.move_to_end(key)  # mark MRU
        return val

    def put(self, track_id: int, frame_seq: int, features: Dict[str, Any]) -> None:
        # Auto-begin a frame on first put so callers may skip begin_frame.
        if frame_seq != self._frame_seq:
            self.begin_frame(frame_seq)
        key = (track_id, frame_seq)
        self._store[key] = features
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)  # evict LRU

    def clear(self) -> None:
        self._store.clear()
        self._frame_seq = None
