# Copyright 2026 Tinker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-track-id state is bounded by lazy eviction of gone ids.

``candidate_consistency`` and ``relative_positions`` are keyed by ByteTrack
track_id. ByteTrack ids increase monotonically for the life of the process (a
fresh id every time someone enters/leaves), so without eviction these dicts grow
one entry per id forever — a slow host-RAM leak feeding the long-run
swap-thrash. ``_prune_track_state`` bounds them WITHOUT changing scoring for
realistic scenes:

- It only prunes once a dict exceeds ``MAX_TRACK_STATE_IDS`` (256), and
- it only evicts ids NOT currently visible.

So a scene with <= cap distinct ids behaves EXACTLY as before, and a
currently-visible id (even one flickering through occlusion) is never evicted.

The tests bind the *real* ``YOLOTracker`` methods onto a lightweight
``SimpleNamespace`` (via ``__get__``) so the YOLO model never loads — we exercise
the genuine method logic, not a reimplementation.
"""
from types import SimpleNamespace

from vision_track.core.tracking_types import TrackingResult
from vision_track.yolo_tracker import YOLOTracker


def _make_tracker():
    """A minimal duck-typed tracker carrying only the per-id state + bound methods.

    The YOLO model is never constructed; we attach the unbound methods from the
    real class so the logic under test is the production logic.
    """
    t = SimpleNamespace()
    t.candidate_consistency = {}
    t.relative_positions = {}
    # CONSISTENCY_WINDOW / CONSISTENCY_THRESHOLD are instance attrs set in the
    # real __init__; replicate the production values so scoring matches.
    t.CONSISTENCY_WINDOW = 5
    t.CONSISTENCY_THRESHOLD = 0.15
    t.MAX_TRACK_STATE_IDS = YOLOTracker.MAX_TRACK_STATE_IDS
    # Bind the real methods to the namespace.
    t._prune_track_state = YOLOTracker._prune_track_state.__get__(t, SimpleNamespace)
    t._update_candidate_consistency = \
        YOLOTracker._update_candidate_consistency.__get__(t, SimpleNamespace)
    t._get_candidate_consistency_score = \
        YOLOTracker._get_candidate_consistency_score.__get__(t, SimpleNamespace)
    t._update_relative_positions = \
        YOLOTracker._update_relative_positions.__get__(t, SimpleNamespace)
    t._check_relative_position_consistency = \
        YOLOTracker._check_relative_position_consistency.__get__(t, SimpleNamespace)
    return t


def _res(track_id, cx=100.0, cy=100.0, class_id=0):
    half = 10
    return TrackingResult(
        track_id=track_id,
        bbox=(int(cx - half), int(cy - half), int(cx + half), int(cy + half)),
        mask=None,
        confidence=0.9,
        class_id=class_id,
        class_name="person" if class_id == 0 else "thing",
    )


def test_max_track_state_ids_constant():
    assert YOLOTracker.MAX_TRACK_STATE_IDS == 256


def test_candidate_consistency_dict_is_bounded():
    """Feeding 1000 distinct ids with few current each frame stays bounded."""
    t = _make_tracker()
    current_ids = set()
    for tid in range(1000):
        # Each "frame" only a couple of ids are visible; the rest are gone.
        t._update_candidate_consistency(tid, 0.8)
        current_ids = {tid}
        t._prune_track_state(current_ids)
    # Bounded near the cap, NOT grown to 1000.
    assert len(t.candidate_consistency) <= t.MAX_TRACK_STATE_IDS + len(current_ids)
    assert len(t.candidate_consistency) < 1000


def test_relative_positions_dict_is_bounded():
    """relative_positions keyed by other-person ids stays bounded over a run."""
    t = _make_tracker()
    target = _res(10 ** 6, cx=320.0, cy=240.0)  # stable target id, never an "other"
    current_ids = set()
    for tid in range(1000):
        other = _res(tid, cx=100.0 + tid, cy=120.0)
        t._update_relative_positions(target, [target, other])
        current_ids = {target.track_id, tid}
        t._prune_track_state(current_ids)
    assert len(t.relative_positions) <= t.MAX_TRACK_STATE_IDS + len(current_ids)
    assert len(t.relative_positions) < 1000


def test_no_eviction_below_cap_consistency_score_unchanged():
    """With <= cap distinct ids, NO eviction happens; scoring is identical."""
    t = _make_tracker()
    # Build consistency history for a handful of ids (far under the cap).
    sims = {1: [0.80, 0.81, 0.79, 0.80], 2: [0.10, 0.90, 0.20, 0.85]}
    for tid, seq in sims.items():
        for s in seq:
            t._update_candidate_consistency(tid, s)

    # Record scores BEFORE any prune.
    before = {tid: t._get_candidate_consistency_score(tid) for tid in sims}
    keys_before = dict(t.candidate_consistency)

    # Prune with all ids "current" — and also with the dict well under cap.
    t._prune_track_state(set(sims.keys()))

    # Nothing evicted, scores identical.
    assert t.candidate_consistency == keys_before
    for tid in sims:
        assert t._get_candidate_consistency_score(tid) == before[tid]


def test_current_id_never_evicted_even_when_over_cap():
    """A currently-visible id survives pruning even after the dict overflows."""
    t = _make_tracker()
    # Stuff the dict well past the cap with gone ids.
    for tid in range(t.MAX_TRACK_STATE_IDS + 50):
        t._update_candidate_consistency(tid, 0.5)
    # A specific id we care about, given a distinctive history.
    keep = 12345
    for s in [0.80, 0.80, 0.80, 0.80]:
        t._update_candidate_consistency(keep, s)
    score_before = t._get_candidate_consistency_score(keep)

    t._prune_track_state({keep})

    # keep is still present with the same history/score; the gone ids were culled.
    assert keep in t.candidate_consistency
    assert t._get_candidate_consistency_score(keep) == score_before
    assert len(t.candidate_consistency) <= t.MAX_TRACK_STATE_IDS + 1


def test_relative_position_lookup_unchanged_for_current_id():
    """A visible other-person's stored relative position survives + reads same."""
    t = _make_tracker()
    target = _res(999999, cx=300.0, cy=200.0)
    # Overflow with gone others.
    for tid in range(t.MAX_TRACK_STATE_IDS + 30):
        t._update_relative_positions(target, [target, _res(tid, cx=50.0 + tid, cy=60.0)])
    # The one we keep visible.
    keep = 7777
    keep_other = _res(keep, cx=420.0, cy=205.0)
    t._update_relative_positions(target, [target, keep_other])
    stored_before = t.relative_positions[keep]

    t._prune_track_state({target.track_id, keep})

    assert keep in t.relative_positions
    assert t.relative_positions[keep] == stored_before
    assert len(t.relative_positions) <= t.MAX_TRACK_STATE_IDS + 2


def test_prune_is_noop_when_exactly_at_cap():
    """At exactly the cap (not over), nothing is evicted (boundary behavior)."""
    t = _make_tracker()
    for tid in range(t.MAX_TRACK_STATE_IDS):
        t._update_candidate_consistency(tid, 0.5)
    assert len(t.candidate_consistency) == t.MAX_TRACK_STATE_IDS
    t._prune_track_state(set())  # no current ids, but at cap -> no eviction
    assert len(t.candidate_consistency) == t.MAX_TRACK_STATE_IDS
