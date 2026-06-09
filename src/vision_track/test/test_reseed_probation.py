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
"""Reseed confirmation gate (Issue 2 / Phase 2).

A reseed (manual dashboard click OR waving auto-reseed — same service path) no
longer instant-locks on a single IoU frame. It seeds a *probation*: the seeded
id must be PRESENT (matched by ByteTrack) AND ReID-confirmed
(sim >= reid_threshold) for ``reseed_confirmation_frames`` consecutive frames
before the lock commits (target_lost flips False). During probation the tracker
reports target_lost=True. A present-but-unconfirmed frame resets the count; an
absent seeded-id frame abandons probation (falls back to normal recovery).
"""
from types import SimpleNamespace

import numpy as np

import vision_track.core.tracking_pipeline as TP
from vision_track.core.lock_state_machine import LockStateMachine
from vision_track.core.tracking_types import (
    TargetAppearance, TrackerState, TrackingResult)
from vision_track.yolo_tracker import YOLOTracker


def _v(*vals, dim=8):
    a = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vals):
        a[i] = x
    return a


# --- _apply_reseed seeds probation (no instant lock) ----------------------


def _bare_tracker():
    t = YOLOTracker.__new__(YOLOTracker)          # bypass heavy __init__
    t.target_appearance = TargetAppearance(class_id=0, class_name="person")
    t.target_appearance.configure_gallery(enabled=True, size=6, novelty_max=0.99,
                                          score_mode="max")
    t.target_appearance.gallery.maybe_add(_v(1, 0))   # pre-existing identity view
    t.target_track_id = 3
    t.original_track_id = 3
    t.frames_lost = 40
    t.state = TrackerState.REIDENTIFYING
    t.lock_state_machine = LockStateMachine()
    t.reseed_probation_id = None
    t.reseed_probation_count = 0
    t.reseed_confirmation_frames = 5
    # __init__ is bypassed, so seed occlusion state explicitly.
    t.is_occluded = True
    t.pre_occlusion_appearance = object()
    return t


def test_apply_reseed_seeds_probation_does_not_instant_lock():
    t = _bare_tracker()
    det = TrackingResult(track_id=9, bbox=(10, 10, 50, 120), mask=None,
                         confidence=0.9, class_id=0, class_name="person")
    fresh = _v(0, 1)
    tid = t._apply_reseed(det, fresh)

    # The service still reports the accepted id ("confirming", not "locked").
    assert tid == 9
    assert t.target_track_id == 9 and t.original_track_id == 9
    # NOT an instant lock: state is REIDENTIFYING, not TRACKING.
    assert t.state == TrackerState.REIDENTIFYING
    # Probation armed.
    assert t.reseed_probation_id == 9
    assert t.reseed_probation_count == 0
    # FSM is probationary (reidentifying), NOT tracking.
    assert t.lock_state_machine._state == "reidentifying"
    assert t.lock_state_machine._committed_id == 9
    # Gallery still got the fresh crop (preserved + appended).
    assert len(t.target_appearance.gallery) == 2
    # Occlusion bookkeeping cleared on re-lock.
    assert t.is_occluded is False
    assert t.pre_occlusion_appearance is None


def test_apply_reseed_none_detection_fails():
    t = _bare_tracker()
    assert t._apply_reseed(None, _v(0, 1)) == -1
    assert t.reseed_probation_id is None          # no probation on failure


# --- per-frame probation step --------------------------------------------


class _StubExtractor:
    def extract_features(self, *a, **kw):
        return {"reid": _v(0, 1)}


def _probation_tracker(sim_value, reseed_frames=5):
    """Minimal tracker for driving _step_reseed_probation, with a controllable
    ReID similarity (monkeypatched at the module level below)."""
    fsm = LockStateMachine()
    fsm.start_probation(7)
    return SimpleNamespace(
        reseed_probation_id=7,
        reseed_probation_count=0,
        reseed_confirmation_frames=reseed_frames,
        reid_threshold=0.55,
        frames_lost=3,
        state=TrackerState.REIDENTIFYING,
        original_track_id=7,
        target_track_id=7,
        lock_state_machine=fsm,
        last_frame_recovery=False,
        last_lock_decision=None,
        target_appearance=TargetAppearance(class_id=0, class_name="person"),
        appearance_extractor=_StubExtractor(),
        embedding_cache=None,
        frame_count=10,
        _with_original_id=lambda r: TrackingResult(
            track_id=7, bbox=r.bbox, mask=r.mask, confidence=r.confidence,
            class_id=r.class_id, class_name=r.class_name),
    )


def _present(track_id=7):
    return TrackingResult(track_id=track_id, bbox=(0, 0, 10, 20), mask=None,
                          confidence=0.9, class_id=0, class_name="person")


def test_present_confirmed_frames_commit_at_threshold(monkeypatch):
    """N present+confirmed frames keep target_lost=True until the count reaches
    reseed_confirmation_frames, then commit (target_lost=False, FSM tracking)."""
    monkeypatch.setattr(TP.ReIDMatcher, "compute_similarity",
                        staticmethod(lambda *a, **kw: 0.90))
    tr = _probation_tracker(0.90, reseed_frames=5)
    results = [_present()]

    committed = None
    for i in range(1, 6):
        handled, out = TP._step_reseed_probation(tr, frame=None, results=results)
        assert handled is True
        assert tr.last_frame_recovery is True
        if i < 5:
            # Still confirming: not committed, target_lost stays True.
            assert out is None
            assert tr.reseed_probation_count == i
            assert tr.last_lock_decision.target_lost is True
            assert tr.last_lock_decision.state == "reidentifying"
            assert tr.reseed_probation_id == 7
        else:
            committed = out

    # Fifth confirmed frame commits.
    assert committed is not None
    assert committed.track_id == 7            # original-id stamped
    assert tr.state == TrackerState.TRACKING
    assert tr.reseed_probation_id is None     # probation cleared on commit
    assert tr.reseed_probation_count == 0
    assert tr.last_lock_decision.target_lost is False
    assert tr.last_lock_decision.state == "tracking"


def test_present_unconfirmed_frame_resets_count(monkeypatch):
    """A present-but-sim<reid_threshold frame resets the streak to 0."""
    sims = iter([0.90, 0.90, 0.30, 0.90])      # third frame dips below threshold
    monkeypatch.setattr(TP.ReIDMatcher, "compute_similarity",
                        staticmethod(lambda *a, **kw: next(sims)))
    tr = _probation_tracker(0.0, reseed_frames=5)
    results = [_present()]

    TP._step_reseed_probation(tr, None, results)   # 0.90 -> count 1
    assert tr.reseed_probation_count == 1
    TP._step_reseed_probation(tr, None, results)   # 0.90 -> count 2
    assert tr.reseed_probation_count == 2
    handled, out = TP._step_reseed_probation(tr, None, results)   # 0.30 -> reset
    assert handled is True
    assert out is None
    assert tr.reseed_probation_count == 0
    assert tr.last_lock_decision.target_lost is True
    # Still in probation (not abandoned, not committed).
    assert tr.reseed_probation_id == 7
    TP._step_reseed_probation(tr, None, results)   # 0.90 -> count climbs again
    assert tr.reseed_probation_count == 1


def test_absent_seeded_id_abandons_probation(monkeypatch):
    """An absent seeded-id frame abandons probation (handled=False so the caller
    falls through to normal recovery)."""
    monkeypatch.setattr(TP.ReIDMatcher, "compute_similarity",
                        staticmethod(lambda *a, **kw: 0.90))
    tr = _probation_tracker(0.90, reseed_frames=5)
    # Seeded id 7 is NOT among the detections (a different id present).
    results = [_present(track_id=42)]

    handled, out = TP._step_reseed_probation(tr, None, results)
    assert handled is False                  # caller must fall through
    assert out is None
    assert tr.reseed_probation_id is None    # probation cleared
    assert tr.reseed_probation_count == 0


def test_no_detections_abandons_probation(monkeypatch):
    monkeypatch.setattr(TP.ReIDMatcher, "compute_similarity",
                        staticmethod(lambda *a, **kw: 0.90))
    tr = _probation_tracker(0.90, reseed_frames=5)
    handled, out = TP._step_reseed_probation(tr, None, results=[])
    assert handled is False
    assert tr.reseed_probation_id is None


def test_update_tracker_routes_probation_before_track_by_id(monkeypatch):
    """update_tracker must run the probation gate BEFORE track_by_id so the
    seeded id is not instant-locked by ByteTrack id."""
    monkeypatch.setattr(TP.ReIDMatcher, "compute_similarity",
                        staticmethod(lambda *a, **kw: 0.90))

    def _boom_track_by_id(*a, **kw):
        raise AssertionError("track_by_id must not run during probation")

    monkeypatch.setattr(TP, "track_by_id", _boom_track_by_id)

    tr = _probation_tracker(0.90, reseed_frames=5)
    # update_tracker touches more attributes; supply the few it needs.
    tr.state = TrackerState.REIDENTIFYING
    tr.frame_count = 0
    tr.embedding_cache = None
    tr.track = lambda frame, persist=True: [_present()]
    tr._t_yolo_ms = 0.0

    out = TP.update_tracker(tr, frame=np.zeros((4, 4, 3), dtype=np.uint8))
    # First probation frame: not committed yet → returns None.
    assert out is None
    assert tr.reseed_probation_count == 1
