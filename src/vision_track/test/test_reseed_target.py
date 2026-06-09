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
"""Gallery-preserving re-seed: _apply_reseed preserves gallery + seeds probation.

Issue 2: a reseed no longer instant-locks. _apply_reseed re-locks the ids and
appends the fresh crop, but enters a REIDENTIFYING *probation* (FSM
start_probation, NOT start) — the per-frame gate in tracking_pipeline confirms
the appearance over N frames before the lock commits. These tests assert the
probation seeding (the gate's per-frame behavior lives in test_reseed_probation).
"""
import numpy as np

from vision_track.core.tracking_types import TargetAppearance, TrackingResult, TrackerState
from vision_track.yolo_tracker import YOLOTracker


def _v(*vals, dim=8):
    a = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vals):
        a[i] = x
    return a


class _Lsm:
    def __init__(self):
        self.started = None
        self.probation_started = None

    def start(self, tid):
        self.started = tid

    def start_probation(self, tid):
        self.probation_started = tid


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
    t.lock_state_machine = _Lsm()
    # Issue 2: __init__ is bypassed, so seed probation state explicitly.
    t.reseed_probation_id = None
    t.reseed_probation_count = 0
    t.reseed_confirmation_frames = 5
    # __init__ is bypassed, so seed occlusion state explicitly (D2 clears it).
    t.is_occluded = True
    t.pre_occlusion_appearance = object()
    return t


def test_apply_reseed_preserves_gallery_and_seeds_probation():
    t = _bare_tracker()
    det = TrackingResult(track_id=9, bbox=(10, 10, 50, 120), mask=None,
                         confidence=0.9, class_id=0, class_name="person")
    fresh = _v(0, 1)                              # a new, distinct confirmed view
    tid = t._apply_reseed(det, fresh)
    assert tid == 9
    assert t.target_track_id == 9 and t.original_track_id == 9
    # Issue 2: probation, NOT an instant TRACKING lock.
    assert t.state == TrackerState.REIDENTIFYING
    assert t.reseed_probation_id == 9
    assert t.reseed_probation_count == 0
    assert t.frames_lost == 0
    # gallery preserved (still has the old view) AND the fresh view appended
    assert len(t.target_appearance.gallery) == 2
    # FSM re-armed PROBATIONALLY on the new id (start_probation, not start).
    assert t.lock_state_machine.probation_started == 9
    assert t.lock_state_machine.started is None
    # D2: occlusion bookkeeping cleared on re-lock
    assert t.is_occluded is False
    assert t.pre_occlusion_appearance is None


def test_apply_reseed_none_detection_fails():
    t = _bare_tracker()
    assert t._apply_reseed(None, _v(0, 1)) == -1
    assert t.target_track_id == 3                 # unchanged on failure


def test_apply_reseed_none_feature_relocks_without_gallery_growth():
    # ReID extraction yielding nothing must still re-lock the ids + seed
    # probation, but must NOT grow the gallery (no fresh view to append) — the
    # common real-world reseed.
    t = _bare_tracker()
    det = TrackingResult(track_id=9, bbox=(10, 10, 50, 120), mask=None,
                         confidence=0.9, class_id=0, class_name="person")
    tid = t._apply_reseed(det, None)
    assert tid == 9
    assert t.target_track_id == 9 and t.original_track_id == 9
    # Issue 2: probation, NOT an instant TRACKING lock.
    assert t.state == TrackerState.REIDENTIFYING
    assert t.reseed_probation_id == 9
    assert t.frames_lost == 0
    assert len(t.target_appearance.gallery) == 1   # preserved, not grown
