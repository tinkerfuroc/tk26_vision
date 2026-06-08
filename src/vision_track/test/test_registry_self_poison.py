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
"""Regression: the lone operator returning under a fresh ByteTrack id must not
poison its own reacquisition.

Reproduces the live failure (operator alone, lost almost immediately after a
2-frame YOLO gap, never reacquired). Two stacked defects:

  Defect 1 (source): ``register_other_persons`` registers the returning target
  itself as an "other person" ghost, because its only exclusion is the *stale*
  ``target_track_id`` (the pre-loss id), not the id the operator now wears.

  Defect 2 (gate): ``check_distinctiveness`` then counts that ghost's ~1.0
  self-match as a competing person and rejects the real target
  (log: "best 0.861, max other 1.000, margin -0.139").

Both tests assert the corrected behaviour, so both are RED on the current code.
"""
import time
import types

import numpy as np

from vision_track.core.registry import PersonRegistry
from vision_track.core.tracking_types import TargetAppearance, TrackingResult
from vision_track.core.tracking_pipeline import register_other_persons


class _FixedExtractor:
    """Returns one fixed feature dict for any crop (the returning operator)."""

    def __init__(self, feats):
        self._feats = feats

    def extract_features(self, frame, bbox, mask, class_id=0):
        return self._feats


def _person_appearance(reid, body):
    ta = TargetAppearance(class_id=0, class_name="person")
    ta.feature_history.append(reid)
    ta.body_color_history.append(body)
    ta.last_seen_time = time.time()
    return ta


def test_returning_target_not_registered_as_other():
    """Defect 1: a detection that matches the target must not become a ghost."""
    reid = np.ones(512, dtype=np.float32)
    body = np.ones(96, dtype=np.float32)
    feats = {"reid": reid.copy(), "body_color": body.copy(),
             "size": np.array([80.0, 200.0], dtype=np.float32)}
    tracker = types.SimpleNamespace(
        appearance_extractor=_FixedExtractor(feats),
        fast_tracking_mode=False,
        frame_count=10,
        reid_extraction_interval=1,
        pending_reid_match=None,
        target_track_id=1,                       # STALE pre-loss id
        target_appearance=_person_appearance(reid, body),
        person_registry=PersonRegistry(),
        reid_threshold=0.55,
    )
    # Lone operator reappears as fresh ByteTrack id 18; crop ~= stored target.
    results = [TrackingResult(track_id=18, bbox=(100, 50, 180, 250), mask=None,
                              confidence=0.9, class_id=0, class_name="person")]
    register_other_persons(tracker, np.zeros((480, 640, 3), np.uint8), results)
    assert tracker.person_registry.get_person(-18) is None, (
        "returning target was cemented as an 'other person' distractor (-18)")


def test_self_match_ghost_ignored_in_distinctiveness():
    """Defect 2: a ~1.0 self-match ghost must not block the real target."""
    reg = PersonRegistry()
    target = TargetAppearance(class_id=0, class_name="person")   # original id 1
    ghost = TargetAppearance(class_id=0, class_name="person")    # self-ghost -18
    reg.register_person(1, target)
    reg.register_person(-18, ghost)

    # Returning operator scores 0.861 to its (drifted) stored target appearance
    # but ~1.000 to its own freshly-registered ghost.
    def sim_func(appearance, feats):
        return 1.0 if appearance is ghost else 0.5

    assert reg.check_distinctiveness(1, {"reid": np.zeros(4)}, 0.861, sim_func) is True, (
        "self-ghost self-match counted as a distinct person and rejected target")
