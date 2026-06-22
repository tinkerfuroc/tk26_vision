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
"""Unit tests for _target_box_color_kind — the per-box color decision helper.

Regression: after any ReID reacquire the live ByteTrack id (target_track_id)
diverges from original_track_id, which is what target_result.track_id carries.
The old logic compared track_id == target_result.track_id, so a fully-locked
target was drawn YELLOW instead of GREEN. The fix uses the LIVE id
(target_track_id) and gates on the FSM state ('tracking').
"""
from types import SimpleNamespace

from vision_track.person_track_node import _target_box_color_kind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decision(state):
    """Return a minimal LockDecision-like object with the given state."""
    return SimpleNamespace(state=state)


def _target_result(track_id):
    """Return a minimal TrackingResult-like object with the given track_id."""
    return SimpleNamespace(track_id=track_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_live_id_match_tracking_state_is_target():
    """THE REGRESSION: live id matches + FSM tracking → green, even when
    target_result.track_id (original_track_id) diverges after a reacquire."""
    live_id = 5          # current ByteTrack id
    original_id = 2      # frozen id stored in target_result (post-reacquire)
    result = _target_result(original_id)   # diverged from live_id
    decision = _decision('tracking')

    kind = _target_box_color_kind(
        track_id=live_id,
        target_result=result,
        target_track_id=live_id,
        decision=decision,
    )
    assert kind == 'target', (
        "A box whose track_id matches target_track_id (live) while FSM is "
        "'tracking' must be 'target' (green) — not 'yolo_target' (yellow)."
    )


def test_live_id_match_reidentifying_state_is_yolo_target():
    """Live id matches but FSM is reidentifying (not yet committed) → yellow."""
    decision = _decision('reidentifying')
    kind = _target_box_color_kind(
        track_id=7,
        target_result=_target_result(7),
        target_track_id=7,
        decision=decision,
    )
    assert kind == 'yolo_target'


def test_live_id_match_no_decision_is_yolo_target():
    """Live id matches but decision is None (strict) → yellow."""
    kind = _target_box_color_kind(
        track_id=3,
        target_result=_target_result(3),
        target_track_id=3,
        decision=None,
    )
    assert kind == 'yolo_target'


def test_non_matching_id_is_other():
    """A box whose track_id does not match target_track_id → blue."""
    decision = _decision('tracking')
    kind = _target_box_color_kind(
        track_id=9,
        target_result=_target_result(1),
        target_track_id=1,
        decision=decision,
    )
    assert kind == 'other'


def test_lost_loop_target_result_none_is_yolo_target():
    """Lost-loop call: target_result=None + id matches → yolo_target (unchanged)."""
    decision = _decision('tracking')
    kind = _target_box_color_kind(
        track_id=4,
        target_result=None,
        target_track_id=4,
        decision=decision,
    )
    assert kind == 'yolo_target'
