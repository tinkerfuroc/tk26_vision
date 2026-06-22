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
"""Tracker vision_log EVENT writes are throttled, not per-frame.

The tracker used to emit a vision_log artifact on every per-frame
tracked<->lost flip (keyed on track_result presence), so a churny scene
(rapid loss/reacquire, id churn) re-fired the "transition" writes over and
over -> a redundant flood. ``_vision_log_due`` replaces that with:

- a debounced state transition (state CHANGED *and* >= ``vision_log_min_gap_s``
  since the last write) -> fire promptly on a real acquire/lost, but swallow
  rapid flip-flop, and
- a steady-state heartbeat (>= ``vision_log_interval_s`` since the last write)
  for both steady tracking and steady lost.

The test binds the *real* ``_vision_log_due`` / ``_mark_vision_logged`` methods
onto a lightweight ``SimpleNamespace`` (via ``__get__``) so the YOLO model /
ROS node never load — we exercise the genuine method logic, not a
reimplementation. Same convention as ``test_track_state_pruning.py``.
"""
from types import SimpleNamespace

from vision_track.person_track_node import PersonTrackNode


def _make_stub():
    """A minimal duck-typed node carrying only the throttle state + bound methods."""
    s = SimpleNamespace()
    s._vlog_last_state = None
    s._vlog_last_time = 0.0
    s._vlog_interval_s = 5.0
    s._vlog_min_gap_s = 1.0
    s._vision_log_due = PersonTrackNode._vision_log_due.__get__(s, SimpleNamespace)
    s._mark_vision_logged = \
        PersonTrackNode._mark_vision_logged.__get__(s, SimpleNamespace)
    return s


def test_throttle_lifecycle():
    """Walk a churny acquire/lost timeline and assert the throttle decisions."""
    s = _make_stub()

    # Initial acquire: last_state None -> due (the first lock logs promptly).
    assert s._vision_log_due('tracked', 10.0) is True
    s._mark_vision_logged('tracked', 10.0)

    # Same state, gap 0.5s < min_gap and < interval -> NOT due.
    assert s._vision_log_due('tracked', 10.5) is False

    # Churn: changed to 'lost' but gap 0.5s < min_gap -> debounced, NOT due.
    assert s._vision_log_due('lost', 10.5) is False

    # Real lost transition: changed AND gap 1.2s >= min_gap -> due.
    assert s._vision_log_due('lost', 11.2) is True
    s._mark_vision_logged('lost', 11.2)

    # Steady lost, gap 1.8s < interval -> NOT due (no heartbeat yet).
    assert s._vision_log_due('lost', 13.0) is False

    # Steady-lost heartbeat: gap 5.1s >= interval -> due.
    assert s._vision_log_due('lost', 16.3) is True


def test_mark_vision_logged_records_state_and_time():
    """_mark_vision_logged stores the new state + timestamp for later compares."""
    s = _make_stub()
    s._mark_vision_logged('tracked', 42.0)
    assert s._vlog_last_state == 'tracked'
    assert s._vlog_last_time == 42.0
