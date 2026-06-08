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
"""Active re-ID hold policy: _is_awaiting_help (wave-to-resume).

Once escalated to NEEDS_ACTIVE_HELP, the tracker must keep coasting (no abort)
so a wave -> reseed can re-lock the operator. With active_help_timeout_sec <= 0
the hold is indefinite (the operator-chosen "wait forever" policy); a positive
value bounds it. active_help_after_frames <= 0 disables active help entirely.
"""
from types import SimpleNamespace

import pytest

PersonTrackNode = pytest.importorskip(
    "vision_track.person_track_node").PersonTrackNode
_awaiting = PersonTrackNode._is_awaiting_help


def _cfg(after_frames, timeout):
    return SimpleNamespace(active_help_after_frames=after_frames,
                           active_help_timeout_sec=timeout)


def test_indefinite_hold_never_expires():
    cfg = _cfg(45, 0.0)                       # timeout<=0 -> forever
    assert _awaiting(cfg, 45, 0.0) is True    # just escalated
    assert _awaiting(cfg, 600, 9999.0) is True  # huge time since seen, still held


def test_not_awaiting_before_escalation():
    cfg = _cfg(45, 0.0)
    assert _awaiting(cfg, 44, 100.0) is False  # one frame short of escalation


def test_bounded_hold_expires():
    cfg = _cfg(45, 20.0)
    assert _awaiting(cfg, 50, 10.0) is True    # within the bounded window
    assert _awaiting(cfg, 50, 25.0) is False   # past the bounded window


def test_active_help_disabled():
    cfg = _cfg(0, 0.0)                          # after_frames<=0 disables help
    assert _awaiting(cfg, 999, 0.0) is False


def test_hold_latches_through_frames_lost_reset():
    # The reappearing operator's pre-commit re-ID resets frames_lost to 0 before
    # the re-lock commits; the latch must keep holding so the hard-lost abort
    # does NOT fire mid-reappearance (the bug that made auto-reclaim impossible).
    cfg = _cfg(45, 0.0)                         # forever
    assert _awaiting(cfg, 50, 5.0) is True      # escalated -> latched
    assert _awaiting(cfg, 0, 6.0) is True       # frames_lost reset, still held
    assert _awaiting(cfg, 3, 7.0) is True       # any sub-threshold value: still held


def test_bounded_hold_latches_but_still_respects_timeout():
    cfg = _cfg(45, 20.0)                        # bounded
    assert _awaiting(cfg, 50, 5.0) is True      # escalated -> latched
    assert _awaiting(cfg, 0, 10.0) is True      # frames_lost reset, within window
    assert _awaiting(cfg, 0, 25.0) is False     # latched but past the time bound


def test_latch_release_enables_next_cycle():
    # On a successful re-lock the node clears _help_latched (see
    # _handle_tracked_frame); a subsequent loss must escalate + hold afresh,
    # so repeated loss->reclaim cycles each get their own hold.
    cfg = _cfg(45, 0.0)
    assert _awaiting(cfg, 50, 5.0) is True      # cycle 1: escalate + latch
    cfg._help_latched = False                   # re-lock clears the latch
    assert _awaiting(cfg, 10, 6.0) is False     # back to tracking-ish, not held
    assert _awaiting(cfg, 50, 7.0) is True      # cycle 2: escalate + latch again
