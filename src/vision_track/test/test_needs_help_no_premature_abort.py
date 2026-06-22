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
"""Regression: the lost goal must NOT abort prematurely after NEEDS_HELP.

ROOT CAUSE (commit 1817a48): the NEEDS_HELP escalation was converted from
frame-count to wall-clock. The latch now activates at active_help_after_sec
(5.0 s), but the FSM recovery-cap (decision.state == 'lost' -> hard_lost) still
fires at max_recovery_frames (45 ≈ 1.5 s @ 30 fps). The abort branch in
_handle_lost_frame ran ``if hard_lost or time_since_seen > lost_timeout``, and
the _is_awaiting_help early-return only shadowed it once the latch was set.
So in the window [~1.5 s, 5.0 s after loss] hard_lost was True but the latch was
not yet engaged -> the goal aborted, instead of coasting into the indefinite
NEEDS_HELP hold. Pre-1817a48 the latch (frames_lost>=45) coincided with
hard_lost (max_recovery_frames=45) and always shadowed the abort.

THE FIX: the FSM recovery-cap (hard_lost) aborts ONLY when active help is
DISABLED (legacy active_help_after_sec <= 0). With active help ENABLED the goal
coasts through the passive-recovery window into the hold. The lost_timeout
absolute ceiling always aborts; the active-help hold (awaiting_help) never does.
"""
import pytest

lost_should_abort = pytest.importorskip(
    "vision_track.person_track_node").lost_should_abort


def test_recovery_cap_does_not_abort_while_active_help_enabled():
    """REGRESSION: in [recovery-cap, escalation) the FSM hard-lost must NOT abort
    while active help is enabled, even before the help-latch engages."""
    assert lost_should_abort(
        hard_lost=True, awaiting_help=False, active_help_enabled=True,
        time_since_seen=2.0, lost_timeout=300.0) is False


def test_recovery_cap_aborts_when_active_help_disabled():
    """Legacy behavior preserved: with active help disabled
    (active_help_after_sec <= 0) the FSM hard-lost still aborts on the cap."""
    assert lost_should_abort(
        hard_lost=True, awaiting_help=False, active_help_enabled=False,
        time_since_seen=2.0, lost_timeout=300.0) is True


def test_awaiting_help_never_aborts_regardless_of_hard_lost():
    """While the active-help hold is engaged the goal coasts: no abort even with
    hard_lost True."""
    assert lost_should_abort(
        hard_lost=True, awaiting_help=True, active_help_enabled=True,
        time_since_seen=10.0, lost_timeout=300.0) is False
    assert lost_should_abort(
        hard_lost=False, awaiting_help=True, active_help_enabled=True,
        time_since_seen=10.0, lost_timeout=300.0) is False


def test_lost_timeout_ceiling_always_aborts():
    """The absolute ceiling fires regardless of active-help / hold / hard_lost."""
    # Past the ceiling even with active help enabled and not yet hard-lost.
    assert lost_should_abort(
        hard_lost=False, awaiting_help=False, active_help_enabled=True,
        time_since_seen=301.0, lost_timeout=300.0) is True
    # Ceiling wins even mid-hold (a runaway hold cannot outlive lost_timeout
    # when the caller passes awaiting_help; here we pass the helper directly).
    assert lost_should_abort(
        hard_lost=False, awaiting_help=False, active_help_enabled=False,
        time_since_seen=301.0, lost_timeout=300.0) is True


def test_no_abort_when_neither_condition_holds():
    """Inside the passive window with active help on and no ceiling breach: hold."""
    assert lost_should_abort(
        hard_lost=False, awaiting_help=False, active_help_enabled=True,
        time_since_seen=1.0, lost_timeout=300.0) is False


def test_old_inline_boolean_would_have_aborted():
    """Documents the REGRESSION: the pre-fix inline condition
    ``hard_lost or time_since_seen > lost_timeout`` aborted in the
    [recovery-cap, escalation) window. The fixed helper does NOT — this is the
    behavioral delta the fix corrects.
    """
    hard_lost, time_since_seen, lost_timeout = True, 2.0, 300.0
    old_inline_abort = hard_lost or time_since_seen > lost_timeout
    assert old_inline_abort is True  # the buggy decision
    # The fix flips this case to a hold while active help is enabled.
    assert lost_should_abort(
        hard_lost=hard_lost, awaiting_help=False, active_help_enabled=True,
        time_since_seen=time_since_seen, lost_timeout=lost_timeout) is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
