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
"""Passive re-ID re-lock while the lock FSM is latched 'lost'.

Regression for the bug where reidentify_target commits a genuine id-swap (after
the full reid_confirmation window) but the FSM is in terminal 'lost' (operator
gone > max_recovery_frames). step() short-circuits 'lost' and squashes the
re-lock to target_lost=True, so the node keeps reporting lost forever (waiting
for a wave) even though appearance/gallery re-ID already re-found the operator.

The fix re-arms the FSM via start() before the present-frame step in the
committed-swap branch — the same way reseed does — so the re-lock surfaces.
"""
from types import SimpleNamespace

import vision_track.core.tracking_pipeline as TP
from vision_track.core.lock_state_machine import LockStateMachine


def _fake_tracker(fsm):
    """Minimal tracker exposing every attribute reidentify_target touches."""
    return SimpleNamespace(
        enable_reid=True,
        frames_lost=100,
        max_frames_lost=600,
        target_track_id=3,
        lock_state_machine=fsm,
        last_frame_recovery=False,
        last_reid_margin=0.0,
        candidate_depths_m={},
        operator_last_depth_m=None,
        crosser_depth_jump_m=0.6,
        _with_original_id=lambda r: r,
    )


def test_committed_swap_relocks_fsm_from_lost(monkeypatch):
    """A committed id-swap while FSM is 'lost' must report target_lost=False."""
    fsm = LockStateMachine()
    # __init__ leaves the FSM in terminal 'lost'; make it explicit.
    assert fsm._state == "lost"

    tracker = _fake_tracker(fsm)
    match = SimpleNamespace(track_id=7, class_id=0, bbox=(0, 0, 10, 10), mask=None)
    results = [match]

    monkeypatch.setattr(TP, "find_best_match_reid", lambda tr, fr, res: (match, 0.9))

    def fake_confirm(tracker, frame, match, sim):
        # Simulate the real id-swap commit: target_track_id mutates.
        tracker.target_track_id = 7
        return match

    monkeypatch.setattr(TP, "_confirm_reid_candidate", fake_confirm)

    out = TP.reidentify_target(tracker, frame=None, results=results)

    # The branch returned the confirmed match.
    assert out is match
    # THE FIX: the re-lock surfaces instead of staying latched lost.
    assert tracker.last_lock_decision.target_lost is False
    assert tracker.last_lock_decision.state == "tracking"
    # start() re-synced the committed id to the new id.
    assert tracker.lock_state_machine._committed_id == 7


def test_fsm_contract_start_before_step_in_lost():
    """The FSM contract the fix relies on: fresh FSM is 'lost' and short-circuits.

    A fresh LockStateMachine() is 'lost'; stepping it (even present=True) returns
    target_lost=True. Only after start() does a present step re-lock (target_lost
    False, state 'tracking') — which is exactly what the committed-swap branch
    now does before mirroring the present frame.
    """
    fsm = LockStateMachine()
    assert fsm._state == "lost"

    # Short-circuit: 'lost' ignores all inputs, even present=True.
    squashed = fsm.step(
        sim_score=0.9, present=True, frames_since_loss=0,
        num_candidates=1, distinct_margin=0.0, depth_consistent=True,
    )
    assert squashed.target_lost is True
    assert squashed.state == "lost"

    # After re-arming, the same present step re-locks.
    fsm.start(7)
    relocked = fsm.step(
        sim_score=0.9, present=True, frames_since_loss=0,
        num_candidates=1, distinct_margin=0.0, depth_consistent=True,
    )
    assert relocked.target_lost is False
    assert relocked.state == "tracking"
    assert relocked.committed_id == 7
