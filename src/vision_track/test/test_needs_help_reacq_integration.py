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
"""Integration reproduction for the NEEDS_HELP passive re-acquisition failure.

The isolated unit test (test_needs_help_recovery.py) drives
``_confirm_reid_candidate`` directly with a STABLE candidate id and passes. The
operator-reported failure is that on the LIVE robot, while latched in
NEEDS_HELP with the operator standing in clear view at a high gallery
similarity, passive re-lock still does not happen.

These tests drive the REAL per-frame recovery path — ``reidentify_target`` with
a real ``LockStateMachine`` — the same way ``test_lookalike_pursuit.py`` does,
but with ``in_needs_help=True`` and the relaxed help bar in play. They isolate
the suspected causes one variable at a time:

  (a) STABLE candidate id, high sim  -> should re-lock (control).
  (b) CHURNING candidate id, high sim -> hypothesis 2 (ByteTrack re-emits the
      returning operator under a fresh track_id every frame; the N-of-M window
      is reset on every id change so it never reaches the commit count).
"""
from types import SimpleNamespace

import pytest

import vision_track.core.tracking_pipeline as TP
from vision_track.core.lock_state_machine import LockStateMachine
from vision_track.reid.reid import ReIDMatcher


class _PersonRegistry:
    """Minimal registry stub: records clear/register calls, no-ops otherwise."""

    def __init__(self):
        self.cleared = 0
        self.registered = []

    def clear(self):
        self.cleared += 1

    def register_person(self, pid, appearance):
        self.registered.append(pid)

    def clear_temporary_ids(self):
        pass

    def get_person(self, pid):
        return None


class _NullCache:
    def begin_frame(self, seq):
        pass

    def get(self, tid, seq):
        return None

    def put(self, tid, seq, features):
        pass


def make_tracker(in_needs_help=True, commit_bar_help=0.62,
                 confirm_frames=12, window=16, preconfirm=3):
    """Stub tracker exposing every attribute the recovery path touches.

    A real LockStateMachine is wired in (started + then driven to terminal
    'lost' by the caller, mirroring an operator gone past max_recovery_frames),
    so the integration path — find_best_match_reid -> _confirm_reid_candidate ->
    FSM step / re-arm — runs exactly as on the robot.
    """
    fsm = LockStateMachine(
        high_bar=0.72, distinct_margin=0.10,
        commit_frames=confirm_frames, max_recovery_frames=45,
        provisional_commit_window=18,
    )
    fsm.start(committed_id=3)
    extractor = SimpleNamespace(
        extract_features=lambda *a, **k: {"reid": [0.0]},
        extract_features_batch=lambda *a, **k: [{"reid": [0.0]}],
    )
    tracker = SimpleNamespace(
        enable_reid=True,
        frames_lost=0,
        max_frames_lost=10_000,
        state=None,
        frame_count=0,
        target_track_id=3,
        original_track_id=3,
        target_class_id=0,
        target_class_name="person",
        target_appearance=SimpleNamespace(class_id=0, class_name="person"),
        appearance_extractor=extractor,
        embedding_cache=_NullCache(),
        person_registry=_PersonRegistry(),
        is_occluded=False,
        pre_occlusion_appearance=None,
        last_camera_motion_time=-1e9,
        reid_threshold=0.55,
        reid_confirmation_frames=12,
        reid_preconfirm_frames=preconfirm,
        consecutive_reid_frames=0,
        pending_reid_match=None,
        reid_fit_streak=0,
        reid_fit_id=None,
        last_reid_switch_time=-1e9,
        reid_switch_cooldown=1.0,
        single_person_pursue_floor=0.55,
        single_person_commit_bar=0.72,
        provisional_commit_window=18,
        reid_confirm_window=[],
        lock_state_machine=fsm,
        last_lock_decision=None,
        last_frame_recovery=False,
        last_reid_margin=float("inf"),
        candidate_depths_m={},
        operator_last_depth_m=None,
        crosser_depth_jump_m=0.6,
        fast_tracking_mode=False,
        # --- Issue 1: NEEDS_HELP relaxed-recovery knobs ---
        in_needs_help=in_needs_help,
        single_person_commit_bar_help=commit_bar_help,
        needs_help_confirm_frames=confirm_frames,
        needs_help_commit_window=window,
        _with_original_id=lambda r: r,
    )
    return tracker


def _match(track_id=7):
    return SimpleNamespace(track_id=track_id, class_id=0,
                           class_name="person", bbox=(0, 0, 10, 10), mask=None)


def _drive(monkeypatch, tracker, sim_sequence, track_ids):
    """Run reidentify_target once per frame.

    ``track_ids[i]`` is the ByteTrack id of the lone returning candidate on
    frame i (lets a variant churn the id every frame). find_best_match_reid is
    stubbed to return that lone candidate at the scripted sim, exactly as it
    would when the operator is alone in clear view above the pursue floor.
    """
    box = {"i": 0}

    def fake_find(tr, fr, res):
        return (_match(track_ids[box["i"]]), sim_sequence[box["i"]])

    def fake_sim(appearance, features, bbox, t, is_person=True, use_gallery=False):
        return sim_sequence[box["i"]]

    monkeypatch.setattr(TP, "find_best_match_reid", fake_find)
    monkeypatch.setattr(ReIDMatcher, "compute_similarity", staticmethod(fake_sim))

    committed = False
    for i in range(len(sim_sequence)):
        box["i"] = i
        tracker.frame_count += 1
        tracker.last_frame_recovery = False
        prev = tracker.target_track_id
        TP.reidentify_target(tracker, frame=None, results=[_match(track_ids[i])])
        if tracker.target_track_id != prev:
            committed = True
            break
    return committed


def test_reacq_stable_id_relocks(monkeypatch):
    """CONTROL: lone returner with a STABLE id at 0.65 (>=0.62 help bar) re-locks
    within the N-of-M window while latched in NEEDS_HELP."""
    tracker = make_tracker(in_needs_help=True)
    n = 16
    committed = _drive(monkeypatch, tracker, [0.65] * n, [7] * n)
    assert committed, "stable-id high-sim returner failed to re-lock"
    assert tracker.target_track_id == 7
    assert tracker.last_lock_decision.target_lost is False


def test_reacq_churning_id_relocks(monkeypatch):
    """REPRODUCTION (hypothesis 2): the SAME lone returner at the SAME 0.65, but
    ByteTrack re-emits it under a NEW track_id every frame (8, 9, 10, ...).

    On the live robot a real loss+reappearance churns the returning operator's
    id. If the N-of-M confirm window is hard-reset on every id change, the
    confirm count never reaches 12 and re-lock NEVER happens — the operator
    stands in clear view and the tracker won't re-lock. This is the failure.

    After the fix the window survives the id churn while latched in NEEDS_HELP,
    so the confirm count accumulates and the commit fires on the LIVE id.
    """
    tracker = make_tracker(in_needs_help=True)
    n = 40
    sims = [0.65] * n
    ids = [8 + i for i in range(n)]  # fresh id every frame
    committed = _drive(monkeypatch, tracker, sims, ids)
    assert committed, (
        "churning-id high-sim returner failed to re-lock: the N-of-M confirm "
        "window is reset on every track_id change, so the confirm count never "
        "reaches the commit threshold even at high similarity"
    )
    # Committed onto whatever live id was current at the commit frame, not the
    # stale pre-loss id.
    assert tracker.target_track_id != 3
    assert tracker.last_lock_decision.target_lost is False


def test_reacq_churning_id_below_help_bar_never_commits(monkeypatch):
    """PRECISION INVARIANT (fix must not over-commit): a CHURNING-id lone
    candidate scoring 0.58 — above the pursue floor (0.55) but BELOW the relaxed
    help bar (0.62) — is pursued but is never a confirm HIT, so it never commits
    even though the window now survives the id churn. The fix only stops the
    window from being WIPED; it does not lower the hit bar."""
    tracker = make_tracker(in_needs_help=True)
    n = 60
    sims = [0.58] * n
    ids = [8 + i for i in range(n)]
    committed = _drive(monkeypatch, tracker, sims, ids)
    assert not committed
    assert tracker.target_track_id == 3
    assert tracker.pending_reid_match is None  # never even armed


def test_churning_id_not_in_help_still_resets(monkeypatch):
    """PRECISION INVARIANT (outside the help gate, behavior is unchanged): with
    in_needs_help=False, a churning-id lone candidate at 0.75 (clears the STRICT
    0.72 bar) must NOT commit — the strict path still hard-resets the window on
    every id change, so the count never accumulates. This proves the relaxation
    is gated strictly on the NEEDS_HELP latch.

    (A stable-id candidate at 0.75 WOULD commit on the strict path; the churn is
    what the strict path legitimately refuses, because outside NEEDS_HELP an
    id-churning lone candidate is not trusted to be the operator.)"""
    tracker = make_tracker(in_needs_help=False)
    n = 40
    sims = [0.75] * n  # clears the strict 0.72 lone commit bar
    ids = [8 + i for i in range(n)]
    committed = _drive(monkeypatch, tracker, sims, ids)
    assert not committed, "strict (non-help) path wrongly accumulated across id churn"
    assert tracker.target_track_id == 3


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
