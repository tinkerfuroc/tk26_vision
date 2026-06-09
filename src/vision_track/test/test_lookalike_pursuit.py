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
"""Phase 3 pipeline tests: pursue look-alikes without lowering the commit bar.

THE PRECISION INVARIANT (test_lone_held_below_bar_never_commits): lowering the
single-candidate PURSUE floor must NOT lower the lone-candidate COMMIT bar. A
lone returner held at sim 0.60 is PURSUED (surfaced as reidentifying,
target_lost True) but its id-swap NEVER commits. The lone commit bar stays 0.72.

N-of-M (test_lone_commit_with_dips): once the returner clears 0.72 on 12 of the
last 18 frames — even with dips interspersed — the id-swap commits. Strict-
consecutive would have failed on any dip.
"""
from types import SimpleNamespace

import pytest

import vision_track.core.tracking_pipeline as TP
from vision_track.core.lock_state_machine import LockStateMachine
from vision_track.core.tracking_types import TrackerState
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


def make_tracker(commit_bar=0.72, window=18, confirm_frames=12, preconfirm=3):
    """Stub tracker exposing every attribute the recovery path touches.

    appearance_extractor is a sentinel object (non-None) so _confirm_reid_candidate
    takes the feature path; compute_similarity is monkeypatched per-test to feed
    the controlled sim sequence, so the actual feature content is irrelevant.
    """
    fsm = LockStateMachine(
        high_bar=commit_bar, distinct_margin=0.10,
        commit_frames=confirm_frames, max_recovery_frames=10_000,
        provisional_commit_window=window,
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
        last_camera_motion_time=-1e9,         # no post-shake extra
        reid_threshold=0.55,
        reid_confirmation_frames=confirm_frames,
        reid_preconfirm_frames=preconfirm,
        consecutive_reid_frames=0,
        pending_reid_match=None,
        reid_fit_streak=0,
        reid_fit_id=None,
        last_reid_switch_time=-1e9,
        reid_switch_cooldown=1.0,
        single_person_pursue_floor=0.55,
        single_person_commit_bar=commit_bar,
        provisional_commit_window=window,
        reid_confirm_window=[],
        lock_state_machine=fsm,
        last_lock_decision=None,
        last_frame_recovery=False,
        last_reid_margin=float("inf"),
        candidate_depths_m={},
        operator_last_depth_m=None,
        crosser_depth_jump_m=0.6,
        fast_tracking_mode=False,
        _with_original_id=lambda r: r,
    )
    # --- extra surface exercised only by the FULL update_tracker loop (Stage 1) ---
    # Occlusion + camera-motion bookkeeping that track_by_id / update_scene_motion
    # touch. detect_occlusion / update_scene_motion are monkeypatched off in the
    # full-loop test, but these keep the stub self-consistent if they run.
    tracker.frames_since_occlusion_ended = 0
    tracker.occlusion_recovery_frames = 45
    tracker.occlusion_start_time = None
    tracker.occlusion_iou_threshold = 0.3
    tracker.camera_motion_detected = False
    tracker.camera_motion_recent_window = 1.0
    tracker.scene_center_history = []
    tracker.camera_motion_vector = (0.0, 0.0)
    tracker.CAMERA_MOTION_THRESHOLD = 50.0
    tracker.CAMERA_MOTION_COOLDOWN = 0.5
    # reid_verification_interval=0 disables periodic_reid_validation (it would
    # otherwise call tracker._find_best_match_reid, an unstubbed method).
    tracker.reid_verification_interval = 0
    tracker.last_debug_scores = {}
    tracker.last_results = []
    tracker._t_yolo_ms = 0.0
    tracker._t_pipeline_ms = 0.0
    # track() returns whatever the test scripts; _passes_spatial_gate always
    # passes (lone candidate, no spatial veto) so the leak — not geometry —
    # is what the test isolates.
    tracker.track = lambda frame, persist=True: list(tracker._scripted_results)
    tracker._scripted_results = []
    tracker._passes_spatial_gate = lambda bbox, use_camera_gate=False: (True, 0.0, 0.0)
    return tracker


def _match(track_id=7):
    return SimpleNamespace(track_id=track_id, class_id=0,
                           class_name="person", bbox=(0, 0, 10, 10), mask=None)


def _drive(monkeypatch, tracker, sim_sequence, match):
    """Run reidentify_target once per sim in the sequence.

    find_best_match_reid always returns the lone candidate (it cleared the pursue
    floor); compute_similarity returns the scripted per-frame sim so the commit
    bar inside _confirm_reid_candidate sees exactly that value.
    """
    results = [match]
    box = {"i": 0}

    def fake_find(tr, fr, res):
        return (match, sim_sequence[box["i"]])

    def fake_sim(appearance, features, bbox, t, is_person=True, use_gallery=False):
        return sim_sequence[box["i"]]

    monkeypatch.setattr(TP, "find_best_match_reid", fake_find)
    monkeypatch.setattr(ReIDMatcher, "compute_similarity", staticmethod(fake_sim))

    outs = []
    for i in range(len(sim_sequence)):
        box["i"] = i
        tracker.frame_count += 1
        tracker.last_frame_recovery = False
        prev = tracker.target_track_id
        outs.append(TP.reidentify_target(tracker, frame=None, results=results))
        if tracker.target_track_id != prev:
            # Commit happened this frame. In the live loop the swapped id would
            # next be picked up by track_by_id (present path); reidentify_target
            # is no longer the authority. Stop driving so last_lock_decision
            # reflects the commit frame (as it would on the real loop).
            break
    return outs


def test_full_loop_lone_below_bar_never_arms_or_locks(monkeypatch):
    """PRECISION INVARIANT — FULL LOOP (Stage 1 + Stage 2).

    The Stage-2-only test (test_lone_held_below_bar_never_commits) misses the
    real leak: the pre-confirm ramp can ARM pending_reid_match on frames that
    only clear reid_threshold (0.55), not the lone commit bar (0.72). Once armed,
    Stage 1 (track_by_id -> _verify_person_candidate -> _confirm_pending_reid)
    adopts the pending id by ByteTrack id and locks it after
    reid_confirmation_frames present frames, NEVER consulting the 0.72 commit bar.

    Drive the FULL update_tracker loop for 25 frames with a LONE candidate (id=7)
    scoring 0.60 every frame and the real operator (id=3) absent. Assert the
    candidate is never armed, target_track_id never changes, and no lock is ever
    reported by ANY path.
    """
    tracker = make_tracker(commit_bar=0.72)
    tracker.state = TrackerState.REIDENTIFYING  # not UNINITIALIZED
    cand = _match(track_id=7)
    tracker._scripted_results = [cand]

    sim = 0.60

    def fake_find(tr, fr, res):
        return (cand, sim)

    def fake_sim(appearance, features, bbox, t, is_person=True, use_gallery=False):
        return sim

    # Off the heavy bits update_tracker would otherwise touch; the leak under
    # test is the arming/lock logic, not occlusion / scene-motion / distractor
    # registration.
    monkeypatch.setattr(TP, "find_best_match_reid", fake_find)
    monkeypatch.setattr(TP, "register_other_persons", lambda tr, fr, res: None)
    monkeypatch.setattr(TP, "update_scene_motion", lambda tr, res, fr=None: None)
    monkeypatch.setattr(TP, "detect_occlusion", lambda tr, tgt, res: (False, None))
    monkeypatch.setattr(ReIDMatcher, "compute_similarity", staticmethod(fake_sim))

    for _ in range(25):
        TP.update_tracker(tracker, frame=None)
        # The id-swap authority must never flip to the lone sub-0.72 candidate,
        # via Stage 1 OR Stage 2.
        assert tracker.target_track_id == 3, "lone <0.72 candidate was locked"
        # The candidate must never be armed (the root cause of the Stage-1 leak).
        assert tracker.pending_reid_match is None, "lone <0.72 candidate armed pending"
        # No lock may be reported by the FSM either.
        if tracker.last_lock_decision is not None:
            assert tracker.last_lock_decision.target_lost is True, "reported a lock"


def test_lone_held_below_bar_never_commits(monkeypatch):
    """PRECISION INVARIANT: a lone candidate at 0.60 for 25 frames is pursued
    but the id-swap NEVER commits (commit bar held at 0.72)."""
    tracker = make_tracker(commit_bar=0.72)
    match = _match(track_id=7)
    sims = [0.60] * 25
    _drive(monkeypatch, tracker, sims, match)

    # The id-swap authority (target_track_id) was never mutated → no commit.
    assert tracker.target_track_id == 3
    # The FSM never reports a re-lock; it stays reidentifying / target_lost.
    assert tracker.last_lock_decision is not None
    assert tracker.last_lock_decision.target_lost is True
    assert tracker.last_lock_decision.state == "reidentifying"
    # frames_lost keeps growing (pursued-but-not-hit frames don't reset it),
    # so NEEDS_HELP can still escalate.
    assert tracker.frames_lost > 0


def test_lone_commit_with_dips(monkeypatch):
    """N-of-M: lone candidate clearing 0.72 on 12 of the last 18 frames commits,
    despite up to 6 dips at 0.60 interspersed. Strict-consecutive would fail."""
    tracker = make_tracker(commit_bar=0.72, window=18, confirm_frames=12, preconfirm=3)
    match = _match(track_id=7)
    # 18 frames: pattern of 2 hits + 1 dip → 12 hits, 6 dips.
    sims = ([0.80, 0.80, 0.60] * 6)
    _drive(monkeypatch, tracker, sims, match)

    # The id-swap committed.
    assert tracker.target_track_id == 7
    assert tracker.last_lock_decision.target_lost is False
    assert tracker.last_lock_decision.state == "tracking"


def test_bystander_spike_rejected(monkeypatch):
    """A 5-frame >=0.72 spike (then sub-bar) within an 18-window never reaches 12
    → no commit."""
    tracker = make_tracker(commit_bar=0.72, window=18, confirm_frames=12, preconfirm=3)
    match = _match(track_id=7)
    sims = [0.80] * 5 + [0.60] * 20  # spike then dips
    _drive(monkeypatch, tracker, sims, match)

    assert tracker.target_track_id == 3  # never committed
    assert tracker.last_lock_decision.target_lost is True


def test_dip_does_not_teardown_pending(monkeypatch):
    """A non-hit frame (sim in [pursue_floor, commit_bar)) keeps the pending /
    window alive — it does not zero the accumulated hits."""
    tracker = make_tracker(commit_bar=0.72, window=18, confirm_frames=12, preconfirm=3)
    match = _match(track_id=7)
    # 11 hits, then a dip, then a 12th hit (all within an 18-window) → commit.
    sims = [0.80] * 11 + [0.60] + [0.80]
    _drive(monkeypatch, tracker, sims, match)
    assert tracker.target_track_id == 7  # the dip did not reset the count


def test_multi_candidate_commit_bar_is_reid_threshold(monkeypatch):
    """With >1 candidate the commit bar is reid_threshold (0.55), not 0.72 — a
    0.60 hit counts. (Distinctiveness/margin gates are upstream in
    find_best_match_reid, which this test bypasses.)"""
    tracker = make_tracker(commit_bar=0.72, window=18, confirm_frames=12, preconfirm=3)
    match = _match(track_id=7)
    other = _match(track_id=8)
    results = [match, other]
    box = {"i": 0}
    sims = [0.60] * 18  # all >= reid_threshold but < single_person_commit_bar

    def fake_find(tr, fr, res):
        return (match, sims[box["i"]])

    def fake_sim(appearance, features, bbox, t, is_person=True, use_gallery=False):
        return sims[box["i"]]

    monkeypatch.setattr(TP, "find_best_match_reid", fake_find)
    monkeypatch.setattr(TP, "register_other_persons", lambda tr, fr, res: None)
    monkeypatch.setattr(ReIDMatcher, "compute_similarity", staticmethod(fake_sim))

    for i in range(len(sims)):
        box["i"] = i
        tracker.frame_count += 1
        tracker.last_frame_recovery = False
        prev = tracker.target_track_id
        TP.reidentify_target(tracker, frame=None, results=results)
        if tracker.target_track_id != prev:
            break

    # 18 frames all >= reid_threshold (multi commit bar) → commits.
    assert tracker.target_track_id == 7


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
