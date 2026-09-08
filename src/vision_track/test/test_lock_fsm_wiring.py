"""Wiring-level regression for the asymmetric-hysteresis lock FSM.

The pure-FSM tests (test_lock_state_machine.py) prove the FSM contract in
isolation. They do NOT exercise the *wiring* in tracking_pipeline.reidentify_target
+ the node's present-step reconciliation, which is where the Critical defect lived:
the recovery path treated ANY non-None return from _confirm_reid_candidate as a
committed present frame and stepped the FSM present=True, flipping target_lost=False
and publishing a committed point on a PARTIAL confirm at sim >= reid_threshold
(0.55) but BELOW the FSM high bar (0.72) — bypassing both the high-bar gate and the
commit_frames slow-commit.

These tests drive the real reidentify_target() decision path (real LockStateMachine,
real _confirm_reid_candidate) with a duck-typed tracker (ROS-free, no cv/torch), and
mirror the node's present-step gate as a tiny helper, proving:

  * a partial confirm at sim 0.6 does NOT clear target_lost and does NOT surface a
    published point — no matter how long it holds — because 0.6 < high_bar;
  * a genuine streak >= high_bar DOES commit (target_lost=False) after exactly
    commit_frames provisionals.

Both must FAIL on the pre-fix wiring (non-None confirm -> present=True) and PASS
after gating the present-step on an actual id-swap + last_frame_recovery.
"""
from typing import List, Optional

import numpy as np

from vision_track.core import tracking_pipeline as tp
from vision_track.core.lock_state_machine import LockStateMachine
from vision_track.core.tracking_types import TrackerState, TrackingResult

REID_THRESHOLD = 0.55
HIGH_BAR = 0.72
COMMIT_FRAMES = 3
# Keep the pipeline's id-swap accumulation far out of reach so _confirm_reid_candidate
# only ever returns PARTIAL (pre-commit) non-None results during the test window.
# This isolates the FSM hysteresis from the pipeline's own (reid_threshold-gated)
# commit, which is what the defect conflated.
NO_SWAP_CONFIRM_FRAMES = 10_000


class _Registry:
    def clear(self):
        pass

    def register_person(self, *_a, **_k):
        pass

    def get_person(self, *_a, **_k):
        return None


class FakeTracker:
    """Minimal duck-typed tracker exercising the real recovery decision path."""

    def __init__(self, fsm: LockStateMachine):
        self.enable_reid = True
        self.frames_lost = 1
        self.max_frames_lost = 90
        self.state = TrackerState.REIDENTIFYING
        self.fast_tracking_mode = False

        # identity: operator is ABSENT (Stage 1 failed) so target_track_id is the
        # last-known operator yolo id; the candidate has a DIFFERENT yolo id.
        self.target_track_id = 1
        self.original_track_id = 7  # stable display id

        # reid confirm machinery (real _confirm_reid_candidate reads these)
        self.appearance_extractor = None  # -> match_similarity stays = best_similarity
        self.target_appearance = None
        self.is_occluded = False
        self.pre_occlusion_appearance = None
        self.reid_threshold = REID_THRESHOLD
        self.reid_confirmation_frames = NO_SWAP_CONFIRM_FRAMES
        self.reid_preconfirm_frames = 1
        self.consecutive_reid_frames = 0
        self.pending_reid_match = None
        self.reid_fit_streak = 0
        self.reid_fit_id = None
        self.last_camera_motion_time = 0.0  # ancient -> post_shake_extra = 0
        self.last_reid_switch_time = 0.0
        self.reid_switch_cooldown = 1.0
        self.person_registry = _Registry()

        # spatial continuity (touched by _with_original_id)
        self.last_known_bbox = None
        self.last_known_center = None

        # FSM wiring
        self.lock_state_machine = fsm
        self.last_lock_decision = None
        self.last_reid_margin = 999.0  # single distinct candidate
        self.last_frame_recovery = False

    def _with_original_id(self, result: TrackingResult) -> TrackingResult:
        self.last_known_bbox = result.bbox
        x1, y1, x2, y2 = result.bbox
        self.last_known_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return TrackingResult(
            track_id=self.original_track_id,
            bbox=result.bbox,
            mask=result.mask,
            confidence=result.confidence,
            class_id=result.class_id,
            class_name=result.class_name,
        )


def _candidate(track_id: int = 2) -> TrackingResult:
    return TrackingResult(
        track_id=track_id, bbox=(10, 10, 50, 90), mask=None,
        confidence=0.9, class_id=0, class_name="person",
    )


def _node_present_step(tracker, track_result):
    """Mirror person_track_node._handle_tracked_frame's present-step reconciliation.

    The node receives whatever reidentify_target returned (a non-None
    TrackingResult => _handle_tracked_frame is called). It must defer to the
    pipeline's authoritative decision on a recovery frame instead of re-stepping
    present=True. Returns the effective LockDecision the node would act on.
    """
    fsm = getattr(tracker, "lock_state_machine", None)
    decision = getattr(tracker, "last_lock_decision", None)
    recovery_frame = bool(getattr(tracker, "last_frame_recovery", False))
    target_present = (
        not recovery_frame
        and tracker.target_track_id is not None
        and track_result.track_id == tracker.original_track_id
        and getattr(tracker, "frames_lost", 0) == 0
    )
    if fsm is not None and target_present:
        decision = fsm.step(
            sim_score=1.0, present=True, frames_since_loss=0,
            num_candidates=1, distinct_margin=float("inf"), depth_consistent=True,
        )
        tracker.last_lock_decision = decision
    return decision


def _drive_frame(tracker, sim, monkeypatch):
    """Run one recovery frame at the given similarity and return (returned_result,
    effective node decision). `returned_result is None` => nothing published."""
    cand = _candidate()
    monkeypatch.setattr(tp, "find_best_match_reid", lambda *a, **k: (cand, sim))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # update_tracker normally clears this each frame; emulate that here.
    tracker.last_frame_recovery = False
    returned = tp.reidentify_target(tracker, frame, [cand])
    decision = None
    if returned is not None:
        decision = _node_present_step(tracker, returned)
    else:
        decision = getattr(tracker, "last_lock_decision", None)
    return returned, decision


def _make_tracker():
    fsm = LockStateMachine(
        high_bar=HIGH_BAR, distinct_margin=0.10,
        commit_frames=COMMIT_FRAMES, max_recovery_frames=90,
    )
    fsm.start(committed_id=7)
    return FakeTracker(fsm)


class TestPartialConfirmDoesNotLeak:
    def test_sim_above_threshold_below_high_bar_never_commits_or_publishes(self, monkeypatch):
        """sim 0.6 is a partial confirm (>= reid_threshold 0.55) but below the FSM
        high bar (0.72): no published point, target_lost stays True, indefinitely.

        Pre-fix wiring: the non-None partial confirm was stepped present=True,
        flipping target_lost=False and returning a committed point on frame 1.
        """
        tracker = _make_tracker()
        for _ in range(COMMIT_FRAMES + 5):
            returned, decision = _drive_frame(tracker, 0.60, monkeypatch)
            assert returned is None, "partial confirm below high bar must not publish"
            assert decision is not None
            assert decision.target_lost is True, "below high bar must stay coasting"
            assert decision.publish is False

    def test_confirm_returns_non_none_partial_but_no_id_swap(self, monkeypatch):
        """Sanity: at sim 0.6 the real _confirm_reid_candidate never swaps
        target_track_id in the window.

        Precision-leak fix: a LONE candidate at 0.60 (below the 0.72 commit bar)
        must NEVER arm pending_reid_match. Arming is now gated on commit-bar hits
        (sum(window)), not on the reid_threshold-counted reid_fit_streak. If
        pending were armed here, Stage 1 (_confirm_pending_reid) would adopt and
        lock the id by its ByteTrack id without re-checking the 0.72 bar — the
        leak this fix closes. (The previous version of this test asserted the
        opposite, encoding the leaky pre-fix behavior.)
        """
        tracker = _make_tracker()
        prev = tracker.target_track_id
        for _ in range(COMMIT_FRAMES + 5):
            _drive_frame(tracker, 0.60, monkeypatch)
        assert tracker.target_track_id == prev, "no id-swap expected in window"
        assert tracker.pending_reid_match is None, (
            "lone sub-commit-bar candidate must NOT arm pending (precision invariant)"
        )


class TestHighBarStreakCommits:
    def test_streak_above_high_bar_commits_after_commit_frames(self, monkeypatch):
        """A sustained sim 0.8 streak (>= high bar) publishes provisionals and flips
        target_lost=False after exactly commit_frames provisionals — the FSM owns
        the timing, not the pipeline's reid_threshold confirm."""
        tracker = _make_tracker()
        commit_frame = None
        decisions = []
        for f in range(1, COMMIT_FRAMES + 4):
            returned, decision = _drive_frame(tracker, 0.80, monkeypatch)
            decisions.append((f, returned, decision))
            # Above the high bar => provisional publishes a point every frame.
            assert returned is not None, "above high bar must surface a provisional point"
            if decision.target_lost is False and commit_frame is None:
                commit_frame = f
        assert commit_frame == COMMIT_FRAMES, (
            f"expected commit at frame {COMMIT_FRAMES}, got {commit_frame}"
        )
        # Frames before the commit are provisional: published but still target_lost.
        for f, returned, decision in decisions:
            if f < COMMIT_FRAMES:
                assert decision.target_lost is True
                assert decision.publish is True
        # The commit frame clears target_lost and reports TRACKING.
        committed = decisions[COMMIT_FRAMES - 1][2]
        assert committed.target_lost is False
        assert committed.state == "tracking"
