"""Pure asymmetric-hysteresis lock-state machine for person tracking.

ROS-free + numpy-free: encodes the recovery policy as a finite-state machine so
the reacquire↔false-target trade can be unit-tested with synthetic per-frame
inputs. The node and tracking_pipeline call into it; they hold no duplicate
transition logic.

States: 'tracking' (committed, present) → 'reidentifying' (coasting / probing) →
'lost' (hard-lost past max_recovery_frames).

Asymmetric hysteresis:
- FAST out (provisional publish): a single high-confidence candidate that clears
  the HIGH bar AND wins the distinctiveness margin AND is depth-consistent gets
  published immediately, but target_lost STAYS True until a full commit.
- SLOW commit: target_lost flips to False (full re-lock) only after the candidate
  holds for commit_frames consecutive frames.
- Bounded coast: after max_recovery_frames absent/unconfirmed frames, declare
  hard-lost and drop committed_id.
"""
from typing import Optional

from .tracking_types import LockDecision


class LockStateMachine:
    """Finite-state recovery policy with asymmetric hysteresis."""

    def __init__(
        self,
        high_bar: float = 0.72,
        distinct_margin: float = 0.10,
        commit_frames: int = 12,
        max_recovery_frames: int = 45,
        provisional_commit_window: int = 18,
    ) -> None:
        """Store policy thresholds; call start() before stepping."""
        self.high_bar = high_bar
        self.distinct_margin = distinct_margin
        self.commit_frames = commit_frames
        self.max_recovery_frames = max_recovery_frames
        # Phase 3 / Option A: N-of-M commit window. The provisional commits once
        # commit_frames (N) clear-bar coast frames occur within the last
        # provisional_commit_window (M) frames, tolerating dips instead of the old
        # strict-consecutive accumulation. N consecutive still commits (superset).
        self.provisional_commit_window = provisional_commit_window
        self._committed_id: Optional[int] = None
        self._state = "lost"
        # Sliding window of the last M coast frames' clear-bar verdicts (bools).
        # _provisional_streak is kept as a back-compat mirror (= current run of
        # trailing hits); the WINDOW is the commit authority.
        self._provisional_streak = 0
        self._provisional_window: list = []

    def start(self, committed_id: int) -> None:
        """Lock onto an initial committed id (called once on init/commit)."""
        self._committed_id = committed_id
        self._state = "tracking"
        self._provisional_streak = 0
        self._provisional_window = []

    def start_probation(self, committed_id: int) -> None:
        """Re-arm onto a candidate id WITHOUT committing (probationary reseed).

        Unlike start() (jumps to 'tracking'), this enters 'reidentifying' so the
        node does not report a lock until the reseed probation confirms; it also
        lifts the machine out of terminal 'lost' so it can be stepped again.
        """
        self._committed_id = committed_id
        self._state = "reidentifying"
        self._provisional_streak = 0
        self._provisional_window = []

    def step(
        self,
        sim_score: float,
        present: bool,
        frames_since_loss: int,
        num_candidates: int,
        distinct_margin: float,
        depth_consistent: bool,
    ) -> LockDecision:
        """Advance one frame; return the publish/target_lost/committed_id decision.

        Args:
            sim_score: best candidate's ReID similarity this frame (0 if absent).
            present: the committed track id was matched by ByteTrack this frame.
            frames_since_loss: consecutive frames since the committed id was
                last directly present (0 while present).
            num_candidates: number of person candidates considered this frame.
            distinct_margin: best - second-best similarity (∞-ish if single).
            depth_consistent: depth gate verdict for the best candidate.
        """
        if self._state == "lost":
            return LockDecision(False, True, None, "lost")

        # Direct, present-by-id tracking: full lock, publish, not lost.
        if present:
            self._state = "tracking"
            self._provisional_streak = 0
            self._provisional_window = []
            return LockDecision(True, False, self._committed_id, "tracking")

        # Absent: coasting. Bound the coast first.
        if frames_since_loss > self.max_recovery_frames:
            self._committed_id = None
            self._state = "lost"
            self._provisional_streak = 0
            self._provisional_window = []
            return LockDecision(False, True, None, "lost")

        self._state = "reidentifying"

        # Provisional FAST-publish gate: HIGH bar + distinctiveness + depth + single
        # OR clearly-distinct multi. target_lost STAYS True (asymmetric). The
        # per-frame bar is UNCHANGED — only the accumulation is windowed.
        clears_bar = sim_score >= self.high_bar and depth_consistent
        distinct_ok = num_candidates <= 1 or distinct_margin >= self.distinct_margin
        is_hit = clears_bar and distinct_ok

        # Phase 3 / Option A: N-of-M window. Append this coast frame's hit verdict,
        # trim to the last M frames, and commit when the window holds >= N hits.
        # A non-hit frame KEEPS the window (does not zero it) so dips are tolerated;
        # old hits expire only when they slide past M frames. N consecutive hits
        # within an M-window still commit (a 12-of-12 in an 18-window has 12 hits),
        # so the old strict-consecutive behaviour is a subset.
        self._provisional_window.append(is_hit)
        if len(self._provisional_window) > self.provisional_commit_window:
            self._provisional_window = self._provisional_window[-self.provisional_commit_window:]

        if is_hit:
            self._provisional_streak += 1
            if sum(self._provisional_window) >= self.commit_frames:
                self._state = "tracking"
                return LockDecision(True, False, self._committed_id, "tracking")
            return LockDecision(True, True, self._committed_id, "reidentifying")

        # Did not clear the bar this frame: the dip does NOT zero the window
        # (accumulated hits persist until they age out), but the consecutive-run
        # mirror resets and we do not surface a provisional publish this frame.
        self._provisional_streak = 0
        return LockDecision(False, True, self._committed_id, "reidentifying")
