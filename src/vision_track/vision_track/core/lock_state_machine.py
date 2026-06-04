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
    ) -> None:
        """Store policy thresholds; call start() before stepping."""
        self.high_bar = high_bar
        self.distinct_margin = distinct_margin
        self.commit_frames = commit_frames
        self.max_recovery_frames = max_recovery_frames
        self._committed_id: Optional[int] = None
        self._state = "lost"
        self._provisional_streak = 0

    def start(self, committed_id: int) -> None:
        """Lock onto an initial committed id (called once on init/commit)."""
        self._committed_id = committed_id
        self._state = "tracking"
        self._provisional_streak = 0

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
            return LockDecision(True, False, self._committed_id, "tracking")

        # Absent: coasting. Bound the coast first.
        if frames_since_loss > self.max_recovery_frames:
            self._committed_id = None
            self._state = "lost"
            self._provisional_streak = 0
            return LockDecision(False, True, None, "lost")

        self._state = "reidentifying"

        # Provisional FAST-publish gate: HIGH bar + distinctiveness + depth + single
        # OR clearly-distinct multi. target_lost STAYS True (asymmetric).
        clears_bar = sim_score >= self.high_bar and depth_consistent
        distinct_ok = num_candidates <= 1 or distinct_margin >= self.distinct_margin
        if clears_bar and distinct_ok:
            self._provisional_streak += 1
            # SLOW commit: only after holding commit_frames do we drop target_lost.
            if self._provisional_streak >= self.commit_frames:
                self._state = "tracking"
                return LockDecision(True, False, self._committed_id, "tracking")
            return LockDecision(True, True, self._committed_id, "reidentifying")

        # Did not clear the bar: coast silently, keep identity, stay lost.
        self._provisional_streak = 0
        return LockDecision(False, True, self._committed_id, "reidentifying")
