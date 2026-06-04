"""Tests for the pure asymmetric-hysteresis lock-state machine."""
from vision_track.core.lock_state_machine import LockStateMachine
from vision_track.core.tracking_types import LockDecision


def make_sm(**kw):
    """Build a state machine with test-friendly defaults."""
    params = dict(
        high_bar=0.72,
        distinct_margin=0.10,
        commit_frames=12,
        max_recovery_frames=45,
    )
    params.update(kw)
    return LockStateMachine(**params)


class TestLockDecisionType:
    def test_fields(self):
        d = LockDecision(publish=True, target_lost=False, committed_id=7, state="tracking")
        assert d.publish is True
        assert d.target_lost is False
        assert d.committed_id == 7
        assert d.state == "tracking"


class TestCoastAndHardLost:
    def test_absent_coasts_with_target_lost_true(self):
        sm = make_sm()
        sm.start(committed_id=3)
        # First few absent frames: coasting, target_lost stays True, no publish.
        for f in range(1, 5):
            d = sm.step(sim_score=0.0, present=False, frames_since_loss=f,
                        num_candidates=0, distinct_margin=0.0, depth_consistent=True)
            assert d.target_lost is True
            assert d.publish is False
            assert d.committed_id == 3  # identity retained while coasting
            assert d.state == "reidentifying"

    def test_hard_lost_after_max_recovery_frames(self):
        sm = make_sm(max_recovery_frames=10)
        sm.start(committed_id=3)
        last = None
        for f in range(1, 12):
            last = sm.step(sim_score=0.0, present=False, frames_since_loss=f,
                           num_candidates=0, distinct_margin=0.0, depth_consistent=True)
        # Past the cap → hard lost, identity dropped.
        assert last.state == "lost"
        assert last.target_lost is True
        assert last.publish is False


class TestAsymmetricHysteresis:
    def test_fast_provisional_publish_above_high_bar(self):
        """A single high-bar, depth-consistent candidate publishes immediately,
        but target_lost stays True until commit_frames are accumulated."""
        sm = make_sm(high_bar=0.72, commit_frames=12)
        sm.start(committed_id=5)
        d = sm.step(sim_score=0.80, present=False, frames_since_loss=1,
                    num_candidates=1, distinct_margin=999.0, depth_consistent=True)
        assert d.publish is True          # FAST out
        assert d.target_lost is True      # but still coasting (asymmetric)
        assert d.committed_id == 5
        assert d.state == "reidentifying"

    def test_no_publish_below_high_bar(self):
        """Below the HIGH bar, no provisional publish even if 'matched-ish'."""
        sm = make_sm(high_bar=0.72)
        sm.start(committed_id=5)
        d = sm.step(sim_score=0.69, present=False, frames_since_loss=1,
                    num_candidates=1, distinct_margin=999.0, depth_consistent=True)
        assert d.publish is False         # conservative: no false target
        assert d.target_lost is True

    def test_ambiguous_multi_candidate_blocks_provisional(self):
        """High score but tiny distinctiveness margin in a crowd → no publish
        (prevents locking a lookalike crosser)."""
        sm = make_sm(high_bar=0.72, distinct_margin=0.10)
        sm.start(committed_id=5)
        d = sm.step(sim_score=0.85, present=False, frames_since_loss=1,
                    num_candidates=3, distinct_margin=0.04, depth_consistent=True)
        assert d.publish is False
        assert d.target_lost is True

    def test_depth_inconsistent_blocks_provisional(self):
        """A toward-camera crosser (depth gate fails) is not published."""
        sm = make_sm(high_bar=0.72)
        sm.start(committed_id=5)
        d = sm.step(sim_score=0.90, present=False, frames_since_loss=1,
                    num_candidates=1, distinct_margin=999.0, depth_consistent=False)
        assert d.publish is False
        assert d.target_lost is True

    def test_sustained_match_commits_and_clears_lost(self):
        """Holding the high bar for commit_frames flips target_lost → False."""
        sm = make_sm(high_bar=0.72, commit_frames=3)
        sm.start(committed_id=5)
        decisions = [
            sm.step(sim_score=0.80, present=False, frames_since_loss=f,
                    num_candidates=1, distinct_margin=999.0, depth_consistent=True)
            for f in range(1, 4)
        ]
        # First two: provisional (publish True, lost True). Third: committed.
        assert decisions[0].target_lost is True and decisions[0].publish is True
        assert decisions[1].target_lost is True
        assert decisions[2].target_lost is False
        assert decisions[2].state == "tracking"

    def test_reacquire_latency_bound_in_frames(self):
        """At a high bar that is met from frame 1, full re-lock happens in
        exactly commit_frames frames (latency = commit_frames / rate)."""
        sm = make_sm(high_bar=0.72, commit_frames=12)
        sm.start(committed_id=5)
        committed_at = None
        for f in range(1, 30):
            d = sm.step(sim_score=0.80, present=False, frames_since_loss=f,
                        num_candidates=1, distinct_margin=999.0, depth_consistent=True)
            if d.target_lost is False:
                committed_at = f
                break
        assert committed_at == 12   # deterministic reacquire bound
