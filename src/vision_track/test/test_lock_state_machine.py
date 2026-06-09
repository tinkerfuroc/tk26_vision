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


class TestStartProbation:
    """Phase 2: probationary re-arm for the reseed confirmation gate."""

    def test_enters_reidentifying_without_committing(self):
        """start_probation arms a candidate id but does NOT jump to 'tracking'."""
        sm = make_sm()
        sm.start_probation(committed_id=7)
        assert sm._state == "reidentifying"
        assert sm._committed_id == 7
        assert sm._provisional_streak == 0

    def test_probation_alone_does_not_yield_target_lost_false(self):
        """No step after start_probation → the machine has not committed a lock.

        Unlike start() (which jumps to 'tracking', so a later present step
        commits immediately), start_probation leaves the machine in a
        non-committed state until the pipeline drives it to commit.
        """
        sm = make_sm()
        sm.start_probation(committed_id=7)
        # A coast (absent / sub-bar) frame must keep target_lost True.
        d = sm.step(sim_score=0.0, present=False, frames_since_loss=1,
                    num_candidates=1, distinct_margin=0.0, depth_consistent=True)
        assert d.target_lost is True
        assert d.state == "reidentifying"

    def test_probation_lifts_machine_out_of_terminal_lost(self):
        """start_probation re-arms a terminal-'lost' machine so it can step again."""
        sm = make_sm()
        assert sm._state == "lost"
        sm.start_probation(committed_id=9)
        # A present-by-id step now re-locks (a fresh 'lost' machine would not).
        d = sm.step(sim_score=0.9, present=True, frames_since_loss=0,
                    num_candidates=1, distinct_margin=0.0, depth_consistent=True)
        assert d.target_lost is False
        assert d.state == "tracking"
        assert d.committed_id == 9

    def test_present_step_after_probation_commits(self):
        """A present-by-id step after probation yields tracking / target_lost False.

        (The pipeline's commit drives start()+present; this asserts the FSM
        contract the commit relies on.)"""
        sm = make_sm()
        sm.start_probation(committed_id=7)
        d = sm.step(sim_score=0.9, present=True, frames_since_loss=0,
                    num_candidates=1, distinct_margin=999.0, depth_consistent=True)
        assert d.state == "tracking"
        assert d.target_lost is False
        assert d.committed_id == 7


class TestNofMProvisionalStreak:
    """Phase 3 / Option A: windowed (N-of-M) provisional commit in the FSM.

    Replaces strict-consecutive accumulation: commit when commit_frames (N)
    clear-bar coast frames occur within the last provisional_commit_window (M)
    frames. N consecutive still commits (superset of the old behaviour); a single
    dip no longer zeroes the accumulated count.
    """

    def _coast(self, sm, sim, f):
        return sm.step(sim_score=sim, present=False, frames_since_loss=f,
                       num_candidates=1, distinct_margin=999.0, depth_consistent=True)

    def test_12_of_18_commits(self):
        """12 clear-bar hits within an 18-frame window → commit, despite 6 dips."""
        sm = make_sm(high_bar=0.72, commit_frames=12, provisional_commit_window=18)
        sm.start(committed_id=5)
        # Pattern: 2 hits, 1 dip, repeated → over 18 frames yields 12 hits, 6 dips.
        pattern = ([0.80, 0.80, 0.60] * 6)  # 18 frames, 12 hits
        committed_at = None
        for i, sim in enumerate(pattern, start=1):
            d = self._coast(sm, sim, i)
            if d.target_lost is False:
                committed_at = i
                break
        assert committed_at is not None
        assert committed_at <= 18
        assert d.state == "tracking"

    def test_11_hits_in_18_never_commits(self):
        """Only 11 clear-bar hits within the window → no commit."""
        sm = make_sm(high_bar=0.72, commit_frames=12, provisional_commit_window=18)
        sm.start(committed_id=5)
        # 11 hits + 7 dips = 18 frames, never 12 within any 18-window.
        sims = [0.80] * 11 + [0.60] * 7
        last = None
        for i, sim in enumerate(sims, start=1):
            last = self._coast(sm, sim, i)
        assert last.target_lost is True
        assert last.state == "reidentifying"

    def test_single_dip_does_not_zero_count(self):
        """11 hits, then a dip, then the 12th hit (all within 18) still commits.

        Strict-consecutive would have zeroed at the dip and required 12 fresh
        consecutive hits; the window keeps the 11 alive so the 12th commits.
        """
        sm = make_sm(high_bar=0.72, commit_frames=12, provisional_commit_window=18)
        sm.start(committed_id=5)
        # 11 hits (frames 1-11).
        for i in range(1, 12):
            d = self._coast(sm, 0.80, i)
            assert d.target_lost is True
        # A dip at frame 12 — must NOT zero the accumulated 11.
        d = self._coast(sm, 0.60, 12)
        assert d.target_lost is True
        # The 12th hit at frame 13 (window now holds 11 old hits + this = 12).
        d = self._coast(sm, 0.80, 13)
        assert d.target_lost is False
        assert d.state == "tracking"

    def test_12_consecutive_still_commits(self):
        """Regression: 12 unbroken hits commits at exactly frame 12."""
        sm = make_sm(high_bar=0.72, commit_frames=12, provisional_commit_window=18)
        sm.start(committed_id=5)
        committed_at = None
        for i in range(1, 30):
            d = self._coast(sm, 0.80, i)
            if d.target_lost is False:
                committed_at = i
                break
        assert committed_at == 12
        assert d.state == "tracking"

    def test_5_frame_spike_then_gone_never_commits(self):
        """A 5-frame >=0.72 spike then sub-bar coast never reaches 12 in any window."""
        sm = make_sm(high_bar=0.72, commit_frames=12, provisional_commit_window=18)
        sm.start(committed_id=5)
        sims = [0.80] * 5 + [0.50] * 40  # spike then gone
        committed = False
        for i, sim in enumerate(sims, start=1):
            d = self._coast(sm, sim, i)
            if d.target_lost is False:
                committed = True
                break
        assert committed is False

    def test_window_slides_old_hits_expire(self):
        """Hits older than M frames drop out of the window (no infinite memory).

        9 hits, then a long sub-bar coast that slides them all out, then 11 fresh
        hits: total >12 hits historically but never 12 within any single M-window.
        """
        sm = make_sm(high_bar=0.72, commit_frames=12, provisional_commit_window=18)
        sm.start(committed_id=5)
        f = 0
        for _ in range(9):       # 9 hits
            f += 1
            self._coast(sm, 0.80, f)
        for _ in range(18):      # 18 dips slide the 9 hits fully out of the window
            f += 1
            self._coast(sm, 0.50, f)
        last = None
        for _ in range(11):      # 11 fresh hits — under 12, must not commit
            f += 1
            last = self._coast(sm, 0.80, f)
        assert last.target_lost is True

    def test_start_resets_window(self):
        """start() clears the windowed accumulator (no carry-over across re-arm)."""
        sm = make_sm(high_bar=0.72, commit_frames=12, provisional_commit_window=18)
        sm.start(committed_id=5)
        for i in range(1, 12):   # 11 hits
            self._coast(sm, 0.80, i)
        sm.start(committed_id=5)  # re-arm: window must reset
        # One fresh hit must NOT commit (would if the 11 carried over).
        d = self._coast(sm, 0.80, 1)
        assert d.target_lost is True
