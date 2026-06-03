"""Tests for ptbench.common.metrics — scoreboard metric computation.

Aligned lists are fabricated with explicit nanosecond timestamps (1e9 ns = 1 s)
so every span/latency in seconds is exact and asserted against concrete numbers.
"""
import math

import pytest

from ptbench.common.align import PredFrame
from ptbench.common.metrics import MetricConfig, compute_metrics
from ptbench.common.schema import GtFrame

S = 1_000_000_000  # 1 second in ns


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def g(t_s, present=True, centroid=(0.0, 0.0, 2.0)):
    """GtFrame at t_s seconds. bbox is irrelevant to metrics."""
    bbox = (0, 0, 10, 10) if present else None
    c = centroid if present else None
    return GtFrame(
        t_ns=int(t_s * S),
        present=present,
        bbox=bbox,
        centroid_field=c,
        centroid_track=c,
    )


def p(t_s, lost=False, xyz=(0.0, 0.0, 2.0), tid=1):
    return PredFrame(t_ns=int(t_s * S), target_lost=lost, target_track_id=tid, point_xyz=xyz)


# ---------------------------------------------------------------------------
# correct_lock_rate
# ---------------------------------------------------------------------------

class TestCorrectLockRate:
    def test_all_correct(self):
        aligned = [
            (g(0.0, centroid=(0.0, 0.0, 2.0)), p(0.0, xyz=(0.0, 0.0, 2.0))),
            (g(0.1, centroid=(0.0, 0.0, 2.0)), p(0.1, xyz=(0.1, 0.0, 2.0))),  # 0.1m < 0.5
        ]
        m = compute_metrics(aligned)
        assert m["correct_lock_rate"] == pytest.approx(1.0)

    def test_half_correct(self):
        aligned = [
            (g(0.0), p(0.0, xyz=(0.0, 0.0, 2.0))),       # dist 0 -> correct
            (g(0.1), p(0.1, xyz=(3.0, 0.0, 2.0))),       # dist 3 -> not correct (wrong)
        ]
        m = compute_metrics(aligned)
        assert m["correct_lock_rate"] == pytest.approx(0.5)

    def test_no_present_frames_zero(self):
        aligned = [
            (g(0.0, present=False), p(0.0)),
            (g(0.1, present=False), None),
        ]
        m = compute_metrics(aligned)
        assert m["correct_lock_rate"] == 0.0

    def test_lost_pred_not_counted_correct(self):
        aligned = [(g(0.0), p(0.0, lost=True, xyz=(0.0, 0.0, 2.0)))]
        m = compute_metrics(aligned)
        assert m["correct_lock_rate"] == 0.0

    def test_none_pred_not_correct(self):
        aligned = [(g(0.0), None)]
        m = compute_metrics(aligned)
        assert m["correct_lock_rate"] == 0.0

    def test_exact_radius_boundary_is_correct(self):
        # dist exactly 0.5 -> correct (<=)
        aligned = [(g(0.0, centroid=(0.0, 0.0, 2.0)), p(0.0, xyz=(0.5, 0.0, 2.0)))]
        m = compute_metrics(aligned)
        assert m["correct_lock_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# wrong_lock_episodes (sustained > 0.5s)
# ---------------------------------------------------------------------------

class TestWrongLockEpisodes:
    def test_sustained_run_counts(self):
        # wrong locks (dist > 0.75) from t=0.0 to t=0.6 -> span 0.6s > 0.5s
        far = (5.0, 0.0, 2.0)
        aligned = [
            (g(0.0), p(0.0, xyz=far)),
            (g(0.2), p(0.2, xyz=far)),
            (g(0.4), p(0.4, xyz=far)),
            (g(0.6), p(0.6, xyz=far)),
        ]
        m = compute_metrics(aligned)
        assert m["wrong_lock_episodes"] == 1

    def test_brief_run_not_counted(self):
        # wrong locks from t=0.0 to t=0.3 -> span 0.3s <= 0.5s
        far = (5.0, 0.0, 2.0)
        aligned = [
            (g(0.0), p(0.0, xyz=far)),
            (g(0.1), p(0.1, xyz=far)),
            (g(0.3), p(0.3, xyz=far)),
        ]
        m = compute_metrics(aligned)
        assert m["wrong_lock_episodes"] == 0

    def test_two_separate_sustained_runs(self):
        far = (5.0, 0.0, 2.0)
        near = (0.0, 0.0, 2.0)
        aligned = [
            (g(0.0), p(0.0, xyz=far)),
            (g(0.6), p(0.6, xyz=far)),   # run 1: span 0.6
            (g(0.7), p(0.7, xyz=near)),  # break (correct)
            (g(1.0), p(1.0, xyz=far)),
            (g(1.6), p(1.6, xyz=far)),   # run 2: span 0.6
        ]
        m = compute_metrics(aligned)
        assert m["wrong_lock_episodes"] == 2

    def test_exact_span_boundary_not_counted(self):
        # span exactly 0.5s is NOT > 0.5s
        far = (5.0, 0.0, 2.0)
        aligned = [
            (g(0.0), p(0.0, xyz=far)),
            (g(0.5), p(0.5, xyz=far)),
        ]
        m = compute_metrics(aligned)
        assert m["wrong_lock_episodes"] == 0


# ---------------------------------------------------------------------------
# reacquire_latency_s
# ---------------------------------------------------------------------------

class TestReacquireLatency:
    def test_reacquire_after_absent_gap(self):
        # absent then present; the operator reappears at t=1.0 (wrong lock) and
        # the first correct lock arrives at t=1.3. One loss episode, anchored at
        # the reappearance frame -> a single 0.3s sample (no spurious 0.0).
        aligned = [
            (g(0.0, present=False), None),
            (g(1.0), p(1.0, xyz=(5.0, 0.0, 2.0))),   # reappears, wrong (seek anchor)
            (g(1.3), p(1.3, xyz=(0.0, 0.0, 2.0))),   # first correct -> latency 0.3
            (g(1.4), p(1.4, xyz=(0.0, 0.0, 2.0))),
        ]
        m = compute_metrics(aligned)
        assert m["reacquire_latency_s"]["samples"] == pytest.approx([0.3])
        assert m["reacquire_latency_s"]["median"] == pytest.approx(0.3)
        assert m["reacquire_latency_s"]["max"] == pytest.approx(0.3)

    def test_reacquire_after_absent_gap_clean(self):
        # absent then present; the FIRST present frame is already a correct lock,
        # so there is exactly one event (latency 0.0) and no spurious follow-on.
        aligned = [
            (g(0.0, present=False), None),
            (g(1.0), p(1.0, xyz=(0.0, 0.0, 2.0))),   # event start, correct -> latency 0.0
            (g(1.1), p(1.1, xyz=(0.0, 0.0, 2.0))),
        ]
        m = compute_metrics(aligned)
        assert m["reacquire_latency_s"]["samples"] == pytest.approx([0.0])

    def test_reacquire_after_lost_run(self):
        # present throughout; initial acquire at t=0 (latency 0.0); the lock
        # breaks (lost) at t=0.5 -> loss onset; reacquired at t=1.0 -> 0.5s.
        aligned = [
            (g(0.0), p(0.0, xyz=(0.0, 0.0, 2.0))),   # initial acquire -> latency 0.0
            (g(0.5), p(0.5, lost=True)),             # lost -> loss onset at 0.5
            (g(1.0), p(1.0, xyz=(0.0, 0.0, 2.0))),   # reacquired -> latency 0.5
            (g(1.2), p(1.2, xyz=(0.0, 0.0, 2.0))),
        ]
        m = compute_metrics(aligned)
        assert sorted(m["reacquire_latency_s"]["samples"]) == pytest.approx([0.0, 0.5])
        assert m["reacquire_latency_s"]["max"] == pytest.approx(0.5)

    def test_censored_when_never_reacquired(self):
        # operator reappears at t=1.0 but is never correctly locked -> one
        # censored sample = onset (reappearance 1.0) to last frame (1.8) = 0.8s.
        far = (5.0, 0.0, 2.0)
        aligned = [
            (g(0.0, present=False), None),
            (g(1.0), p(1.0, xyz=far)),
            (g(1.8), p(1.8, xyz=far)),   # 1.0..1.8 = 0.8s, never correct (censored)
        ]
        m = compute_metrics(aligned)
        assert m["reacquire_latency_s"]["samples"] == pytest.approx([0.8])
        assert m["reacquire_latency_s"]["max"] == pytest.approx(0.8)

    def test_empty_default(self):
        m = compute_metrics([])
        assert m["reacquire_latency_s"] == {"median": 0.0, "max": 0.0, "samples": []}

    def test_no_events_when_all_absent(self):
        aligned = [(g(0.0, present=False), None), (g(1.0, present=False), None)]
        m = compute_metrics(aligned)
        assert m["reacquire_latency_s"]["samples"] == []


# ---------------------------------------------------------------------------
# pos_error_lateral_m / pos_error_range_m
# ---------------------------------------------------------------------------

class TestPosError:
    def test_median_and_p95(self):
        # all correct locks (within 0.5m). lateral = dx, range = dz.
        aligned = [
            (g(0.0, centroid=(0.0, 0.0, 2.0)), p(0.0, xyz=(0.1, 0.0, 2.0))),  # lat .1 rng 0
            (g(0.1, centroid=(0.0, 0.0, 2.0)), p(0.1, xyz=(0.2, 0.0, 2.1))),  # lat .2 rng .1
            (g(0.2, centroid=(0.0, 0.0, 2.0)), p(0.2, xyz=(0.3, 0.0, 2.2))),  # lat .3 rng .2
        ]
        m = compute_metrics(aligned)
        assert m["pos_error_lateral_m"]["median"] == pytest.approx(0.2)
        assert m["pos_error_range_m"]["median"] == pytest.approx(0.1)
        # p95 over [.1,.2,.3] = .29
        assert m["pos_error_lateral_m"]["p95"] == pytest.approx(0.29, abs=1e-9)

    def test_only_correct_locks_contribute(self):
        # one correct, one wrong (excluded)
        aligned = [
            (g(0.0, centroid=(0.0, 0.0, 2.0)), p(0.0, xyz=(0.1, 0.0, 2.0))),  # correct lat .1
            (g(0.1, centroid=(0.0, 0.0, 2.0)), p(0.1, xyz=(5.0, 0.0, 2.0))),  # wrong, excluded
        ]
        m = compute_metrics(aligned)
        assert m["pos_error_lateral_m"]["median"] == pytest.approx(0.1)

    def test_empty_default(self):
        m = compute_metrics([])
        assert m["pos_error_lateral_m"] == {"median": 0.0, "p95": 0.0}
        assert m["pos_error_range_m"] == {"median": 0.0, "p95": 0.0}


# ---------------------------------------------------------------------------
# false_target_rate
# ---------------------------------------------------------------------------

class TestFalseTargetRate:
    def test_half_false(self):
        aligned = [
            (g(0.0, present=False), p(0.0, lost=False)),  # false target
            (g(0.1, present=False), p(0.1, lost=True)),   # lost -> not false
            (g(0.2, present=False), None),                # no pred -> not false
            (g(0.3, present=False), p(0.3, lost=False)),  # false target
        ]
        m = compute_metrics(aligned)
        # 2 false / 4 absent = 0.5
        assert m["false_target_rate"] == pytest.approx(0.5)

    def test_no_absent_zero(self):
        aligned = [(g(0.0), p(0.0))]
        m = compute_metrics(aligned)
        assert m["false_target_rate"] == 0.0

    def test_all_false(self):
        aligned = [
            (g(0.0, present=False), p(0.0)),
            (g(0.1, present=False), p(0.1)),
        ]
        m = compute_metrics(aligned)
        assert m["false_target_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# throughput passthrough + empty defaults
# ---------------------------------------------------------------------------

class TestThroughputAndEmpty:
    def test_throughput_passthrough(self):
        aligned = [(g(0.0), p(0.0))]
        m = compute_metrics(aligned, throughput_hz=14.5)
        assert m["throughput_hz"] == 14.5

    def test_throughput_none_default(self):
        m = compute_metrics([(g(0.0), p(0.0))])
        assert m["throughput_hz"] is None

    def test_empty_input_all_defaults(self):
        m = compute_metrics([])
        assert m["correct_lock_rate"] == 0.0
        assert m["wrong_lock_episodes"] == 0
        assert m["reacquire_latency_s"] == {"median": 0.0, "max": 0.0, "samples": []}
        assert m["pos_error_lateral_m"] == {"median": 0.0, "p95": 0.0}
        assert m["pos_error_range_m"] == {"median": 0.0, "p95": 0.0}
        assert m["false_target_rate"] == 0.0
        assert m["throughput_hz"] is None

    def test_does_not_raise_on_degenerate(self):
        # mix of None preds, absent frames, lost preds — must not raise
        aligned = [
            (g(0.0, present=False), None),
            (g(0.1), p(0.1, lost=True)),
            (g(0.2), None),
        ]
        m = compute_metrics(aligned)  # should not raise
        assert isinstance(m, dict)

    def test_config_radii_respected(self):
        # With a tighter correct radius, a 0.4m pred is no longer correct.
        aligned = [(g(0.0, centroid=(0.0, 0.0, 2.0)), p(0.0, xyz=(0.4, 0.0, 2.0)))]
        cfg = MetricConfig(correct_radius_m=0.3)
        m = compute_metrics(aligned, cfg=cfg)
        assert m["correct_lock_rate"] == 0.0


class TestGatesOnField:
    def _gt(self, t_ns, field, track, present=True):
        from ptbench.common.schema import GtFrame
        return GtFrame(
            t_ns=t_ns, present=present,
            bbox=(0, 0, 10, 10) if present else None,
            centroid_field=field, centroid_track=track,
        )

    def _pred(self, t_ns, xyz, lost=False):
        from ptbench.common.align import PredFrame
        return PredFrame(t_ns=t_ns, target_lost=lost, target_track_id=1, point_xyz=xyz)

    def test_correct_lock_uses_field_not_track(self):
        from ptbench.common.metrics import compute_metrics
        # pred matches TRACK exactly but is 1.0 m from FIELD → NOT a correct lock.
        aligned = [(
            self._gt(1000, field=(1.0, 0.0, 3.0), track=(0.0, 0.0, 3.0)),
            self._pred(1000, (0.0, 0.0, 3.0)),
        )]
        m = compute_metrics(aligned)
        assert m["correct_lock_rate"] == 0.0  # 1.0 m > correct_radius 0.5

    def test_lateral_error_measured_against_field(self):
        from ptbench.common.metrics import compute_metrics
        # pred == field → correct lock with ~0 lateral error.
        aligned = [(
            self._gt(1000, field=(0.0, 0.0, 3.0), track=(0.3, 0.0, 3.0)),
            self._pred(1000, (0.0, 0.0, 3.0)),
        )]
        m = compute_metrics(aligned)
        assert m["correct_lock_rate"] == 1.0
        assert m["pos_error_lateral_m"]["median"] < 1e-6

    def test_centroid_track_diagnostic_present(self):
        from ptbench.common.metrics import compute_metrics
        aligned = [(
            self._gt(1000, field=(0.0, 0.0, 3.0), track=(0.3, 0.0, 3.0)),
            self._pred(1000, (0.0, 0.0, 3.0)),
        )]
        m = compute_metrics(aligned)
        assert "centroid_track_diag" in m
        diag = m["centroid_track_diag"]
        # pred (0,0,3) vs track (0.3,0,3) → lateral ~0.3 in the diagnostic.
        assert diag["pos_error_lateral_m"]["median"] == pytest.approx(0.3, abs=1e-6)
