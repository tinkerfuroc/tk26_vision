import math
from vision_track.sim_track_server import TrackBuffer


def test_finite_point_sets_tracking_and_updates_position():
    b = TrackBuffer(stale_timeout_sec=1.0)
    b.on_point(1.0, 2.0, "map", t=10.0)
    assert b.lost(now=10.05) is False
    assert b.position() == (1.0, 2.0, "map")


def test_nan_sentinel_marks_lost_and_holds_position():
    b = TrackBuffer(stale_timeout_sec=1.0)
    b.on_point(1.0, 2.0, "map", t=10.0)
    b.on_point(float("nan"), float("nan"), "map", t=10.1)
    assert b.lost(now=10.10) is True
    assert b.position() == (1.0, 2.0, "map")        # last finite point HELD


def test_staleness_marks_lost():
    b = TrackBuffer(stale_timeout_sec=1.0)
    b.on_point(1.0, 2.0, "map", t=10.0)
    assert b.lost(now=11.5) is True                 # >1 s since last finite point


def test_reacq_passthrough_and_default():
    b = TrackBuffer()
    assert b.reacq() == 0                            # default TRACKING
    b.on_reacq(2)
    assert b.reacq() == 2
