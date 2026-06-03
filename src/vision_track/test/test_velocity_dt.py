"""_update_target_velocity must use a supplied frame-stamp dt, not wall clock."""
from vision_track.yolo_tracker import YOLOTracker


def _bare_tracker():
    # Construct without loading YOLO/torch: bypass __init__ and set just the
    # fields _update_target_velocity touches.
    t = YOLOTracker.__new__(YOLOTracker)
    t.last_known_center = (100.0, 100.0)
    t.last_position_time = 0.0
    t.target_velocity = (0.0, 0.0)
    t.target_velocity_history = []
    return t


def test_velocity_uses_supplied_dt():
    t = _bare_tracker()
    # Move 50 px in x over dt=0.5 s of scene time → raw vx = 100 px/s.
    # EMA alpha=0.3 from a zero start → 0.3 * 100 = 30 px/s.
    t._update_target_velocity((150.0, 100.0), dt=0.5)
    vx, vy = t.target_velocity
    assert abs(vx - 30.0) < 1e-6
    assert abs(vy - 0.0) < 1e-6


def test_zero_dt_is_ignored():
    t = _bare_tracker()
    t._update_target_velocity((150.0, 100.0), dt=0.0)
    assert t.target_velocity == (0.0, 0.0)
