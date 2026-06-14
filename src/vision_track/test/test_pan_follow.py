import math
import pytest
from vision_track.core.pan_follow import PanFollower


def _f(**kw):
    # Permissive throttle so single-call tests exercise the control law, not gating.
    base = dict(pan_sign=1.0, deadband_rad=math.radians(1.0),
                min_change_rad=0.0, min_interval_s=0.0, ema_alpha=1.0,
                pan_min_rad=math.radians(-90.0), pan_max_rad=math.radians(90.0))
    base.update(kw)
    return PanFollower(**base)


def test_center_person_right_of_axis_turns_head_toward_them():
    # Person to the right of the optical axis (u > cx): theta > 0. follow_head's
    # convention is world_pan = cur_pan + atan2(x_cam, z_cam) with +x=right and the
    # URDF pan axis "0 0 -1" (positive pan = turn right), i.e. pan_sign=+1 -> the
    # head turns RIGHT (target_pan > current_pan), TOWARD the person.
    foll = _f(pan_sign=1.0)
    fx, cx, u = 600.0, 320.0, 320.0 + 600.0  # atan2(600,600)=45deg
    out = foll.center(u=u, cx=cx, fx=fx, current_pan=0.0, now=1.0)
    assert out == pytest.approx(math.radians(45.0), abs=1e-6)


def test_center_is_absolute_no_accumulation():
    # Same pixel error from a different current_pan yields target = current_pan +
    # sign*theta — it tracks live state, it does NOT integrate.
    foll = _f(pan_sign=1.0)
    fx, cx, u = 600.0, 320.0, 320.0 + 600.0
    out = foll.center(u=u, cx=cx, fx=fx, current_pan=math.radians(10.0), now=1.0)
    assert out == pytest.approx(math.radians(10.0) + math.radians(45.0), abs=1e-6)


def test_center_within_deadband_holds():
    foll = _f(deadband_rad=math.radians(5.0))
    # u == cx -> theta 0 -> target == current_pan -> within deadband -> None.
    assert foll.center(u=320.0, cx=320.0, fx=600.0, current_pan=0.0, now=1.0) is None


def test_center_requires_current_pan_for_absolute():
    foll = _f()
    assert foll.center(u=900.0, cx=320.0, fx=600.0, current_pan=None, now=1.0) is None


def test_center_clamps_to_limits():
    foll = _f(pan_sign=1.0, pan_max_rad=math.radians(30.0))
    # Big positive theta would exceed +30deg; clamp.
    out = foll.center(u=320.0 + 6000.0, cx=320.0, fx=600.0, current_pan=0.0, now=1.0)
    assert out == pytest.approx(math.radians(30.0), abs=1e-6)


def test_no_recenter_method_holds_during_needs_help():
    # The head must HOLD (not recenter) on any loss, including NEEDS_HELP — so the
    # camera keeps pointing where the operator was last seen and can re-detect them.
    # The caller simply stops calling center(); there is no recenter path.
    assert not hasattr(PanFollower, "recenter")


def test_min_interval_throttles_commands():
    foll = _f(min_interval_s=1.0)
    a = foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=10.0)
    assert a is not None                      # first command issues
    b = foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=10.5)
    assert b is None                          # within 1s -> throttled
    c = foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=11.1)
    assert c is not None                      # after the interval -> issues


def test_min_change_suppresses_micro_commands():
    foll = _f(min_change_rad=math.radians(5.0), min_interval_s=0.0)
    first = foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=1.0)
    assert first is not None
    # A target within 5deg of the last command is suppressed.
    again = foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=2.0)
    assert again is None


def test_reset_clears_throttle_and_ema():
    foll = _f(min_interval_s=100.0)
    foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=1.0)
    foll.reset()
    # After reset the interval clock no longer blocks the next command.
    assert foll.center(u=1000.0, cx=320.0, fx=600.0, current_pan=0.0, now=1.5) is not None
