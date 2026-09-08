import math
from pan_tilt.calibration.utils import wrap_to_pi
from pan_tilt.calibration.run_calibration import _params_to_dict
from pan_tilt.calibration.pan_tilt_model import PanTiltParams


def test_wrap_to_pi_basic():
    assert math.isclose(wrap_to_pi(0.0), 0.0, abs_tol=1e-12)
    # +478.31 deg (the real wzx_0529 value) -> +118.31 deg
    assert math.isclose(wrap_to_pi(8.348085384508424), 2.0649000773, abs_tol=1e-6)
    # rotation-equivalence: differs from input by a multiple of 2*pi
    x = 8.348085384508424
    assert math.isclose((x - wrap_to_pi(x)) % (2 * math.pi), 0.0, abs_tol=1e-9)
    # spec: the interval is half-open (-pi, pi], so -pi maps to +pi (not -pi).
    # A lenient bounds check would pass even if the boundary fixup were deleted.
    assert math.isclose(wrap_to_pi(-math.pi), math.pi, abs_tol=1e-12)


def test_params_to_dict_normalizes_offsets():
    p = PanTiltParams()
    p.theta_t_offset = 8.348085384508424   # +478.31 deg, un-normalized
    p.theta_p_offset = 3.1400375561        # already in range
    d = _params_to_dict(p)
    assert math.isclose(d["theta_t_offset_rad"], 2.0649000773, abs_tol=1e-6)
    assert math.isclose(d["theta_t_offset_deg"], 118.3098, abs_tol=1e-2)
    assert math.isclose(d["theta_p_offset_rad"], 3.1400375561, abs_tol=1e-9)
    # _deg must be derived from the SAME normalized value
    assert math.isclose(d["theta_t_offset_deg"], math.degrees(d["theta_t_offset_rad"]), abs_tol=1e-9)


def test_render_offsets_yaml_normalizes():
    # The per-robot offsets renderer must wrap un-normalized offsets (from
    # OLD result JSONs that predate the solver-side wrap) to (-pi, pi].
    from pan_tilt.calibration.apply_to_urdf import render_offsets_yaml
    out = render_offsets_yaml("tinker1", 3.1400375561, 8.348085384508424)
    assert "tilt_offset_rad: 2.0649" in out          # +478 deg wrapped to +118 deg
    assert "tilt_offset_rad: 8.34" not in out
    assert "pan_offset_rad: 3.1400375561" in out      # already in range, unchanged
    # Renders the tinker_robot_config profile shape the state publisher reads.
    assert "pan_tilt:" in out and "offsets:" in out
