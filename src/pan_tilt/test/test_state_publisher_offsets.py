import math


class _FakeLogger:
    def __init__(self):
        self.warnings = []
    def warning(self, msg):
        self.warnings.append(msg)


def test_resolve_offset_wraps_and_warns():
    from pan_tilt.pan_tilt_state_publisher import _resolve_offset
    log = _FakeLogger()
    out = _resolve_offset(8.348085384508424, "tilt_offset_rad", log)
    assert math.isclose(out, 2.0649000773, abs_tol=1e-6)
    assert len(log.warnings) == 1
    assert "tilt_offset_rad" in log.warnings[0]


def test_resolve_offset_in_range_no_warn():
    from pan_tilt.pan_tilt_state_publisher import _resolve_offset
    log = _FakeLogger()
    out = _resolve_offset(1.3306144109, "tilt_offset_rad", log)
    assert math.isclose(out, 1.3306144109, abs_tol=1e-9)
    assert log.warnings == []
