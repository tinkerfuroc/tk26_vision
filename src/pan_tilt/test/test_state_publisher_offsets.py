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


def test_per_robot_offsets_win_over_params(monkeypatch):
    from pan_tilt import pan_tilt_state_publisher as sp

    class FakeCfg:
        def get(self, key, default=None):
            return {'pan_tilt.offsets.pan_offset_rad': 1.25,
                    'pan_tilt.offsets.tilt_offset_rad': -0.5}.get(key, default)

    monkeypatch.setattr(sp, '_load_profile', lambda: FakeCfg())
    assert sp._load_per_robot_offsets(_FakeLogger()) == (1.25, -0.5)


def test_per_robot_offsets_absent_falls_back(monkeypatch):
    from pan_tilt import pan_tilt_state_publisher as sp
    monkeypatch.setattr(sp, '_load_profile', lambda: None)
    assert sp._load_per_robot_offsets(_FakeLogger()) is None


def test_per_robot_offsets_partial_falls_back(monkeypatch):
    from pan_tilt import pan_tilt_state_publisher as sp

    class FakeCfg:
        def get(self, key, default=None):
            return 1.25 if key.endswith('pan_offset_rad') else None

    monkeypatch.setattr(sp, '_load_profile', lambda: FakeCfg())
    assert sp._load_per_robot_offsets(_FakeLogger()) is None
