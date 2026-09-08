from handeye_calib.handeye_web import validate_pose_set, diff_payload


def test_validate_pose_set_flags_short_sets():
    ok, msg = validate_pose_set([{"joints": [0] * 7} for _ in range(5)])
    assert ok is False and "at least" in msg


def test_validate_pose_set_accepts_enough():
    ok, msg = validate_pose_set([{"joints": [0] * 7} for _ in range(15)])
    assert ok is True


def test_diff_payload_shows_before_after():
    d = diff_payload(old_xyz="0.06746 -0.0175 0.0237",
                     new_xyz="0.1 0.2 0.3",
                     old_rpy="3.14159 -1.5708 0", new_rpy="0 0 0")
    assert d["xyz"]["old"] == "0.06746 -0.0175 0.0237"
    assert d["xyz"]["new"] == "0.1 0.2 0.3"
    assert d["changed"] is True
