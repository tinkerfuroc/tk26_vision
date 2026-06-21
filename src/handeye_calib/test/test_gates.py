import numpy as np
from scipy.spatial.transform import Rotation as R
from handeye_calib import transforms as tf
from handeye_calib import gates


def test_stability_tracker_flags_when_steady():
    trk = gates.StabilityTracker(window=3, rot_tol_deg=0.1, trans_tol_m=0.0003)
    T = tf.T_from_Rt(np.eye(3), [0, 0, 0.5])
    assert trk.update(T) is False        # need a full window
    assert trk.update(T) is False
    assert trk.update(T) is True         # 3 steady frames -> stable


def test_stability_tracker_rejects_jitter():
    trk = gates.StabilityTracker(window=3, rot_tol_deg=0.1, trans_tol_m=0.0003)
    for k in range(5):
        T = tf.T_from_Rt(np.eye(3), [0, 0, 0.5 + 0.01 * k])   # moving 1 cm/frame
        assert trk.update(T) is False


def test_is_diverse():
    accepted = [tf.T_from_Rt(np.eye(3), [0, 0, 0.5])]
    near = tf.T_from_Rt(R.from_euler('z', 10, degrees=True).as_matrix(), [0, 0, 0.5])
    far = tf.T_from_Rt(R.from_euler('z', 40, degrees=True).as_matrix(), [0, 0, 0.5])
    assert gates.is_diverse(near, accepted, min_deg=30) is False
    assert gates.is_diverse(far, accepted, min_deg=30) is True
    assert gates.is_diverse(near, [], min_deg=30) is True     # first pose always ok


def test_quality_ok_reasons():
    ok, reason = gates.quality_ok(n_corners=16, reproj_px=0.8, area_frac=0.2)
    assert ok and reason == "ok"
    ok, reason = gates.quality_ok(n_corners=4, reproj_px=0.8, area_frac=0.2)
    assert not ok and "corners" in reason
    ok, reason = gates.quality_ok(n_corners=16, reproj_px=3.0, area_frac=0.2)
    assert not ok and "reproj" in reason


def test_quality_ok_board_too_small():
    # area_frac well below the (now-0.01) floor — board so distant the
    # bbox spans <10% of one image dimension.
    ok, reason = gates.quality_ok(n_corners=16, reproj_px=0.8, area_frac=0.005)
    assert not ok and "board too small" in reason


def test_stability_tracker_rejects_slow_drift():
    # Safety-critical: each frame is within tol of its neighbor, but the window
    # spans > tol end-to-end. The all-vs-most-recent comparison must reject this
    # (a naive neighbor-vs-neighbor check would wrongly accept it).
    trk = gates.StabilityTracker(window=3, rot_tol_deg=0.1, trans_tol_m=0.0003)
    out = False
    for k in range(3):
        T = tf.T_from_Rt(np.eye(3), [0, 0, 0.5 + 0.0002 * k])  # 0.2 mm/frame
        out = trk.update(T)
    assert out is False  # oldest vs newest = 0.4 mm > 0.3 mm tol
