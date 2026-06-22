import numpy as np
import pytest
from handeye_calib import transforms as tf


def test_vec_roundtrip():
    v = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    T = tf.T_from_vec(v)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(tf.vec_from_T(T), v, atol=1e-9)


def test_invert_is_inverse():
    v = np.array([0.3, 0.1, -0.2, 1.0, 2.0, -3.0])
    T = tf.T_from_vec(v)
    np.testing.assert_allclose(tf.invert(T) @ T, np.eye(4), atol=1e-9)


def test_se3_average_of_identical_is_identity_member():
    T = tf.T_from_vec(np.array([0.2, 0.0, 0.1, 0.5, 0.5, 0.5]))
    avg = tf.se3_average([T, T, T])
    np.testing.assert_allclose(avg, T, atol=1e-9)


def test_rotation_angle_deg():
    from scipy.spatial.transform import Rotation as R
    R1 = np.eye(3)
    R2 = R.from_euler('z', 30, degrees=True).as_matrix()
    assert abs(tf.rotation_angle_deg(R1, R2) - 30.0) < 1e-6


def test_T_from_Rt_roundtrip():
    from scipy.spatial.transform import Rotation as R
    Rm = R.from_euler('xyz', [10, -20, 30], degrees=True).as_matrix()
    t = np.array([1.0, -2.0, 3.0])
    T = tf.T_from_Rt(Rm, t)
    np.testing.assert_allclose(T[:3, :3], Rm, atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], t, atol=1e-12)
    np.testing.assert_allclose(tf.invert(T) @ T, np.eye(4), atol=1e-9)


def test_se3_average_small_spread_midpoint():
    from scipy.spatial.transform import Rotation as R
    T1 = tf.T_from_Rt(R.from_euler('z', -5, degrees=True).as_matrix(), [0.0, 0.0, 1.0])
    T2 = tf.T_from_Rt(R.from_euler('z', 5, degrees=True).as_matrix(), [0.0, 2.0, 1.0])
    avg = tf.se3_average([T1, T2])
    assert tf.rotation_angle_deg(avg[:3, :3], np.eye(3)) < 1e-6      # halfway between -5 and +5
    np.testing.assert_allclose(avg[:3, 3], [0.0, 1.0, 1.0], atol=1e-9)


def test_se3_average_antipodal_sign_alignment():
    # +170 and -170 deg about z: the short-way midpoint is 180 deg. Without quaternion
    # sign-alignment a naive mean would wrongly collapse to identity (0 deg).
    from scipy.spatial.transform import Rotation as R
    T1 = tf.T_from_Rt(R.from_euler('z', 170, degrees=True).as_matrix(), [0.0, 0.0, 0.0])
    T2 = tf.T_from_Rt(R.from_euler('z', -170, degrees=True).as_matrix(), [0.0, 0.0, 0.0])
    avg = tf.se3_average([T1, T2])
    assert abs(tf.rotation_angle_deg(np.eye(3), avg[:3, :3]) - 180.0) < 1e-6


def test_se3_average_empty_raises():
    with pytest.raises(ValueError):
        tf.se3_average([])


def test_rigid_3d_3d_recovers_known_transform():
    # rigid_3d_3d(src, dst) returns T s.t. dst ≈ R @ src + t (no scale).
    # Used for the FFS-depth metric cross-check: src = board-frame corner
    # model points, dst = FFS-deprojected camera-frame points => T_cam_board.
    from scipy.spatial.transform import Rotation as R
    rng = np.random.default_rng(0)
    Rm = R.from_euler('xyz', [25, -40, 12], degrees=True).as_matrix()
    t = np.array([0.3, -0.15, 0.55])
    src = rng.uniform(-0.1, 0.1, size=(20, 3))
    dst = (Rm @ src.T).T + t
    T = tf.rigid_3d_3d(src, dst)
    np.testing.assert_allclose(T[:3, :3], Rm, atol=1e-9)
    np.testing.assert_allclose(T[:3, 3], t, atol=1e-9)
    # Applying T to src reproduces dst.
    src_h = np.c_[src, np.ones(len(src))]
    np.testing.assert_allclose((T @ src_h.T).T[:, :3], dst, atol=1e-9)


def test_rigid_3d_3d_no_reflection_under_noise():
    # Coplanar/near-degenerate points + noise must still yield a proper
    # rotation (det = +1), never a reflection (the classic Kabsch SVD trap).
    from scipy.spatial.transform import Rotation as R
    rng = np.random.default_rng(3)
    Rm = R.from_euler('xyz', [5, 8, -3], degrees=True).as_matrix()
    t = np.array([0.1, 0.2, 0.5])
    src = np.c_[rng.uniform(-0.08, 0.08, size=(16, 2)), np.zeros(16)]  # planar board
    dst = (Rm @ src.T).T + t + rng.normal(0, 1e-3, size=(16, 3))
    T = tf.rigid_3d_3d(src, dst)
    assert np.linalg.det(T[:3, :3]) > 0.99           # proper rotation, not a flip
    assert tf.rotation_angle_deg(T[:3, :3], Rm) < 1.0
