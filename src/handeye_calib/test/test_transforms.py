import numpy as np
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
