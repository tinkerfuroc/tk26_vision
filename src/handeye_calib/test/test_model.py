import numpy as np
from handeye_calib import transforms as tf
from handeye_calib import handeye_model as hm


def test_board_corners_count_and_centered():
    pts = hm.board_corners(squares_x=5, squares_y=5, square_len=0.04)
    assert pts.shape == (16, 3)              # (5-1)*(5-1) inner corners
    assert np.allclose(pts[:, 2], 0.0)       # planar board
    np.testing.assert_allclose(pts[:, :2].mean(axis=0), [0, 0], atol=1e-9)


def test_project_known_point_on_axis():
    K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
    # board 1 m in front of camera, axes aligned -> corner at board origin maps to principal point
    T_cam_board = tf.T_from_Rt(np.eye(3), [0, 0, 1.0])
    px = hm.project_corners(np.array([[0., 0., 0.]]), T_cam_board, K, dist=None)
    np.testing.assert_allclose(px[0], [320., 240.], atol=1e-9)


def test_project_offset_point():
    K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
    T_cam_board = tf.T_from_Rt(np.eye(3), [0, 0, 2.0])
    px = hm.project_corners(np.array([[0.1, 0.0, 0.0]]), T_cam_board, K, dist=None)
    np.testing.assert_allclose(px[0], [320. + 600 * 0.1 / 2.0, 240.], atol=1e-9)
