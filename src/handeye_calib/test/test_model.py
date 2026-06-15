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


def test_board_corners_match_cv2_charuco_order():
    # Pin board_corners() to OpenCV's CharucoBoard inner-corner ordering: the
    # load-bearing invariant is that corner_idx lines up with cv2's detected
    # charuco IDs, so board_corners(5,5,0.04) must equal cv2's getChessboardCorners
    # grid (board-centered). Constructor signature matches the installed OpenCV
    # (4.10.0) — same form used by pan_tilt.calibration.aruco_detect.build_board.
    import cv2
    aruco = cv2.aruco
    dic = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    board = aruco.CharucoBoard((5, 5), 0.04, 0.03, dic)
    cv_corners = board.getChessboardCorners()             # (16,3) in board frame
    cv_centered = cv_corners.copy()
    cv_centered[:, 0] -= cv_centered[:, 0].mean()
    cv_centered[:, 1] -= cv_centered[:, 1].mean()
    np.testing.assert_allclose(hm.board_corners(5, 5, 0.04), cv_centered, atol=1e-6)
