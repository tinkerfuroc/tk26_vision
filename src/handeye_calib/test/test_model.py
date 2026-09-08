import numpy as np
from handeye_calib import transforms as tf
from handeye_calib import handeye_model as hm


def test_board_corners_count_and_origin():
    """Inner-corner grid: shape, planarity, OpenCV bottom-left origin.

    The first inner corner sits at ``(square_len, square_len, 0)`` per
    OpenCV's ``CharucoBoard.getChessboardCorners()`` convention — NOT
    centered at origin. The earlier centered layout was a latent bug
    (caused a ~141mm BA residual once mixed with PnP's OpenCV-frame
    T_cam_board); see ``hm.board_corners`` docstring for the full
    incident notes."""
    pts = hm.board_corners(squares_x=5, squares_y=5, square_len=0.04)
    assert pts.shape == (16, 3)              # (5-1)*(5-1) inner corners
    assert np.allclose(pts[:, 2], 0.0)       # planar board
    # First inner corner at (square_len, square_len) — OpenCV convention
    np.testing.assert_allclose(pts[0, :2], [0.04, 0.04], atol=1e-9)
    # Last inner corner at (4*square_len, 4*square_len)
    np.testing.assert_allclose(pts[-1, :2], [0.16, 0.16], atol=1e-9)


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
    """The load-bearing invariant: ``board_corners`` must equal cv2's
    ``CharucoBoard.getChessboardCorners()`` BYTE-FOR-BYTE, including the
    bottom-left origin. Per-frame PnP uses ``board.matchImagePoints`` which
    returns object points in cv2's coords; the BA later indexes into
    ``board_corners[corner_idx]`` and projects via the SAME T_cam_board
    PnP solved. If the two grids disagree by any constant offset, every
    BA projection picks up that offset rotated into pixel coords, which
    BA cannot absorb into a single X+Tbb (different rotations per pose
    project the offset differently) — manifests as bimodal per-sample
    reproj and a train_trans ~= |offset|. Don't re-center either side."""
    import cv2
    aruco = cv2.aruco
    dic = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    board = aruco.CharucoBoard((5, 5), 0.04, 0.03, dic)
    cv_corners = board.getChessboardCorners()             # (16,3) in board frame
    np.testing.assert_allclose(hm.board_corners(5, 5, 0.04), cv_corners, atol=1e-6)
