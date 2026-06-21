"""Board geometry, the calibration Sample, and the pinhole reprojection used by the solver."""
from dataclasses import dataclass
import numpy as np
import cv2


def board_corners(squares_x=5, squares_y=5, square_len=0.04):
    """Inner ChArUco corner positions (meters), z=0 plane.

    Origin is at the board's bottom-left corner, MATCHING OpenCV's
    ``cv2.aruco.CharucoBoard.getChessboardCorners()`` convention. The first
    inner corner is at ``(square_len, square_len, 0)``.

    Order is row-major over the (squares_x-1) x (squares_y-1) inner grid.

    History: prior to 2026-06-21 this function CENTERED the grid at origin,
    which conflicted with OpenCV's bottom-left-origin convention. Per-frame
    PnP uses ``board.matchImagePoints`` which returns OpenCV-coord object
    points, so it succeeded with sub-pixel reproj — but the bundle adjust
    later mixed OpenCV-frame ``T_cam_board`` with our-frame ``board_pts``
    and every projection was off by a constant ``(square_len * (nx-1) / 2,
    square_len * (ny-1) / 2)`` in board frame. For a 5x5x40mm board that
    constant offset is ``sqrt(80² + 80²)/2 ≈ 141 mm``, which exactly
    matched the operator's reported 142mm train_trans_rmse — a frame
    misalignment masquerading as a calibration failure.
    """
    nx, ny = squares_x - 1, squares_y - 1
    xs = (np.arange(1, squares_x)) * square_len
    ys = (np.arange(1, squares_y)) * square_len
    return np.array(
        [[xs[i], ys[j], 0.0] for j in range(ny) for i in range(nx)], float)


def project_corners(board_pts, T_cam_board, K, dist=None):
    """Project board points (N,3, board frame) into pixels via T_cam_board and K."""
    board_pts = np.ascontiguousarray(board_pts, dtype=float).reshape(-1, 3)
    rvec, _ = cv2.Rodrigues(np.asarray(T_cam_board)[:3, :3])
    tvec = np.asarray(T_cam_board)[:3, 3]
    if dist is None:
        dist = np.zeros(5)
    px, _ = cv2.projectPoints(board_pts, rvec, tvec, np.asarray(K, float), np.asarray(dist, float))
    return px.reshape(-1, 2)


@dataclass
class Sample:
    """One accepted calibration pose."""
    T_base_eef: np.ndarray        # 4x4, A_i (from TF/FK)
    T_cam_board: np.ndarray       # 4x4, B_i (from PnP) — seed input
    obs_px: np.ndarray            # (M,2) observed corner pixels
    corner_idx: np.ndarray        # (M,) indices into board_corners() for obs_px
