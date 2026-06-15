"""Board geometry, the calibration Sample, and the pinhole reprojection used by the solver."""
from dataclasses import dataclass
import numpy as np
import cv2


def board_corners(squares_x=5, squares_y=5, square_len=0.04):
    """Inner ChArUco corner positions (meters), board-centered, z=0 plane.

    Order is row-major over the (squares_x-1) x (squares_y-1) inner grid, matching
    cv2.aruco CharucoBoard chessboard-corner ordering.
    """
    nx, ny = squares_x - 1, squares_y - 1
    xs = (np.arange(1, squares_x)) * square_len
    ys = (np.arange(1, squares_y)) * square_len
    pts = np.array([[xs[i], ys[j], 0.0] for j in range(ny) for i in range(nx)], float)
    pts[:, 0] -= pts[:, 0].mean()
    pts[:, 1] -= pts[:, 1].mean()
    return pts


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
