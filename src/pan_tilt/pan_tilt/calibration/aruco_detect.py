"""ChArUco board detection + pose estimation.

Independent of ROS. Given:
  - an 8-bit RGB (or grayscale) image
  - a camera matrix K (3x3) + distortion coefficients
  - a pre-built CharucoBoard

returns the board pose in the **OpenCV optical frame** (z-forward, x-right, y-down)
along with detection-quality metrics for outlier gating.

Convention reminder: the caller is responsible for converting the returned
optical-frame pose to the ROS body frame via `utils.optical_to_body`. We keep
this module pure-OpenCV so it can also be used stand-alone for intrinsic
calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ---- default board config (plan spec) ---------------------------------------
#
# Dictionary: DICT_5X5_100 — ample ID budget for 5x7 board (only 24 markers used).
# Square size 40 mm, marker size 30 mm. Callers can override to match a physical
# board that differs.

DEFAULT_SQUARES_X = 5
DEFAULT_SQUARES_Y = 7
DEFAULT_SQUARE_LEN_M = 0.040
DEFAULT_MARKER_LEN_M = 0.030
DEFAULT_DICT_ID = cv2.aruco.DICT_5X5_100


# ---- data classes -----------------------------------------------------------

@dataclass
class BoardSpec:
    squares_x: int = DEFAULT_SQUARES_X
    squares_y: int = DEFAULT_SQUARES_Y
    square_len_m: float = DEFAULT_SQUARE_LEN_M
    marker_len_m: float = DEFAULT_MARKER_LEN_M
    dict_id: int = DEFAULT_DICT_ID

    @property
    def n_inner_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)


@dataclass
class Detection:
    pose_optical: np.ndarray               # 4x4 T_cam_optical_to_marker (identity on failure)
    n_corners: int                         # chessboard corners actually matched
    reprojection_rms_px: float             # post-PnP reprojection error; inf on failure
    success: bool

    def valid(self, *, min_corners: int = 10, max_reproj_px: float = 1.5) -> bool:
        return (
            self.success
            and self.n_corners >= min_corners
            and self.reprojection_rms_px <= max_reproj_px
        )


# ---- board + detector builders ----------------------------------------------

def build_board(spec: Optional[BoardSpec] = None) -> cv2.aruco.CharucoBoard:
    spec = spec or BoardSpec()
    adict = cv2.aruco.getPredefinedDictionary(spec.dict_id)
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_len_m,
        spec.marker_len_m,
        adict,
    )


def build_detector(board: cv2.aruco.CharucoBoard) -> cv2.aruco.CharucoDetector:
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    charuco_params = cv2.aruco.CharucoParameters()
    return cv2.aruco.CharucoDetector(
        board,
        charuco_params,
        detector_params,
    )


# ---- main entry -------------------------------------------------------------

def detect_pose(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    board: Optional[cv2.aruco.CharucoBoard] = None,
    detector: Optional[cv2.aruco.CharucoDetector] = None,
) -> Detection:
    """Detect a ChArUco board and return its pose in the camera optical frame.

    Pose is `T_cam_optical_to_marker` (i.e., maps a marker-frame point into the
    camera optical frame). Translation is in meters.

    On any failure (too few corners, PnP divergence, unexpected exception) this
    returns a `Detection(success=False)` rather than raising, so the collector
    can skip the sample and continue. Log the underlying exception via your
    node logger if needed — we do not log here (no ROS dependency).
    """
    if board is None:
        board = build_board()
    if detector is None:
        detector = build_detector(board)

    gray = _to_gray(image)
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)

    if ch_corners is None or ch_ids is None or len(ch_ids) < 4:
        return _fail_detection()

    try:
        obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    except cv2.error:
        return _fail_detection()

    if obj_pts is None or len(obj_pts) < 4:
        return _fail_detection()

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return _fail_detection()

    reproj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    reproj = reproj.reshape(-1, 2)
    img_pts_flat = img_pts.reshape(-1, 2)
    residuals = reproj - img_pts_flat
    rms = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.flatten()

    return Detection(
        pose_optical=T,
        n_corners=int(len(ch_ids)),
        reprojection_rms_px=rms,
        success=True,
    )


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _fail_detection() -> Detection:
    return Detection(
        pose_optical=np.eye(4),
        n_corners=0,
        reprojection_rms_px=float("inf"),
        success=False,
    )


# ---- averaging across multiple detections (per-cell outlier rejection) ------
#
# Collector captures N=10 frames per cell. We MAD-reject in translation
# (rotation outliers handled downstream by SE(3) averaging being insensitive
# to rare noise spikes when the rotation cluster is tight).

def robust_average(detections: list[Detection], *, mad_k: float = 3.0) -> Optional[Detection]:
    valid = [d for d in detections if d.valid()]
    if len(valid) < 3:
        return None

    translations = np.stack([d.pose_optical[:3, 3] for d in valid])
    median = np.median(translations, axis=0)
    abs_dev = np.abs(translations - median)
    mad = np.median(abs_dev, axis=0) + 1e-9
    normed = abs_dev / mad
    keep = np.all(normed < mad_k, axis=1)
    kept = [d for d, k in zip(valid, keep) if k]

    if len(kept) < 3:
        return None

    # SE(3) chordal mean of the kept detections.
    from .optimize import _average_se3  # reuse

    T_mean = _average_se3([d.pose_optical for d in kept])
    corners_mean = float(np.mean([d.n_corners for d in kept]))
    rms_mean = float(np.mean([d.reprojection_rms_px for d in kept]))
    return Detection(
        pose_optical=T_mean,
        n_corners=int(corners_mean),
        reprojection_rms_px=rms_mean,
        success=True,
    )
