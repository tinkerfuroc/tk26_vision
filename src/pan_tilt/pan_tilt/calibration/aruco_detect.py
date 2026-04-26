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

PnP strategy
------------
A planar target like ChArUco has a structural two-fold ambiguity under IPPE
(see Collins & Bartoli, "Infinitesimal Plane-based Pose Estimation"). The two
candidates reproject within sub-pixel of each other at glancing views, so any
single-criterion picker (lowest reprojection / front-facing sign / closest to
seed) fails on a non-trivial fraction of cells.

This module sidesteps that by promoting `SOLVEPNP_ITERATIVE` to the default
when corner count >= 6: iterative PnP has no two-fold ambiguity and always
converges to a single local minimum. IPPE is still used as the seed (cheap,
analytic, planar-aware) and as the fallback for the 4-5 corner case where
iterative is under-determined. In the IPPE fallback we return BOTH candidates
and let the cell-level cluster_consensus voter pick the right one across many
frames -- a single frame is never asked to commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import utils
from .optimize import _average_se3


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

# Below this corner count, iterative PnP is under-determined; fall through
# to two-candidate IPPE and let cluster_consensus disambiguate.
ITERATIVE_MIN_CORNERS = 6


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
class PoseCandidate:
    """One pose hypothesis for a single frame.

    For an iterative-PnP frame this is the unique solution. For an IPPE
    fallback frame each `Detection` carries one or two of these — the cell
    voter consumes them all.
    """
    pose_optical: np.ndarray   # 4x4 SE(3) in camera optical frame
    reproj_rms_px: float       # post-PnP reprojection RMS


@dataclass
class Detection:
    pose_optical: np.ndarray               # 4x4 — primary candidate (lowest-reproj after refine)
    n_corners: int                         # chessboard corners actually matched
    reprojection_rms_px: float             # primary candidate's reproj RMS; inf on failure
    success: bool
    candidates: list[PoseCandidate] = field(default_factory=list)
    method: str = ""                       # "iterative" | "ippe" | "" on failure

    def valid(self, *, min_corners: int = 10, max_reproj_px: float = 1.5) -> bool:
        # Restored to the original strict thresholds. Briefly loosened to
        # (6, 2.5px) earlier this session to fix an operator "0 valid
        # detections" warning, but the regression audit on
        # 0426_newset showed 2 samples with reproj 2.0-2.2px passed the
        # lax gate and dragged hand-eye RMSE from 4.8 mm -> 6.8 mm. The
        # actual cause of the operator's warning was the 0.3 px image-
        # stability gate timing out (also fixed -- now 1.0 px + falls
        # back to fixed wait instead of skipping the cell), so the
        # per-frame gate doesn't need to be lax. Cell-level
        # cluster_consensus + iterative MAD/RANSAC handle residual jitter
        # in the few-frames case; the per-frame gate's job is to keep
        # genuinely noisy detections out of the cell aggregate.
        return (
            self.success
            and self.n_corners >= min_corners
            and self.reprojection_rms_px <= max_reproj_px
        )

    def reject_reason(self, *, min_corners: int = 10, max_reproj_px: float = 1.5) -> str:
        """Why valid() returned False, as a short diagnostic string."""
        if not self.success:
            return "no detection"
        if self.n_corners < min_corners:
            return f"only {self.n_corners} corners (<{min_corners})"
        if self.reprojection_rms_px > max_reproj_px:
            return f"reproj RMS {self.reprojection_rms_px:.2f}px (>{max_reproj_px:.1f})"
        return "ok"


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
    seed_pose: Optional[np.ndarray] = None,
) -> Detection:
    """Detect a ChArUco board and return its pose in the camera optical frame.

    Pose is `T_cam_optical_to_marker` (i.e., maps a marker-frame point into the
    camera optical frame). Translation is in meters.

    On any failure (too few corners, PnP divergence, unexpected exception) this
    returns a `Detection(success=False)` rather than raising, so the collector
    can skip the sample and continue.

    `seed_pose` is consumed only as a tie-breaker in the IPPE fallback path
    (4-5 corners). With ≥6 corners we use iterative PnP, which has a single
    local minimum and doesn't need a seed.
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
        return _fail_detection(n_corners=int(len(ch_ids)))

    if obj_pts is None or len(obj_pts) < 4:
        return _fail_detection(n_corners=int(len(ch_ids)))

    n_corners = int(len(ch_ids))
    if n_corners >= ITERATIVE_MIN_CORNERS:
        candidates, method = _solve_iterative(obj_pts, img_pts, camera_matrix, dist_coeffs)
    else:
        candidates, method = _solve_ippe(obj_pts, img_pts, camera_matrix, dist_coeffs,
                                         seed_pose=seed_pose)

    if not candidates:
        return _fail_detection(n_corners=n_corners)

    primary = min(candidates, key=lambda c: c.reproj_rms_px)
    return Detection(
        pose_optical=primary.pose_optical.copy(),
        n_corners=n_corners,
        reprojection_rms_px=primary.reproj_rms_px,
        success=True,
        candidates=candidates,
        method=method,
    )


# ---- PnP backends ------------------------------------------------------------

def _solve_iterative(obj_pts, img_pts, K, D) -> tuple[list[PoseCandidate], str]:
    """SOLVEPNP_IPPE seed → SOLVEPNP_ITERATIVE refine. Single unambiguous solution."""
    try:
        n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return [], ""
    if n_sol == 0:
        return [], ""

    # Pick the lower-reproj IPPE branch as the seed -- iterative PnP will then
    # converge to whichever local minimum the seed sits in. With a planar
    # target the iterative solver typically pulls both seeds to the same
    # solution; we run from the better seed for slightly faster convergence.
    seed_errs = []
    for rvec, tvec in zip(rvecs, tvecs):
        seed_errs.append(_reprojection_rms(obj_pts, img_pts, rvec, tvec, K, D))
    best_seed = int(np.argmin(seed_errs))

    rvec0 = rvecs[best_seed].copy()
    tvec0 = tvecs[best_seed].copy()
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, D, rvec0, tvec0)
    except cv2.error:
        return [], ""

    rms = _reprojection_rms(obj_pts, img_pts, rvec, tvec, K, D)
    if not np.isfinite(rms):
        return [], ""

    return [PoseCandidate(pose_optical=_to_se3(rvec, tvec), reproj_rms_px=rms)], "iterative"


def _solve_ippe(obj_pts, img_pts, K, D,
                *, seed_pose: Optional[np.ndarray]) -> tuple[list[PoseCandidate], str]:
    """Two-candidate IPPE for the under-determined 4-5 corner case.

    Returns BOTH refined candidates so the cell-level voter can majority-rule
    across N frames. We do NOT commit to a single candidate here unless the
    multi-criterion gate (refined-reproj ratio, front-facing dot product,
    seed distance) clearly favors one — in that case we return only the
    winner.
    """
    try:
        n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return [], ""
    if n_sol == 0:
        return [], ""

    refined = []
    for rvec, tvec in zip(rvecs, tvecs):
        try:
            rv, tv = cv2.solvePnPRefineLM(obj_pts, img_pts, K, D, rvec.copy(), tvec.copy())
        except cv2.error:
            continue
        rms = _reprojection_rms(obj_pts, img_pts, rv, tv, K, D)
        if np.isfinite(rms):
            refined.append(PoseCandidate(pose_optical=_to_se3(rv, tv), reproj_rms_px=rms))

    if not refined:
        return [], ""
    if len(refined) == 1:
        return refined, "ippe"

    winner_idx = _disambiguate_ippe(refined, seed_pose=seed_pose)
    if winner_idx is None:
        # Ambiguous — return both, let cluster_consensus handle it.
        return refined, "ippe"
    return [refined[winner_idx]], "ippe"


def _disambiguate_ippe(candidates: list[PoseCandidate],
                       *, seed_pose: Optional[np.ndarray]) -> Optional[int]:
    """Multi-criterion IPPE picker. Returns winner index or None if ambiguous.

    Decision tree:
      1. Refined reprojection RMS ratio > 1.5  → lower-rms wins.
      2. Front-facing dot product spread > 0.1 → more-negative-dot wins
         (more head-on view of the printed face).
      3. Seed pose supplied                    → smallest SO(3) distance to seed.
      4. Otherwise                             → None (caller keeps both).
    """
    a, b = candidates[0], candidates[1]
    rms_a, rms_b = a.reproj_rms_px, b.reproj_rms_px
    if max(rms_a, rms_b) > 1.5 * min(rms_a, rms_b):
        return 0 if rms_a < rms_b else 1

    f_a = _front_facing_dot(a.pose_optical)
    f_b = _front_facing_dot(b.pose_optical)
    if abs(f_a - f_b) > 0.1:
        # The candidate facing more head-on (more negative dot) is the
        # physically-correct one in nearly all observed cases.
        return 0 if f_a < f_b else 1

    if seed_pose is not None:
        seed_R = seed_pose[:3, :3]
        d_a = float(np.linalg.norm(utils.so3_log(a.pose_optical[:3, :3] @ seed_R.T)))
        d_b = float(np.linalg.norm(utils.so3_log(b.pose_optical[:3, :3] @ seed_R.T)))
        return 0 if d_a < d_b else 1

    return None


def _front_facing_dot(T: np.ndarray) -> float:
    """Dot product of the marker outward normal with -tvec/|tvec|.

    Positive = marker faces camera (correct). Negative = marker faces away.
    Used as a tie-breaker for IPPE's two-fold ambiguity: at glancing angles
    both candidates can be "front-facing" (positive dot), but the physically
    correct one usually has the larger dot product."""
    R = T[:3, :3]
    t = T[:3, 3]
    n = R[:, 2]                              # marker's local +Z in cam frame
    rng = float(np.linalg.norm(t))
    if rng < 1e-6:
        return 0.0
    return float(np.dot(n, -t / rng))


def _reprojection_rms(obj_pts, img_pts, rvec, tvec, K, D) -> float:
    try:
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    except cv2.error:
        return float("inf")
    proj = proj.reshape(-1, 2)
    img = img_pts.reshape(-1, 2)
    res = proj - img
    return float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))


def _to_se3(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.flatten()
    return T


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _fail_detection(n_corners: int = 0) -> Detection:
    return Detection(
        pose_optical=np.eye(4),
        n_corners=int(n_corners),
        reprojection_rms_px=float("inf"),
        success=False,
        candidates=[],
        method="",
    )


# ---- cell-level consensus ---------------------------------------------------
#
# Collector captures N=10 frames per cell. We greedy-cluster every frame's
# rotation candidate(s) in SO(3), pick the largest cluster, MAD-reject
# translation outliers within it, and SE(3)-mean what remains. A cell that
# can't reach a quorum fraction of frames in one cluster is dropped — better
# to skip a noisy cell than to commit to a smeared average.

def cluster_consensus(
    detections: list[Detection],
    *,
    cluster_rot_tol_rad: float = 0.087,    # ~5 deg
    min_cluster_frac: float = 0.6,
    mad_k: float = 3.0,
) -> Optional[Detection]:
    """Pick the dominant pose cluster across multiple frames; SE(3)-mean it.

    Robust to mixed-IPPE-branch cells: if 6/10 frames pick the right branch
    and 4 pick the wrong one, the right cluster wins. Returns None when the
    cell is too noisy to commit (no cluster reaches quorum).

    Each input Detection may carry 1 or 2 PoseCandidates (iterative vs IPPE
    fallback). All candidates feed the cluster vote; one frame contributes at
    most one vote to any single cluster.
    """
    valid = [d for d in detections if d.valid()]
    n_frames = len(valid)
    if n_frames < 3:
        return None

    # Build a flat list of (frame_idx, candidate) pairs.
    flat: list[tuple[int, PoseCandidate]] = []
    for fi, d in enumerate(valid):
        for c in d.candidates:
            flat.append((fi, c))
    if not flat:
        return None

    # Greedy cluster by rotation distance. Process candidates in ascending
    # reproj order so cleaner detections seed clusters.
    flat.sort(key=lambda x: x[1].reproj_rms_px)
    clusters: list[dict] = []   # each: {"frames": set[int], "members": list[(fi, cand)]}
    for fi, cand in flat:
        rv = utils.so3_log(cand.pose_optical[:3, :3])
        placed = False
        for cl in clusters:
            seed_rv = utils.so3_log(cl["members"][0][1].pose_optical[:3, :3])
            if np.linalg.norm(rv - seed_rv) < cluster_rot_tol_rad:
                if fi in cl["frames"]:
                    # Same frame already represented in this cluster (its
                    # twin IPPE candidate). Skip — one vote per frame per
                    # cluster.
                    placed = True
                    break
                cl["frames"].add(fi)
                cl["members"].append((fi, cand))
                placed = True
                break
        if not placed:
            clusters.append({"frames": {fi}, "members": [(fi, cand)]})

    # Largest cluster by frame count (each frame contributes at most one
    # member per cluster after the dedup above, so cluster size = frame count).
    clusters.sort(key=lambda cl: len(cl["frames"]), reverse=True)
    dom = clusters[0]
    if len(dom["frames"]) < max(3, int(np.ceil(min_cluster_frac * n_frames))):
        return None

    # Translation MAD-reject within the dominant cluster.
    members = dom["members"]
    Ts = np.stack([m[1].pose_optical for m in members])
    translations = Ts[:, :3, 3]
    median = np.median(translations, axis=0)
    abs_dev = np.abs(translations - median)
    mad = np.median(abs_dev, axis=0) + 1e-9
    keep = np.all(abs_dev / mad < mad_k, axis=1)
    kept = [m for m, k in zip(members, keep) if k]
    if len(kept) < 3:
        return None

    T_mean = _average_se3([m[1].pose_optical for m in kept])
    rms_mean = float(np.mean([m[1].reproj_rms_px for m in kept]))
    # corners reported = mean of the source detections' corner counts (each
    # member's frame index resolves back into `valid`).
    frame_idxs = sorted({m[0] for m in kept})
    corners_mean = float(np.mean([valid[i].n_corners for i in frame_idxs]))
    return Detection(
        pose_optical=T_mean,
        n_corners=int(corners_mean),
        reprojection_rms_px=rms_mean,
        success=True,
        candidates=[PoseCandidate(pose_optical=T_mean, reproj_rms_px=rms_mean)],
        method="consensus",
    )


# Backward-compat alias. Existing callers used `robust_average`; we keep the
# name but route to cluster_consensus, which subsumes its behavior and adds
# multi-candidate clustering. Old `mad_k` / `rot_outlier_rad` kwargs are
# accepted but ignored — the new voter doesn't need them.
def robust_average(detections: list[Detection], *,
                   mad_k: float = 3.0,
                   rot_outlier_rad: float = 0.175) -> Optional[Detection]:
    return cluster_consensus(detections, mad_k=mad_k)
