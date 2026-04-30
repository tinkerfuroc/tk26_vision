"""Non-linear least-squares fits for the pan-tilt calibration problem.

Exposed entry points:
    fit_chain(samples, *, initial=None, fit_pan_offset=False, loss='soft_l1')
        -> PanTiltParams, optimization report

    fit_joint(samples, *, initial, fit_tb_rotation=False, fit_pan_offset=False,
              loss='soft_l1')
        -> PanTiltParams, optimization report

    solve_handeye(samples)
        -> (T_ee_marker, T_base_cam_ref, per_pose_residuals)

Each `sample` is a dict with keys:
    theta_pan_rad, theta_tilt_rad, t_base_ee, t_cam_marker_body
See `utils.sample_to_matrices`.

The chain fit assumes `t_ee_marker` is known (from `solve_handeye`) and computes
per-sample ground truth `T_base_cam_i = T_base_ee @ T_ee_marker @ inv(T_cam_marker_body)`
which is then compared to the FK prediction.

The joint fit treats `t_ee_marker` as a free parameter and uses the full closure
equation directly, comparing the marker pose observed in camera to the marker
pose predicted via `FK^-1 @ T_base_ee @ T_ee_marker`.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .pan_tilt_model import (
    PanTiltParams,
    forward_kinematics,
    pack_chain,
    pack_joint,
    unpack_chain,
    unpack_joint,
)
from .utils import (
    body_yaw_from_rotvec,
    invert_transform,
    pose_error_scalars,
    sample_to_matrices,
    se3_log_residual,
)


_log = logging.getLogger(__name__)


# ---- forward-camera invariant ----------------------------------------------
#
# For a forward-facing head camera (the configuration we ship), the body-frame
# yaw of `t_b_rotvec` (camera_mount_joint rpy yaw) must stay within ±π/2 of
# zero. A value near ±π is the smoking-gun signature of a flipped optical→body
# conversion somewhere upstream and was the root cause of the 2026-04-30
# backward-camera incident. Anyone calibrating a genuinely backward-mounted
# camera must opt in via `allow_flipped_camera=True`.

_FORWARD_YAW_LIMIT_RAD = math.pi / 2.0
_FLIPPED_YAW_LIMIT_RAD = math.pi


def _t_b_rotvec_z_index_chain(fit_pan_offset: bool) -> int:
    """Index of the t_b_rotvec Z component in `pack_chain` output.

    pack_chain layout (only when fit_tb_rotation=True):
        t_a (3) + t_b_trans (3) + theta_t (1) [+ theta_p (1)] + t_b_rotvec (3)

    The Z component is the last entry of t_b_rotvec.
    """
    base = 3 + 3 + 1 + (1 if fit_pan_offset else 0)
    return base + 2


def _t_b_rotvec_z_index_joint(fit_pan_offset: bool) -> int:
    """Index of the t_b_rotvec Z component in `pack_joint` output.

    pack_joint layout (only when fit_tb_rotation=True):
        t_a (3) + t_b_trans (3) + t_ee_marker_rot (3) + t_ee_marker_trans (3)
        + theta_t (1) [+ theta_p (1)] + t_b_rotvec (3)
    """
    base = 3 + 3 + 3 + 3 + 1 + (1 if fit_pan_offset else 0)
    return base + 2


def _build_bounds(
    n_params: int,
    yaw_index: Optional[int],
    *,
    allow_flipped_camera: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds for least_squares: open on every parameter except t_b_rotvec[Z].

    `yaw_index=None` means t_b_rotvec is locked at the warm-start value and
    not in the parameter vector — return fully-open bounds.
    """
    lo = np.full(n_params, -np.inf)
    hi = np.full(n_params, +np.inf)
    if yaw_index is not None:
        limit = (
            _FLIPPED_YAW_LIMIT_RAD
            if allow_flipped_camera
            else _FORWARD_YAW_LIMIT_RAD
        )
        lo[yaw_index] = -limit
        hi[yaw_index] = +limit
    return lo, hi


def _clip_initial_to_bounds(x0: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Project the initial guess into the bound box.

    scipy.least_squares raises if x0 is strictly outside `bounds`, so when the
    warm-start lands at e.g. yaw=+π (the bug we're guarding against) we must
    clip first or the solver call dies. Clipping pushes the seed onto the
    boundary, which is the right behavior — the bound is the operator's
    declaration that the truth lives inside.
    """
    return np.clip(x0, lo, hi)


@dataclass
class OptReport:
    success: bool
    message: str
    cost: float
    n_samples: int
    trans_rmse_m: float
    rot_rmse_rad: float
    trans_rmse_per_sample: np.ndarray = field(default_factory=lambda: np.empty(0))
    rot_rmse_per_sample: np.ndarray = field(default_factory=lambda: np.empty(0))
    per_sample_residuals: np.ndarray = field(default_factory=lambda: np.empty(0))

    def summary(self) -> str:
        return (
            f"{'OK ' if self.success else 'FAIL'}  "
            f"n={self.n_samples:<4d}  "
            f"cost={self.cost:.4g}  "
            f"trans_rmse={self.trans_rmse_m * 1000:.2f}mm  "
            f"rot_rmse={np.degrees(self.rot_rmse_rad):.3f}deg  "
            f"[{self.message}]"
        )


# ---- chain fit --------------------------------------------------------------

def _predict_chain_gt(samples, t_ee_marker: np.ndarray):
    """Per-sample ground-truth T_base_cam given a known T_ee_marker."""
    out = []
    for s in samples:
        _, _, T_be, T_cm = sample_to_matrices(s)
        T_base_cam = T_be @ t_ee_marker @ invert_transform(T_cm)
        out.append(T_base_cam)
    return out


def _chain_residuals(x, samples, template, fit_pan_offset, fit_tb_rotation, t_base_cam_gt):
    params = unpack_chain(
        x, template,
        fit_pan_offset=fit_pan_offset,
        fit_tb_rotation=fit_tb_rotation,
    )
    res = []
    for s, T_gt in zip(samples, t_base_cam_gt):
        theta_p, theta_t, _, _ = sample_to_matrices(s)
        T_pred = forward_kinematics(theta_p, theta_t, params)
        res.append(se3_log_residual(T_pred, T_gt))
    return np.concatenate(res)


def fit_chain(
    samples: list,
    *,
    t_ee_marker: np.ndarray,
    initial: Optional[PanTiltParams] = None,
    fit_pan_offset: bool = False,
    fit_tb_rotation: bool = False,
    loss: str = "soft_l1",
    verbose: int = 0,
    allow_flipped_camera: bool = False,
) -> tuple[PanTiltParams, OptReport]:
    """Phase-2 fit: solve pan-tilt chain params given a known T_ee_marker.

    Parameters
    ----------
    samples
        List of sample dicts (see module docstring).
    t_ee_marker
        4x4 transform from Phase-1 hand-eye calibration.
    initial
        Starting params. **You should seed T_B via `warm_start_t_b_rotation`
        from the Phase-1 `t_base_cam_ref` before calling this** — otherwise T_B
        rotation stays at identity, which is wrong for any robot whose camera
        is not axis-aligned with the tilt arm (e.g. the ~90 deg perpendicular
        mount on this setup).
    fit_pan_offset
        Whether to include theta_p_offset in the parameter vector.
    fit_tb_rotation
        Default **False**. T_B rotation about the tilt axis (+Y) is
        mathematically degenerate with `theta_t_offset` — any Y-rotation of
        T_B gets absorbed into the scalar offset. Freezing T_B rotation at
        the warm-start value avoids that degeneracy. Unlock only during the
        polish phase, where Phase-1 data jointly anchors T_ee_marker and
        breaks the degeneracy.
    loss
        scipy.optimize.least_squares `loss` kwarg; soft_l1 is robust to outliers.
    allow_flipped_camera
        Default **False** (forward-facing camera). When True the body-frame
        yaw bound on T_B widens from ±π/2 to ±π — set this only when the
        camera is genuinely mounted backward on the head. See module-level
        forward-camera invariant.
    """
    template = initial or PanTiltParams()
    x0 = pack_chain(
        template,
        fit_pan_offset=fit_pan_offset,
        fit_tb_rotation=fit_tb_rotation,
    )

    yaw_idx = (
        _t_b_rotvec_z_index_chain(fit_pan_offset) if fit_tb_rotation else None
    )
    lo, hi = _build_bounds(
        x0.size, yaw_idx, allow_flipped_camera=allow_flipped_camera,
    )
    x0 = _clip_initial_to_bounds(x0, lo, hi)

    t_base_cam_gt = _predict_chain_gt(samples, t_ee_marker)

    result = least_squares(
        _chain_residuals,
        x0,
        args=(samples, template, fit_pan_offset, fit_tb_rotation, t_base_cam_gt),
        method="trf",
        loss=loss,
        verbose=verbose,
        bounds=(lo, hi),
    )

    params = unpack_chain(
        result.x, template,
        fit_pan_offset=fit_pan_offset,
        fit_tb_rotation=fit_tb_rotation,
    )
    report = _build_report(result, samples, params, t_base_cam_gt=t_base_cam_gt)
    return params, report


def warm_start_t_b_rotation(
    template: PanTiltParams,
    t_base_cam_ref: np.ndarray,
    park_pan_rad: float = 0.0,
    park_tilt_rad: float = 0.0,
) -> PanTiltParams:
    """Back-solve T_B from the Phase-1 reference pose `Z_park = T_base_cam(park)`.

    The reference pose Z_park was captured with the pan-tilt held at
    (park_pan_rad, park_tilt_rad) FIRMWARE radians during Phase 1 data
    collection -- NOT necessarily firmware zero. The FK identity we invert is

        Z_park = translate(t_a)
                 @ R_z(-(theta_p_off + park_pan))
                 @ translate(L_pan z)
                 @ R_y(theta_t_off + park_tilt)
                 @ T_B

    Defaults (0, 0) preserve legacy callers that parked at servo-zero.
    This pulls a (potentially large, e.g. 90 deg) T_B rotation into the init
    so the chain optimizer starts inside the right convergence basin.
    """
    T_a = np.eye(4)
    T_a[:3, 3] = template.t_a
    R_pan0 = np.eye(4)
    R_pan0[:3, :3] = Rotation.from_euler('z', -(template.theta_p_offset + park_pan_rad)).as_matrix()
    T_lp = np.eye(4)
    T_lp[:3, 3] = [0, 0, template.l_pan]
    R_tilt0 = np.eye(4)
    R_tilt0[:3, :3] = Rotation.from_euler('y', template.theta_t_offset + park_tilt_rad).as_matrix()

    pre = T_a @ R_pan0 @ T_lp @ R_tilt0
    T_b = invert_transform(pre) @ t_base_cam_ref

    rotvec = Rotation.from_matrix(T_b[:3, :3]).as_rotvec()
    yaw = body_yaw_from_rotvec(rotvec)
    # Forward-facing cameras land near 0; the original 2026-04-30
    # backward-camera incident showed up here as ~+π. Suppress the warning
    # only when the caller explicitly seeded a flipped-basin exploration
    # (theta_p_offset ≈ ±π) — cmd_chain's two-basin search hits warm_start
    # twice (basin0 + basinπ) and the basinπ branch always produces a
    # flipped rotvec by construction, which is not a bug. The basin0 branch
    # still triggers the warning if the bug recurs.
    seeded_pi_basin = abs(abs(template.theta_p_offset) - math.pi) < 0.1
    if abs(yaw) > math.pi / 4.0 and not seeded_pi_basin:
        msg = (
            f"warm-start t_b_rotvec yaw={yaw:.3f} rad "
            f"({math.degrees(yaw):.1f}°) is far from 0 — this usually means "
            f"the optical→body conversion was missed in the Phase-1 "
            f"hand-eye result. Calibration may converge to a flipped basin; "
            f"re-check Phase-1 conventions before trusting the URDF patch."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        _log.warning(msg)

    out = PanTiltParams(
        t_a=template.t_a.copy(),
        t_b_trans=T_b[:3, 3].copy(),
        t_b_rotvec=rotvec,
        t_ee_marker_rotvec=template.t_ee_marker_rotvec.copy(),
        t_ee_marker_trans=template.t_ee_marker_trans.copy(),
        theta_t_offset=template.theta_t_offset,
        theta_p_offset=template.theta_p_offset,
        l_pan=template.l_pan,
    )
    return out


# ---- joint (polish) fit -----------------------------------------------------

def _joint_residuals(x, samples, template, fit_tb_rotation, fit_pan_offset):
    params = unpack_joint(
        x,
        template,
        fit_tb_rotation=fit_tb_rotation,
        fit_pan_offset=fit_pan_offset,
    )
    T_ee_m = params.t_ee_marker()
    res = []
    for s in samples:
        theta_p, theta_t, T_be, T_cm = sample_to_matrices(s)
        T_base_cam = forward_kinematics(theta_p, theta_t, params)
        # Predicted marker pose in base_link = T_be @ T_ee_m.
        # Measured marker pose in base_link = T_base_cam @ T_cm.
        # Their residual lives in base_link; use SE(3) log of the difference.
        T_pred_marker = T_be @ T_ee_m
        T_meas_marker = T_base_cam @ T_cm
        res.append(se3_log_residual(T_pred_marker, T_meas_marker))
    return np.concatenate(res)


def fit_joint(
    samples: list,
    *,
    initial: PanTiltParams,
    fit_tb_rotation: bool = False,
    fit_pan_offset: bool = False,
    loss: str = "soft_l1",
    verbose: int = 0,
    allow_flipped_camera: bool = False,
) -> tuple[PanTiltParams, OptReport]:
    """Phase-3 polish: joint fit over all parameters including T_ee_marker.

    `allow_flipped_camera` widens the body-frame yaw bound on T_B from ±π/2
    to ±π — see module-level forward-camera invariant.
    """
    x0 = pack_joint(
        initial, fit_tb_rotation=fit_tb_rotation, fit_pan_offset=fit_pan_offset
    )

    yaw_idx = (
        _t_b_rotvec_z_index_joint(fit_pan_offset) if fit_tb_rotation else None
    )
    lo, hi = _build_bounds(
        x0.size, yaw_idx, allow_flipped_camera=allow_flipped_camera,
    )
    x0 = _clip_initial_to_bounds(x0, lo, hi)

    result = least_squares(
        _joint_residuals,
        x0,
        args=(samples, initial, fit_tb_rotation, fit_pan_offset),
        method="trf",
        loss=loss,
        verbose=verbose,
        bounds=(lo, hi),
    )

    params = unpack_joint(
        result.x,
        initial,
        fit_tb_rotation=fit_tb_rotation,
        fit_pan_offset=fit_pan_offset,
    )
    report = _build_joint_report(result, samples, params)
    return params, report


# ---- hand-eye (Phase 1) -----------------------------------------------------
#
# Geometry: camera is fixed in base (pan-tilt frozen at servo zero); ChArUco is
# rigidly attached to the xArm EE. We want T_ee_marker (fixed, unknown) and
# T_base_cam at the reference pose (fixed, unknown).
#
# For any sample i:
#     T_base_cam @ T_cam_marker_i = T_base_ee_i @ T_ee_marker       (closure)
#
# Take pairs (i, j). Let
#     A_ij = T_base_ee_j @ inv(T_base_ee_i)         (relative EE motion in base)
#     B_ij = T_cam_marker_j @ inv(T_cam_marker_i)   (relative marker motion in cam)
#
# From the closure:
#     T_base_cam @ B_ij = A_ij @ T_base_cam        =>  A X = X B  with X = T_base_cam
#
# We solve the AX=XB system via the Park-Martin linear method: rotation first
# (solve R A R^T = R B in axis-angle form via a 3x3 linear system over pairs),
# then translation (closed-form linear system). Then T_ee_marker is recovered
# per sample and averaged on SE(3).

def solve_handeye(samples: list):
    """Eye-to-hand solve for (T_base_cam_ref, T_ee_marker) via AX=XB (Park-Martin).

    Returns
    -------
    t_ee_marker : (4,4) ndarray
    t_base_cam_ref : (4,4) ndarray
    per_pose_residual : list of (trans_err_m, rot_err_rad) for each input sample
    """
    if len(samples) < 3:
        raise ValueError(f"solve_handeye needs >=3 samples, got {len(samples)}")

    return _park_martin_solve(samples)


def solve_handeye_with_consensus(
    samples: list,
    *,
    pre_filter_rot_deg: float = 5.0,
    min_samples: int = 8,
    ransac_iters: int = 50,
    ransac_subset_size: int = 5,
    ransac_seed: Optional[int] = 0,
) -> tuple[np.ndarray, np.ndarray, list, list[int]]:
    """Two-stage hand-eye solve: RANSAC pre-pass to find inliers, then Park-Martin refinement.

    Stage 1 (RANSAC)
        - For `ransac_iters` trials, pick a random subset of `ransac_subset_size`
          samples, run Park-Martin on the subset, and count inliers across
          ALL samples (a sample is an inlier if its implied T_ee_marker
          rotation matches the trial's T_ee_marker within `pre_filter_rot_deg`).
        - The trial with the most inliers wins. Robust to up to ~50%
          IPPE-flip contamination because, with reasonable subset size and
          enough trials, at least one trial samples an all-clean subset.

    Stage 2 (final solve)
        - Park-Martin on the inliers. Returns final
          (t_em, t_bc, per_pose, rejected_indices).

    A handful of samples surviving stage 1 may still get the iterative
    MAD-sigma refinement applied by `cmd_handeye`; this function only does
    the absolute-threshold RANSAC pre-filter.
    """
    if len(samples) < max(3, min_samples // 2):
        raise ValueError(f"need >=3 samples, got {len(samples)}")

    rot_thresh = np.deg2rad(pre_filter_rot_deg)
    N = len(samples)

    # Pre-extract matrices (cheap, but we do this many times in RANSAC).
    T_be_list, T_cm_list = [], []
    for s in samples:
        _, _, T_be, T_cm = sample_to_matrices(s)
        T_be_list.append(T_be)
        T_cm_list.append(T_cm)

    rng = np.random.default_rng(ransac_seed)
    subset_size = max(3, min(ransac_subset_size, N))
    best_inliers: list[int] = []
    best_t_em = best_t_bc = None

    for _ in range(ransac_iters):
        idx = rng.choice(N, size=subset_size, replace=False)
        subset = [samples[i] for i in idx]
        try:
            t_em_trial, t_bc_trial, _ = _park_martin_solve(subset)
        except ValueError:
            # Subset didn't have enough rotational diversity; skip.
            continue

        # Score: inliers across ALL samples for this trial.
        inliers = []
        for k in range(N):
            T_em_k = invert_transform(T_be_list[k]) @ t_bc_trial @ T_cm_list[k]
            _, r = pose_error_scalars(T_em_k, t_em_trial)
            if r < rot_thresh:
                inliers.append(k)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_t_em = t_em_trial
            best_t_bc = t_bc_trial

    rejected = sorted(set(range(N)) - set(best_inliers))
    if best_t_em is None or len(best_inliers) < 3:
        # RANSAC didn't find any consistent subset -- fall back to plain
        # Park-Martin on the full set, no rejections. The downstream
        # MAD-sigma loop will at least try to clean up.
        t_em, t_bc, per_pose = _park_martin_solve(samples)
        return t_em, t_bc, per_pose, []

    # Final refinement on the inlier set.
    kept = [samples[i] for i in best_inliers]
    if len(kept) >= 3:
        try:
            t_em, t_bc, per_pose = _park_martin_solve(kept)
            return t_em, t_bc, per_pose, rejected
        except ValueError:
            pass
    # Fall back to the RANSAC trial's solution.
    per_pose = []
    for s in kept:
        _, _, T_be, T_cm = sample_to_matrices(s)
        T_em_k = invert_transform(T_be) @ best_t_bc @ T_cm
        per_pose.append(pose_error_scalars(best_t_em, T_em_k))
    return best_t_em, best_t_bc, per_pose, rejected


def _park_martin_solve(samples: list):
    """Inner Park-Martin solve. Pure linear AX=XB, no outlier handling."""
    N = len(samples)
    A_list, B_list = [], []
    for i in range(N):
        _, _, T_bei, T_cmi = sample_to_matrices(samples[i])
        T_ebi = invert_transform(T_bei)
        T_mci = invert_transform(T_cmi)
        for j in range(i + 1, N):
            _, _, T_bej, T_cmj = sample_to_matrices(samples[j])
            A = T_bej @ T_ebi
            B = T_cmj @ T_mci
            if float(np.linalg.norm(Rotation.from_matrix(A[:3, :3]).as_rotvec())) < 0.08:
                continue
            A_list.append(A)
            B_list.append(B)

    if len(A_list) < 3:
        raise ValueError("Not enough rotationally-distinct sample pairs for hand-eye solve.")

    R_x = _park_martin_rotation(A_list, B_list)
    t_x = _park_martin_translation(A_list, B_list, R_x)

    t_base_cam_ref = np.eye(4)
    t_base_cam_ref[:3, :3] = R_x
    t_base_cam_ref[:3, 3] = t_x

    candidates = []
    for s in samples:
        _, _, T_be, T_cm = sample_to_matrices(s)
        T_ee_marker_i = invert_transform(T_be) @ t_base_cam_ref @ T_cm
        candidates.append(T_ee_marker_i)
    t_ee_marker = _average_se3(candidates)

    per_pose = [pose_error_scalars(t_ee_marker, T) for T in candidates]
    return t_ee_marker, t_base_cam_ref, per_pose


def _park_martin_rotation(A_list, B_list) -> np.ndarray:
    """Solve the rotational part of AX=XB via orthogonal Procrustes on axis-angle vectors.

    If `R_A X = X R_B` then for axis-angle vectors `alpha = log(R_A)`, `beta = log(R_B)`
    the relation is `alpha = R_X beta`. We minimise ``sum || alpha - R_X beta ||^2``
    over `R_X in SO(3)`.
    """
    C = np.zeros((3, 3))
    for A, B in zip(A_list, B_list):
        alpha = Rotation.from_matrix(A[:3, :3]).as_rotvec()
        beta = Rotation.from_matrix(B[:3, :3]).as_rotvec()
        C += np.outer(alpha, beta)

    U, _, Vt = np.linalg.svd(C)
    D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])  # reflect if needed
    return U @ D @ Vt


def _park_martin_translation(A_list, B_list, R_X) -> np.ndarray:
    """Solve the translational part of AX=XB given R_X.

    From AX = XB (4x4):
        R_A t_X + t_A = R_X t_B + t_X
    =>  (I - R_A) t_X = t_A - R_X t_B
    """
    L = np.zeros((3 * len(A_list), 3))
    r = np.zeros(3 * len(A_list))
    for k, (A, B) in enumerate(zip(A_list, B_list)):
        L[3 * k: 3 * k + 3, :] = np.eye(3) - A[:3, :3]
        r[3 * k: 3 * k + 3] = A[:3, 3] - R_X @ B[:3, 3]
    t_X, *_ = np.linalg.lstsq(L, r, rcond=None)
    return t_X


def _average_se3(Ts: list) -> np.ndarray:
    """Chordal mean of a list of SE(3) transforms."""
    Ts = np.asarray(Ts)
    t_mean = np.mean(Ts[:, :3, 3], axis=0)
    R_sum = np.sum(Ts[:, :3, :3], axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    R_mean = U @ D @ Vt
    M = np.eye(4)
    M[:3, :3] = R_mean
    M[:3, 3] = t_mean
    return M


# ---- reporting --------------------------------------------------------------

def _build_report(result, samples, params, t_base_cam_gt):
    residuals = result.fun.reshape(-1, 6)
    trans_errs, rot_errs = [], []
    for s, T_gt in zip(samples, t_base_cam_gt):
        theta_p, theta_t, _, _ = sample_to_matrices(s)
        T_pred = forward_kinematics(theta_p, theta_t, params)
        te, re = pose_error_scalars(T_pred, T_gt)
        trans_errs.append(te)
        rot_errs.append(re)
    trans_errs = np.asarray(trans_errs)
    rot_errs = np.asarray(rot_errs)
    return OptReport(
        success=result.success,
        message=result.message,
        cost=float(result.cost),
        n_samples=len(samples),
        trans_rmse_m=float(np.sqrt(np.mean(trans_errs ** 2))),
        rot_rmse_rad=float(np.sqrt(np.mean(rot_errs ** 2))),
        trans_rmse_per_sample=trans_errs,
        rot_rmse_per_sample=rot_errs,
        per_sample_residuals=residuals,
    )


def _build_joint_report(result, samples, params):
    residuals = result.fun.reshape(-1, 6)
    trans_errs, rot_errs = [], []
    T_ee_m = params.t_ee_marker()
    for s in samples:
        theta_p, theta_t, T_be, T_cm = sample_to_matrices(s)
        T_pred_marker = T_be @ T_ee_m
        T_meas_marker = forward_kinematics(theta_p, theta_t, params) @ T_cm
        te, re = pose_error_scalars(T_pred_marker, T_meas_marker)
        trans_errs.append(te)
        rot_errs.append(re)
    trans_errs = np.asarray(trans_errs)
    rot_errs = np.asarray(rot_errs)
    return OptReport(
        success=result.success,
        message=result.message,
        cost=float(result.cost),
        n_samples=len(samples),
        trans_rmse_m=float(np.sqrt(np.mean(trans_errs ** 2))),
        rot_rmse_rad=float(np.sqrt(np.mean(rot_errs ** 2))),
        trans_rmse_per_sample=trans_errs,
        rot_rmse_per_sample=rot_errs,
        per_sample_residuals=residuals,
    )
