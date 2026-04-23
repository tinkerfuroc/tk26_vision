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
    invert_transform,
    pose_error_scalars,
    sample_to_matrices,
    se3_log_residual,
)


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


def _chain_residuals(x, samples, template, fit_pan_offset, t_base_cam_gt):
    params = unpack_chain(x, template, fit_pan_offset=fit_pan_offset)
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
    loss: str = "soft_l1",
    verbose: int = 0,
) -> tuple[PanTiltParams, OptReport]:
    """Phase-2 fit: solve pan-tilt chain params given a known T_ee_marker.

    Parameters
    ----------
    samples
        List of sample dicts (see module docstring).
    t_ee_marker
        4x4 transform from Phase-1 hand-eye calibration.
    initial
        Optional starting params; defaults to URDF guesses + theta_t_offset = -pi/4.
    fit_pan_offset
        Whether to include theta_p_offset in the parameter vector.
    loss
        scipy.optimize.least_squares `loss` kwarg; soft_l1 is robust to outliers.
    """
    template = initial or PanTiltParams()
    x0 = pack_chain(template, fit_pan_offset=fit_pan_offset)
    t_base_cam_gt = _predict_chain_gt(samples, t_ee_marker)

    result = least_squares(
        _chain_residuals,
        x0,
        args=(samples, template, fit_pan_offset, t_base_cam_gt),
        method="trf",
        loss=loss,
        verbose=verbose,
    )

    params = unpack_chain(result.x, template, fit_pan_offset=fit_pan_offset)
    report = _build_report(result, samples, params, t_base_cam_gt=t_base_cam_gt)
    return params, report


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
) -> tuple[PanTiltParams, OptReport]:
    """Phase-3 polish: joint fit over all parameters including T_ee_marker."""
    x0 = pack_joint(
        initial, fit_tb_rotation=fit_tb_rotation, fit_pan_offset=fit_pan_offset
    )

    result = least_squares(
        _joint_residuals,
        x0,
        args=(samples, initial, fit_tb_rotation, fit_pan_offset),
        method="trf",
        loss=loss,
        verbose=verbose,
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

    # Build pairs (i, j) with i < j. For N=15 that's 105 pairs — plenty.
    N = len(samples)
    A_list, B_list = [], []
    for i in range(N):
        _, _, T_bei, T_cmi = sample_to_matrices(samples[i])
        T_ebi = invert_transform(T_bei)
        T_mci = invert_transform(T_cmi)
        for j in range(i + 1, N):
            _, _, T_bej, T_cmj = sample_to_matrices(samples[j])
            A = T_bej @ T_ebi           # relative EE motion in base
            B = T_cmj @ T_mci           # relative marker motion in cam
            # Prune near-zero-rotation pairs; they add no rotational constraint.
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

    # Recover T_ee_marker per sample from the closure and average on SE(3).
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
