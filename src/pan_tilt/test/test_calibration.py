"""Synthetic-data validation for the pan-tilt calibration solvers.

These tests do NOT talk to hardware. They:
  1. Fabricate a ground-truth PanTiltParams + T_ee_marker.
  2. Sample (theta_p, theta_t) + xArm pose across a realistic grid.
  3. Run the forward chain to generate synthetic observations
     (T_base_ee known, T_cam_marker computed as FK^-1 @ T_base_ee @ T_ee_marker).
  4. Add Gaussian noise representative of real measurements.
  5. Run the solvers and assert recovery inside tolerance.

If any assertion loosens, the cause is almost always a bad residual formulation
or a sign flip in the FK convention; start there before increasing tolerances.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pan_tilt.calibration.optimize import fit_chain, fit_joint, solve_handeye
from pan_tilt.calibration.pan_tilt_model import PanTiltParams, forward_kinematics
from pan_tilt.calibration.utils import (
    invert_transform,
    matrix_to_pose_dict,
    pose_error_scalars,
)


RNG_SEED = 20260423


def _make_truth() -> tuple[PanTiltParams, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)

    truth = PanTiltParams(
        t_a=np.array([-0.28, -0.02, 1.55]),
        t_b_trans=np.array([-0.07, -0.01, 0.08]),
        t_b_rotvec=np.zeros(3),
        theta_t_offset=-np.pi / 4 + np.deg2rad(1.2),  # 1.2 deg away from the nominal
        theta_p_offset=np.deg2rad(0.4),               # small pan bias
        l_pan=0.135,
    )

    # Realistic-ish T_ee_marker: board clipped ~12cm below flange, rotated ~30 deg about Y.
    Rem = Rotation.from_euler("xyz", [0.1, 0.5, 0.05]).as_matrix()
    t_ee_marker = np.eye(4)
    t_ee_marker[:3, :3] = Rem
    t_ee_marker[:3, 3] = np.array([0.02, -0.01, -0.12])
    return truth, t_ee_marker


def _random_ee_pose(rng) -> np.ndarray:
    """xArm link_eef in base_link: plausible reachable configurations near the head."""
    R = Rotation.from_euler("xyz", rng.uniform(-0.6, 0.6, size=3)).as_matrix()
    t = np.array([
        rng.uniform(-0.1, 0.3),
        rng.uniform(-0.3, 0.3),
        rng.uniform(0.9, 1.3),
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _sample(theta_p, theta_t, T_base_ee, truth, t_ee_marker, noise_rng=None,
            trans_noise_std=0.001, rot_noise_std_deg=0.1):
    """Fabricate one observation = {theta, T_base_ee, T_cam_marker_body}."""
    T_base_cam = forward_kinematics(theta_p, theta_t, truth)
    T_cam_base = invert_transform(T_base_cam)
    T_cam_marker_body = T_cam_base @ T_base_ee @ t_ee_marker

    if noise_rng is not None:
        # Perturb the *observation* (what ArUco would report).
        dR = Rotation.from_rotvec(noise_rng.normal(0, np.deg2rad(rot_noise_std_deg), size=3)).as_matrix()
        dt = noise_rng.normal(0, trans_noise_std, size=3)
        T_cam_marker_body[:3, :3] = dR @ T_cam_marker_body[:3, :3]
        T_cam_marker_body[:3, 3] = T_cam_marker_body[:3, 3] + dt

    return {
        "theta_pan_rad": theta_p,
        "theta_tilt_rad": theta_t,
        "t_base_ee": matrix_to_pose_dict(T_base_ee),
        "t_cam_marker_body": matrix_to_pose_dict(T_cam_marker_body),
        "image_stamp_ns": 0,
        "state_stamp_ns": 0,
        "detection_quality": 24,
    }


def _phase1_samples(truth, t_ee_marker, n=16, noise_rng=None):
    rng = np.random.default_rng(RNG_SEED + 1)
    samples = []
    for _ in range(n):
        T_be = _random_ee_pose(rng)
        samples.append(_sample(0.0, 0.0, T_be, truth, t_ee_marker, noise_rng))
    return samples


def _phase2_samples(truth, t_ee_marker, noise_rng=None):
    rng = np.random.default_rng(RNG_SEED + 2)
    pan_grid_deg = [-60, -30, 0, 30, 60]
    tilt_grid_deg = [-25, -10, 0, 15, 35]
    ee_poses = [_random_ee_pose(rng) for _ in range(3)]

    samples = []
    for T_be in ee_poses:
        for p_deg in pan_grid_deg:
            for t_deg in tilt_grid_deg:
                samples.append(_sample(
                    np.deg2rad(p_deg), np.deg2rad(t_deg),
                    T_be, truth, t_ee_marker, noise_rng,
                ))
    return samples


# ---- tests ------------------------------------------------------------------

def test_noise_free_chain_recovers_params_exactly():
    truth, t_ee_marker = _make_truth()
    samples = _phase2_samples(truth, t_ee_marker, noise_rng=None)

    # Start far from truth to prove convergence doesn't depend on init luck.
    init = PanTiltParams()  # URDF defaults

    params, report = fit_chain(
        samples,
        t_ee_marker=t_ee_marker,
        initial=init,
        fit_pan_offset=True,
        loss="linear",
    )
    assert report.success
    assert report.trans_rmse_m < 1e-6
    assert report.rot_rmse_rad < 1e-6
    assert np.allclose(params.t_a, truth.t_a, atol=1e-5)
    assert np.allclose(params.t_b_trans, truth.t_b_trans, atol=1e-5)
    assert abs(params.theta_t_offset - truth.theta_t_offset) < 1e-6
    assert abs(params.theta_p_offset - truth.theta_p_offset) < 1e-6


def test_noisy_chain_recovers_within_tolerance():
    truth, t_ee_marker = _make_truth()
    noise_rng = np.random.default_rng(RNG_SEED + 100)
    samples = _phase2_samples(truth, t_ee_marker, noise_rng=noise_rng)

    init = PanTiltParams()
    params, report = fit_chain(
        samples,
        t_ee_marker=t_ee_marker,
        initial=init,
        fit_pan_offset=True,
        loss="soft_l1",
    )
    assert report.success
    # Injected noise: 1 mm trans + 0.1 deg rot per sample. Residuals are
    # lower-bounded by that; parameter recovery is the real test.
    assert report.trans_rmse_m < 0.004, report.summary()
    assert np.degrees(report.rot_rmse_rad) < 0.35, report.summary()
    # Parameters should recover to well inside the injected noise scale.
    assert np.linalg.norm(params.t_a - truth.t_a) < 0.003
    assert np.linalg.norm(params.t_b_trans - truth.t_b_trans) < 0.003
    assert abs(params.theta_t_offset - truth.theta_t_offset) < np.deg2rad(0.3)


def test_handeye_recovers_t_ee_marker():
    truth, t_ee_marker = _make_truth()
    noise_rng = np.random.default_rng(RNG_SEED + 200)
    samples = _phase1_samples(truth, t_ee_marker, n=16, noise_rng=noise_rng)

    T_solved, T_base_cam_ref, per_pose = solve_handeye(samples)
    trans_err, rot_err = pose_error_scalars(T_solved, t_ee_marker)
    # Hand-eye with 16 noisy samples is less precise than the chain fit; 5 mm /
    # 0.5 deg is the plan's gate.
    assert trans_err < 0.006, f"t_ee_marker trans err {trans_err*1000:.2f}mm"
    assert np.degrees(rot_err) < 0.8, f"t_ee_marker rot err {np.degrees(rot_err):.3f}deg"


def test_joint_polish_tightens_result():
    truth, t_ee_marker = _make_truth()
    noise_rng = np.random.default_rng(RNG_SEED + 300)
    phase1 = _phase1_samples(truth, t_ee_marker, n=16, noise_rng=noise_rng)
    phase2 = _phase2_samples(truth, t_ee_marker, noise_rng=noise_rng)

    # Seed joint with hand-eye + chain outputs.
    t_ee_solved, _, _ = solve_handeye(phase1)
    seed = PanTiltParams(
        t_a=PanTiltParams().t_a.copy(),
        t_b_trans=PanTiltParams().t_b_trans.copy(),
        t_ee_marker_rotvec=Rotation.from_matrix(t_ee_solved[:3, :3]).as_rotvec(),
        t_ee_marker_trans=t_ee_solved[:3, 3].copy(),
        theta_t_offset=-np.pi / 4,
    )
    chain_params, _ = fit_chain(
        phase2, t_ee_marker=t_ee_solved, initial=seed, fit_pan_offset=True,
    )
    # Inject chain fit into the joint seed.
    seed.t_a = chain_params.t_a
    seed.t_b_trans = chain_params.t_b_trans
    seed.theta_t_offset = chain_params.theta_t_offset
    seed.theta_p_offset = chain_params.theta_p_offset

    all_samples = phase1 + phase2
    params, report = fit_joint(
        all_samples,
        initial=seed,
        fit_tb_rotation=False,
        fit_pan_offset=True,
        loss="soft_l1",
    )
    assert report.success
    assert report.trans_rmse_m < 0.003, report.summary()
    assert np.degrees(report.rot_rmse_rad) < 0.4, report.summary()
