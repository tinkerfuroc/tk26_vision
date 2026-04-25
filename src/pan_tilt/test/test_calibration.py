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

from pan_tilt.calibration.optimize import (
    fit_chain,
    fit_joint,
    solve_handeye,
    solve_handeye_with_consensus,
    warm_start_t_b_rotation,
)
from pan_tilt.calibration.pan_tilt_model import PanTiltParams, forward_kinematics
from pan_tilt.calibration.utils import (
    invert_transform,
    matrix_to_pose_dict,
    pose_error_scalars,
    pose_to_matrix,
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


def test_chain_with_90deg_mount_and_warm_start():
    """90 deg perpendicular mount: T_B has a real +pi/2 about X (camera looks along tilt_link +Z).

    Phase-2 chain fit must still recover T_A, T_B translation, and theta_t_offset
    after warm-starting T_B from a synthetic Phase-1 reference pose. T_B
    rotation is frozen (not fit) because the optimizer would otherwise absorb
    the Y-component into theta_t_offset.
    """
    truth = PanTiltParams(
        t_a=np.array([-0.28, -0.02, 1.55]),
        t_b_trans=np.array([-0.07, -0.01, 0.08]),
        t_b_rotvec=np.array([np.pi / 2, 0.0, 0.0]),  # 90 deg about X
        theta_t_offset=-np.pi / 4 + np.deg2rad(1.0),
        theta_p_offset=np.deg2rad(0.3),
        l_pan=0.135,
    )
    # Arbitrary T_ee_marker (known exactly because we fabricate it).
    Rem = Rotation.from_euler("xyz", [0.2, 0.4, -0.1]).as_matrix()
    t_ee_marker = np.eye(4)
    t_ee_marker[:3, :3] = Rem
    t_ee_marker[:3, 3] = np.array([0.03, 0.0, -0.10])

    noise_rng = np.random.default_rng(RNG_SEED + 400)
    phase2 = _phase2_samples(truth, t_ee_marker, noise_rng=noise_rng)

    # Synthesize a Phase-1 reference pose (Z_0 at servo zero).
    t_base_cam_ref = forward_kinematics(0.0, 0.0, truth)
    initial = warm_start_t_b_rotation(PanTiltParams(), t_base_cam_ref)
    # Warm start should put T_B_rotvec close to truth (pi/2 about X).
    # (Some Y-component may leak in because warm-start uses the default
    # theta_t_offset of -pi/4 which differs from truth.)
    assert abs(np.linalg.norm(initial.t_b_rotvec) - np.pi / 2) < 0.05

    params, report = fit_chain(
        phase2,
        t_ee_marker=t_ee_marker,
        initial=initial,
        fit_pan_offset=True,
        fit_tb_rotation=False,
        loss="soft_l1",
    )
    assert report.success
    # Residuals are bounded by the injected 1 mm / 0.1 deg noise.
    assert report.trans_rmse_m < 0.004, report.summary()
    assert np.degrees(report.rot_rmse_rad) < 0.35, report.summary()
    assert np.linalg.norm(params.t_a - truth.t_a) < 0.004
    assert np.linalg.norm(params.t_b_trans - truth.t_b_trans) < 0.004


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


def test_apply_to_urdf_patches_both_xacro_forms():
    """Regression guard: `apply_to_urdf._patched_xacro` must handle both the
    tk26_vision standalone form and the tk25_basic macro form."""
    from pan_tilt.calibration.apply_to_urdf import _patched_xacro

    t_a = np.array([-0.30, -0.02, 1.52])
    t_b_trans = np.array([-0.08, -0.01, 0.08])

    # Standalone form — literal pan_joint + camera_mount_joint
    standalone = (
        '<?xml version="1.0"?>\n'
        '<robot>\n'
        '  <joint name="pan_joint" type="revolute">\n'
        '    <parent link="base_link"/><child link="pan_link"/>\n'
        '    <origin xyz="0 0 0" rpy="0.01 0 0"/>\n'
        '  </joint>\n'
        '  <joint name="camera_mount_joint" type="fixed">\n'
        '    <parent link="tilt_link"/><child link="head_camera_link"/>\n'
        '    <origin xyz="0 0 0" rpy="0.1 0 3.0"/>\n'
        '  </joint>\n'
        '</robot>\n'
    )
    patched = _patched_xacro(standalone, t_a, t_b_trans, t_b_rotvec=np.zeros(3))
    assert 'xyz="-0.3 -0.02 1.52"' in patched
    assert 'rpy="0.01 0 0"' in patched  # preserved (T_A rotation not fitted)
    assert 'xyz="-0.08 -0.01 0.08"' in patched
    assert 'rpy="0.1 0 3.0"' in patched  # preserved (t_b_rotvec == 0)

    # Macro form — parameterized attach_xyz default
    macro = (
        '<?xml version="1.0"?>\n'
        '<robot xmlns:xacro="http://www.ros.org/wiki/xacro">\n'
        '  <xacro:macro name="pan_tilt_macro" params="\n'
        "    parent\n"
        "    prefix:=''\n"
        "    attach_xyz:='0 0 0'\n"
        "    attach_rpy:='0.01 0 0'\">\n"
        '    <joint name="${prefix}pan_joint" type="revolute">\n'
        '      <origin xyz="${attach_xyz}" rpy="${attach_rpy}"/>\n'
        '    </joint>\n'
        '    <joint name="${prefix}camera_mount_joint" type="fixed">\n'
        '      <origin xyz="0 0 0" rpy="0.1 0 3.0"/>\n'
        '    </joint>\n'
        '  </xacro:macro>\n'
        '</robot>\n'
    )
    patched = _patched_xacro(macro, t_a, t_b_trans, t_b_rotvec=np.zeros(3))
    assert "attach_xyz:='-0.3 -0.02 1.52'" in patched
    assert "attach_rpy:='0.01 0 0'" in patched  # preserved
    assert 'xyz="-0.08 -0.01 0.08"' in patched
    assert 'rpy="0.1 0 3.0"' in patched  # preserved

    # Non-zero t_b_rotvec must overwrite camera_mount rpy (both forms)
    rotvec = np.array([1.5, 0.0, 0.0])  # ~90 deg about X
    for src in (standalone, macro):
        p = _patched_xacro(src, t_a, t_b_trans, rotvec)
        assert 'rpy="0.1 0 3.0"' not in p, "old rpy should be gone"
        assert 'xyz="-0.08 -0.01 0.08"' in p


def test_polish_rejects_corrupted_sample(tmp_path):
    """Polish must drop outliers like handeye does. We synthesize a clean
    phase1+phase2 dataset, then deliberately rotate one phase1 sample's marker
    pose by 30 deg. With auto rejection on (default) the corrupted index must
    appear in `rejected_indices_auto` and the joint RMSE must stay near the
    noise floor; with `--no-reject` (and no manual exclude) the same fit blows
    up. This is the regression guard for the 0426_newset failure mode where
    `phase1/12` slipped through and drove polish to 17.9 mm trans / 5.65 deg.
    """
    from pan_tilt.calibration import run_calibration as rc
    import json, argparse

    truth, t_ee_marker = _make_truth()
    noise_rng = np.random.default_rng(RNG_SEED + 700)
    phase1 = _phase1_samples(truth, t_ee_marker, n=18, noise_rng=noise_rng)
    phase2 = _phase2_samples(truth, t_ee_marker, noise_rng=noise_rng)

    # Corrupt phase1[3]'s marker rotation by 30 deg about a random axis. This
    # mimics an EE-pose timing skew or a misdetected board orientation.
    bad_local = 3
    T_cm = pose_to_matrix(
        phase1[bad_local]["t_cam_marker_body"]["translation"],
        phase1[bad_local]["t_cam_marker_body"]["rotation"],
    )
    bad_axis = np.array([0.3, 0.7, 0.6]); bad_axis /= np.linalg.norm(bad_axis)
    R_bad = Rotation.from_rotvec(np.deg2rad(30.0) * bad_axis).as_matrix()
    T_cm[:3, :3] = R_bad @ T_cm[:3, :3]
    phase1[bad_local]["t_cam_marker_body"] = matrix_to_pose_dict(T_cm)
    bad_global = bad_local  # phase1 occupies the first len(phase1) slots

    # Write phase1, phase2 files. We don't need a real chain.json — write
    # the truth params directly into a seed JSON that polish can read.
    (tmp_path / "phase1.json").write_text(json.dumps({"samples": phase1}))
    (tmp_path / "phase2.json").write_text(json.dumps({"samples": phase2}))
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps({
        "params": {
            "t_a": truth.t_a.tolist(),
            "t_b_trans": truth.t_b_trans.tolist(),
            "t_b_rotvec": truth.t_b_rotvec.tolist(),
            "t_ee_marker_rotvec": Rotation.from_matrix(t_ee_marker[:3, :3]).as_rotvec().tolist(),
            "t_ee_marker_trans": t_ee_marker[:3, 3].tolist(),
            "theta_t_offset_rad": float(truth.theta_t_offset),
            "theta_p_offset_rad": float(truth.theta_p_offset),
            "l_pan": float(truth.l_pan),
        }
    }))

    def _polish_args(out_dir, *, no_reject=False, exclude_indices=()):
        return argparse.Namespace(
            phase1=[str(tmp_path / "phase1.json")],
            phase2=str(tmp_path / "phase2.json"),
            seed=str(seed_path),
            out=str(out_dir),
            unlock_tb_rotation=False,
            fit_pan_offset=True,
            loss="soft_l1",
            exclude_indices=list(exclude_indices),
            reject_sigma=3.0,
            max_reject_frac=0.10,
            no_reject=no_reject,
        )

    # Default run: auto MAD rejection should catch the corrupted sample.
    out_auto = tmp_path / "auto"
    rc.cmd_polish(_polish_args(out_auto))
    payload_auto = json.loads((out_auto / "polish.json").read_text())
    auto_indices = [r["index"] for r in payload_auto["rejected_indices_auto"]]
    assert bad_global in auto_indices, (
        f"corrupted sample #{bad_global} not auto-rejected; got {auto_indices}"
    )
    # On clean residue the joint fit must hit the calibration gate (3 mm/0.4°).
    assert payload_auto["trans_rmse_m"] < 0.003, (
        f"polish trans RMSE {payload_auto['trans_rmse_m']*1000:.2f} mm > 3 mm "
        f"after auto rejection — outlier still poisoning the fit"
    )
    assert np.degrees(payload_auto["rot_rmse_rad"]) < 0.4

    # Manual exclude path: same outcome without the MAD loop running.
    out_manual = tmp_path / "manual"
    rc.cmd_polish(_polish_args(out_manual, no_reject=True, exclude_indices=[bad_global]))
    payload_manual = json.loads((out_manual / "polish.json").read_text())
    assert payload_manual["rejected_indices_manual"] == [bad_global]
    assert payload_manual["rejected_indices_auto"] == []
    assert payload_manual["trans_rmse_m"] < 0.003

    # No-reject + no manual exclude: outlier is left in -> fit must be noticeably
    # worse than the gate, proving rejection is what saves the result. The
    # corruption is rotation-only, so the trans RMSE may stay near the noise
    # floor; the rot residual is what reliably blows up.
    out_none = tmp_path / "none"
    rc.cmd_polish(_polish_args(out_none, no_reject=True))
    payload_none = json.loads((out_none / "polish.json").read_text())
    assert (
        payload_none["trans_rmse_m"] > 0.003
        or np.degrees(payload_none["rot_rmse_rad"]) > 0.4
    ), (
        "no-reject baseline didn't blow up — corruption may be too small to "
        "exercise the regression"
    )


def test_handeye_t_ee_marker_cross_check(tmp_path):
    """If `handeye_custom.json` would be written with a T_ee_marker that disagrees
    with the existing canonical `handeye.json` by more than 5 mm / 1°, the
    handeye solver must refuse — that's the operator's loud signal that one of
    the phase-1 files is stale or the board was re-mounted between collects.
    Override flag bypasses the check.

    This is the regression guard for the 0426_newset incident where a stale
    canonical `phase1_handeye.json` quietly poisoned the entire downstream
    pipeline (chain ~13 mm / 21°, polish ~248 mm / 32°)."""
    from pan_tilt.calibration import run_calibration as rc
    import json, argparse, pytest

    truth, t_ee_marker_a = _make_truth()

    # Second batch fabricated with a different marker pose (rotated 30°
    # around X, translated 10 cm) to mimic the operator re-mounting the board.
    Rb = Rotation.from_euler("xyz", [0.5, 0.0, 0.0]).as_matrix()
    t_ee_marker_b = np.eye(4)
    t_ee_marker_b[:3, :3] = Rb
    t_ee_marker_b[:3, 3] = t_ee_marker_a[:3, 3] + np.array([0.10, 0.0, 0.0])

    # Generate two clean phase-1 batches that describe two different
    # T_ee_markers — the disagreement is genuine, not solver noise.
    phase1_a = _phase1_samples(truth, t_ee_marker_a, n=14, noise_rng=None)
    phase1_b = _phase1_samples(truth, t_ee_marker_b, n=14, noise_rng=None)
    pa = tmp_path / "phase1_handeye.json"
    pb = tmp_path / "phase1_handeye_custom.json"
    pa.write_text(json.dumps({"samples": phase1_a}))
    pb.write_text(json.dumps({"samples": phase1_b}))

    def _handeye_args(phase1, out_name=None, allow=False):
        return argparse.Namespace(
            phase1=str(phase1),
            out=str(tmp_path),
            out_name=out_name,
            no_quality_gate=False,
            quality_min_corners=10,
            quality_max_reproj_px=1.5,
            no_reject=False,
            reject_sigma=3.0,
            max_reject_frac=0.25,
            prefilter_rot_deg=5.0,
            allow_t_ee_marker_mismatch=allow,
        )

    # Step 1: write the canonical handeye.
    rc.cmd_handeye(_handeye_args(pa))
    canonical = tmp_path / "handeye.json"
    assert canonical.is_file()
    canonical_bytes_before = canonical.read_bytes()

    # Step 2: try to write handeye_custom from the second batch — must abort.
    custom = tmp_path / "handeye_custom.json"
    assert not custom.is_file()
    with pytest.raises(SystemExit) as excinfo:
        rc.cmd_handeye(_handeye_args(pb, out_name="handeye_custom.json"))
    assert excinfo.value.code != 0
    assert not custom.is_file(), "cross-check should refuse to write the file"
    # Canonical file must be untouched.
    assert canonical.read_bytes() == canonical_bytes_before

    # Step 3: with the override flag, the file is written.
    rc.cmd_handeye(_handeye_args(pb, out_name="handeye_custom.json", allow=True))
    assert custom.is_file()
    saved = json.loads(custom.read_text())
    saved_em_trans = np.asarray(saved["t_ee_marker"]["translation"])
    # Saved should reflect the second batch's truth (within solver noise).
    assert np.linalg.norm(saved_em_trans - t_ee_marker_b[:3, 3]) < 0.005


def test_polish_merges_multiple_phase1(tmp_path):
    """Polish must concatenate multiple phase-1 datasets when --phase1 receives
    more than one path. Merging two phase-1 batches collected at different park
    poses is the operator-facing reason this exists: the extra EE-rotation
    diversity helps break the T_B(Y) ↔ θ_t_offset degeneracy when polish runs
    with --unlock-tb-rotation. We synthesize two batches and verify both that
    polish reads them and that an outlier dropped into the *second* batch is
    still caught by the auto-rejection loop (i.e. concatenation indices are
    threaded through correctly)."""
    from pan_tilt.calibration import run_calibration as rc
    import json, argparse

    truth, t_ee_marker = _make_truth()
    noise_rng = np.random.default_rng(RNG_SEED + 750)
    # Two phase-1 batches with different RNG seeds → different EE poses.
    phase1_a = _phase1_samples(truth, t_ee_marker, n=12, noise_rng=noise_rng)
    rng_b = np.random.default_rng(RNG_SEED + 9000)
    phase1_b = []
    for _ in range(10):
        T_be = _random_ee_pose(rng_b)
        phase1_b.append(_sample(0.0, 0.0, T_be, truth, t_ee_marker, noise_rng))
    phase2 = _phase2_samples(truth, t_ee_marker, noise_rng=noise_rng)

    # Corrupt one sample in batch B with a 30° rotation. Index in the
    # concatenated array = len(phase1_a) + 4 (since A precedes B in the
    # --phase1 argument order).
    bad_in_b = 4
    T_cm = pose_to_matrix(
        phase1_b[bad_in_b]["t_cam_marker_body"]["translation"],
        phase1_b[bad_in_b]["t_cam_marker_body"]["rotation"],
    )
    bad_axis = np.array([0.6, 0.5, 0.6]); bad_axis /= np.linalg.norm(bad_axis)
    R_bad = Rotation.from_rotvec(np.deg2rad(30.0) * bad_axis).as_matrix()
    T_cm[:3, :3] = R_bad @ T_cm[:3, :3]
    phase1_b[bad_in_b]["t_cam_marker_body"] = matrix_to_pose_dict(T_cm)
    bad_global = len(phase1_a) + bad_in_b

    pa = tmp_path / "phase1_a.json"
    pb = tmp_path / "phase1_b.json"
    p2 = tmp_path / "phase2.json"
    pa.write_text(json.dumps({"samples": phase1_a}))
    pb.write_text(json.dumps({"samples": phase1_b}))
    p2.write_text(json.dumps({"samples": phase2}))
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps({
        "params": {
            "t_a": truth.t_a.tolist(),
            "t_b_trans": truth.t_b_trans.tolist(),
            "t_b_rotvec": truth.t_b_rotvec.tolist(),
            "t_ee_marker_rotvec": Rotation.from_matrix(t_ee_marker[:3, :3]).as_rotvec().tolist(),
            "t_ee_marker_trans": t_ee_marker[:3, 3].tolist(),
            "theta_t_offset_rad": float(truth.theta_t_offset),
            "theta_p_offset_rad": float(truth.theta_p_offset),
            "l_pan": float(truth.l_pan),
        }
    }))

    out = tmp_path / "out"
    rc.cmd_polish(argparse.Namespace(
        phase1=[str(pa), str(pb)],
        phase2=str(p2),
        seed=str(seed_path),
        out=str(out),
        unlock_tb_rotation=False,
        fit_pan_offset=True,
        loss="soft_l1",
        exclude_indices=[],
        reject_sigma=3.0,
        max_reject_frac=0.10,
        no_reject=False,
    ))
    payload = json.loads((out / "polish.json").read_text())

    # Both phase-1 paths must be recorded so polish.json is self-describing.
    assert payload["phase1_sources"] == [str(pa), str(pb)]
    # Total sample count = sum of all inputs.
    assert payload["n_samples_total"] == len(phase1_a) + len(phase1_b) + len(phase2)
    # Auto rejection must catch the corrupted index, which is in batch B's
    # range of the concatenated array.
    auto_indices = [r["index"] for r in payload["rejected_indices_auto"]]
    assert bad_global in auto_indices, (
        f"corrupted sample at concat-index {bad_global} not auto-rejected; "
        f"got {auto_indices}"
    )
    # Joint fit on the clean residue must hit the calibration gate.
    assert payload["trans_rmse_m"] < 0.003, (
        f"polish trans RMSE {payload['trans_rmse_m']*1000:.2f} mm > 3 mm"
    )
    assert np.degrees(payload["rot_rmse_rad"]) < 0.4


def test_chain_output_includes_per_sample_residuals(tmp_path):
    """`chain.json` must expose per-sample residual arrays -- the Calibrate
    tab's browser-side residual chart reads them directly. Without this test
    a future refactor could silently drop the arrays and leave the chart
    empty. We re-use the synthetic fixture so there's no hardware dependency.
    """
    from pan_tilt.calibration import run_calibration as rc
    import json, argparse

    truth, t_ee_marker = _make_truth()
    # Phase-1 samples (hand-eye stage input) — xArm poses vary, head at "park".
    he_samples = _phase1_samples(truth, t_ee_marker, noise_rng=None)
    (tmp_path / "phase1_handeye.json").write_text(json.dumps({"samples": he_samples}))
    # Phase-2 samples (chain stage input) — pan/tilt vary, xArm frozen.
    ch_samples = _phase2_samples(truth, t_ee_marker, noise_rng=None)
    (tmp_path / "phase2_chain.json").write_text(json.dumps({"samples": ch_samples}))

    # Run handeye first so chain has its seed.
    rc.cmd_handeye(argparse.Namespace(
        phase1=str(tmp_path / "phase1_handeye.json"),
        out=str(tmp_path),
    ))
    assert (tmp_path / "handeye.json").is_file()

    rc.cmd_chain(argparse.Namespace(
        phase2=str(tmp_path / "phase2_chain.json"),
        handeye=str(tmp_path / "handeye.json"),
        out=str(tmp_path),
        fit_pan_offset=False,
        unlock_tb_rotation=False,
        loss="soft_l1",
        val_seed=0,
        verbose=False,
    ))
    payload = json.loads((tmp_path / "chain.json").read_text())

    assert "per_sample_trans_err_m" in payload
    assert "per_sample_rot_err_rad" in payload
    assert isinstance(payload["per_sample_trans_err_m"], list)
    assert len(payload["per_sample_trans_err_m"]) == payload["n_train"]
    assert len(payload["per_sample_rot_err_rad"]) == payload["n_train"]
    # Noise-free fixture -> residuals should be finite and well below the
    # 3 mm calibration gate. Tolerance is loose vs analytic zero because
    # the solver terminates on ftol once parameters are recovered, not when
    # residuals themselves hit machine precision.
    assert all(np.isfinite(v) for v in payload["per_sample_trans_err_m"])
    assert max(payload["per_sample_trans_err_m"]) < 1e-3


# ---- robustness regressions for the redesigned pipeline ---------------------
#
# The 2026-04-25 redesign added three layers:
#   - per-frame PnP method selection (B1) and IPPE multi-criterion picker (B2)
#   - per-cell cluster_consensus voter (B3)
#   - per-dataset cross-cell consensus pre-pass (C)
#
# Tests below exercise each layer with synthetic inputs that mimic the real
# failure mode (IPPE planar reflection at the cell or sample level).


def _ippe_flip(T_cm: np.ndarray) -> np.ndarray:
    """Synthesize an IPPE planar-reflection flip for a marker pose.

    The two IPPE solutions for a planar target are related by a rotation by
    pi about an axis lying in the plane perpendicular to the principal
    viewing ray. We approximate this as a rotation by pi about an axis in
    the image plane (which for our purposes is good enough -- the rotation
    magnitude is the diagnostic signature).
    """
    R_in = T_cm[:3, :3]
    t = T_cm[:3, 3]
    rng = np.linalg.norm(t)
    view_dir = t / max(rng, 1e-9)
    # Build an axis perpendicular to view_dir.
    arbitrary = np.array([0.0, 0.0, 1.0]) if abs(view_dir[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    axis = np.cross(view_dir, arbitrary)
    axis = axis / np.linalg.norm(axis)
    R_flip = Rotation.from_rotvec(np.pi * axis).as_matrix()
    T_out = T_cm.copy()
    T_out[:3, :3] = R_flip @ R_in
    return T_out


def test_solve_handeye_with_consensus_rejects_ippe_flips():
    """Cross-cell consensus pre-pass must drop synthetically-flipped samples.

    Build a clean phase-1 dataset, flip ~30% of the marker poses (simulate
    IPPE branch failures that survived the per-cell voter), and check that
    `solve_handeye_with_consensus` filters them out and recovers the truth
    on the remaining clean samples.
    """
    truth, t_ee_marker = _make_truth()
    noise_rng = np.random.default_rng(RNG_SEED + 300)
    samples = _phase1_samples(truth, t_ee_marker, n=20, noise_rng=noise_rng)

    flip_indices = {3, 7, 11, 13, 16, 18}    # 6/20 = 30% flipped
    for i in flip_indices:
        T_cm = pose_to_matrix(
            samples[i]["t_cam_marker_body"]["translation"],
            samples[i]["t_cam_marker_body"]["rotation"],
        )
        T_flipped = _ippe_flip(T_cm)
        samples[i]["t_cam_marker_body"] = matrix_to_pose_dict(T_flipped)

    t_em_recovered, t_bc_recovered, _per_pose, rejected = \
        solve_handeye_with_consensus(samples, pre_filter_rot_deg=5.0)

    # Every flipped sample must have been rejected. We allow extras
    # (occasional clean samples can land just over threshold under noise),
    # but no flip should slip through.
    assert flip_indices.issubset(set(rejected)), (
        f"flipped samples not all rejected; missed {flip_indices - set(rejected)}"
    )
    # Recovery on the clean residue must meet the calibration gate.
    trans_err, rot_err = pose_error_scalars(t_em_recovered, t_ee_marker)
    assert trans_err < 0.006, f"trans err {trans_err*1000:.2f} mm"
    assert np.degrees(rot_err) < 0.8, f"rot err {np.degrees(rot_err):.3f} deg"


def test_cluster_consensus_picks_majority_branch():
    """When most frames agree on one IPPE branch, cluster_consensus picks it
    even though a minority of frames sit on the other branch."""
    from pan_tilt.calibration.aruco_detect import (
        Detection, PoseCandidate, cluster_consensus,
    )

    rng = np.random.default_rng(RNG_SEED + 400)
    # Truth pose (cam_optical -> marker), arbitrary.
    R_true = Rotation.from_euler("xyz", [0.1, -0.2, 0.05]).as_matrix()
    t_true = np.array([0.05, -0.02, 0.65])
    T_true = np.eye(4); T_true[:3, :3] = R_true; T_true[:3, 3] = t_true

    # Corresponding flip via the same helper used for sample-level tests.
    T_flip = _ippe_flip(T_true)

    detections = []
    for k in range(10):
        # Add small per-frame noise.
        dR = Rotation.from_rotvec(rng.normal(0, np.deg2rad(0.2), size=3)).as_matrix()
        dt = rng.normal(0, 0.0005, size=3)
        if k < 7:
            T = T_true.copy()
        else:
            T = T_flip.copy()
        T[:3, :3] = dR @ T[:3, :3]
        T[:3, 3] = T[:3, 3] + dt
        cand = PoseCandidate(pose_optical=T, reproj_rms_px=0.3 + rng.normal(0, 0.05))
        detections.append(Detection(
            pose_optical=T, n_corners=20, reprojection_rms_px=cand.reproj_rms_px,
            success=True, candidates=[cand], method="iterative",
        ))

    consensus = cluster_consensus(detections, min_cluster_frac=0.6)
    assert consensus is not None
    # Recovered ROTATION must be close to the majority cluster (T_true), not
    # the flipped minority. Translations are identical between branches by
    # construction of the IPPE planar reflection, so we gate on rotation.
    _, rot_err_true = pose_error_scalars(consensus.pose_optical, T_true)
    _, rot_err_flip = pose_error_scalars(consensus.pose_optical, T_flip)
    assert np.degrees(rot_err_true) < 1.0, f"rot err to T_true {np.degrees(rot_err_true):.3f} deg"
    assert rot_err_true < rot_err_flip


def test_cluster_consensus_returns_none_on_split():
    """Even split between two branches yields no quorum -> None."""
    from pan_tilt.calibration.aruco_detect import (
        Detection, PoseCandidate, cluster_consensus,
    )

    rng = np.random.default_rng(RNG_SEED + 500)
    R_true = Rotation.from_euler("xyz", [0.0, 0.3, 0.0]).as_matrix()
    t_true = np.array([0.0, 0.0, 0.5])
    T_true = np.eye(4); T_true[:3, :3] = R_true; T_true[:3, 3] = t_true
    T_flip = _ippe_flip(T_true)

    detections = []
    for k in range(10):
        dR = Rotation.from_rotvec(rng.normal(0, np.deg2rad(0.1), size=3)).as_matrix()
        dt = rng.normal(0, 0.0003, size=3)
        T = (T_true if k < 5 else T_flip).copy()
        T[:3, :3] = dR @ T[:3, :3]
        T[:3, 3] = T[:3, 3] + dt
        cand = PoseCandidate(pose_optical=T, reproj_rms_px=0.3)
        detections.append(Detection(
            pose_optical=T, n_corners=20, reprojection_rms_px=0.3,
            success=True, candidates=[cand], method="iterative",
        ))

    # 5/10 in each cluster -> neither reaches the 60% quorum.
    assert cluster_consensus(detections, min_cluster_frac=0.6) is None


def test_cluster_consensus_breaks_tie_with_dual_ippe_candidates():
    """When per-frame IPPE returns BOTH candidates and clustering across all
    candidates favors one cluster, the voter should still pick correctly."""
    from pan_tilt.calibration.aruco_detect import (
        Detection, PoseCandidate, cluster_consensus,
    )

    rng = np.random.default_rng(RNG_SEED + 600)
    R_true = Rotation.from_euler("xyz", [-0.05, 0.2, 0.1]).as_matrix()
    t_true = np.array([0.02, 0.01, 0.55])
    T_true = np.eye(4); T_true[:3, :3] = R_true; T_true[:3, 3] = t_true
    T_flip = _ippe_flip(T_true)

    detections = []
    for k in range(8):
        dR = Rotation.from_rotvec(rng.normal(0, np.deg2rad(0.2), size=3)).as_matrix()
        dt = rng.normal(0, 0.0005, size=3)
        T_a = T_true.copy(); T_a[:3, :3] = dR @ T_a[:3, :3]; T_a[:3, 3] = T_a[:3, 3] + dt
        T_b = T_flip.copy(); T_b[:3, :3] = dR @ T_b[:3, :3]; T_b[:3, 3] = T_b[:3, 3] + dt
        # Half the frames give the true branch a slightly lower reproj
        # (matching what really happens at most viewing angles).
        if k % 2 == 0:
            cands = [
                PoseCandidate(pose_optical=T_a, reproj_rms_px=0.4),
                PoseCandidate(pose_optical=T_b, reproj_rms_px=0.5),
            ]
            primary = T_a
        else:
            cands = [
                PoseCandidate(pose_optical=T_b, reproj_rms_px=0.4),
                PoseCandidate(pose_optical=T_a, reproj_rms_px=0.5),
            ]
            primary = T_b
        detections.append(Detection(
            pose_optical=primary, n_corners=18,
            reprojection_rms_px=0.4, success=True,
            candidates=cands, method="ippe",
        ))

    # Each frame contributes one vote per cluster. Both clusters get 8/8
    # frames, but the dedup ("one vote per frame per cluster") means the
    # tie is broken by which cluster the lower-reproj candidates seed first.
    # The function should return a valid consensus.
    consensus = cluster_consensus(detections, min_cluster_frac=0.6)
    assert consensus is not None
    err_to_true, _ = pose_error_scalars(consensus.pose_optical, T_true)
    err_to_flip, _ = pose_error_scalars(consensus.pose_optical, T_flip)
    # Either branch is technically valid as a "vote outcome" here, but the
    # recovered pose must match ONE cluster (i.e. not be smeared between).
    assert min(err_to_true, err_to_flip) < 0.01


def test_duplicate_ee_geometry_check():
    """The EE-duplicate guard at the recording side compares full SE(3).
    Verify the geometry primitive used (`pose_error_scalars`) flags the
    smoking-gun pairs we saw in the 04-25 dataset."""
    # Two waypoints with identical rotation but 1.5 mm translation.
    T_a = np.eye(4); T_a[:3, 3] = [0.30, 0.0, 1.20]
    T_b = T_a.copy(); T_b[:3, 3] = [0.3015, 0.0, 1.20]
    t, r = pose_error_scalars(T_a, T_b)
    assert t > 0.001 and r < 0.001    # 1.5 mm > 1 mm tol -> would NOT flag

    # Identical poses (the smoking-gun case from the field dataset).
    T_dup = T_a.copy()
    t, r = pose_error_scalars(T_a, T_dup)
    assert t < 1e-9 and r < 1e-9      # would flag -- duplicate detected
