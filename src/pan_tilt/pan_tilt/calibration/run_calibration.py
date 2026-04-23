"""Calibration CLI.

Subcommands:
    intrinsic  <images_dir> <--board spec.json>
        Estimate Orbbec RGB intrinsics from a bag of ChArUco images. Writes
        `<out>/intrinsic.json` with K, dist, per-image reprojection RMSE.

    handeye    <phase1.json> <--out results/>
        Solve Phase-1 hand-eye from a collected dataset. Emits
        `results/<session>/handeye.json` with T_ee_marker + reference T_base_cam.

    chain      <phase2.json> <--handeye handeye.json> <--out results/>
        Solve Phase-2 pan-tilt chain fit. Emits `chain.json` with fitted
        PanTiltParams (T_A_trans, T_B_trans, theta_t_offset, optional theta_p_offset)
        and per-cell residual arrays.

    polish     <phase1.json> <phase2.json> <--seed chain.json> <--out results/>
        Optional Phase-3 joint refinement. Emits `polish.json`.

    validate   <results_dir>
        Load whatever is in results/ and print a pass/fail summary against
        the plan's gates. Also writes `residuals.png` if matplotlib is available.

Each subcommand is pure-Python / no ROS — safe to run after-the-fact on saved data.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .aruco_detect import BoardSpec, build_board, detect_pose
from .optimize import fit_chain, fit_joint, solve_handeye, warm_start_t_b_rotation
from .pan_tilt_model import PanTiltParams, forward_kinematics
from .utils import (
    invert_transform,
    matrix_to_pose,
    matrix_to_pose_dict,
    pose_error_scalars,
    pose_to_matrix,
    sample_to_matrices,
)


# ---- I/O helpers ------------------------------------------------------------

def _load_samples(path: Path) -> list[dict]:
    blob = json.loads(path.read_text())
    return blob["samples"] if "samples" in blob else blob


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _params_to_dict(p: PanTiltParams) -> dict:
    return {
        "t_a": p.t_a.tolist(),
        "t_b_trans": p.t_b_trans.tolist(),
        "t_b_rotvec": p.t_b_rotvec.tolist(),
        "t_ee_marker_rotvec": p.t_ee_marker_rotvec.tolist(),
        "t_ee_marker_trans": p.t_ee_marker_trans.tolist(),
        "theta_t_offset_rad": float(p.theta_t_offset),
        "theta_t_offset_deg": float(np.degrees(p.theta_t_offset)),
        "theta_p_offset_rad": float(p.theta_p_offset),
        "theta_p_offset_deg": float(np.degrees(p.theta_p_offset)),
        "l_pan": float(p.l_pan),
    }


def _params_from_dict(d: dict) -> PanTiltParams:
    return PanTiltParams(
        t_a=np.asarray(d["t_a"], dtype=float),
        t_b_trans=np.asarray(d["t_b_trans"], dtype=float),
        t_b_rotvec=np.asarray(d.get("t_b_rotvec", [0, 0, 0]), dtype=float),
        t_ee_marker_rotvec=np.asarray(d.get("t_ee_marker_rotvec", [0, 0, 0]), dtype=float),
        t_ee_marker_trans=np.asarray(d.get("t_ee_marker_trans", [0, 0, 0]), dtype=float),
        theta_t_offset=float(d.get("theta_t_offset_rad", -np.pi / 4)),
        theta_p_offset=float(d.get("theta_p_offset_rad", 0.0)),
        l_pan=float(d.get("l_pan", 0.135)),
    )


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---- subcommands ------------------------------------------------------------

def cmd_intrinsic(args):
    """Calibrate camera intrinsics from a directory of ChArUco images."""
    import cv2

    img_paths = sorted(Path(args.images_dir).glob("*.png")) + sorted(
        Path(args.images_dir).glob("*.jpg")
    )
    if len(img_paths) < 10:
        raise SystemExit(f"Need >=10 images, found {len(img_paths)}")

    spec = _load_board_spec(args.board)
    board = build_board(spec)
    detector = _build_calib_detector(board)

    all_obj, all_img, image_size = [], [], None
    for p in img_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        ch_corners, ch_ids, _, _ = detector.detectBoard(img)
        if ch_ids is None or len(ch_ids) < 6:
            print(f"  skip {p.name}: only {0 if ch_ids is None else len(ch_ids)} corners")
            continue
        obj, imgpts = board.matchImagePoints(ch_corners, ch_ids)
        all_obj.append(obj)
        all_img.append(imgpts)

    if len(all_obj) < 10:
        raise SystemExit(f"Only {len(all_obj)} usable images; need >=10.")

    rms, K, dist, _, _ = cv2.calibrateCamera(all_obj, all_img, image_size, None, None)

    out_path = Path(args.out) / "intrinsic.json"
    _save_json(out_path, {
        "rms_px": float(rms),
        "K": K.tolist(),
        "dist": dist.flatten().tolist(),
        "image_size": image_size,
        "n_images": len(all_obj),
    })
    print(f"Intrinsic RMS: {rms:.3f} px  (plan gate < 0.5 px)")
    print(f"Wrote {out_path}")


def cmd_handeye(args):
    samples = _load_samples(Path(args.phase1))
    print(f"Loaded {len(samples)} Phase-1 samples")

    t_ee_marker, t_base_cam_ref, per_pose = solve_handeye(samples)
    trans_errs = np.array([p[0] for p in per_pose])
    rot_errs = np.array([p[1] for p in per_pose])

    out_path = Path(args.out) / "handeye.json"
    _save_json(out_path, {
        "t_ee_marker": matrix_to_pose_dict(t_ee_marker),
        "t_base_cam_ref": matrix_to_pose_dict(t_base_cam_ref),
        "n_samples": len(samples),
        "per_sample_trans_err_m": trans_errs.tolist(),
        "per_sample_rot_err_rad": rot_errs.tolist(),
        "trans_rmse_m": float(np.sqrt(np.mean(trans_errs ** 2))),
        "rot_rmse_rad": float(np.sqrt(np.mean(rot_errs ** 2))),
    })
    print(
        f"Hand-eye trans RMSE: {np.sqrt(np.mean(trans_errs**2))*1000:.2f} mm  "
        f"rot RMSE: {np.degrees(np.sqrt(np.mean(rot_errs**2))):.3f} deg"
    )
    print(f"Wrote {out_path}")


def cmd_chain(args):
    samples = _load_samples(Path(args.phase2))
    handeye = json.loads(Path(args.handeye).read_text())
    t_ee_marker = pose_to_matrix(
        handeye["t_ee_marker"]["translation"],
        handeye["t_ee_marker"]["rotation"],
    )
    t_base_cam_ref = pose_to_matrix(
        handeye["t_base_cam_ref"]["translation"],
        handeye["t_base_cam_ref"]["rotation"],
    )

    # Hold out a random 20% split for validation.
    rng = np.random.default_rng(args.val_seed)
    idxs = rng.permutation(len(samples))
    n_val = max(1, len(samples) // 5)
    val_idxs = set(idxs[:n_val].tolist())
    train = [s for i, s in enumerate(samples) if i not in val_idxs]
    val = [s for i, s in enumerate(samples) if i in val_idxs]
    print(f"train={len(train)} val={len(val)}")

    # Warm-start T_B rotation from Phase 1's reference pose. On this robot the
    # camera is mounted ~90 deg off the tilt arm, so starting T_B_rot at identity
    # would drop the optimizer into a bad basin.
    initial = warm_start_t_b_rotation(PanTiltParams(), t_base_cam_ref)
    if args.verbose:
        print(
            f"T_B warm start: trans={np.round(initial.t_b_trans, 4)}  "
            f"rotvec={np.round(initial.t_b_rotvec, 4)} "
            f"(norm={np.linalg.norm(initial.t_b_rotvec):.3f} rad)"
        )

    params, report = fit_chain(
        train,
        t_ee_marker=t_ee_marker,
        initial=initial,
        fit_pan_offset=args.fit_pan_offset,
        fit_tb_rotation=not args.lock_tb_rotation,
        loss=args.loss,
    )
    print("TRAIN:", report.summary())

    _, val_report = fit_chain(
        val,
        t_ee_marker=t_ee_marker,
        initial=params,
        fit_pan_offset=args.fit_pan_offset,
        loss="linear",
    )
    # val_report re-optimizes on val; to truly measure generalization we want
    # the predicted residuals for `val` with `params` held fixed.
    val_trans, val_rot = _eval_params_on_samples(params, val, t_ee_marker)
    print(
        f"VAL:   n={len(val)} trans_rmse={val_trans*1000:.2f} mm "
        f"rot_rmse={np.degrees(val_rot):.3f} deg"
    )

    out_path = Path(args.out) / "chain.json"
    _save_json(out_path, {
        "params": _params_to_dict(params),
        "train_trans_rmse_m": report.trans_rmse_m,
        "train_rot_rmse_rad": report.rot_rmse_rad,
        "val_trans_rmse_m": float(val_trans),
        "val_rot_rmse_rad": float(val_rot),
        "n_train": len(train),
        "n_val": len(val),
        "fit_pan_offset": args.fit_pan_offset,
        "fit_tb_rotation": not args.lock_tb_rotation,
        "handeye_source": str(Path(args.handeye)),
    })
    print(f"Wrote {out_path}")


def cmd_polish(args):
    phase1 = _load_samples(Path(args.phase1))
    phase2 = _load_samples(Path(args.phase2))
    seed = _params_from_dict(json.loads(Path(args.seed).read_text())["params"])

    samples = phase1 + phase2
    params, report = fit_joint(
        samples,
        initial=seed,
        fit_tb_rotation=args.unlock_tb_rotation,
        fit_pan_offset=args.fit_pan_offset,
        loss=args.loss,
    )
    print("POLISH:", report.summary())

    out_path = Path(args.out) / "polish.json"
    _save_json(out_path, {
        "params": _params_to_dict(params),
        "trans_rmse_m": report.trans_rmse_m,
        "rot_rmse_rad": report.rot_rmse_rad,
        "n_samples": len(samples),
        "fit_tb_rotation": args.unlock_tb_rotation,
        "fit_pan_offset": args.fit_pan_offset,
    })
    print(f"Wrote {out_path}")


def cmd_validate(args):
    results_dir = Path(args.results_dir)
    gates = [
        ("intrinsic.json", "rms_px", 0.5, "< 0.5 px"),
        ("handeye.json", "trans_rmse_m", 0.003, "< 3 mm trans"),
        ("handeye.json", "rot_rmse_rad", np.deg2rad(0.5), "< 0.5 deg rot"),
        ("chain.json", "val_trans_rmse_m", 0.003, "< 3 mm trans"),
        ("chain.json", "val_rot_rmse_rad", np.deg2rad(0.4), "< 0.4 deg rot"),
    ]
    for fname, key, thresh, label in gates:
        p = results_dir / fname
        if not p.exists():
            print(f"  [skip ] {fname}:{key}  (missing)")
            continue
        val = json.loads(p.read_text()).get(key)
        if val is None:
            print(f"  [skip ] {fname}:{key}  (key missing)")
            continue
        ok = val < thresh
        flag = "  OK  " if ok else " FAIL "
        unit = "mm" if "trans" in key else ("deg" if "rot" in key else "px")
        disp = val * 1000 if "trans" in key else (np.degrees(val) if "rot" in key else val)
        print(f"  [{flag}] {fname}:{key} = {disp:.3f} {unit}  ({label})")


# ---- helpers ---------------------------------------------------------------

def _eval_params_on_samples(
    params: PanTiltParams, samples: list, t_ee_marker: np.ndarray
):
    """RMSE of `forward_kinematics(params)` vs ground truth from hand-eye X."""
    trans_errs, rot_errs = [], []
    for s in samples:
        theta_p, theta_t, T_be, T_cm = sample_to_matrices(s)
        T_gt = T_be @ t_ee_marker @ invert_transform(T_cm)
        T_pred = forward_kinematics(theta_p, theta_t, params)
        te, re = pose_error_scalars(T_pred, T_gt)
        trans_errs.append(te)
        rot_errs.append(re)
    trans_errs = np.asarray(trans_errs)
    rot_errs = np.asarray(rot_errs)
    return float(np.sqrt(np.mean(trans_errs ** 2))), float(np.sqrt(np.mean(rot_errs ** 2)))


def _load_board_spec(path: Optional[str]) -> BoardSpec:
    if not path:
        return BoardSpec()
    d = json.loads(Path(path).read_text())
    import cv2
    return BoardSpec(
        squares_x=int(d["squares_x"]),
        squares_y=int(d["squares_y"]),
        square_len_m=float(d["square_len_m"]),
        marker_len_m=float(d["marker_len_m"]),
        dict_id=getattr(cv2.aruco, d.get("dict", "DICT_5X5_100")),
    )


def _build_calib_detector(board):
    import cv2
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), params)


# ---- CLI --------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Pan-tilt calibration driver.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("intrinsic", help="Calibrate Orbbec RGB intrinsics.")
    pi.add_argument("images_dir")
    pi.add_argument("--board", help="board spec JSON", default=None)
    pi.add_argument("--out", default="results")
    pi.set_defaults(func=cmd_intrinsic)

    ph = sub.add_parser("handeye", help="Phase 1: hand-eye at servo zero.")
    ph.add_argument("phase1")
    ph.add_argument("--out", default="results")
    ph.set_defaults(func=cmd_handeye)

    pc = sub.add_parser("chain", help="Phase 2: pan-tilt chain fit.")
    pc.add_argument("phase2")
    pc.add_argument("--handeye", required=True)
    pc.add_argument("--out", default="results")
    pc.add_argument("--fit-pan-offset", action="store_true")
    pc.add_argument("--lock-tb-rotation", action="store_true",
                    help="freeze T_B rotation at init (default: fit it, since the "
                         "physical camera mount has a ~90 deg twist vs the tilt arm)")
    pc.add_argument("--loss", default="soft_l1")
    pc.add_argument("--val-seed", type=int, default=0)
    pc.add_argument("--verbose", action="store_true")
    pc.set_defaults(func=cmd_chain)

    pp = sub.add_parser("polish", help="Phase 3: joint refinement.")
    pp.add_argument("phase1")
    pp.add_argument("phase2")
    pp.add_argument("--seed", required=True)
    pp.add_argument("--out", default="results")
    pp.add_argument("--unlock-tb-rotation", action="store_true")
    pp.add_argument("--fit-pan-offset", action="store_true")
    pp.add_argument("--loss", default="soft_l1")
    pp.set_defaults(func=cmd_polish)

    pv = sub.add_parser("validate", help="Check results against plan gates.")
    pv.add_argument("results_dir")
    pv.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
