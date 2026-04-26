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

    validate   --phase4 phase4.json --params polish.json --out results/
        Phase-4 end-to-end check (xArm-independent, board fixed in
        base_link). Compose `T_base_marker_pred` for each held-out
        (pan, tilt) view through the FK chain under test and report the
        spread vs the centroid. Writes `validation.json` with a
        PASS/WARN/FAIL verdict against 5 mm / 0.5° (PASS) and 10 mm / 1° (WARN).

    gates      <results_dir>
        Legacy: print PASS/FAIL summary for each per-phase residual against
        the static plan gates. Superseded by `validate` for end-to-end checks.

Each subcommand is pure-Python / no ROS — safe to run after-the-fact on saved data.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .aruco_detect import BoardSpec, build_board, detect_pose
from .optimize import (
    fit_chain,
    fit_joint,
    solve_handeye,
    solve_handeye_with_consensus,
    warm_start_t_b_rotation,
)
from .pan_tilt_model import PanTiltParams, forward_kinematics
from .utils import (
    invert_transform,
    matrix_to_pose,
    matrix_to_pose_dict,
    pose_error_scalars,
    pose_to_matrix,
    sample_to_matrices,
)


# Phase gates -- single source of truth for both `validate` (this module)
# and the calib_web Calibrate-tab dashboard. Tuple shape: (filename, json key,
# threshold in SI units, human label, unit). The unit string drives display
# formatting downstream; the SI threshold is what we actually compare.
GATES: list[tuple] = [
    ("intrinsic.json", "rms_px",            0.5,                "< 0.5 px",        "px"),
    ("handeye.json",   "trans_rmse_m",      0.003,              "< 3 mm trans",    "mm"),
    ("handeye.json",   "rot_rmse_rad",      np.deg2rad(0.5),    "< 0.5 deg rot",   "deg"),
    ("chain.json",     "val_trans_rmse_m",  0.003,              "< 3 mm trans",    "mm"),
    ("chain.json",     "val_rot_rmse_rad",  np.deg2rad(0.4),    "< 0.4 deg rot",   "deg"),
]


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
    n_loaded = len(samples)
    print(f"Loaded {n_loaded} Phase-1 samples")

    # Solver-side per-sample quality gate: drop samples whose stored
    # detection_quality (corner count) or reprojection_rms_px would have
    # failed the per-frame Detection.valid() gate. This catches noisy
    # samples baked into JSONs collected when the per-frame gate was
    # briefly relaxed. Defaults match the strict gate in aruco_detect.
    quality_min_corners = int(getattr(args, "quality_min_corners", 10))
    quality_max_reproj_px = float(getattr(args, "quality_max_reproj_px", 1.5))
    if not bool(getattr(args, "no_quality_gate", False)):
        kept = []
        for s in samples:
            nc = int(s.get("detection_quality", quality_min_corners))
            rms = float(s.get("reprojection_rms_px", 0.0))
            if nc < quality_min_corners or rms > quality_max_reproj_px:
                print(
                    f"  quality-gate reject sample {s.get('label','?')}: "
                    f"corners={nc} (<{quality_min_corners}) or "
                    f"reproj={rms:.2f}px (>{quality_max_reproj_px:.1f})"
                )
            else:
                kept.append(s)
        samples = kept
        if len(samples) < n_loaded:
            print(f"After quality gate: {len(samples)}/{n_loaded} samples")

    # Two-stage outlier handling:
    #
    # 1. Cross-cell consensus pre-pass: provisional Park-Martin solve, then
    #    drop any sample whose implied T_ee_marker rotation deviates from
    #    the provisional median by more than `--prefilter-rot-deg` (default
    #    5°). This catches per-cell IPPE flips at the absolute level, before
    #    they pollute MAD statistics.
    #
    # 2. Iterative MAD-sigma refinement on what remains: continue dropping
    #    the worst-residual sample until either the threshold passes or the
    #    rejection cap kicks in.
    #
    # Knobs:
    #   --prefilter-rot-deg: absolute rot-residual threshold for stage 1.
    #   --max-reject-frac: cap on stage-2 rejections (default 0.25).
    #   --reject-sigma: stage-2 MAD threshold (default 3.0).
    #   --no-reject: skip both stages (e.g. for test fixtures).
    prefilter_rot_deg = float(getattr(args, "prefilter_rot_deg", 5.0))
    max_reject_frac = float(getattr(args, "max_reject_frac", 0.25))
    reject_sigma = float(getattr(args, "reject_sigma", 3.0))
    do_reject = not bool(getattr(args, "no_reject", False))

    if do_reject:
        _, _, _, prefilter_rejected = solve_handeye_with_consensus(
            samples, pre_filter_rot_deg=prefilter_rot_deg,
        )
    else:
        prefilter_rejected = []
    keep_idx = np.array([i for i in range(len(samples)) if i not in set(prefilter_rejected)])
    if len(prefilter_rejected) > 0:
        for gi in prefilter_rejected:
            print(
                f"  pre-filter reject sample #{gi} "
                f"({samples[gi].get('label', '?')}): "
                f"implied marker rot >{prefilter_rot_deg:.1f} deg from cluster"
            )

    rejected_history: list[tuple[int, float, float]] = [(gi, float("nan"), float("nan"))
                                                         for gi in prefilter_rejected]
    while True:
        kept = [samples[i] for i in keep_idx]
        t_ee_marker, t_base_cam_ref, per_pose = solve_handeye(kept)
        trans_errs = np.array([p[0] for p in per_pose])
        rot_errs = np.array([p[1] for p in per_pose])
        if not do_reject:
            break
        combined = np.sqrt(trans_errs ** 2 + rot_errs ** 2)
        med = float(np.median(combined))
        mad = float(np.median(np.abs(combined - med))) + 1e-9
        threshold = med + reject_sigma * 1.4826 * mad
        bad_local = np.where(combined > threshold)[0]
        if len(bad_local) == 0:
            break
        n_dropped = len(samples) - len(keep_idx)
        if n_dropped >= int(max_reject_frac * len(samples)):
            break
        if len(keep_idx) - 1 < 3:
            break
        worst_local = int(np.argmax(combined))
        worst_global = int(keep_idx[worst_local])
        rejected_history.append((
            worst_global,
            float(trans_errs[worst_local]),
            float(rot_errs[worst_local]),
        ))
        keep_idx = np.delete(keep_idx, worst_local)
        print(
            f"  MAD reject sample #{worst_global} ({samples[worst_global].get('label','?')}): "
            f"residual {trans_errs[worst_local]*1000:.1f} mm / "
            f"{np.degrees(rot_errs[worst_local]):.2f} deg (threshold "
            f"{threshold*1000:.1f} mm-equiv)"
        )

    if rejected_history:
        print(f"Rejected {len(rejected_history)}/{len(samples)} samples "
              f"({len(prefilter_rejected)} pre-filter + "
              f"{len(rejected_history) - len(prefilter_rejected)} MAD); "
              f"final solve uses {len(keep_idx)}")

    # Record the pan-tilt park pose used during Phase 1. All samples share
    # the same park pose by construction (Phase 1 parks the head once and
    # then cycles xArm poses), so we average across samples to suppress
    # per-sample servo jitter rather than trusting a single reading.
    park_pan_rad = float(np.mean([float(s["theta_pan_rad"]) for s in samples]))
    park_tilt_rad = float(np.mean([float(s["theta_tilt_rad"]) for s in samples]))

    out_name = getattr(args, "out_name", None) or "handeye.json"
    out_path = Path(args.out) / out_name

    # T_ee_marker is the pose of the marker on the xArm flange — a fixed
    # mechanical attachment. Both `handeye.json` (canonical 45° park) and
    # `handeye_custom.json` (operator-chosen park) describe the SAME physical
    # board, so their solved T_ee_markers must agree within solver noise.
    # When they don't, it almost always means: (a) the board was re-mounted
    # between collects, or (b) one of the two phase-1 sample files is stale
    # (the operator forgot to re-collect after touching the EE/board). Either
    # way, the downstream chain + polish silently produce huge errors.
    # Catch it here, before anyone wastes 8 minutes on a poisoned solve.
    sibling = "handeye_custom.json" if out_name == "handeye.json" else "handeye.json"
    sibling_path = Path(args.out) / sibling
    if sibling_path.is_file() and not getattr(args, "allow_t_ee_marker_mismatch", False):
        try:
            sib = json.loads(sibling_path.read_text())
            em_sib = pose_to_matrix(
                sib["t_ee_marker"]["translation"],
                sib["t_ee_marker"]["rotation"],
            )
        except (KeyError, ValueError, OSError) as exc:
            print(f"  [warn] could not read {sibling_path} for cross-check: {exc}")
        else:
            te, re_ = pose_error_scalars(t_ee_marker, em_sib)
            if te > 0.005 or re_ > np.deg2rad(1.0):
                import datetime as _dt
                sib_mtime = _dt.datetime.fromtimestamp(
                    sibling_path.stat().st_mtime
                ).strftime('%Y-%m-%d %H:%M:%S')
                phase1_path = Path(args.phase1)
                phase1_mtime = _dt.datetime.fromtimestamp(
                    phase1_path.stat().st_mtime
                ).strftime('%Y-%m-%d %H:%M:%S') if phase1_path.is_file() else "?"
                msg = (
                    f"\nT_ee_marker sibling cross-check FAILED.\n"
                    f"  new solve   ({out_name}, from {phase1_path.name} @ {phase1_mtime}):\n"
                    f"    trans={np.round(t_ee_marker[:3,3],4).tolist()}\n"
                    f"  sibling     ({sibling} @ {sib_mtime}):\n"
                    f"    trans={np.round(em_sib[:3,3],4).tolist()}\n"
                    f"  disagreement: {te*1000:.1f} mm trans, {np.degrees(re_):.2f} deg rot\n"
                    f"  (gate: 5 mm / 1 deg)\n\n"
                    f"T_ee_marker is the rigid pose of the marker on the EE flange — both\n"
                    f"handeye solves describe the same physical board, so they must agree.\n"
                    f"Likely causes:\n"
                    f"  • One of the phase-1 sample files is stale (you re-collected one\n"
                    f"    park pose but not the other). Check the mtimes above.\n"
                    f"  • The board was re-mounted on the EE between collects.\n"
                    f"Recovery:\n"
                    f"  • Re-collect BOTH phase-1 datasets in one sitting without\n"
                    f"    touching the board, the EE, or the xArm zero. Re-run handeye.\n"
                    f"  • OR pass --allow-t-ee-marker-mismatch if you genuinely intended\n"
                    f"    to remount (e.g. swapping marker boards for evaluation).\n"
                    f"Refusing to write {out_path}; existing sibling file is untouched.\n"
                )
                print(msg)
                sys.exit(2)

    _save_json(out_path, {
        "t_ee_marker": matrix_to_pose_dict(t_ee_marker),
        "t_base_cam_ref": matrix_to_pose_dict(t_base_cam_ref),
        "phase1_park_pan_rad": park_pan_rad,
        "phase1_park_tilt_rad": park_tilt_rad,
        "n_samples_total": len(samples),
        "n_samples_used": int(len(keep_idx)),
        "rejected_sample_indices": [r[0] for r in rejected_history],
        "per_sample_trans_err_m": trans_errs.tolist(),
        "per_sample_rot_err_rad": rot_errs.tolist(),
        "trans_rmse_m": float(np.sqrt(np.mean(trans_errs ** 2))),
        "rot_rmse_rad": float(np.sqrt(np.mean(rot_errs ** 2))),
    })
    print(
        f"Hand-eye trans RMSE: {np.sqrt(np.mean(trans_errs**2))*1000:.2f} mm  "
        f"rot RMSE: {np.degrees(np.sqrt(np.mean(rot_errs**2))):.3f} deg  "
        f"(on {len(keep_idx)}/{len(samples)} samples)"
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
    # would drop the optimizer into a bad basin. We thread the Phase-1 park
    # angles (firmware radians, recorded in handeye.json) through so the
    # back-solve uses the FK at the actual park pose, not firmware zero.
    park_pan = float(handeye.get("phase1_park_pan_rad", 0.0))
    park_tilt = float(handeye.get("phase1_park_tilt_rad", 0.0))

    # Two-basin warm-start. The pan axis is rotationally periodic: solutions
    # with theta_p_offset = θ and (θ + π) are both kinematically valid because
    # the same FK chain can be expressed by flipping pan sign and rotating T_B
    # by 180° about Z. The handeye solver doesn't pin pan basin (it parks the
    # head and only sees one pan angle), so the URDF-default seed at
    # theta_p_off = 0 may land us in the wrong half of the kinematic torus on
    # hardware where the firmware reports pan with the opposite sign of what
    # the FK assumes. Symptom on bad-basin: locked-T_B chain rot RMSE ~20°.
    # We try both basins (θ_p_off ∈ {0, π}), run a locked-T_B chain fit from
    # each, and pick the lower-residual result. Cheap (two scipy.least_squares
    # calls); robust against firmware sign conventions; no operator action.
    candidates = []
    for basin_label, theta_p_seed in [("basin0", 0.0), ("basinπ", np.pi)]:
        seed_template = PanTiltParams(theta_p_offset=theta_p_seed)
        warm = warm_start_t_b_rotation(
            seed_template, t_base_cam_ref,
            park_pan_rad=park_pan, park_tilt_rad=park_tilt,
        )
        if args.verbose:
            print(
                f"T_B warm start [{basin_label}]: trans={np.round(warm.t_b_trans, 4)}  "
                f"rotvec={np.round(warm.t_b_rotvec, 4)} "
                f"(norm={np.linalg.norm(warm.t_b_rotvec):.3f} rad)"
            )
        params_c, report_c = fit_chain(
            train,
            t_ee_marker=t_ee_marker,
            initial=warm,
            fit_pan_offset=args.fit_pan_offset,
            fit_tb_rotation=args.unlock_tb_rotation,
            loss=args.loss,
        )
        candidates.append((basin_label, warm, params_c, report_c))
        print(
            f"  {basin_label}:  trans_rmse {report_c.trans_rmse_m*1000:.2f} mm  "
            f"rot_rmse {np.degrees(report_c.rot_rmse_rad):.3f} deg"
        )
    # Pick the basin with the lower rot residual. Trans residual alone isn't
    # discriminative — a wrong-basin fit can still place the camera roughly
    # in the right spot (chain has 7 DOF and trans contributes 3); but the
    # rotation residual sits ~20° in the wrong basin and well under 1° in
    # the right one, making it a clean separator.
    best = min(candidates, key=lambda c: c[3].rot_rmse_rad)
    basin_label, initial, params, report = best
    print(f"Chosen basin: {basin_label}")
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
        # Per-sample residuals on the training set — the solver already
        # computes these (OptReport.*_rmse_per_sample). Surfaced so the
        # browser Calibrate tab can render residual histogram + scatter
        # without re-running the fit.
        "per_sample_trans_err_m": report.trans_rmse_per_sample.tolist(),
        "per_sample_rot_err_rad": report.rot_rmse_per_sample.tolist(),
        "fit_pan_offset": args.fit_pan_offset,
        "fit_tb_rotation": args.unlock_tb_rotation,
        "handeye_source": str(Path(args.handeye)),
    })
    print(f"Wrote {out_path}")


def cmd_polish(args):
    # Phase 1 supports concatenation of multiple datasets (e.g. canonical-park
    # phase1_handeye.json + custom-park phase1_handeye_custom.json) so polish can
    # exercise more EE-rotation diversity. Files concatenate in the order given
    # to --phase1; that order is also the index basis for --exclude-indices.
    phase1_paths = [Path(p) for p in args.phase1]
    phase1: list[dict] = []
    for p in phase1_paths:
        phase1 += _load_samples(p)
    phase2 = _load_samples(Path(args.phase2))
    seed = _params_from_dict(json.loads(Path(args.seed).read_text())["params"])

    samples = phase1 + phase2
    n_total = len(samples)
    if len(phase1_paths) > 1:
        print(f"Merged {len(phase1_paths)} phase-1 datasets ({len(phase1)} samples) "
              f"+ phase 2 ({len(phase2)} samples) = {n_total} total")

    # Manual exclusions first. Indices are zero-based into the concatenated
    # `phase1 + phase2` array: the first len(phase1) are phase1, the rest are phase2.
    # This is the operator's escape hatch when they already know which sample is bad
    # (e.g. handeye flagged it).
    exclude_set = set(int(i) for i in (args.exclude_indices or []))
    bad = [i for i in exclude_set if i < 0 or i >= n_total]
    if bad:
        raise ValueError(
            f"--exclude-indices values out of range [0,{n_total}): {sorted(bad)}"
        )
    rejected_manual = sorted(exclude_set)
    if rejected_manual:
        for gi in rejected_manual:
            print(
                f"  manual reject sample #{gi} "
                f"({samples[gi].get('label','?')})"
            )

    keep_idx = np.array(
        [i for i in range(n_total) if i not in exclude_set], dtype=int
    )

    # Iterative MAD-sigma refinement. Mirrors the handeye loop above: solve,
    # find the worst-residual sample, drop it if it exceeds the MAD threshold,
    # re-solve. Stops when no sample is above threshold, the rejection cap is
    # hit, or we'd drop below the safety floor for a joint fit.
    reject_sigma = float(getattr(args, "reject_sigma", 3.0))
    max_reject_frac = float(getattr(args, "max_reject_frac", 0.10))
    do_reject = not bool(getattr(args, "no_reject", False))
    rejected_auto: list[dict] = []
    params = seed
    report = None
    while True:
        kept = [samples[i] for i in keep_idx]
        params, report = fit_joint(
            kept,
            initial=seed,
            fit_tb_rotation=args.unlock_tb_rotation,
            fit_pan_offset=args.fit_pan_offset,
            loss=args.loss,
        )
        if not do_reject:
            break
        trans_e = np.asarray(report.trans_rmse_per_sample)
        rot_e = np.asarray(report.rot_rmse_per_sample)
        combined = np.sqrt(trans_e ** 2 + rot_e ** 2)
        med = float(np.median(combined))
        mad = float(np.median(np.abs(combined - med))) + 1e-9
        threshold = med + reject_sigma * 1.4826 * mad
        worst_local = int(np.argmax(combined))
        if combined[worst_local] <= threshold:
            break
        n_dropped_total = len(rejected_auto) + len(rejected_manual)
        if n_dropped_total >= int(max_reject_frac * n_total):
            print(f"  MAD reject cap reached ({n_dropped_total}/{n_total}); stopping")
            break
        if len(keep_idx) - 1 < 8:
            print(f"  MAD reject floor reached (kept={len(keep_idx)-1} < 8); stopping")
            break
        worst_global = int(keep_idx[worst_local])
        rejected_auto.append({
            "index": worst_global,
            "label": samples[worst_global].get("label", "?"),
            "trans_err_m": float(trans_e[worst_local]),
            "rot_err_rad": float(rot_e[worst_local]),
        })
        keep_idx = np.delete(keep_idx, worst_local)
        print(
            f"  MAD reject sample #{worst_global} "
            f"({samples[worst_global].get('label','?')}): residual "
            f"{trans_e[worst_local]*1000:.1f} mm / "
            f"{np.degrees(rot_e[worst_local]):.2f} deg "
            f"(threshold {threshold*1000:.1f} mm-equiv)"
        )

    print("POLISH:", report.summary())
    if rejected_manual or rejected_auto:
        print(
            f"Polish dropped {len(rejected_manual) + len(rejected_auto)}/{n_total} "
            f"samples ({len(rejected_manual)} manual + {len(rejected_auto)} MAD); "
            f"final solve uses {len(keep_idx)}"
        )

    out_path = Path(args.out) / "polish.json"
    _save_json(out_path, {
        "params": _params_to_dict(params),
        "trans_rmse_m": report.trans_rmse_m,
        "rot_rmse_rad": report.rot_rmse_rad,
        "n_samples_total": n_total,
        "n_samples_used": int(len(keep_idx)),
        "kept_indices": keep_idx.tolist(),
        "rejected_indices_manual": rejected_manual,
        "rejected_indices_auto": rejected_auto,
        "per_sample_trans_err_m": report.trans_rmse_per_sample.tolist(),
        "per_sample_rot_err_rad": report.rot_rmse_per_sample.tolist(),
        "fit_tb_rotation": args.unlock_tb_rotation,
        "fit_pan_offset": args.fit_pan_offset,
        "reject_sigma": reject_sigma,
        "max_reject_frac": max_reject_frac,
        "phase1_sources": [str(p) for p in phase1_paths],
        "phase2_source": str(args.phase2),
    })
    print(f"Wrote {out_path}")


def cmd_gates(args):
    """Print PASS/FAIL summary for each phase result against the static GATES."""
    results_dir = Path(args.results_dir)
    for fname, key, thresh, label, unit in GATES:
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
        disp = val * 1000 if unit == "mm" else (np.degrees(val) if unit == "deg" else val)
        print(f"  [{flag}] {fname}:{key} = {disp:.3f} {unit}  ({label})")


def cmd_validate(args):
    """Phase-4 end-to-end validation.

    The ChArUco board was held stationary in `base_link` (mounted to a
    tripod / wall / fixture — anywhere fixed, not on the EE) while the
    pan-tilt swept N (pan, tilt) views. Compose `T_base_marker_pred_i =
    forward_kinematics(theta_p_i, theta_t_i, params) @ T_cam_marker_i` for
    each view; if the calibration is correct, all views project the marker
    to the same base-frame pose. Spread across views = end-to-end error.
    No xArm or T_ee_marker assumption is involved.
    """
    phase4 = json.loads(Path(args.phase4).read_text())
    samples = phase4.get("samples", [])
    n_total = len(samples)
    if n_total < 3:
        raise SystemExit(
            f"validate: need >=3 samples in phase4_validation, got {n_total}"
        )

    params_blob = json.loads(Path(args.params).read_text())
    params = _params_from_dict(params_blob["params"])

    # Compose per-sample base-frame marker predictions through the FK chain
    # under test. `t_cam_marker_body` is already in body coords (post
    # optical_to_body conversion at collect time), so no extra rotation here.
    pred_T: list[np.ndarray] = []
    pan_arr: list[float] = []
    tilt_arr: list[float] = []
    for s in samples:
        theta_p = float(s["theta_pan_rad"])
        theta_t = float(s["theta_tilt_rad"])
        t_cam_marker = pose_to_matrix(
            s["t_cam_marker_body"]["translation"],
            s["t_cam_marker_body"]["rotation"],
        )
        t_base_cam = forward_kinematics(theta_p, theta_t, params)
        pred_T.append(t_base_cam @ t_cam_marker)
        pan_arr.append(theta_p)
        tilt_arr.append(theta_t)

    # Self-consistency centroid. Translation is arithmetic mean; rotation is
    # the chordal mean over the per-sample rotation matrices (same primitive
    # the chain solver uses elsewhere). The choice doesn't matter much when
    # views agree closely (the regime that gates PASS); it just keeps the
    # residual definition well-posed when they don't.
    trans_arr = np.array([T[:3, 3] for T in pred_T])
    rot_arr = np.array([T[:3, :3] for T in pred_T])
    centroid_trans = trans_arr.mean(axis=0)
    centroid_rot = Rotation.from_matrix(rot_arr).mean().as_matrix()
    centroid_T = np.eye(4)
    centroid_T[:3, :3] = centroid_rot
    centroid_T[:3, 3] = centroid_trans

    trans_errs = np.zeros(n_total)
    rot_errs = np.zeros(n_total)
    for i, T in enumerate(pred_T):
        te, re = pose_error_scalars(T, centroid_T)
        trans_errs[i] = te
        rot_errs[i] = re

    trans_rmse_self = float(np.sqrt(np.mean(trans_errs ** 2)))
    rot_rmse_self = float(np.sqrt(np.mean(rot_errs ** 2)))
    trans_max_self = float(np.max(trans_errs))
    rot_max_self = float(np.max(rot_errs))
    # Per-axis std on translation: an operator can read "Z dominates" → look
    # at T_B Y rotation / theta_t_offset, "X dominates" → look at theta_p.
    trans_std_xyz = trans_arr.std(axis=0).tolist()

    # Verdict on self-consistency: spread of T_base_marker_pred across views.
    trans_pass = args.trans_pass_mm * 1e-3
    rot_pass = np.deg2rad(args.rot_pass_deg)
    trans_warn = args.trans_warn_mm * 1e-3
    rot_warn = np.deg2rad(args.rot_warn_deg)
    if trans_rmse_self <= trans_pass and rot_rmse_self <= rot_pass:
        verdict = "PASS"
    elif trans_rmse_self <= trans_warn and rot_rmse_self <= rot_warn:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    out_path = Path(args.out) / args.out_name
    payload = {
        "phase": "validation",
        "params_source": str(args.params),
        "phase4_source": str(args.phase4),
        "n_samples_total": n_total,
        "n_samples_used": n_total,
        "self_consistency": {
            "trans_rmse_m": trans_rmse_self,
            "rot_rmse_rad": rot_rmse_self,
            "trans_max_m": trans_max_self,
            "rot_max_rad": rot_max_self,
            "trans_std_xyz_m": trans_std_xyz,
            "centroid": matrix_to_pose_dict(centroid_T),
        },
        "per_sample": [
            {
                "i": i,
                "pan_rad": pan_arr[i],
                "tilt_rad": tilt_arr[i],
                "trans_err_m": float(trans_errs[i]),
                "rot_err_rad": float(rot_errs[i]),
                "T_base_marker_pred": matrix_to_pose_dict(pred_T[i]),
            }
            for i in range(n_total)
        ],
        "verdict": verdict,
        "thresholds": {
            "trans_pass_mm": args.trans_pass_mm,
            "rot_pass_deg": args.rot_pass_deg,
            "trans_warn_mm": args.trans_warn_mm,
            "rot_warn_deg": args.rot_warn_deg,
        },
    }
    _save_json(out_path, payload)

    print(
        f"VALIDATE: {verdict}  n={n_total}  "
        f"trans_rmse={trans_rmse_self*1000:.2f}mm "
        f"rot_rmse={np.degrees(rot_rmse_self):.3f}deg  "
        f"trans_max={trans_max_self*1000:.2f}mm "
        f"rot_max={np.degrees(rot_max_self):.3f}deg"
    )
    print(f"Wrote {out_path}")


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
    ph.add_argument("--prefilter-rot-deg", type=float, default=5.0,
                    help="Absolute rot-residual threshold (deg) for the cross-cell "
                         "consensus pre-pass (default 5.0). Drops cells whose implied "
                         "T_ee_marker is geometrically inconsistent with the bulk before "
                         "the MAD-sigma loop runs.")
    ph.add_argument("--max-reject-frac", type=float, default=0.25,
                    help="Cap on fraction of samples that may be auto-rejected by the "
                         "MAD-sigma stage (default 0.25). Pre-filter rejections are not "
                         "counted toward this cap.")
    ph.add_argument("--reject-sigma", type=float, default=3.0,
                    help="MAD-sigma threshold above which a sample is treated as an outlier (default 3.0).")
    ph.add_argument("--no-reject", action="store_true",
                    help="Disable both the consensus pre-filter and iterative outlier rejection.")
    ph.add_argument("--out-name", default=None,
                    help="Output filename within --out (default: handeye.json). "
                         "Use 'handeye_custom.json' when solving the operator's "
                         "custom-park dataset so the canonical handeye.json is preserved.")
    ph.add_argument("--quality-min-corners", type=int, default=10,
                    help="Drop samples with fewer than N detected corners (default 10).")
    ph.add_argument("--quality-max-reproj-px", type=float, default=1.5,
                    help="Drop samples with per-frame reprojection RMS above N px (default 1.5).")
    ph.add_argument("--no-quality-gate", action="store_true",
                    help="Solve on every sample regardless of stored detection_quality / reproj.")
    ph.add_argument("--allow-t-ee-marker-mismatch", action="store_true",
                    help="Skip the T_ee_marker cross-check against the sibling handeye solve "
                         "(handeye.json ↔ handeye_custom.json). Use only when intentionally "
                         "re-mounting the board between collects (e.g. evaluating a different "
                         "marker board) — otherwise this gate catches stale-phase-1-file mistakes.")
    ph.set_defaults(func=cmd_handeye)

    pc = sub.add_parser("chain", help="Phase 2: pan-tilt chain fit.")
    pc.add_argument("phase2")
    pc.add_argument("--handeye", required=True)
    pc.add_argument("--out", default="results")
    pc.add_argument("--fit-pan-offset", action="store_true")
    pc.add_argument("--unlock-tb-rotation", action="store_true",
                    help="Fit T_B rotation in the chain phase (default: lock at warm-start). "
                         "T_B rotation about the tilt-Y axis is degenerate with theta_t_offset "
                         "during Phase-2-only fitting; the warm-start from Phase-1 anchors it. "
                         "Unlock only for debugging or comparison runs; the joint polish phase "
                         "is the right place to refine T_B rotation against Phase-1 data.")
    pc.add_argument("--loss", default="soft_l1")
    pc.add_argument("--val-seed", type=int, default=0)
    pc.add_argument("--verbose", action="store_true")
    pc.set_defaults(func=cmd_chain)

    pp = sub.add_parser("polish", help="Phase 3: joint refinement.")
    pp.add_argument("--phase1", nargs="+", required=True,
                    help="One or more phase-1 sample JSONs to concatenate. Use multiple "
                         "to merge datasets collected at different park poses (e.g. "
                         "phase1_handeye.json + phase1_handeye_custom.json) — extra "
                         "EE-rotation diversity helps break the T_B(Y) ↔ theta_t_offset "
                         "degeneracy when --unlock-tb-rotation is also set.")
    pp.add_argument("--phase2", required=True,
                    help="Phase-2 sample JSON (single file).")
    pp.add_argument("--seed", required=True)
    pp.add_argument("--out", default="results")
    pp.add_argument("--unlock-tb-rotation", action="store_true")
    pp.add_argument("--fit-pan-offset", action="store_true")
    pp.add_argument("--loss", default="soft_l1")
    pp.add_argument("--exclude-indices", type=int, nargs="+", default=[],
                    help="Manual indices into the concatenated (phase1 + phase2) sample array "
                         "to drop before fitting. Indices [0..len(phase1)-1] are phase1; the "
                         "rest are phase2. Use this when handeye already flagged a sample as "
                         "an outlier (it does not propagate to polish automatically).")
    pp.add_argument("--reject-sigma", type=float, default=3.0,
                    help="MAD-sigma threshold for the iterative auto-rejection loop (default 3.0).")
    pp.add_argument("--max-reject-frac", type=float, default=0.10,
                    help="Cap on fraction of samples (manual + auto) that may be dropped "
                         "(default 0.10). Once this is hit the loop stops even if the MAD "
                         "threshold still flags samples.")
    pp.add_argument("--no-reject", action="store_true",
                    help="Disable the iterative MAD-sigma rejection. --exclude-indices still applies.")
    pp.set_defaults(func=cmd_polish)

    pg = sub.add_parser("gates", help="Print PASS/FAIL summary for static phase gates.")
    pg.add_argument("results_dir")
    pg.set_defaults(func=cmd_gates)

    pv = sub.add_parser(
        "validate",
        help="Phase 4: end-to-end pan-tilt sweep against a stationary "
             "in-base_link board (xArm-independent).",
    )
    pv.add_argument("--phase4", required=True,
                    help="phase4_validation.json (board fixed in base_link).")
    pv.add_argument("--params", required=True,
                    help="polish.json (preferred) or chain.json — params under test.")
    pv.add_argument("--out", default="results",
                    help="Session dir; writes <out>/<out-name>.")
    pv.add_argument("--out-name", default="validation.json")
    pv.add_argument("--trans-pass-mm", type=float, default=5.0)
    pv.add_argument("--rot-pass-deg", type=float, default=0.5)
    pv.add_argument("--trans-warn-mm", type=float, default=10.0)
    pv.add_argument("--rot-warn-deg", type=float, default=1.0)
    pv.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
