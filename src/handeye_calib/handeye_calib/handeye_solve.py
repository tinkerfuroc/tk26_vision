"""Eye-in-hand solver: multi-method seed -> bundle-adjust refine -> held-out evaluation."""
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.optimize import least_squares

from handeye_calib import transforms as tf
from handeye_calib import handeye_model as hm

_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _reproj_rms(X, Tbb, samples, K, dist, board_pts, *, per_sample=False):
    """Reprojection RMS over all corners across ``samples``.

    When ``per_sample`` is ``True``, also returns a list[float] of per-sample
    RMS values (one per element of ``samples``, in order) so the caller can
    surface per-frame residuals without re-doing the projection math. Empty
    ``samples`` returns ``(0.0, [])`` in that path to keep the contract total.
    """
    sq = []
    per_sample_vals = []
    for s in samples:
        T_cam_board = tf.invert(s.T_base_eef @ X) @ Tbb
        pred = hm.project_corners(board_pts[s.corner_idx], T_cam_board, K, dist)
        diff_sq = np.sum((pred - s.obs_px) ** 2, axis=1)
        sq.append(diff_sq)
        if per_sample:
            per_sample_vals.append(float(np.sqrt(np.mean(diff_sq))) if diff_sq.size else 0.0)
    if per_sample:
        agg = float(np.sqrt(np.mean(np.concatenate(sq)))) if sq else 0.0
        return agg, per_sample_vals
    return float(np.sqrt(np.mean(np.concatenate(sq))))


def _estimate_board_in_base(X, samples):
    return tf.se3_average([s.T_base_eef @ X @ s.T_cam_board for s in samples])


def seed_handeye(samples, K, dist, board_pts, *, methods=None):
    """Run hand-eye methods, return the X with lowest reprojection RMS.

    ``methods`` (optional): a dict[str, int] of {name: cv2.CALIB_HAND_EYE_*}
    subset to run. ``None`` (default) preserves the historical behaviour of
    running every entry in :data:`_METHODS` (all five OpenCV methods). Pass a
    single-key dict to restrict to one specific method — the rest of the
    pipeline (best-of, bundle-adjust, gate) treats that one method as the
    sole candidate.
    """
    use_methods = _METHODS if methods is None else methods
    R_g2b = [np.asarray(s.T_base_eef)[:3, :3] for s in samples]
    t_g2b = [np.asarray(s.T_base_eef)[:3, 3] for s in samples]
    R_t2c = [np.asarray(s.T_cam_board)[:3, :3] for s in samples]
    t_t2c = [np.asarray(s.T_cam_board)[:3, 3] for s in samples]
    per_method = []
    for name, flag in use_methods.items():
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=flag)
        except cv2.error:
            continue
        X = tf.T_from_Rt(R_c2g, t_c2g.reshape(3))
        Tbb = _estimate_board_in_base(X, samples)
        per_method.append({"name": name, "X": X, "Tbb": Tbb,
                           "reproj_px": _reproj_rms(X, Tbb, samples, K, dist, board_pts)})
    if not per_method:
        raise RuntimeError("all calibrateHandEye methods failed")
    best = min(per_method, key=lambda m: m["reproj_px"])
    return best["X"], best["Tbb"], per_method


def seed_from_board_anchor(samples, anchor_Tbb):
    """Closed-form warm-start ``X = T_eef_cam`` from a KNOWN board pose in base.

    The board pose in the arm-base frame (``anchor_Tbb`` = T_base_board) is
    measured by an EXTERNAL, already-calibrated sensor (the pan-tilt head
    Orbbec, composed through TF into the arm base frame). Each sample then
    closes the kinematic loop directly::

        A_i @ X @ B_i = Tbb   =>   X_i = inv(A_i) @ Tbb @ inv(B_i)

    with A_i = T_base_eef (FK) and B_i = T_cam_board (wrist PnP). Unlike AX=XB
    this needs NO rotation diversity — a single pose determines X — so it is a
    basin-immune seed for the bundle adjust. Returns ``Tbb_seed = anchor_Tbb``;
    the bundle adjust keeps Tbb FREE, so the head's absolute bias is used only
    to choose the convergence basin and is NOT injected into the final X.
    """
    Xs = []
    for s in samples:
        A = np.asarray(s.T_base_eef, float)
        B = np.asarray(s.T_cam_board, float)
        Xs.append(tf.invert(A) @ np.asarray(anchor_Tbb, float) @ tf.invert(B))
    return tf.se3_average(Xs), np.asarray(anchor_Tbb, float)


def average_board_anchors(anchors):
    """SE(3)-average a list of board-in-base measurements and report scatter.

    ``anchors`` is a list of 4x4 ``T_base_board`` observations (e.g. the head
    at several pan/tilt poses). Returns ``(Tbb_mean, scatter)`` where scatter is
    ``{"trans_mm", "rot_deg", "n"}`` — the RMS deviation of the observations
    from their mean, a data-driven confidence readout (large scatter => the
    anchor is unreliable; widen the prior / re-check the head TF).
    """
    Ts = [np.asarray(T, float) for T in anchors]
    if not Ts:
        raise ValueError("average_board_anchors needs >=1 anchor")
    mean = tf.se3_average(Ts)
    inv_mean = tf.invert(mean)
    t_dev, r_dev = [], []
    for T in Ts:
        D = inv_mean @ T
        t_dev.append(float(np.linalg.norm(D[:3, 3])))
        r_dev.append(np.radians(tf.rotation_angle_deg(np.eye(3), D[:3, :3])))
    scatter = {
        "trans_mm": float(np.sqrt(np.mean(np.square(t_dev))) * 1000.0),
        "rot_deg": float(np.degrees(np.sqrt(np.mean(np.square(r_dev))))),
        "n": len(Ts),
    }
    return mean, scatter


def _residuals(params, samples, K, dist, board_pts,
               depth_weight=0.0, depth_sigma_m=0.005):
    """Stacked residual: per-corner pixel reprojection, plus (when enabled) a
    metric 3D point residual against FFS-measured camera-frame corner points.

    The depth block ``(P_pred - P_meas) * depth_weight / depth_sigma_m`` is in
    units of "sigmas of depth error", so with ``depth_weight=1`` a corner whose
    predicted position is one ``depth_sigma_m`` off contributes like a 1 px
    reprojection error. The pixel block is left unscaled so ``depth_weight=0``
    reproduces the original monocular residual byte-for-byte (and is skipped
    entirely for samples without ``obs_xyz_cam``, i.e. graceful fallback when
    FFS was unavailable at capture). Reprojection pins rotation + lateral
    translation; the depth block pins the optical-axis DOF planar PnP can't see.
    """
    X = tf.T_from_vec(params[:6])
    Tbb = tf.T_from_vec(params[6:])
    res = []
    for s in samples:
        T_cam_board = tf.invert(s.T_base_eef @ X) @ Tbb
        bp = board_pts[s.corner_idx]
        pred = hm.project_corners(bp, T_cam_board, K, dist)
        res.append((pred - s.obs_px).ravel())
        if depth_weight > 0.0 and getattr(s, "obs_xyz_cam", None) is not None:
            meas = np.asarray(s.obs_xyz_cam, float)
            valid = (np.asarray(s.obs_xyz_valid, bool)
                     if getattr(s, "obs_xyz_valid", None) is not None
                     else np.ones(len(meas), bool))
            # deproject_corners writes NaN rows for holes ("never a fake 0"); AND
            # the mask with finiteness so a NaN can never reach least_squares
            # (which rejects a non-finite initial residual) even if the mask was
            # dropped in a round-trip — keeps the NaN contract self-consistent.
            valid = valid & np.isfinite(meas).all(axis=1)
            if valid.any():
                P_pred = (T_cam_board[:3, :3] @ bp.T).T + T_cam_board[:3, 3]
                d = (P_pred[valid] - meas[valid]) * (depth_weight / depth_sigma_m)
                res.append(d.ravel())
    return np.concatenate(res)


def bundle_adjust(samples, K, dist, board_pts, X0, Tbb0,
                  depth_weight=0.0, depth_sigma_m=0.005):
    """Jointly refine X (T_eef_cam) and Tbb (T_base_board) minimizing corner
    reprojection, plus the optional FFS-depth 3D residual (``depth_weight>0``).

    ``depth_weight`` defaults to ``0.0`` so existing direct callers (and the
    pixel-only synthetic tests) get the unchanged monocular solve; the higher-
    level :func:`solve` passes a non-zero default which no-ops anyway on samples
    that carry no depth.
    """
    p0 = np.concatenate([tf.vec_from_T(X0), tf.vec_from_T(Tbb0)])
    sol = least_squares(_residuals, p0, loss="soft_l1", method="trf",
                        args=(samples, K, dist, board_pts,
                              depth_weight, depth_sigma_m))
    X = tf.T_from_vec(sol.x[:6])
    Tbb = tf.T_from_vec(sol.x[6:])
    info = {"final_reproj_px": _reproj_rms(X, Tbb, samples, K, dist, board_pts),
            "success": bool(sol.success), "cost": float(sol.cost)}
    return X, Tbb, info


@dataclass
class SolveResult:
    X: np.ndarray
    Tbb: np.ndarray
    train_metrics: dict
    heldout_metrics: dict
    status: str
    per_method: list


# pan-tilt parity thresholds
_PASS = {"trans_rmse_m": 0.003, "rot_rmse_rad": 0.00873, "reproj_px": 1.5}
_WARN = {"trans_rmse_m": 0.006, "rot_rmse_rad": 0.01745, "reproj_px": 3.0}


def split_train_test(samples, heldout_frac, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_te = max(1, int(round(len(samples) * heldout_frac)))
    te = sorted(idx[:n_te].tolist())
    tr = sorted(idx[n_te:].tolist())
    return [samples[i] for i in tr], [samples[i] for i in te]


def _depth_point_metrics(X, Tbb, samples, board_pts):
    """Depth-grounded accuracy: RMS distance (mm) between the solved chain's
    predicted camera-frame corner points and the FFS-measured points, over all
    valid corners. Unlike ``trans_rmse_m`` (which compares against the monocular
    PnP pose — itself biased when intrinsics/board-scale are off), this compares
    against an independent *metric* measurement.

    Honest only on the HELD-OUT split: those poses never entered the bundle
    adjust, so the metric genuinely cross-validates the depth-grounded solve.
    The TRAIN value is in-sample (X/Tbb were fit to these same points with
    ``depth_weight>0``), so it can read deceptively low if the depth term
    over-fits a systematic FFS bias — read the held-out value as the real-world
    error budget. Returns ``(rmse_mm_or_None, n_corners)``."""
    sq = []
    for s in samples:
        if getattr(s, "obs_xyz_cam", None) is None:
            continue
        meas = np.asarray(s.obs_xyz_cam, float)
        valid = (np.asarray(s.obs_xyz_valid, bool)
                 if getattr(s, "obs_xyz_valid", None) is not None
                 else np.ones(len(meas), bool))
        valid = valid & np.isfinite(meas).all(axis=1)  # NaN holes never count
        if not valid.any():
            continue
        T_cam_board = tf.invert(s.T_base_eef @ X) @ Tbb
        bp = board_pts[s.corner_idx]
        P_pred = (T_cam_board[:3, :3] @ bp.T).T + T_cam_board[:3, 3]
        sq.append(np.sum((P_pred[valid] - meas[valid]) ** 2, axis=1))
    if not sq:
        return None, 0
    allsq = np.concatenate(sq)
    return float(np.sqrt(np.mean(allsq)) * 1000.0), int(allsq.size)


def evaluate(X, Tbb, samples, K, dist, board_pts):
    trans_e, rot_e = [], []
    for s in samples:
        T_pred = tf.invert(s.T_base_eef @ X) @ Tbb     # predicted board-in-cam
        T_obs = s.T_cam_board                           # observed (PnP)
        trans_e.append(np.linalg.norm(T_pred[:3, 3] - T_obs[:3, 3]))
        rot_e.append(np.radians(tf.rotation_angle_deg(T_pred[:3, :3], T_obs[:3, :3])))
    out = {"trans_rmse_m": float(np.sqrt(np.mean(np.square(trans_e)))),
           "rot_rmse_rad": float(np.sqrt(np.mean(np.square(rot_e)))),
           "reproj_px": _reproj_rms(X, Tbb, samples, K, dist, board_pts)}
    depth_rmse_mm, n_depth = _depth_point_metrics(X, Tbb, samples, board_pts)
    if depth_rmse_mm is not None:
        out["depth_point_rmse_mm"] = depth_rmse_mm
        out["n_depth_corners"] = n_depth
    return out


def gate(metrics):
    def ok(th):
        return all(metrics[k] <= th[k] for k in th)
    if ok(_PASS):
        return "PASS"
    if ok(_WARN):
        return "WARN"
    return "FAIL"


def solve(samples, K, dist, board_pts, heldout_frac=0.2, rng_seed=0, *,
          methods=None, reject_sigma=None, max_reject_frac=0.5,
          depth_weight=1.0, depth_sigma_m=0.005):
    """Full solve pipeline. ``methods`` is forwarded to :func:`seed_handeye`
    so a single-method run (e.g. ``methods={"TSAI": cv2.CALIB_HAND_EYE_TSAI}``)
    skips the multi-method best-of step; ``None`` keeps the default 5-method
    sweep.

    ``reject_sigma``: when set, iteratively drop samples whose post-BA
    per-sample reprojection RMS exceeds ``reject_sigma × median`` and re-solve.
    Loop terminates when no further outliers exceed the threshold OR when
    cumulative drops reach ``max_reject_frac`` (default 50%). Mirrors the
    polish-phase rejection in pan_tilt's calibration. Use when the operator's
    physical setup yields a bimodal per-sample reproj (some good, some bad)
    — the rejection isolates the consistent-data subset so the calibration
    is usable even before the mechanical disturbance is found and fixed.
    ``rejected_indices`` (indices into the original ``samples``) is attached
    to the returned ``SolveResult.per_method[-1]`` for operator visibility.
    """
    train, test = split_train_test(samples, heldout_frac, rng_seed)
    X0, Tbb0, per_method = seed_handeye(train, K, dist, board_pts, methods=methods)
    X, Tbb, _ = bundle_adjust(train, K, dist, board_pts, X0, Tbb0,
                              depth_weight=depth_weight, depth_sigma_m=depth_sigma_m)

    rejected = []
    if reject_sigma is not None and len(train) >= 6:
        # Iterative MAD-based outlier rejection on the TRAIN set only —
        # held-out stays intact as the honest evaluator.
        #
        # Threshold: median + sigma * MAD (one-sided, since reproj is
        # bounded below by zero — only the high tail is "outlier"). MAD
        # is robust to existing outliers (unlike stdev), so the metric
        # stays stable as the rejection loop iterates.
        active = list(range(len(train)))
        n_orig = len(train)
        for _ in range(20):  # safety cap on iterations
            _, per_sample = _reproj_rms(X, Tbb, [train[i] for i in active],
                                        K, dist, board_pts, per_sample=True)
            arr = np.asarray(per_sample, float)
            if arr.size < 6:
                break
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            # Floor MAD at a tiny value so a perfectly-clean active set
            # doesn't divide-by-zero on the next-round threshold.
            threshold = med + float(reject_sigma) * max(mad, 1e-6)
            keep_mask = arr <= threshold
            n_drop = int((~keep_mask).sum())
            cumulative_drop = n_orig - int(keep_mask.sum())
            if n_drop == 0:
                break
            if cumulative_drop > int(max_reject_frac * n_orig):
                break
            new_active = [active[i] for i, k in enumerate(keep_mask) if k]
            if len(new_active) == len(active):
                break
            rejected.extend([active[i] for i, k in enumerate(keep_mask) if not k])
            active = new_active
            sub = [train[i] for i in active]
            X0r, Tbb0r, _ = seed_handeye(sub, K, dist, board_pts, methods=methods)
            X, Tbb, _ = bundle_adjust(sub, K, dist, board_pts, X0r, Tbb0r,
                                      depth_weight=depth_weight,
                                      depth_sigma_m=depth_sigma_m)
        train = [train[i] for i in active]

    train_m = evaluate(X, Tbb, train, K, dist, board_pts)
    held_m = evaluate(X, Tbb, test, K, dist, board_pts)
    if rejected:
        per_method = list(per_method) + [{"name": "rejected_indices",
                                          "rejected_indices": sorted(rejected),
                                          "X": X, "Tbb": Tbb,
                                          "reproj_px": float("nan")}]
    return SolveResult(X, Tbb, train_m, held_m, gate(held_m), per_method)
