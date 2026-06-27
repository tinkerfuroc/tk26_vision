"""Eye-in-hand solver: multi-method seed -> bundle-adjust refine -> held-out evaluation."""
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as _R

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


def _per_sample_chain_errors(X, Tbb, samples):
    """Per-sample SE(3) chain error against the wrist PnP observation.

    For each sample computes:
        T_pred = inv(T_base_eef @ X) @ Tbb   (predicted board-in-cam)
    then measures the SE(3) deviation from the observed T_cam_board.  Returns
    ``(trans_m, rot_rad)`` as float arrays of length ``len(samples)``.  This is
    the same metric as :func:`evaluate` but without the RMS aggregation, so the
    caller can score outliers per sample independently on each axis.
    """
    trans_e, rot_e = [], []
    for s in samples:
        T_pred = tf.invert(s.T_base_eef @ X) @ Tbb
        T_obs = s.T_cam_board
        trans_e.append(float(np.linalg.norm(T_pred[:3, 3] - T_obs[:3, 3])))
        rot_e.append(float(np.radians(tf.rotation_angle_deg(T_pred[:3, :3],
                                                             T_obs[:3, :3]))))
    return np.asarray(trans_e, float), np.asarray(rot_e, float)


def _modified_zscores(arr):
    """Modified z-score: 0.6745 * (x − median(x)) / max(MAD(x), 1e-6).

    Robust to existing outliers (unlike standard z-score which uses the mean
    and stdev — both of which are pulled by extreme values).  For Gaussian
    data, 0.6745 × MAD ≈ stdev, so z-scores are comparable across methods.
    One-sided upper-tail rejection (z > threshold) is the intended use.

    The floor ``1e-6`` is a practical calibration floor (1 µm / 1 µrad)
    rather than a purely numerical epsilon: when all residuals cluster at the
    sub-micron / sub-microradian level — as happens on noiseless synthetic
    data — the MAD is driven to zero by floating-point coincidences, and a
    pure numeric floor (e.g. 1e-9) would amplify those coincidences into
    spurious z-score spikes that reject perfectly good samples.  At 1e-6 the
    floor is still far below the MAD of any realistic noisy-data chain-error
    distribution (typical noisy MAD: 1e-4 – 1e-3 m / rad), so it does not
    blunt detection of genuine physical outliers.
    """
    arr = np.asarray(arr, float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return 0.6745 * (arr - med) / max(mad, 1e-6)


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


def _solve_once(samples, K, dist, board_pts, *, methods=None,
                depth_weight=0.0, depth_sigma_m=0.005, anchor_Tbb=None):
    """Multi-start seed -> bundle-adjust; return (X, Tbb, per_method, seed_used).

    Candidate seeds: the best-of-5 closed-form ``calibrateHandEye`` seed, plus
    (when ``anchor_Tbb`` is given) the basin-immune board-anchor seed. Each is
    bundle-adjusted; the converged result with the lowest reprojection RMS wins.
    On degenerate (low-rotation) sets where calibrateHandEye returns a poor or
    flipped seed, the anchor branch rescues the solve.
    """
    X0, Tbb0, per_method = seed_handeye(samples, K, dist, board_pts, methods=methods)
    candidates = [("closed_form", X0, Tbb0)]
    if anchor_Tbb is not None:
        Xa, Tba = seed_from_board_anchor(samples, anchor_Tbb)
        candidates.append(("board_anchor", Xa, Tba))
    best = None
    for name, Xs, Tbs in candidates:
        X, Tbb, info = bundle_adjust(samples, K, dist, board_pts, Xs, Tbs,
                                     depth_weight=depth_weight,
                                     depth_sigma_m=depth_sigma_m)
        reproj = info["final_reproj_px"]
        if best is None or reproj < best[3]:
            best = (X, Tbb, name, reproj)
    return best[0], best[1], per_method, best[2]


@dataclass
class SolveResult:
    X: np.ndarray
    Tbb: np.ndarray
    train_metrics: dict
    heldout_metrics: dict
    status: str
    per_method: list
    # Which seed produced the promoted X: "closed_form" (best-of-5
    # calibrateHandEye) or "board_anchor" (the head warm-start). Surfaced so an
    # operator can confirm whether the head anchor actually rescued the solve.
    seed_used: str = ""


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
          methods=None, reject_sigma=2.5, max_reject_frac=0.25,
          reject_min_trans_m=0.01, reject_min_rot_rad=np.radians(3.0),
          depth_weight=1.0, depth_sigma_m=0.005, anchor_Tbb=None):
    """Full solve pipeline. ``methods`` is forwarded to :func:`seed_handeye`
    so a single-method run (e.g. ``methods={"TSAI": cv2.CALIB_HAND_EYE_TSAI}``)
    skips the multi-method best-of step; ``None`` keeps the default 5-method
    sweep.

    ``anchor_Tbb``: an external board-in-base measurement (e.g. from the
    pan-tilt head via TF) used as a basin-immune warm-start seed; forwarded
    through the rejection loop via :func:`_solve_once` so the anchor is
    available on every re-solve. ``None`` (default) skips the anchor branch.

    ``reject_sigma``: default-on (2.5) iterative per-axis outlier rejection
    on the TRAIN split only — held-out is never touched.  Each iteration
    computes per-sample SE(3) chain errors (translation in metres, rotation
    in radians) via :func:`_per_sample_chain_errors`, turns each axis into
    one-sided modified z-scores via :func:`_modified_zscores`, and drops the
    SINGLE worst qualifying sample, then RE-SOLVES via :func:`_solve_once`
    before scoring again.  Single-worst-then-resolve (not batch-drop) is
    deliberate: after the worst outlier is removed and the fit improves,
    borderline samples that looked marginal often fall back under threshold
    and are kept — batch dropping would over-reject them.

    Absolute physical floor: a sample must be BOTH a statistical outlier
    (z > ``reject_sigma``) AND beyond an absolute physical band
    (translation chain error > ``reject_min_trans_m`` = 10 mm, OR rotation
    chain error > ``reject_min_rot_rad`` = 3.0°) on the SAME axis to be
    dropped, so clean right-skewed residuals at small n (where the symmetric
    modified z-score over-fires on the upper tail) aren't trimmed.  Real FK
    outliers (≥ several cm / degrees) clear the floor with a high z-score and
    are still caught.  On clean data nothing qualifies, so the result is
    identical to ``reject_sigma=None``.  Use ``reject_sigma=None`` to disable
    entirely.  The loop stops when no sample qualifies OR when the active set
    has shrunk to the min-keep floor
    ``max(6, ceil((1 - max_reject_frac) * n_orig))`` (default 25% max drop),
    checked at the TOP of each round so a would-be over-drop simply stops
    rather than dropping zero.  ``rejected_indices`` (indices into the train
    list) is attached to ``SolveResult.per_method[-1]`` for operator
    visibility.
    """
    train, test = split_train_test(samples, heldout_frac, rng_seed)
    X, Tbb, per_method, seed_used = _solve_once(
        train, K, dist, board_pts, methods=methods,
        depth_weight=depth_weight, depth_sigma_m=depth_sigma_m,
        anchor_Tbb=anchor_Tbb)

    rejected = []
    if reject_sigma is not None and len(train) >= 6:
        # Single-worst-drop-then-resolve outlier rejection on the TRAIN set
        # only — held-out stays intact as the honest evaluator.
        active = list(range(len(train)))
        n_orig = len(train)
        # Min-keep floor checked at the TOP of each round: never drop below
        # max(6, ceil((1 - max_reject_frac) * n_orig)) samples.
        min_keep = max(6, int(np.ceil((1.0 - max_reject_frac) * n_orig)))
        for _ in range(20):  # safety cap on iterations
            sub = [train[i] for i in active]
            if len(sub) <= min_keep:
                break
            t_e, r_e = _per_sample_chain_errors(X, Tbb, sub)
            zt = _modified_zscores(t_e)
            zr = _modified_zscores(r_e)
            # A sample qualifies only if it is BOTH a statistical outlier AND
            # physically significant on the SAME axis (z AND abs-floor anded
            # per axis, then OR'd across axes).
            cand_t = (zt > reject_sigma) & (t_e > reject_min_trans_m)
            cand_r = (zr > reject_sigma) & (r_e > reject_min_rot_rad)
            cand = cand_t | cand_r
            if not cand.any():
                break
            score = np.where(cand, np.maximum(zt, zr), -np.inf)
            k = int(np.argmax(score))
            rejected.append(active.pop(k))
            sub = [train[i] for i in active]
            X, Tbb, _pm, seed_used = _solve_once(
                sub, K, dist, board_pts, methods=methods,
                depth_weight=depth_weight, depth_sigma_m=depth_sigma_m,
                anchor_Tbb=anchor_Tbb)
        train = [train[i] for i in active]

    train_m = evaluate(X, Tbb, train, K, dist, board_pts)
    held_m = evaluate(X, Tbb, test, K, dist, board_pts)
    if rejected:
        per_method = list(per_method) + [{"name": "rejected_indices",
                                          "rejected_indices": sorted(rejected),
                                          "X": X, "Tbb": Tbb,
                                          "reproj_px": float("nan")}]
    return SolveResult(X, Tbb, train_m, held_m, gate(held_m), per_method,
                       seed_used=seed_used)


def rotation_observability(samples, *, min_singular=0.3):
    """Diagnose AX=XB rotation observability of the accepted pose set.

    Eye-in-hand identifiability needs >= 2 non-parallel relative-rotation axes;
    a set whose flange rotations all share one axis (or are pure translation)
    leaves X's rotation unobservable and lets a rotation error in X hide in the
    board pose Tbb. We collect the unit axis of every pairwise relative rotation
    R_ij = R_j R_i^T (skipping pairs that rotate < 2 deg, which have no
    well-defined axis), stack them into a 3xK matrix, and SVD. The 2nd singular
    value measures how much the axes span a second dimension; below
    ``min_singular`` the set is effectively single-axis. Returns a JSON-safe
    dict; ``ok`` is the gate the UI shows as a WARN badge.
    """
    Rs = [np.asarray(s.T_base_eef, float)[:3, :3] for s in samples]
    axes = []
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            rv = _R.from_matrix(Rs[j] @ Rs[i].T).as_rotvec()
            ang = float(np.linalg.norm(rv))
            if np.degrees(ang) >= 2.0:
                axes.append(rv / ang)
    if len(axes) < 2:
        return {"ok": False, "n_axes": len(axes), "second_singular": 0.0,
                "detail": "fewer than 2 usable rotation axes — X rotation "
                          "unobservable; add poses that rotate the flange"}
    sv = np.linalg.svd(np.asarray(axes, float).T, compute_uv=False)
    second = float(sv[1]) if len(sv) >= 2 else 0.0
    ok = bool(second >= min_singular)
    return {"ok": ok, "n_axes": len(axes), "second_singular": second,
            "detail": ("rotation axes span >= 2 dimensions" if ok else
                       "rotation axes nearly collinear — add poses that rotate "
                       "the flange about a DIFFERENT axis")}


def consensus_corners(frames, *, min_frac=0.6):
    """Per-corner sub-pixel consensus across N steady frames.

    ``frames`` is a list of ``(ids, px)`` for one frame each: ``ids`` is an
    ``(M,)`` int array of ChArUco corner indices, ``px`` an ``(M,2)`` array of
    sub-pixel corner pixels. A corner id is kept only if it was detected in at
    least ``ceil(min_frac * N)`` frames; its consensus pixel is the per-corner
    MEDIAN over the frames that saw it (robust to the occasional mis-localized
    corner). Returns ``(ids, px)`` sorted by id, or ``(None, None)`` only when
    no corner reaches quorum (caller falls back to the single-frame pose; the
    downstream PnP path enforces its own >=6-corner floor).
    """
    n = len(frames)
    if n == 0:
        return None, None
    quorum = max(1, int(np.ceil(min_frac * n)))
    acc = {}
    for ids, px in frames:
        ids = np.asarray(ids).reshape(-1).astype(int)
        px = np.asarray(px, float).reshape(-1, 2)
        for cid, p in zip(ids, px):
            acc.setdefault(int(cid), []).append(p)
    kept_ids, kept_px = [], []
    for cid in sorted(acc):
        pts = np.asarray(acc[cid], float)
        if len(pts) >= quorum:
            kept_ids.append(cid)
            kept_px.append(np.median(pts, axis=0))
    if not kept_ids:
        return None, None
    return np.asarray(kept_ids, int), np.asarray(kept_px, float)
