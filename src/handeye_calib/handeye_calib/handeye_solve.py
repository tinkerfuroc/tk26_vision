"""Eye-in-hand solver: multi-method seed -> bundle-adjust refine -> full-set evaluation.

The solve fits X (T_eef_cam) + Tbb on ALL captured samples (no train/held-out
split — pan-tilt-calibration parity). Outlier rejection (MAD) screens every
sample, and the reported residual is over the full surviving set. A separate
validation pose set is recorded later to measure generalization independently.
"""
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


def _anchor_devs(Ts, mean):
    """Per-observation SE(3) deviation (trans_m, rot_rad arrays) from ``mean``."""
    inv_mean = tf.invert(mean)
    t_dev, r_dev = [], []
    for T in Ts:
        D = inv_mean @ T
        t_dev.append(float(np.linalg.norm(D[:3, 3])))
        r_dev.append(float(np.radians(tf.rotation_angle_deg(np.eye(3), D[:3, :3]))))
    return np.asarray(t_dev, float), np.asarray(r_dev, float)


def average_board_anchors(anchors, *, reject_sigma=2.5, max_reject_frac=0.34,
                          reject_min_trans_m=0.01,
                          reject_min_rot_rad=np.radians(3.0),
                          min_obs_for_reject=4):
    """Robustly SE(3)-average board-in-base measurements (head warm-start).

    ``anchors`` is a list of 4x4 ``T_base_board`` observations (e.g. the head
    at several pan/tilt poses). Before averaging, the same per-axis MAD outlier
    rejection the wrist solve uses screens the observations themselves, so a
    single bad pan/tilt read (a TF glitch or a misdetection at one head pose)
    can't drag the anchor off — directly parity with how :func:`solve` weeds the
    wrist samples. Each iteration takes the SE(3) median, scores every
    observation's translation + rotation deviation as one-sided modified
    z-scores, and drops the single worst that is BOTH a statistical outlier
    (z > ``reject_sigma``) AND past the absolute physical floor
    (``reject_min_trans_m`` / ``reject_min_rot_rad``) — then re-medians.
    Screening only runs with at least ``min_obs_for_reject`` observations (MAD
    needs a few points to be meaningful) and never drops below the min-keep
    floor ``max(2, ceil((1 - max_reject_frac) * n))``.

    Returns ``(Tbb_mean, scatter)`` where scatter is
    ``{"trans_mm", "rot_deg", "n", "n_total", "n_rejected", "rejected"}``:
    RMS deviation of the SURVIVING observations from their mean (a data-driven
    confidence readout — large scatter => widen the prior / re-check the head
    TF), the kept count ``n``, the original count ``n_total``, and the ORIGINAL
    indices ``rejected`` that were dropped.
    """
    Ts = [np.asarray(T, float) for T in anchors]
    if not Ts:
        raise ValueError("average_board_anchors requires at least 1 anchor; got 0")
    n_total = len(Ts)
    active = list(range(n_total))
    rejected = []
    if reject_sigma is not None and n_total >= min_obs_for_reject:
        min_keep = max(2, int(np.ceil((1.0 - max_reject_frac) * n_total)))
        for _ in range(n_total):
            if len(active) <= min_keep:
                break
            sub = [Ts[i] for i in active]
            m = tf.se3_average(sub)
            t_dev, r_dev = _anchor_devs(sub, m)
            zt, zr = _modified_zscores(t_dev), _modified_zscores(r_dev)
            cand = ((zt > reject_sigma) & (t_dev > reject_min_trans_m)) | \
                   ((zr > reject_sigma) & (r_dev > reject_min_rot_rad))
            if not cand.any():
                break
            score = np.where(cand, np.maximum(zt, zr), -np.inf)
            k = int(np.argmax(score))
            rejected.append(active.pop(k))
    kept = [Ts[i] for i in active]
    mean = tf.se3_average(kept)
    t_dev, r_dev = _anchor_devs(kept, mean)
    scatter = {
        "trans_mm": float(np.sqrt(np.mean(np.square(t_dev))) * 1000.0),
        "rot_deg": float(np.degrees(np.sqrt(np.mean(np.square(r_dev))))),
        "n": len(kept),
        "n_total": n_total,
        "n_rejected": len(rejected),
        "rejected": sorted(rejected),
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
                  depth_weight=0.0, depth_sigma_m=0.005, loss="soft_l1",
                  xtol=None, ftol=None, gtol=None, max_nfev=None):
    """Jointly refine X (T_eef_cam) and Tbb (T_base_board) minimizing corner
    reprojection, plus the optional FFS-depth 3D residual (``depth_weight>0``).

    ``depth_weight`` defaults to ``0.0`` so existing direct callers (and the
    pixel-only synthetic tests) get the unchanged monocular solve; the higher-
    level :func:`solve` passes a non-zero default which no-ops anyway on samples
    that carry no depth.

    ``loss`` is forwarded to :func:`scipy.optimize.least_squares`.  Default
    ``"soft_l1"`` is robust to large outliers; ``"linear"`` (L2) gives tighter
    convergence on a clean, already-screened sample set (used by :func:`polish`).
    Pass ``xtol``/``ftol``/``gtol``/``max_nfev`` to override scipy defaults — the
    L2 polish path uses ``1e-12``/``20000`` for tight convergence.
    """
    p0 = np.concatenate([tf.vec_from_T(X0), tf.vec_from_T(Tbb0)])
    kw = dict(loss=loss, method="trf",
              args=(samples, K, dist, board_pts, depth_weight, depth_sigma_m))
    if xtol is not None:
        kw["xtol"] = xtol
    if ftol is not None:
        kw["ftol"] = ftol
    if gtol is not None:
        kw["gtol"] = gtol
    if max_nfev is not None:
        kw["max_nfev"] = max_nfev
    sol = least_squares(_residuals, p0, **kw)
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
    # Residual over the FULL surviving sample set (post-MAD-rejection). There is
    # no train/held-out split — the calibration is fit on, and scored against,
    # all kept samples. A separately-recorded validation set measures
    # generalization later.
    metrics: dict
    status: str
    per_method: list
    # Which seed produced the promoted X: "closed_form" (best-of-5
    # calibrateHandEye) or "board_anchor" (the head warm-start). Surfaced so an
    # operator can confirm whether the head anchor actually rescued the solve.
    seed_used: str = ""
    # ORIGINAL-sample indices (into the `samples` passed to solve) that the
    # MAD rejection dropped. Surfaced so the UI / dumps can mark them and
    # exclude them from the per-sample residual view.
    rejected_indices: list = None
    # Per-drop diagnostics, one dict per rejected sample (sorted by idx):
    # {idx, trans_mm, rot_deg, reproj_px, z_trans, z_rot, z_reproj} captured at
    # the moment of rejection. Drives the pan-tilt-style "why was this dropped"
    # readout in the UI. None/empty when nothing was rejected.
    rejection_log: list = None


# pan-tilt parity thresholds
_PASS = {"trans_rmse_m": 0.003, "rot_rmse_rad": 0.00873, "reproj_px": 1.5}
_WARN = {"trans_rmse_m": 0.006, "rot_rmse_rad": 0.01745, "reproj_px": 3.0}


def _depth_point_metrics(X, Tbb, samples, board_pts):
    """Depth-grounded accuracy: RMS distance (mm) between the solved chain's
    predicted camera-frame corner points and the FFS-measured points, over all
    valid corners. Unlike ``trans_rmse_m`` (which compares against the monocular
    PnP pose — itself biased when intrinsics/board-scale are off), this compares
    against an independent *metric* measurement.

    This is an IN-SAMPLE metric (all samples enter the fit — there is no held-out
    split), so it can read deceptively low if the depth term over-fits a
    systematic FFS bias; treat the separately-recorded validation set as the
    real-world error budget. Returns ``(rmse_mm_or_None, n_corners)``."""
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


def solve(samples, K, dist, board_pts, *,
          methods=None, reject_sigma=2.5, max_reject_frac=0.25,
          reject_min_trans_m=0.01, reject_min_rot_rad=np.radians(3.0),
          reject_min_reproj_px=3.0,
          depth_weight=1.0, depth_sigma_m=0.005, anchor_Tbb=None,
          progress_cb=None):
    """Full solve pipeline over ALL samples — there is NO train/held-out split
    (pan-tilt-calibration parity).  X/Tbb are fit on, and the residual reported
    over, every surviving sample; generalization is measured separately later
    with a freshly-recorded validation pose set.

    ``methods`` is forwarded to :func:`seed_handeye` so a single-method run
    (e.g. ``methods={"TSAI": cv2.CALIB_HAND_EYE_TSAI}``) skips the multi-method
    best-of step; ``None`` keeps the default 5-method sweep.

    ``anchor_Tbb``: an external board-in-base measurement (e.g. from the
    pan-tilt head via TF) used as a basin-immune warm-start seed; forwarded
    through the rejection loop via :func:`_solve_once` so the anchor is
    available on every re-solve. ``None`` (default) skips the anchor branch.

    ``reject_sigma``: default-on (2.5) iterative per-axis MAD outlier rejection
    over the FULL sample set.  Each iteration fits X/Tbb on the active set,
    computes per-sample SE(3) chain errors (translation in metres, rotation in
    radians) via :func:`_per_sample_chain_errors` plus per-sample reprojection
    (px), turns each axis into one-sided modified z-scores via
    :func:`_modified_zscores`, and drops the SINGLE worst qualifying sample,
    then RE-SOLVES via :func:`_solve_once` before scoring again.
    Single-worst-then-resolve (not batch-drop) is deliberate: after the worst
    outlier is removed and the fit improves, borderline samples that looked
    marginal often fall back under threshold and are kept — batch dropping
    would over-reject them.  Screening the WHOLE set (not a train subset) means
    an outlier is caught wherever it sits — the prior train-only loop left
    outliers that happened to land in the held-out split un-weeded.

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
    rather than dropping zero.  A pose can also be dropped on the
    REPROJECTION axis (per-sample reproj > ``reject_min_reproj_px`` AND a
    statistical outlier) — this catches a camera-to-flange-inconsistent pose
    (mid-ring / mount-flex capture) whose chain rotation sits below the 3°
    floor, which is the dominant real-hardware failure mode.
    ``rejected_indices`` (ORIGINAL indices into ``samples``, so they map onto
    the per-sample residual arrays / the gallery) is on
    ``SolveResult.rejected_indices`` and attached to
    ``SolveResult.per_method[-1]``; per-drop residual+z diagnostics are on
    ``SolveResult.rejection_log``.
    """
    def _emit(ev):
        # ``progress_cb`` lets a caller (the web node) stream per-iteration MAD
        # rejection events to the UI as they happen. A callback bug must never
        # break the solve, so every call is guarded.
        if progress_cb is None:
            return
        try:
            progress_cb(ev)
        except Exception:
            pass

    orig_of = {id(s): i for i, s in enumerate(samples)}
    X, Tbb, per_method, seed_used = _solve_once(
        samples, K, dist, board_pts, methods=methods,
        depth_weight=depth_weight, depth_sigma_m=depth_sigma_m,
        anchor_Tbb=anchor_Tbb)

    rejected = []          # ORIGINAL-sample indices (into `samples`)
    rejection_log = []     # per-drop {idx, trans_mm, rot_deg, reproj_px, z_*}
    active = list(range(len(samples)))
    if reject_sigma is not None and len(samples) >= 6:
        # Single-worst-drop-then-resolve MAD rejection over the WHOLE set.
        n_orig = len(samples)
        # Min-keep floor checked at the TOP of each round: never drop below
        # max(6, ceil((1 - max_reject_frac) * n_orig)) samples.
        min_keep = max(6, int(np.ceil((1.0 - max_reject_frac) * n_orig)))
        _emit({"phase": "start", "n_orig": n_orig, "n_active": len(active),
               "min_keep": min_keep, "iteration": 0, "last_drop": None})
        for _ in range(n_orig + 1):  # safety cap; min_keep bounds real drops
            sub = [samples[i] for i in active]
            if len(sub) <= min_keep:
                break
            t_e, r_e = _per_sample_chain_errors(X, Tbb, sub)
            _, reproj_ps = _reproj_rms(X, Tbb, sub, K, dist, board_pts,
                                       per_sample=True)
            reproj_ps = np.asarray(reproj_ps, float)
            zt = _modified_zscores(t_e)
            zr = _modified_zscores(r_e)
            zp = _modified_zscores(reproj_ps)
            # A sample qualifies only if it is BOTH a statistical outlier AND
            # physically significant on the SAME axis: translation chain error
            # (> reject_min_trans_m), rotation chain error (> reject_min_rot_rad),
            # OR reprojection (> reject_min_reproj_px). The reprojection axis is
            # the discriminator that catches a pose whose camera-to-flange
            # transform is inconsistent with the rest (a mid-ring / mount-flex
            # capture) even when its chain rotation sits below the 3° floor —
            # the dominant real-hardware failure mode. Clean data reprojects
            # well under the px floor, so nothing qualifies there.
            cand_t = (zt > reject_sigma) & (t_e > reject_min_trans_m)
            cand_r = (zr > reject_sigma) & (r_e > reject_min_rot_rad)
            cand_p = (zp > reject_sigma) & (reproj_ps > reject_min_reproj_px)
            cand = cand_t | cand_r | cand_p
            if not cand.any():
                break
            score = np.where(cand, np.maximum(np.maximum(zt, zr), zp), -np.inf)
            k = int(np.argmax(score))
            oi = orig_of[id(sub[k])]
            rejection_log.append({
                "idx": oi,
                "trans_mm": float(t_e[k] * 1000.0),
                "rot_deg": float(np.degrees(r_e[k])),
                "reproj_px": float(reproj_ps[k]),
                "z_trans": float(zt[k]), "z_rot": float(zr[k]),
                "z_reproj": float(zp[k]),
            })
            rejected.append(oi)
            active.pop(k)
            _emit({"phase": "rejecting", "n_orig": n_orig,
                   "n_active": len(active), "min_keep": min_keep,
                   "iteration": len(rejected), "last_drop": rejection_log[-1]})
            X, Tbb, _pm, seed_used = _solve_once(
                [samples[i] for i in active], K, dist, board_pts, methods=methods,
                depth_weight=depth_weight, depth_sigma_m=depth_sigma_m,
                anchor_Tbb=anchor_Tbb)

    kept = [samples[i] for i in active]
    metrics = evaluate(X, Tbb, kept, K, dist, board_pts)
    rejected = sorted(rejected)
    # Re-score each rejected pose's residual MAGNITUDES against the FINAL clean
    # fit (the kept-set solution), so the reported numbers show how far the pose
    # truly sits from the converged calibration. The at-drop-time snapshot is
    # measured on a fit that still contains that pose plus any not-yet-dropped
    # outliers — so the FIRST drop (the worst outlier) reads absurdly large
    # (e.g. 757 mm) purely because it was polluting its own reference fit. The
    # z-scores are LEFT as captured: they are population-relative trigger
    # evidence ("how much it stood out at the moment it was dropped"), which is
    # exactly what justified the rejection and is not meaningful post-hoc.
    if rejection_log:
        rej_samples = [samples[e["idx"]] for e in rejection_log]
        rt, rr = _per_sample_chain_errors(X, Tbb, rej_samples)
        _, rpx = _reproj_rms(X, Tbb, rej_samples, K, dist, board_pts,
                             per_sample=True)
        for e, t, r, p in zip(rejection_log, rt, rr, rpx):
            e["trans_mm"] = float(t * 1000.0)
            e["rot_deg"] = float(np.degrees(r))
            e["reproj_px"] = float(p)
    rejection_log = sorted(rejection_log, key=lambda d: d["idx"])
    if rejected:
        per_method = list(per_method) + [{"name": "rejected_indices",
                                          "rejected_indices": rejected,
                                          "X": X, "Tbb": Tbb,
                                          "reproj_px": float("nan")}]
    return SolveResult(X, Tbb, metrics, gate(metrics), per_method,
                       seed_used=seed_used, rejected_indices=rejected,
                       rejection_log=rejection_log)


def polish(result, samples, K, dist, board_pts, *,
           n_restarts=12, polish_sigma=2.5, max_reject_frac=0.20,
           depth_weight=0.0, depth_sigma_m=0.005, perturb_scale=5e-4):
    """L2 polish pass — pan-tilt calibration parity.

    Starting from the MAD-cleaned :class:`SolveResult`, runs a multi-restart
    L2 (``loss='linear'``) bundle adjust over the **kept** samples, then
    applies a tight per-axis MAD rejection loop (sigma=``polish_sigma``,
    max_reject_frac=``max_reject_frac``) where each re-solve also uses L2.
    Unlike the soft-L1 used in :func:`solve`, L2 rewards tight inlier agreement
    rather than robustness to large outliers — so it is appropriate only AFTER
    the main outlier-rejection pass has cleaned the sample set.

    Returns a fresh :class:`SolveResult`.  ``rejected_indices`` and
    ``rejection_log`` on the return value accumulate ALL rejections (MAD phase +
    polish phase); ``seed_used`` gains a ``+polish_l2`` suffix.

    On non-rigid mounts where rigid_closure_deg dominates the error floor, this
    pass will not break through the metric ceiling — but on genuinely rigid-mount
    data it typically tightens reproj_px by 5–15% vs the soft_l1 minimum.
    """
    rng = np.random.default_rng(42)

    # Reconstruct the kept set from the seed result's rejection list.
    rej_set = set(result.rejected_indices or [])
    kept = [s for i, s in enumerate(samples) if i not in rej_set]
    if len(kept) < 6:
        return result

    # Map Python object id → original index in `samples` for bookkeeping.
    orig_of = {id(s): i for i, s in enumerate(samples)}

    def _ba_l2(X0, Tbb0, samps):
        Xr, Tbbr, info = bundle_adjust(
            samps, K, dist, board_pts, X0, Tbb0,
            depth_weight=depth_weight, depth_sigma_m=depth_sigma_m,
            loss="linear", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000)
        return Xr, Tbbr, info["cost"]

    # Multi-restart: seed + n_restarts-1 small random perturbations.
    p_seed = np.concatenate([tf.vec_from_T(result.X), tf.vec_from_T(result.Tbb)])
    best_cost = None
    X, Tbb = result.X.copy(), result.Tbb.copy()
    for r in range(n_restarts):
        p = p_seed + (rng.standard_normal(p_seed.shape) * perturb_scale
                      if r > 0 else np.zeros_like(p_seed))
        Xr, Tbbr, cost = _ba_l2(tf.T_from_vec(p[:6]), tf.T_from_vec(p[6:]), kept)
        if best_cost is None or cost < best_cost:
            best_cost, X, Tbb = cost, Xr, Tbbr

    # Tight MAD rejection loop with L2 re-solve (no absolute floor — the main
    # solve pass has already screened gross outliers; L2 is sensitive to them).
    n_orig = len(kept)
    min_keep = max(6, int(np.ceil((1.0 - max_reject_frac) * n_orig)))
    rejected_extra = []
    rejection_log_extra = []
    active_kept = list(range(n_orig))

    for _ in range(n_orig + 1):
        sub = [kept[i] for i in active_kept]
        if len(sub) <= min_keep:
            break
        t_e, r_e = _per_sample_chain_errors(X, Tbb, sub)
        _, reproj_ps = _reproj_rms(X, Tbb, sub, K, dist, board_pts, per_sample=True)
        reproj_ps = np.asarray(reproj_ps, float)
        zt = _modified_zscores(t_e)
        zr = _modified_zscores(r_e)
        zp = _modified_zscores(reproj_ps)
        cand = (zt > polish_sigma) | (zr > polish_sigma) | (zp > polish_sigma)
        if not cand.any():
            break
        score = np.where(cand, np.maximum(np.maximum(zt, zr), zp), -np.inf)
        k = int(np.argmax(score))
        oi = orig_of[id(sub[k])]
        rejection_log_extra.append({
            "idx": oi,
            "trans_mm": float(t_e[k] * 1000.0),
            "rot_deg": float(np.degrees(r_e[k])),
            "reproj_px": float(reproj_ps[k]),
            "z_trans": float(zt[k]), "z_rot": float(zr[k]), "z_reproj": float(zp[k]),
        })
        rejected_extra.append(oi)
        active_kept.pop(k)
        sub2 = [kept[i] for i in active_kept]
        X, Tbb, _ = _ba_l2(X, Tbb, sub2)

    final_kept = [kept[i] for i in active_kept]
    metrics = evaluate(X, Tbb, final_kept, K, dist, board_pts)

    all_rejected = sorted((result.rejected_indices or []) + rejected_extra)
    # Re-score rejected-in-polish poses against the final clean fit.
    if rejection_log_extra:
        rs = [samples[e["idx"]] for e in rejection_log_extra]
        rt, rr = _per_sample_chain_errors(X, Tbb, rs)
        _, rpx = _reproj_rms(X, Tbb, rs, K, dist, board_pts, per_sample=True)
        for e, t, r, p in zip(rejection_log_extra, rt, rr, rpx):
            e["trans_mm"] = float(t * 1000.0)
            e["rot_deg"] = float(np.degrees(r))
            e["reproj_px"] = float(p)
    all_rej_log = sorted(
        (result.rejection_log or []) + rejection_log_extra,
        key=lambda d: d["idx"])

    per_method_out = list(result.per_method or []) + [{
        "name": "polish_l2",
        "n_restarts": n_restarts,
        "n_dropped_in_polish": len(rejected_extra),
        "reproj_px": metrics["reproj_px"],
    }]
    return SolveResult(X, Tbb, metrics, gate(metrics), per_method_out,
                       seed_used=(result.seed_used or "") + "+polish_l2",
                       rejected_indices=all_rejected,
                       rejection_log=all_rej_log)


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


def rigid_closure_deg(samples, *, min_rel_deg=5.0, rigid_tol_deg=0.5):
    """Intrinsics-free rigid-MOUNT diagnostic (AX=XB rotation conjugacy).

    For a RIGID camera-to-flange mount the flange motion between two poses
    ``A_ij = inv(A_i) @ A_j`` and the camera motion ``B_ij = inv(B_i) @ B_j``
    are conjugate (``A_ij X = X B_ij``), so they have the SAME rotation ANGLE
    regardless of the hand-eye X, the camera intrinsics, board scale,
    translation, or depth. The per-pair residual ``|angle(R_A_ij) -
    angle(R_B_ij)|`` over all pose pairs with a meaningful relative rotation
    (> ``min_rel_deg``) therefore isolates MECHANICAL non-rigidity (wrist-camera
    bracket flex / arm-FK compliance) from every camera-side error — none of
    which can move it.

    This is a conservative LOWER bound on non-rigidity (it compares rotation
    magnitudes only, not the conjugated axis), so a clean pass means the mount is
    genuinely rigid, while any meaningful residual is a definitive flex signal.
    Returns a JSON-safe dict ``{"ok","mean_deg","median_deg","max_deg",
    "n_pairs","detail"}``. ``ok`` (median < ``rigid_tol_deg``) is the go/no-go an
    operator should check BEFORE trusting a solve: if it FAILs, no hand-eye solve
    can reach the gate — rigidify the mount (and/or fix the arm kinematics) and
    recapture; tuning the solver is wasted effort.
    """
    A = [np.asarray(s.T_base_eef, float) for s in samples]
    B = [np.asarray(s.T_cam_board, float) for s in samples]
    diffs = []
    n = len(samples)
    for i in range(n):
        for j in range(i + 1, n):
            Aij = tf.invert(A[i]) @ A[j]
            Bij = tf.invert(B[i]) @ B[j]
            a = tf.rotation_angle_deg(np.eye(3), Aij[:3, :3])
            b = tf.rotation_angle_deg(np.eye(3), Bij[:3, :3])
            if a < min_rel_deg and b < min_rel_deg:
                continue  # near-static pair: no well-defined motion angle
            diffs.append(abs(a - b))
    if not diffs:
        return {"ok": False, "mean_deg": 0.0, "median_deg": 0.0,
                "max_deg": 0.0, "n_pairs": 0,
                "detail": "no pose pairs exceed the relative-rotation threshold "
                          "— add poses that rotate the flange"}
    diffs = np.asarray(diffs, float)
    med = float(np.median(diffs))
    ok = bool(med < rigid_tol_deg)
    return {"ok": ok, "mean_deg": float(diffs.mean()), "median_deg": med,
            "max_deg": float(diffs.max()), "n_pairs": int(diffs.size),
            "detail": (f"rigid mount (median pair closure {med:.2f}° < "
                       f"{rigid_tol_deg:.1f}°)" if ok else
                       f"NON-RIGID: {med:.2f}° median flange-vs-camera rotation "
                       "mismatch — the camera bracket flexes or the arm FK is "
                       "pose-dependent. NO hand-eye solve can reach the gate; "
                       "rigidify the mount and recapture (tuning the solver "
                       "will not help).")}


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


def _residuals_multi(p, placements_samples, K, dist, board_pts,
                     depth_weight, depth_sigma_m):
    """Stacked residuals for a joint multi-placement solve.

    ``p = [X_6dof | Tbb_0_6dof | ... | Tbb_{n-1}_6dof]``.  Each placement's
    residual is computed by delegating to :func:`_residuals` with the shared X
    and that placement's Tbb slice, then all residual vectors are concatenated.
    """
    X_vec = p[:6]
    all_res = []
    for i, samples in enumerate(placements_samples):
        Tbb_vec = p[6 + 6 * i: 12 + 6 * i]
        all_res.append(_residuals(np.concatenate([X_vec, Tbb_vec]), samples,
                                  K, dist, board_pts, depth_weight, depth_sigma_m))
    return np.concatenate(all_res)


def bundle_adjust_multi(placements_samples, K, dist, board_pts, X0, Tbb0s,
                        depth_weight=0.0, depth_sigma_m=0.005, loss="soft_l1",
                        xtol=None, ftol=None, gtol=None, max_nfev=None):
    """Joint bundle adjust over multiple placements sharing a single X (T_eef_cam).

    Each placement has its own independent Tbb (T_base_board).  The parameter
    vector is ``[X_6dof | Tbb_0_6dof | ... | Tbb_{n-1}_6dof]``.  Returns
    ``(X, Tbbs, info)`` where ``Tbbs`` is a list of 4x4 matrices in the same
    order as ``placements_samples`` and ``info`` contains ``success`` and ``cost``.
    """
    n = len(placements_samples)
    p0 = np.concatenate([tf.vec_from_T(X0)] + [tf.vec_from_T(T) for T in Tbb0s])
    kw = dict(loss=loss, method="trf",
              args=(placements_samples, K, dist, board_pts, depth_weight, depth_sigma_m))
    if xtol is not None:
        kw["xtol"] = xtol
    if ftol is not None:
        kw["ftol"] = ftol
    if gtol is not None:
        kw["gtol"] = gtol
    if max_nfev is not None:
        kw["max_nfev"] = max_nfev
    sol = least_squares(_residuals_multi, p0, **kw)
    X = tf.T_from_vec(sol.x[:6])
    Tbbs = [tf.T_from_vec(sol.x[6 + 6 * i: 12 + 6 * i]) for i in range(n)]
    return X, Tbbs, {"success": bool(sol.success), "cost": float(sol.cost)}


@dataclass
class MultiPlacementSolveResult:
    X: np.ndarray
    placement_Tbbs: list
    placement_results: list
    combined_metrics: dict
    status: str
    seed_placement_id: str


def solve_multi_placement(placements, K, dist, board_pts, *,
                          methods=None, reject_sigma=2.5, max_reject_frac=0.25,
                          depth_weight=1.0, depth_sigma_m=0.005,
                          anchor_Tbbs=None, progress_cb=None):
    """Jointly calibrate X (T_eef_cam) across multiple board placements.

    ``placements: list[tuple[str, list[Sample]]]`` — each element is a
    ``(placement_id, samples)`` pair where the board is at a fixed but unknown
    position for that placement.  A shared X is fit across all placements
    simultaneously, with one independent Tbb (T_base_board) per placement.

    ``anchor_Tbbs: dict[str, np.ndarray] | None`` — optional per-placement
    head-measured board anchor, forwarded to :func:`solve` as ``anchor_Tbb``.

    Algorithm:
    1. Independent :func:`solve` per placement.
    2. Seed the joint optimise from the placement with the lowest
       ``trans_rmse_m`` (best per-placement result).
    3. :func:`bundle_adjust_multi` over ALL original samples.
    4. Aggregate per-placement :func:`evaluate` into combined RMS metrics.
    5. :func:`gate` the combined metrics.

    Raises ``ValueError`` if any placement has fewer than 6 samples.
    """
    short = [pid for pid, samples in placements if len(samples) < 6]
    if short:
        raise ValueError(f"placements with fewer than 6 samples: {short}")

    placement_results = []
    for pid, samples in placements:
        anchor_Tbb = (anchor_Tbbs or {}).get(pid)
        result = solve(samples, K, dist, board_pts,
                       methods=methods,
                       reject_sigma=reject_sigma,
                       max_reject_frac=max_reject_frac,
                       depth_weight=depth_weight,
                       depth_sigma_m=depth_sigma_m,
                       anchor_Tbb=anchor_Tbb,
                       progress_cb=progress_cb)
        placement_results.append(result)

    seed_idx = int(np.argmin([r.metrics["trans_rmse_m"] for r in placement_results]))
    seed_placement_id = placements[seed_idx][0]
    best_result = placement_results[seed_idx]

    placements_samples = [samples for _, samples in placements]
    X_joint, Tbbs, _ = bundle_adjust_multi(
        placements_samples, K, dist, board_pts,
        X0=best_result.X,
        Tbb0s=[r.Tbb for r in placement_results],
        depth_weight=depth_weight,
        depth_sigma_m=depth_sigma_m)

    per_placement_metrics = [
        evaluate(X_joint, Tbbs[i], placements_samples[i], K, dist, board_pts)
        for i in range(len(placements))
    ]
    all_trans = [m["trans_rmse_m"] for m in per_placement_metrics]
    all_rot = [m["rot_rmse_rad"] for m in per_placement_metrics]
    all_repr = [m["reproj_px"] for m in per_placement_metrics]
    combined_metrics = {
        "trans_rmse_m": float(np.sqrt(np.mean(np.square(all_trans)))),
        "rot_rmse_rad": float(np.sqrt(np.mean(np.square(all_rot)))),
        "reproj_px": float(np.sqrt(np.mean(np.square(all_repr)))),
    }

    return MultiPlacementSolveResult(
        X=X_joint,
        placement_Tbbs=Tbbs,
        placement_results=placement_results,
        combined_metrics=combined_metrics,
        status=gate(combined_metrics),
        seed_placement_id=seed_placement_id,
    )
