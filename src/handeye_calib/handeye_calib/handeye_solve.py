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


def _reproj_rms(X, Tbb, samples, K, dist, board_pts):
    sq = []
    for s in samples:
        T_cam_board = tf.invert(s.T_base_eef @ X) @ Tbb
        pred = hm.project_corners(board_pts[s.corner_idx], T_cam_board, K, dist)
        sq.append(np.sum((pred - s.obs_px) ** 2, axis=1))
    return float(np.sqrt(np.mean(np.concatenate(sq))))


def _estimate_board_in_base(X, samples):
    return tf.se3_average([s.T_base_eef @ X @ s.T_cam_board for s in samples])


def seed_handeye(samples, K, dist, board_pts):
    """Run all OpenCV hand-eye methods, return the X with lowest reprojection RMS."""
    R_g2b = [np.asarray(s.T_base_eef)[:3, :3] for s in samples]
    t_g2b = [np.asarray(s.T_base_eef)[:3, 3] for s in samples]
    R_t2c = [np.asarray(s.T_cam_board)[:3, :3] for s in samples]
    t_t2c = [np.asarray(s.T_cam_board)[:3, 3] for s in samples]
    per_method = []
    for name, flag in _METHODS.items():
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


def _residuals(params, samples, K, dist, board_pts):
    X = tf.T_from_vec(params[:6])
    Tbb = tf.T_from_vec(params[6:])
    res = []
    for s in samples:
        T_cam_board = tf.invert(s.T_base_eef @ X) @ Tbb
        pred = hm.project_corners(board_pts[s.corner_idx], T_cam_board, K, dist)
        res.append((pred - s.obs_px).ravel())
    return np.concatenate(res)


def bundle_adjust(samples, K, dist, board_pts, X0, Tbb0):
    """Jointly refine X (T_eef_cam) and Tbb (T_base_board) minimizing corner reprojection."""
    p0 = np.concatenate([tf.vec_from_T(X0), tf.vec_from_T(Tbb0)])
    sol = least_squares(_residuals, p0, loss="soft_l1", method="trf",
                        args=(samples, K, dist, board_pts))
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


def evaluate(X, Tbb, samples, K, dist, board_pts):
    trans_e, rot_e = [], []
    for s in samples:
        T_pred = tf.invert(s.T_base_eef @ X) @ Tbb     # predicted board-in-cam
        T_obs = s.T_cam_board                           # observed (PnP)
        trans_e.append(np.linalg.norm(T_pred[:3, 3] - T_obs[:3, 3]))
        rot_e.append(np.radians(tf.rotation_angle_deg(T_pred[:3, :3], T_obs[:3, :3])))
    return {"trans_rmse_m": float(np.sqrt(np.mean(np.square(trans_e)))),
            "rot_rmse_rad": float(np.sqrt(np.mean(np.square(rot_e)))),
            "reproj_px": _reproj_rms(X, Tbb, samples, K, dist, board_pts)}


def gate(metrics):
    def ok(th):
        return all(metrics[k] <= th[k] for k in th)
    if ok(_PASS):
        return "PASS"
    if ok(_WARN):
        return "WARN"
    return "FAIL"


def solve(samples, K, dist, board_pts, heldout_frac=0.2, rng_seed=0):
    train, test = split_train_test(samples, heldout_frac, rng_seed)
    X0, Tbb0, per_method = seed_handeye(train, K, dist, board_pts)
    X, Tbb, _ = bundle_adjust(train, K, dist, board_pts, X0, Tbb0)
    train_m = evaluate(X, Tbb, train, K, dist, board_pts)
    held_m = evaluate(X, Tbb, test, K, dist, board_pts)
    return SolveResult(X, Tbb, train_m, held_m, gate(held_m), per_method)
