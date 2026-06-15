"""Pure-logic collection gates: settle/stability, pose diversity, per-frame quality."""
import numpy as np
from handeye_calib import transforms as tf


class StabilityTracker:
    """Returns True once the last `window` board poses agree within tolerance.

    Absorbs the 1-2 s mount ring: feed live PnP poses; capture only when it
    returns True (or treat repeated False past a timeout as 'did not settle').
    """
    def __init__(self, window=5, rot_tol_deg=0.1, trans_tol_m=0.0003):
        self.window = window
        self.rot_tol_deg = rot_tol_deg
        self.trans_tol_m = trans_tol_m
        self._buf = []

    def reset(self):
        self._buf = []

    def update(self, T_cam_board):
        self._buf.append(np.asarray(T_cam_board))
        if len(self._buf) > self.window:
            self._buf.pop(0)
        if len(self._buf) < self.window:
            return False
        ref = self._buf[-1]
        for T in self._buf[:-1]:
            if tf.rotation_angle_deg(T[:3, :3], ref[:3, :3]) > self.rot_tol_deg:
                return False
            if np.linalg.norm(T[:3, 3] - ref[:3, 3]) > self.trans_tol_m:
                return False
        return True


def is_diverse(T_base_eef_new, accepted, min_deg=30.0):
    """True if the new flange orientation differs from every accepted pose by >= min_deg."""
    if not accepted:
        return True
    Rn = np.asarray(T_base_eef_new)[:3, :3]
    return all(tf.rotation_angle_deg(np.asarray(T)[:3, :3], Rn) >= min_deg for T in accepted)


def quality_ok(n_corners, reproj_px, area_frac,
               min_corners=10, max_reproj_px=1.5, min_area_frac=0.05):
    if n_corners < min_corners:
        return False, f"too few corners ({n_corners}<{min_corners})"
    if reproj_px > max_reproj_px:
        return False, f"reproj too high ({reproj_px:.2f}>{max_reproj_px})"
    if area_frac < min_area_frac:
        return False, f"board too small ({area_frac:.2f}<{min_area_frac})"
    return True, "ok"
