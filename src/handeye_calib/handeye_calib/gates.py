"""Pure-logic collection gates: settle/stability, pose diversity, per-frame quality."""
import numpy as np
from handeye_calib import transforms as tf


class StabilityTracker:
    """Returns True once the last `window` board poses agree within tolerance.

    Absorbs the 1-2 s mount ring: feed live PnP poses; capture only when it
    returns True (or treat repeated False past a timeout as 'did not settle').

    Default thresholds are calibrated for camera-only ChArUco PnP at typical
    handeye distances (~30-60 cm):
      - ``trans_tol_m=0.003`` (3 mm): single-pixel charuco corner noise on a
        ~5 mm square at ~500 mm yields ~2-5 mm depth noise. Sub-mm thresholds
        (e.g. 0.3 mm) are unreachable without dead-still optical conditions
        and would report ``not steady`` even on a perfectly stationary arm.
      - ``rot_tol_deg=0.5``: corresponds to the rotation jitter induced by
        the same sub-pixel corner noise around the optical axis.
    Override via the ``stability_trans_tol_m`` / ``stability_rot_tol_deg``
    ROS params if your setup is dramatically tighter (or noisier).
    """
    def __init__(self, window=5, rot_tol_deg=0.5, trans_tol_m=0.003):
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


def is_diverse(T_base_eef_new, accepted, min_deg=5.0):
    """True if the new flange orientation differs from EVERY accepted pose by >= min_deg.

    NOTE on default: the original 30° threshold was wrong for hand-eye
    calibration. SO(3) packing puts an upper bound of ~12-15 mutually-
    30°-separated rotations in the reachable manifold, so a 30° all-vs-all
    gate would reject ~half of any operator-authored 20+ waypoint set
    regardless of how diverse the input actually was. The hand-eye solver
    needs the *accepted set* to span SO(3) (so the linear system has rank),
    not that every pair exceeds a large threshold. 5° is enough to dedup
    near-duplicates (camera shake of the same pose) while letting genuinely
    distinct poses through. Set ``min_deg=0`` to disable the gate entirely.
    """
    if not accepted or min_deg <= 0:
        return True
    Rn = np.asarray(T_base_eef_new)[:3, :3]
    return all(tf.rotation_angle_deg(np.asarray(T)[:3, :3], Rn) >= min_deg for T in accepted)


def quality_ok(n_corners, reproj_px, area_frac,
               min_corners=10, max_reproj_px=1.5, min_area_frac=0.01):
    # ``min_area_frac`` is the corner-bbox area / full-image area. At handeye
    # distances (60-80 cm) on a 1280x720 stream a fully-visible 5x5 board
    # bboxes to ~0.03-0.04 of the frame, so a 0.05 floor rejected poses where
    # every corner was in frame. 0.01 still catches the genuinely under-
    # resolved regime (~10%x10% of the frame, ~128x72 px on 720p) where
    # sub-pixel corner noise dominates PnP depth.
    if n_corners < min_corners:
        return False, f"too few corners ({n_corners}<{min_corners})"
    if reproj_px > max_reproj_px:
        return False, f"reproj too high ({reproj_px:.2f}>{max_reproj_px})"
    if area_frac < min_area_frac:
        return False, f"board too small ({area_frac:.3f}<{min_area_frac})"
    return True, "ok"
