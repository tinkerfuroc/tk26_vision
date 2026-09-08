"""Sample FFS metric depth at ChArUco corners and back-project to 3D.

The hand-eye calibration's per-view ``T_cam_board`` is otherwise pure
monocular planar PnP, whose optical-axis (depth) translation is the weakest-
constrained DOF. FoundationStereo (FFS) provides metric depth *aligned to the
color frame* (``/foundation_stereo/get_depth`` with ``align_to_color=True``),
expressed with the **same** color intrinsics the solver already uses. So
deprojecting a corner pixel ``(u, v)`` with its FFS depth ``d`` yields a metric
point directly in the camera optical frame ``T_cam_board`` maps into — a clean,
intrinsics-consistent measurement that the bundle adjust can use to pin scale.

Pure numpy / ROS-free so it unit-tests under the plain venv (mirrors the import
discipline of the rest of handeye_calib). The pinhole back-projection mirrors
``vision_util/_pc_utils`` and the robust local-median mirrors
``vision_track/core/depth_gate.roi_median_depth``.
"""
import numpy as np


def deproject_corners(obs_px, depth_m, K, *, win=2, z_min=0.05, z_max=2.0):
    """Back-project corner pixels to camera-frame 3D points using metric depth.

    Parameters
    ----------
    obs_px : (M, 2) array
        Detected corner pixels ``(u, v)`` = (col, row), sub-pixel floats.
    depth_m : (H, W) array
        Metric depth in **meters**, aligned to the color frame so
        ``depth_m[v, u]`` corresponds to color pixel ``(u, v)``.
    K : (3, 3) array
        Color pinhole intrinsics (the same K the solver reprojects with).
    win : int
        Half-size of the local median window (``(2*win+1)²`` pixels). Robust
        to FFS speckle / single-pixel holes at the corner. ``win=0`` samples
        only the nearest integer pixel.
    z_min, z_max : float
        Valid metric-depth band (meters). Values outside are treated as holes.

    Returns
    -------
    xyz : (M, 3) float array
        Camera-frame metric points; rows for invalid corners are ``nan`` (never
        a fake ``0`` — a 0 would silently bias the solve toward the camera).
    valid : (M,) bool array
        ``True`` where a finite in-band depth was found in the window.
    """
    depth = np.asarray(depth_m, np.float32)
    H, W = depth.shape[:2]
    obs = np.asarray(obs_px, float).reshape(-1, 2)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    xyz = np.full((len(obs), 3), np.nan, float)
    valid = np.zeros(len(obs), bool)

    for i, (u, v) in enumerate(obs):
        uc, vc = int(round(u)), int(round(v))
        if uc < 0 or uc >= W or vc < 0 or vc >= H:
            continue  # corner center off the depth image -> invalid
        r0, r1 = max(0, vc - win), min(H, vc + win + 1)
        c0, c1 = max(0, uc - win), min(W, uc + win + 1)
        patch = depth[r0:r1, c0:c1].reshape(-1)
        good = patch[np.isfinite(patch) & (patch > z_min) & (patch < z_max)]
        if good.size == 0:
            continue
        z = float(np.median(good))
        # Back-project with the SUB-PIXEL corner (u, v) — the detector localizes
        # corners to ~0.1 px, far finer than the integer window used only to pick
        # a robust depth.
        xyz[i] = ((u - cx) / fx * z, (v - cy) / fy * z, z)
        valid[i] = True

    return xyz, valid
