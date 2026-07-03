"""
Depth-image -> 3D-points reprojection for the Orbbec, resolution-agnostic.

Standard pinhole back-projection in the camera's own optical frame
(x=right, y=down, z=forward -- the convention the intrinsics describe, and
the same one `vision_util/_pc_utils.py:build_xy_table_cuda` uses on GPU).
Dimensions are always read from the depth image's own shape, never assumed,
so this works at whatever resolution the camera driver is launched with.

Shared by object_detection_new/object_seg_yolo.py and
vision_util/door_detection.py, which used to carry independent copies of
this reprojection hardcoded to 720x1280 (the Orbbec's old default color
resolution) -- see
docs/superpowers/specs/2026-07-03-orbbec-hri-resolution-bump-design.md.
"""
from __future__ import annotations

import numpy as np


def decode_depth_metres(depth_arr: np.ndarray) -> np.ndarray:
    """
    Coerce a decoded depth image to float32 metres.

    Orbbec Y16 depth decodes to uint16 millimetres via cv_bridge
    ``passthrough``; FoundationStereo-style depth is already float32 metres.
    """
    if depth_arr.dtype == np.uint16:
        return depth_arr.astype(np.float32) * 0.001
    if depth_arr.dtype == np.float32:
        return depth_arr
    raise ValueError(
        f'Unsupported depth dtype {depth_arr.dtype}; expected uint16 mm or '
        'float32 m.'
    )


def depth_image_to_points(depth_m: np.ndarray, k) -> np.ndarray:
    """
    Back-project a metres depth image to an (H, W, 3) points array.

    Parameters
    ----------
    depth_m : np.ndarray
        (H, W) depth in metres.
    k : indexable
        9-element row-major camera intrinsic matrix (CameraInfo.k):
        fx = k[0], fy = k[4], cx = k[2], cy = k[5].

    Returns
    -------
    np.ndarray
        (H, W, 3) float32 array; [..., 0]=x, [..., 1]=y, [..., 2]=z(=depth_m),
        in the optical frame the intrinsics describe. H, W always match
        depth_m's own shape -- never hardcoded.

    """
    h, w = depth_m.shape
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    us, vs = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    z = depth_m.astype(np.float32)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    return np.stack([x, y, z], axis=-1)
