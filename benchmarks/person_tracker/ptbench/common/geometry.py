"""bbox + depth -> 3D centroid (mirrors PersonTrackNode math).

Camera optical-frame convention (matches
``PersonTrackNode._depth_image_to_points`` / ``_calculate_centroid``):

    x = (u - cx) * z / fx   (right)
    y = (v - cy) * z / fy    (down)
    z = depth                 (forward)

So range = z and lateral = sqrt(x^2 + y^2). The centroid uses the median over
x/y/z with z-outlier rejection (ptbench-local ``ptbench.common.centroid.reduce_centroid``,
a pure mirror of ``vision_track.core.centroid``) so the offline benchmark and
the live node reduce points identically — enforced by a ROS-free parity test.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ptbench.common.centroid import reduce_centroid


def _unpack_K(K):
    """Return (fx, fy, cx, cy) from a len-9 row-major list or 3x3 ndarray."""
    arr = np.asarray(K, dtype=np.float64).reshape(-1)
    fx = float(arr[0])
    fy = float(arr[4])
    cx = float(arr[2])
    cy = float(arr[5])
    return fx, fy, cx, cy


def centroid_from_bbox_depth(
    depth_mm,
    K,
    bbox,
    mask=None,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> Optional[Tuple[float, float, float]]:
    """Compute a 3D centroid (m) from a depth image, intrinsics, and a bbox.

    Args:
        depth_mm: HxW depth image in millimeters (uint16 or float). Converted to
            meters via *0.001.
        K: camera intrinsics — len-9 row-major list or 3x3 ndarray.
        bbox: (x1, y1, x2, y2) in pixels; clamped to image bounds.
        mask: optional HxW mask; pixels with mask>0 are kept. If the masked set
            has <10 valid points, falls back to all valid points in the bbox.
        min_depth, max_depth: valid depth range in meters (exclusive bounds).

    Returns:
        (x, y, z) plain Python floats, or None if <10 valid points after the
        fallback.
    """
    fx, fy, cx, cy = _unpack_K(K)

    depth = np.asarray(depth_mm).astype(np.float32) * 0.001
    h, w = depth.shape[:2]

    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None

    roi_depth = depth[y1:y2, x1:x2]
    roi_valid = (roi_depth > min_depth) & (roi_depth < max_depth)

    if mask is not None:
        roi_mask = np.asarray(mask)[y1:y2, x1:x2]
        combined = roi_mask.astype(float) * roi_valid.astype(float)
        if combined.sum() < 10:
            combined = roi_valid.astype(float)
    else:
        combined = roi_valid.astype(float)

    if combined.sum() < 10:
        return None

    # Per-pixel 3D in the ROI. u,v are absolute image coordinates.
    u, v = np.meshgrid(
        np.arange(x1, x2, dtype=np.float32),
        np.arange(y1, y2, dtype=np.float32),
    )
    z = roi_depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)

    obj_pts = points[np.nonzero(combined)]
    if obj_pts.ndim != 2 or obj_pts.shape[0] == 0:
        return None

    cx_m, cy_m, cz_m = reduce_centroid(obj_pts)
    return cx_m, cy_m, cz_m


def dist3d(a, b) -> float:
    """Euclidean distance between two 3D points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def lateral_range(a, b) -> Tuple[float, float]:
    """Return (lateral, range_err) between two 3D points.

    lateral = sqrt(dx^2 + dy^2); range_err = abs(dz).
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy), abs(dz)
