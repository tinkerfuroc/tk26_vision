"""ROS-free robust 3D-centroid reduction shared by the node and ptbench.

Both ``person_track_node._calculate_centroid`` and
``ptbench.common.geometry.centroid_from_bbox_depth`` MUST reduce a set of
per-pixel 3D points through this one function so the live tracker and the
benchmark never silently disagree (enforced by a parity test).

Reduction (camera optical frame, x=right, y=down, z=forward):
  1. Compute median z.
  2. Drop points with |z - median_z| > Z_OUTLIER_M (depth-noise rejection).
     If that leaves nothing (degenerate), keep the original set.
  3. Lateral x, y = MEDIAN over the kept set (robust to limb/edge pixels);
     z = MEDIAN over the kept set.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

Z_OUTLIER_M = 0.4


def reduce_centroid(obj_pts: np.ndarray) -> Tuple[float, float, float]:
    """Reduce an (N, 3) array of 3D points to one robust centroid.

    Args:
        obj_pts: (N, 3) float array of camera-frame XYZ points (meters).

    Returns:
        (x, y, z) as plain Python floats. Caller guarantees N >= 1.
    """
    pts = np.asarray(obj_pts, dtype=np.float64)
    z = pts[:, 2]
    median_z = float(np.median(z))
    keep = np.abs(z - median_z) <= Z_OUTLIER_M
    kept = pts[keep]
    if kept.shape[0] == 0:
        kept = pts
    x = float(np.median(kept[:, 0]))
    y = float(np.median(kept[:, 1]))
    zc = float(np.median(kept[:, 2]))
    return x, y, zc
