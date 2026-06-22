"""ROS-free robust 3D-centroid reduction — ptbench-local mirror.

This is a pure, numpy-only copy of
``vision_track.core.centroid.reduce_centroid`` kept inside ptbench so the
benchmark stays portable (no colcon workspace / ROS on the path). The live
node keeps importing its own copy; a ROS-free parity test
(``tests/test_centroid_reduction.py``) loads the vision_track copy by file
path and asserts the two produce identical output, guarding against silent
desync.

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
