"""ROS-free per-frame perf/quality diagnostic for the person tracker.

Computes mask/valid pixel counts, whether the <10px mask→bbox fallback would
fire (used_mask), depth z IQR over the kept points, BOTH the mask-filtered and
bbox-only centroids (via the shared reduce_centroid), and a no_centroid flag.
Logged only when perf_logging_enabled. Pure; no rclpy/torch.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .centroid import reduce_centroid


def _roi(arr, bbox):
    x1, y1, x2, y2 = bbox
    h, w = arr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return arr[y1:y2, x1:x2], (x1, y1, x2, y2)


def _centroid_from(points_roi, sel_mask) -> Optional[tuple]:
    if sel_mask.sum() < 10:
        return None
    obj = points_roi[np.nonzero(sel_mask)]
    if obj.ndim != 2 or obj.shape[0] == 0:
        return None
    return reduce_centroid(obj)


def compute_frame_diag(points, mask, valid_mask, bbox) -> dict:
    """Return a per-frame diagnostic dict (see module docstring)."""
    points_roi, (x1, y1, x2, y2) = _roi(points, bbox)
    valid_roi, _ = _roi(valid_mask, bbox)
    valid_roi_b = valid_roi.astype(bool)

    if mask is not None and mask.shape[0] > 0 and mask.shape[1] > 0:
        mask_roi = mask[y1:y2, x1:x2].astype(bool)
    else:
        mask_roi = np.zeros_like(valid_roi_b)

    mask_sel = mask_roi & valid_roi_b
    mask_pixel_count = int(mask_roi.sum())
    valid_pixel_count = int(valid_mask.astype(bool).sum())
    used_mask = bool(mask_sel.sum() >= 10)

    bbox_centroid = _centroid_from(points_roi, valid_roi_b)
    mask_centroid = _centroid_from(points_roi, mask_sel)

    # z IQR over the chosen point set (mask if used, else bbox-valid).
    sel = mask_sel if used_mask else valid_roi_b
    if sel.sum() >= 2:
        zvals = points_roi[np.nonzero(sel)][:, 2]
        q75, q25 = np.percentile(zvals, [75, 25])
        depth_z_iqr = float(q75 - q25)
    else:
        depth_z_iqr = 0.0

    no_centroid = (bbox_centroid is None) and (mask_centroid is None)

    return {
        "mask_pixel_count": mask_pixel_count,
        "valid_pixel_count": valid_pixel_count,
        "used_mask": used_mask,
        "depth_z_iqr": depth_z_iqr,
        "mask_centroid": mask_centroid,
        "bbox_centroid": bbox_centroid,
        "no_centroid": no_centroid,
    }
