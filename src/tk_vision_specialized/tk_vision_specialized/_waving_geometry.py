"""Pure box/depth helpers for the waving VLM fallback.

No ROS, no network: these turn a VLM whole-person box into a 3D centroid using
the back-projected XYZ grid the waving server already computes. A box that
overlaps a YOLO person seg-mask reuses that mask (clean silhouette median);
otherwise it falls back to the valid depth inside the box, expanding once if the
box is too sparse.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def box_iou(a, b) -> float:
    """Intersection-over-union of two xyxy boxes. 0.0 if they do not overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_in_box(box, other) -> bool:
    """Return True if the centre of box lies inside other."""
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return other[0] <= cx <= other[2] and other[1] <= cy <= other[3]


def is_duplicate_box(box, existing_boxes: Sequence, *, iou_thresh: float) -> bool:
    """True if box duplicates any existing box (IoU >= thresh or center inside)."""
    for other in existing_boxes:
        if box_iou(box, other) >= iou_thresh or _center_in_box(box, other):
            return True
    return False


def _centroid_over_mask(points: np.ndarray, mask: np.ndarray):
    """Mean XY + median Z over the True pixels of mask, or None if empty."""
    if not mask.any():
        return None
    pts = points[mask]
    centroid = np.mean(pts, axis=0)
    centroid[2] = np.median(pts[:, 2])
    return centroid


def _expand(box, factor, w, h):
    """Return a scaled-up version of box clamped to image bounds (w, h)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * factor, (y2 - y1) * factor
    nx1 = max(0, int(round(cx - bw / 2.0)))
    ny1 = max(0, int(round(cy - bh / 2.0)))
    nx2 = min(w, int(round(cx + bw / 2.0)))
    ny2 = min(h, int(round(cy + bh / 2.0)))
    return nx1, ny1, nx2, ny2


def centroid_from_box(points: np.ndarray, validmask: np.ndarray, box_xyxy,
                      person_records: Sequence, *, mask_iou_thresh: float = 0.3,
                      min_valid: int = 10):
    """Return (centroid_xyz, used_mask) for a VLM box, or None.

    Tier 1: if box_xyxy overlaps a person_records seg-mask (box-vs-mask-bbox IoU
            >= mask_iou_thresh and that record has a mask), reuse mask & valid.
    Tier 2: else the box rectangle & valid; if < min_valid px, expand once x1.5.
    Returns the XYZ centroid (mean XY, median Z) and the bool mask actually used
    (so the caller can log it), or None when no usable depth exists.
    """
    h, w = validmask.shape

    best_mask = None
    best_iou = mask_iou_thresh
    for rec in person_records:
        rx1, ry1, rx2, ry2, rmask = rec[0], rec[1], rec[2], rec[3], rec[4]
        if rmask is None:
            continue
        iou = box_iou(box_xyxy, (rx1, ry1, rx2, ry2))
        if iou >= best_iou:
            best_iou = iou
            best_mask = rmask
    if best_mask is not None:
        combined = best_mask & validmask
        if combined.sum() >= min_valid:
            centroid = _centroid_over_mask(points, combined)
            if centroid is not None:
                return centroid, combined

    for factor in (1.0, 1.5):
        x1, y1, x2, y2 = _expand(box_xyxy, factor, w, h)
        rect = np.zeros((h, w), dtype=bool)
        rect[y1:y2, x1:x2] = True
        combined = rect & validmask
        if combined.sum() >= min_valid:
            centroid = _centroid_over_mask(points, combined)
            if centroid is not None:
                return centroid, combined
    return None
