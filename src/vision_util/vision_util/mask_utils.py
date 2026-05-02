"""Mask post-processing helpers shared across vision nodes."""

from __future__ import annotations

import cv2
import numpy as np


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return the single largest connected component of a binary mask.

    Connectivity is 4 — diagonal-only neighbours are not joined, so thin
    bridges split. Background is label 0; the largest non-background
    component wins by area.

    Fast paths:
      * Empty mask (sum == 0): returned as-is.
      * Already a single component: returned as-is, no copy.

    The output dtype matches the input dtype (bool stays bool, uint8 stays
    uint8). Any non-zero input pixel counts as foreground.
    """
    if mask is None:
        return mask
    if mask.size == 0 or not mask.any():
        return mask

    src = mask.astype(np.uint8, copy=False)
    if src.dtype != np.uint8:
        src = src.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        src, connectivity=4
    )
    if n_labels <= 2:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    winner = 1 + int(np.argmax(areas))
    return (labels == winner).astype(mask.dtype, copy=False)


def largest_connected_component_in_bbox(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Return the largest connected component after confining mask to bbox."""
    if mask is None:
        return mask
    if mask.size == 0 or not mask.any():
        return mask

    h, w = mask.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros_like(mask)

    confined = np.zeros_like(mask)
    confined[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return largest_connected_component(confined)
