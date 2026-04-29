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
