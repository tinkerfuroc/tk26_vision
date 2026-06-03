"""Unit tests for _waving_geometry.py — pure box/depth helpers."""

from __future__ import annotations

import numpy as np

from tk_vision_specialized._waving_geometry import (
    box_iou,
    is_duplicate_box,
    centroid_from_box,
)


def test_box_iou_identical_is_one():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_box_iou_disjoint_is_zero():
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_box_iou_half_overlap():
    # 10x10 boxes sharing a 5x10 strip -> inter=50, union=150 -> 1/3.
    iou = box_iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert abs(iou - (50.0 / 150.0)) < 1e-6


def test_is_duplicate_box_by_iou():
    existing = [(0, 0, 10, 10)]
    assert is_duplicate_box((1, 1, 11, 11), existing, iou_thresh=0.3) is True
    assert is_duplicate_box((40, 40, 50, 50), existing, iou_thresh=0.3) is False


def test_is_duplicate_box_by_center_inside():
    # Low IoU but the new box's center sits inside an existing box -> duplicate.
    existing = [(0, 0, 100, 100)]
    assert is_duplicate_box((40, 40, 60, 60), existing, iou_thresh=0.99) is True


def _grid_with_depth(h, w, z_value, valid_region=None):
    """Build a (points, validmask) pair where points[...,2]=z_value.

    XY are arbitrary linear ramps; only Z (and validity) matter for the asserts.
    valid_region = (y0, y1, x0, x1) marks the only valid pixels (else all valid).
    """
    xs = np.tile(np.arange(w, dtype=float), (h, 1))
    ys = np.tile(np.arange(h, dtype=float)[:, None], (1, w))
    zs = np.full((h, w), float(z_value))
    points = np.stack([xs, ys, zs], axis=2)
    validmask = np.zeros((h, w), dtype=bool)
    if valid_region is None:
        validmask[:] = True
    else:
        y0, y1, x0, x1 = valid_region
        validmask[y0:y1, x0:x1] = True
    return points, validmask


def test_centroid_from_box_reuses_overlapping_mask():
    points, validmask = _grid_with_depth(100, 100, z_value=2.0)
    # A YOLO person mask covering a 20x20 patch at known depth 5.0.
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    points[10:30, 10:30, 2] = 5.0
    person_records = [(10, 10, 30, 30, mask)]
    out = centroid_from_box(points, validmask, (12, 12, 28, 28), person_records)
    assert out is not None
    centroid, used_mask = out
    assert abs(centroid[2] - 5.0) < 1e-6     # median Z from the reused mask
    assert used_mask.sum() > 0


def test_centroid_from_box_box_center_fallback_when_no_mask():
    points, validmask = _grid_with_depth(100, 100, z_value=3.0)
    out = centroid_from_box(points, validmask, (40, 40, 60, 60), person_records=[])
    assert out is not None
    centroid, _ = out
    assert abs(centroid[2] - 3.0) < 1e-6


def test_centroid_from_box_none_when_no_valid_depth():
    # Valid pixels only in a far corner; box + its expansion never reach them.
    points, validmask = _grid_with_depth(
        200, 200, z_value=3.0, valid_region=(190, 200, 190, 200))
    out = centroid_from_box(points, validmask, (10, 10, 20, 20), person_records=[])
    assert out is None
