"""_calculate_centroid must NOT fall back to all-bbox depth.

When the segmentation mask overlaps fewer than 10 valid-depth pixels, the
old code substituted every valid depth pixel in the bbox — the median of the
BACKGROUND behind a small/false detection, i.e. a concrete 3D point where
nobody stands (2026-07-02 phantom follow-point incident). It must return
None instead; the caller already treats None as "no publishable point".
"""
import types
from types import SimpleNamespace

import numpy as np
import pytest

PersonTrackNode = pytest.importorskip(
    "vision_track.person_track_node").PersonTrackNode


def _node():
    node = SimpleNamespace(torso_band_enabled=False)
    node._calculate_centroid = types.MethodType(
        PersonTrackNode._calculate_centroid, node)
    return node


def _scene():
    """40x40 scene: 'person' mask has zero valid depth; background has lots."""
    h = w = 40
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:, :, 2] = 5.0                     # background depth: 5 m everywhere
    valid = np.ones((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[10:20, 10:20] = 1.0                  # 100-px person mask...
    valid[10:20, 10:20] = False               # ...with no valid depth on it
    return points, mask, valid


def test_no_all_bbox_depth_fallback():
    points, mask, valid = _scene()
    result = _node()._calculate_centroid(points, mask, valid, (0, 0, 40, 40))
    assert result is None  # old code: background Point at z ~= 5.0


def test_masked_depth_still_produces_centroid():
    points, mask, valid = _scene()
    valid[10:20, 10:20] = True                # now the mask HAS real depth
    points[10:20, 10:20, 2] = 2.0             # person at 2 m
    result = _node()._calculate_centroid(points, mask, valid, (0, 0, 40, 40))
    assert result is not None
    assert abs(result.z - 2.0) < 1e-3
