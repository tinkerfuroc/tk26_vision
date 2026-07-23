"""Golden tests for the node-local depth reprojection variants."""
from __future__ import annotations

import numpy as np

from vision_util.depth_reproject import (
    decode_depth_metres,
    depth_image_to_points,
    follow_head_optical_points,
    realsense_body_axes_points,
    tracking_optical_points,
    waving_optical_points,
)


K = np.array([
    [4.0, 0.0, 1.0],
    [0.0, 5.0, 0.5],
    [0.0, 0.0, 1.0],
])


def _old_waving(depth_mm):
    depth = depth_mm.astype(float) / 1000.0
    height, width = depth.shape
    valid = (depth > 1e-6) & (depth < 10.0)
    clipped = np.clip(depth, 0.0, 10.0)
    u = np.arange(width, dtype=float)[None, :]
    v = np.arange(height, dtype=float)[:, None]
    x = (u - K[0, 2]) * clipped / K[0, 0]
    y = (v - K[1, 2]) * clipped / K[1, 1]
    return np.stack([x, y, clipped], axis=2), valid


def _old_realsense_body(depth):
    height, width = depth.shape
    rows = np.repeat(np.arange(height)[:, None], width, axis=1)
    cols = np.repeat(np.arange(width)[None, :], height, axis=0)
    x = (rows - K[0, 2]) * depth / K[0, 0]
    y = (cols - K[1, 2]) * depth / K[1, 1]
    valid = np.ones_like(depth)
    valid[depth > 10.0] = 0
    valid[depth < 1e-6] = 0
    clipped = depth.copy()
    clipped[clipped > 10.0] = 10.0
    clipped[clipped < 1e-6] = 0.0
    return np.stack([x, y, clipped], axis=2), valid


def _old_optical(depth, valid_band):
    height, width = depth.shape
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = (v - K[1, 2]) * depth / K[1, 1]
    points = np.stack([x, y, depth], axis=-1)
    valid = (depth > valid_band[0]) & (depth < valid_band[1])
    return points, valid


def test_waving_variant_matches_local_math():
    depth_mm = np.array([[0, 1000, 12000], [500, 2000, 3000]], dtype=np.uint16)
    expected_points, expected_valid = _old_waving(depth_mm)

    points, valid = waving_optical_points(decode_depth_metres(depth_mm), K)

    np.testing.assert_allclose(points, expected_points)
    np.testing.assert_array_equal(valid, expected_valid)


def test_realsense_body_axes_variant_matches_bug_compatible_math():
    depth = np.array(
        [[0.0, 1.0, 12.0], [0.5, 2.0, 3.0]], dtype=np.float64
    )
    expected_points, expected_valid = _old_realsense_body(depth)

    points, valid = realsense_body_axes_points(depth, K)

    np.testing.assert_allclose(points, expected_points)
    np.testing.assert_array_equal(valid, expected_valid)


def test_person_track_variant_matches_roi_local_math():
    depth = np.ones((40, 40), dtype=np.float32)
    depth[10, 10] = 0.05
    depth[30, 30] = 11.0
    expected_points, expected_valid = _old_optical(depth, (0.1, 10.0))
    roi_mask = np.zeros_like(expected_valid)
    roi_mask[4:37, 4:37] = True
    expected_points[~roi_mask] = 0
    expected_valid &= roi_mask

    points, valid = tracking_optical_points(
        depth, K, roi=(20, 20, 21, 21)
    )

    np.testing.assert_allclose(points, expected_points)
    np.testing.assert_array_equal(valid, expected_valid)


def test_follow_head_variant_matches_local_math():
    depth = np.array(
        [[0.0, 0.001, 0.002], [1.0, 9.0, 10.0]], dtype=np.float32
    )
    expected_points, expected_valid = _old_optical(depth, (1e-3, 10.0))

    points, valid = follow_head_optical_points(depth, K)

    np.testing.assert_allclose(points, expected_points)
    np.testing.assert_array_equal(valid, expected_valid)


def test_yolo_orbbec_explicit_negative_valid_band_preserves_zero_depth():
    depth = np.array([[0.0, 2.0, 11.0]], dtype=np.float32)

    points, valid = depth_image_to_points(
        depth,
        K,
        valid_band=(-10.0, 10.0),
        return_valid_mask=True,
    )

    assert points.shape == (1, 3, 3)
    np.testing.assert_array_equal(valid, [[True, True, False]])
