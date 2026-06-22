"""Tests for ptbench.common.geometry — bbox+depth → 3D centroid."""
import math

import numpy as np
import pytest

from ptbench.common.geometry import (
    centroid_from_bbox_depth,
    dist3d,
    lateral_range,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_depth_mm(H: int, W: int, depth_m: float, bbox=None, fill_depth_m: float = 0.0) -> np.ndarray:
    """Make a HxW uint16 depth image (mm). bbox region filled with depth_m."""
    arr = np.full((H, W), int(fill_depth_m * 1000), dtype=np.uint16)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        arr[y1:y2, x1:x2] = int(depth_m * 1000)
    return arr


def pinhole_K(fx=500.0, fy=500.0, cx=320.0, cy=240.0) -> list:
    """Return ROS row-major K = [fx,0,cx, 0,fy,cy, 0,0,1]."""
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Basic centroid correctness
# ---------------------------------------------------------------------------

class TestCentroidFromBboxDepth:
    def test_constant_depth_rectangle_closed_form(self):
        """Centroid of a filled rectangle at constant depth must match closed-form."""
        H, W = 480, 640
        fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
        K = pinhole_K(fx, fy, cx, cy)
        depth_m = 3.0
        bbox = (100, 80, 200, 160)  # x1,y1,x2,y2

        depth = make_depth_mm(H, W, depth_m, bbox=bbox)
        result = centroid_from_bbox_depth(depth, K, bbox)

        assert result is not None
        xc, yc, zc = result

        # Expected: centroid pixel = midpoint of bbox
        u_mid = (100 + 200) / 2.0  # but mid-pixel of range [100,200)
        # The actual midpoint of pixels 100..199 is 149.5
        u_mid = (100 + 199) / 2.0
        v_mid = (80 + 159) / 2.0
        expected_x = (u_mid - cx) * depth_m / fx
        expected_y = (v_mid - cy) * depth_m / fy
        expected_z = depth_m

        assert abs(xc - expected_x) < 0.05
        assert abs(yc - expected_y) < 0.05
        assert abs(zc - expected_z) < 0.002  # median z == constant depth

    def test_median_z_robustness(self):
        """Adding outlier-depth pixels should not shift the median z significantly."""
        H, W = 480, 640
        K = pinhole_K()
        depth_m = 2.5
        bbox = (50, 50, 250, 250)  # 200x200 pixels = 40000 total

        depth = make_depth_mm(H, W, depth_m, bbox=bbox)
        # Corrupt a few pixels (< 5 %) with outlier depth
        outlier_depth_m = 9.0
        depth[51:53, 51:60] = int(outlier_depth_m * 1000)  # 20 pixels out of 40000

        result = centroid_from_bbox_depth(depth, K, bbox)
        assert result is not None
        _, _, zc = result
        # Median should still be very close to the true depth
        assert abs(zc - depth_m) < 0.05

    def test_mask_restricts_region(self):
        """With a mask that covers only the left half, centroid shifts left."""
        H, W = 480, 640
        fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
        K = pinhole_K(fx, fy, cx, cy)
        depth_m = 2.0
        bbox = (100, 100, 300, 200)  # 200 wide, 100 tall

        depth = make_depth_mm(H, W, depth_m, bbox=bbox)

        # Mask: only left half of the bbox (columns 100..199 in image coords)
        mask = np.zeros((H, W), dtype=np.float32)
        mask[100:200, 100:200] = 1.0  # left half of bbox

        result_masked = centroid_from_bbox_depth(depth, K, bbox, mask=mask)
        result_unmasked = centroid_from_bbox_depth(depth, K, bbox)

        assert result_masked is not None
        assert result_unmasked is not None
        xm, _, _ = result_masked
        xu, _, _ = result_unmasked
        # Masked centroid (left half) should be to the left (smaller x) than unmasked (full)
        assert xm < xu

    def test_too_few_valid_returns_none(self):
        """If <10 valid depth pixels, return None."""
        H, W = 480, 640
        K = pinhole_K()
        depth_m = 2.0
        bbox = (100, 100, 104, 102)  # 4x2 = 8 pixels — below threshold

        depth = make_depth_mm(H, W, depth_m, bbox=bbox)
        result = centroid_from_bbox_depth(depth, K, bbox)
        assert result is None

    def test_mask_fallback_when_mask_too_sparse(self):
        """If mask gives <10 points, fall back to valid-without-mask."""
        H, W = 480, 640
        K = pinhole_K()
        depth_m = 2.0
        bbox = (100, 100, 200, 200)  # 100x100 = 10000 pixels, all valid

        depth = make_depth_mm(H, W, depth_m, bbox=bbox)

        # Mask only 3 pixels — triggers fallback
        mask = np.zeros((H, W), dtype=np.float32)
        mask[100:101, 100:103] = 1.0

        result = centroid_from_bbox_depth(depth, K, bbox, mask=mask)
        # After fallback to no-mask, 10000 valid pixels → should succeed
        assert result is not None

    def test_zero_depth_excluded(self):
        """Pixels with depth=0 (invalid) must be excluded."""
        H, W = 480, 640
        K = pinhole_K()
        bbox = (50, 50, 150, 150)  # 100x100

        depth = np.zeros((H, W), dtype=np.uint16)  # all zero → all invalid
        result = centroid_from_bbox_depth(depth, K, bbox)
        assert result is None

    def test_K_as_3x3_ndarray(self):
        """K may be passed as 3x3 ndarray."""
        H, W = 480, 640
        K_list = pinhole_K()
        K_arr = np.array(K_list).reshape(3, 3)
        depth_m = 3.0
        bbox = (100, 80, 200, 160)
        depth = make_depth_mm(H, W, depth_m, bbox=bbox)

        r1 = centroid_from_bbox_depth(depth, K_list, bbox)
        r2 = centroid_from_bbox_depth(depth, K_arr, bbox)
        assert r1 is not None and r2 is not None
        for a, b in zip(r1, r2):
            assert abs(a - b) < 1e-9

    def test_bbox_clamped_to_image_bounds(self):
        """bbox extending outside image bounds should be clamped, not error."""
        H, W = 480, 640
        K = pinhole_K()
        depth_m = 2.0
        # bbox extends 50px beyond right/bottom
        bbox = (600, 460, 700, 550)
        depth = make_depth_mm(H, W, depth_m, bbox=(600, 460, 640, 480))
        # After clamping: 40 x 20 = 800 pixels — should succeed or not raise
        result = centroid_from_bbox_depth(depth, K, bbox)
        # Either valid result or None (if clamped region has fewer than 10 valid)
        # The clamped area is 40x20=800, all filled → should succeed
        assert result is not None

    def test_depth_outside_range_excluded(self):
        """Pixels outside [min_depth, max_depth] must be excluded."""
        H, W = 480, 640
        K = pinhole_K()
        bbox = (50, 50, 150, 150)

        # Fill with depth just outside max_depth=10.0 → all invalid
        depth = np.full((H, W), int(11.0 * 1000), dtype=np.uint16)
        result = centroid_from_bbox_depth(depth, K, bbox, max_depth=10.0)
        assert result is None

    def test_returns_python_floats(self):
        """Return type must be tuple of plain Python floats, not numpy scalars."""
        H, W = 480, 640
        K = pinhole_K()
        depth_m = 2.0
        bbox = (100, 100, 200, 200)
        depth = make_depth_mm(H, W, depth_m, bbox=bbox)
        result = centroid_from_bbox_depth(depth, K, bbox)
        assert result is not None
        x, y, z = result
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)


# ---------------------------------------------------------------------------
# dist3d and lateral_range
# ---------------------------------------------------------------------------

class TestDist3d:
    def test_zero_distance(self):
        assert dist3d((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == pytest.approx(0.0)

    def test_unit_distance(self):
        assert dist3d((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == pytest.approx(1.0)

    def test_3d_distance(self):
        # sqrt(3² + 4² + 0²) = 5
        assert dist3d((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == pytest.approx(5.0)

    def test_negative_coords(self):
        assert dist3d((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == pytest.approx(2.0)


class TestLateralRange:
    def test_pure_range(self):
        """Same x,y → lateral=0, range=|dz|."""
        lat, rng = lateral_range((0.0, 0.0, 2.0), (0.0, 0.0, 3.5))
        assert lat == pytest.approx(0.0)
        assert rng == pytest.approx(1.5)

    def test_pure_lateral(self):
        """Same z → range=0, lateral=sqrt(dx²+dy²)."""
        lat, rng = lateral_range((0.0, 0.0, 2.0), (3.0, 4.0, 2.0))
        assert lat == pytest.approx(5.0)
        assert rng == pytest.approx(0.0)

    def test_combined(self):
        a = (1.0, 2.0, 3.0)
        b = (4.0, 6.0, 5.0)
        dx, dy, dz = 3.0, 4.0, 2.0
        lat, rng = lateral_range(a, b)
        assert lat == pytest.approx(math.sqrt(dx ** 2 + dy ** 2))
        assert rng == pytest.approx(abs(dz))
