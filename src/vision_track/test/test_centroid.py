"""Unit tests for the ROS-free robust-centroid reduction helper."""
import numpy as np

from vision_track.core.centroid import reduce_centroid, Z_OUTLIER_M


def test_median_lateral_pure():
    # Three points: lateral x has one outlier; median ignores it.
    pts = np.array(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [10.0, 0.0, 2.0]], dtype=np.float64
    )
    x, y, z = reduce_centroid(pts)
    assert abs(x - 0.0) < 1e-9   # median x is 0.0, not the mean 3.33
    assert abs(y - 0.0) < 1e-9
    assert abs(z - 2.0) < 1e-9


def test_z_outlier_rejected_before_reduce():
    # 10 inliers at z=3.0 with x=1.0; one far-z outlier at z=9.0, x=5.0.
    inliers = np.array([[1.0, 0.0, 3.0]] * 10, dtype=np.float64)
    outlier = np.array([[5.0, 0.0, 9.0]], dtype=np.float64)
    pts = np.concatenate([inliers, outlier], axis=0)
    x, y, z = reduce_centroid(pts)
    # outlier z (|9-3|=6 > 0.4) dropped → x median stays 1.0, z stays 3.0
    assert abs(x - 1.0) < 1e-9
    assert abs(z - 3.0) < 1e-9


def test_z_outlier_threshold_constant():
    assert Z_OUTLIER_M == 0.4


def test_returns_python_floats():
    pts = np.array([[1.0, 2.0, 3.0]] * 12, dtype=np.float64)
    x, y, z = reduce_centroid(pts)
    assert isinstance(x, float) and isinstance(y, float) and isinstance(z, float)


def test_all_dropped_falls_back_to_unfiltered_median():
    # Degenerate: only 1 point (can't reject); reduce must still return it.
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    x, y, z = reduce_centroid(pts)
    assert (x, y, z) == (1.0, 2.0, 3.0)
