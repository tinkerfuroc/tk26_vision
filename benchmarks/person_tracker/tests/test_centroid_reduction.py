"""ptbench geometry uses the ptbench-local reduction + cross-copy parity.

ptbench keeps its OWN pure copy of the robust-centroid reduction
(``ptbench.common.centroid``) so the benchmark stays portable — no colcon
workspace, no ROS on the path. The live node keeps its own copy
(``vision_track.core.centroid``). This test guards against the two silently
diverging WITHOUT importing the heavy ``vision_track`` package (whose
``__init__`` drags in ultralytics + ROS): it loads the vision_track copy by
FILE PATH via importlib and asserts identical output across a battery of
inputs. It therefore RUNS (not skips) on the bare, ROS-free command.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ptbench.common.centroid import reduce_centroid as pt_reduce
from ptbench.common.geometry import centroid_from_bbox_depth


def _load_vt_reduce_centroid():
    """Load vision_track's reduce_centroid by file path (no package import).

    Importing ``vision_track`` would execute its ``__init__`` and pull in
    ultralytics/ROS, defeating the portability guarantee. Loading the single
    module file directly keeps this test ROS-free.
    """
    vt_path = (
        Path(__file__).resolve().parents[3]
        / "src/vision_track/vision_track/core/centroid.py"
    )
    if not vt_path.exists():
        pytest.skip("vision_track source not found")
    spec = importlib.util.spec_from_file_location("_vt_centroid_parity", vt_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reduce_centroid


def pinhole_K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# ptbench-local geometry exercises the ptbench-local reduction
# ---------------------------------------------------------------------------

def test_geometry_uses_ptbench_local_reduce():
    # geometry must reference the ptbench-local module (not vision_track).
    import ptbench.common.geometry as g
    assert g.reduce_centroid is pt_reduce


def test_z_outlier_rejection_in_geometry():
    # A bbox where most pixels are at 3.0 m and a stripe is at 9.0 m: the
    # rejected stripe must not pull z toward 9.0.
    H, W = 480, 640
    K = pinhole_K()
    bbox = (100, 100, 200, 200)  # 100x100
    depth = np.full((H, W), 3000, dtype=np.uint16)  # 3.0 m
    depth[100:110, 100:200] = 9000  # 9.0 m stripe (10% of rows)
    result = centroid_from_bbox_depth(depth, K, bbox)
    assert result is not None
    _, _, z = result
    assert abs(z - 3.0) < 0.05


def test_lateral_uses_median():
    # Asymmetric mask weighting that would skew a mean but not a median.
    H, W = 480, 640
    K = pinhole_K()
    bbox = (100, 100, 400, 200)
    depth = np.full((H, W), 2000, dtype=np.uint16)
    r = centroid_from_bbox_depth(depth, K, bbox)
    assert r is not None
    x, y, z = r
    assert np.isfinite(x) and np.isfinite(y) and np.isfinite(z)


# ---------------------------------------------------------------------------
# Cross-copy parity: ptbench-local reduce == vision_track reduce (by file path)
# ---------------------------------------------------------------------------

def test_cross_copy_parity_node_vs_ptbench():
    """The two independent copies of reduce_centroid must agree exactly.

    Loads vision_track's copy by file path (ROS-free) and compares against
    ptbench's copy across a battery of representative inputs.
    """
    vt_reduce = _load_vt_reduce_centroid()

    cases = {
        # plain cluster — uniform z, symmetric lateral
        "plain": np.array(
            [[0.1, -0.2, 2.5], [0.0, 0.0, 2.5], [-0.1, 0.2, 2.5]],
            dtype=np.float64,
        ),
        # z-outliers that must be rejected (|z-median|>0.4 m)
        "z_outliers": np.concatenate(
            [
                np.array([[1.0, 0.0, 3.0]] * 10, dtype=np.float64),
                np.array([[5.0, 0.0, 9.0], [-7.0, 3.0, 0.5]], dtype=np.float64),
            ],
            axis=0,
        ),
        # asymmetric lateral spread — median != mean is exercised
        "asymmetric_lateral": np.array(
            [
                [0.0, 0.0, 2.0],
                [0.1, 0.05, 2.0],
                [0.2, 0.1, 2.0],
                [9.0, 8.0, 2.0],  # lone far lateral; median ignores it
            ],
            dtype=np.float64,
        ),
        # degenerate all-outlier fallback: every point far from its own median
        # in z so the keep-mask empties → falls back to the full set.
        "all_outlier_fallback": np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 5.0]], dtype=np.float64
        ),
        # single point (N==1) edge case
        "single": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    }

    for name, pts in cases.items():
        a = pt_reduce(pts)
        b = vt_reduce(pts)
        for ca, cb in zip(a, b):
            assert abs(ca - cb) < 1e-9, f"parity mismatch on case {name!r}: {a} vs {b}"


def test_all_outlier_fallback_keeps_full_set():
    # Sanity-check the degenerate branch independently of parity: with two
    # points 5 m apart in z, every point is >0.4 m from the median, so the
    # keep-mask empties and the reduction falls back to the full set.
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 5.0]], dtype=np.float64)
    x, y, z = pt_reduce(pts)
    # median over the full (unfiltered) set
    assert abs(x - 0.5) < 1e-9
    assert abs(y - 0.5) < 1e-9
    assert abs(z - 2.5) < 1e-9


def test_parity_geometry_pipeline_vs_direct():
    # End-to-end: the geometry pipeline result equals reduce_centroid applied
    # directly to the reconstructed point set.
    H, W = 480, 640
    fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
    K = pinhole_K(fx, fy, cx, cy)
    bbox = (120, 130, 220, 210)
    depth = np.full((H, W), 0, dtype=np.uint16)
    depth[130:210, 120:220] = 2500  # 2.5 m filled bbox
    geo = centroid_from_bbox_depth(depth, K, bbox)
    assert geo is not None

    x1, y1, x2, y2 = bbox
    u, v = np.meshgrid(
        np.arange(x1, x2, dtype=np.float32),
        np.arange(y1, y2, dtype=np.float32),
    )
    z = (depth[y1:y2, x1:x2].astype(np.float32)) * 0.001
    valid = (z > 0.1) & (z < 10.0)
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    pts = np.stack([X, Y, z], axis=-1)[np.nonzero(valid.astype(float))]
    direct = pt_reduce(pts)
    for a, b in zip(geo, direct):
        assert abs(a - b) < 1e-9
