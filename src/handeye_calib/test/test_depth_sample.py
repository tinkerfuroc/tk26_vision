"""TDD for handeye_calib.depth_sample.deproject_corners.

Samples FFS metric depth (aligned to color, same K the solver uses) at the
detected ChArUco corner pixels and back-projects to camera-frame 3D points,
with a robust local-median window + validity mask. These points feed the
depth residual in the bundle adjust, which is what finally constrains the
optical-axis DOF that monocular planar PnP leaves weak.
"""
import numpy as np
from handeye_calib import depth_sample as ds


_K = np.array([[615.0, 0.0, 320.0],
               [0.0, 615.0, 240.0],
               [0.0, 0.0, 1.0]])


def _expected_xyz(u, v, z, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])


def test_deproject_flat_plane_recovers_pinhole_points():
    Z0 = 0.5
    depth = np.full((480, 640), Z0, np.float32)
    obs_px = np.array([[320.0, 240.0], [100.0, 80.0], [500.0, 400.0]])
    xyz, valid = ds.deproject_corners(obs_px, depth, _K)
    assert valid.all()
    for i, (u, v) in enumerate(obs_px):
        np.testing.assert_allclose(xyz[i], _expected_xyz(u, v, Z0, _K), atol=1e-4)


def test_deproject_marks_invalid_depth():
    depth = np.full((480, 640), 0.5, np.float32)
    depth[200:280, 280:360] = 0.0          # a hole around (320,240)
    obs_px = np.array([[320.0, 240.0], [100.0, 80.0]])
    xyz, valid = ds.deproject_corners(obs_px, depth, _K)
    assert valid[0] == False               # corner inside the hole
    assert valid[1] == True
    assert not np.isfinite(xyz[0]).all()    # invalid corner -> nan, never a fake 0


def test_deproject_robust_to_window_outliers():
    # Mostly Z0 in the window with a couple of speckle outliers -> median wins.
    Z0 = 0.7
    depth = np.full((480, 640), Z0, np.float32)
    depth[240, 320] = 5.0                   # speckle at the exact corner pixel
    depth[239, 319] = 0.0
    obs_px = np.array([[320.4, 240.3]])     # sub-pixel corner
    xyz, valid = ds.deproject_corners(obs_px, depth, _K, win=2)
    assert valid[0]
    np.testing.assert_allclose(xyz[0, 2], Z0, atol=1e-4)   # depth = median, not the speckle


def test_deproject_out_of_bounds_corner_is_invalid_no_crash():
    depth = np.full((480, 640), 0.5, np.float32)
    obs_px = np.array([[640.0, 240.0], [320.0, -3.0], [320.0, 240.0]])
    xyz, valid = ds.deproject_corners(obs_px, depth, _K)
    assert valid[0] == False and valid[1] == False
    assert valid[2] == True


def test_deproject_respects_z_range():
    depth = np.full((480, 640), 9.0, np.float32)   # beyond default z_max
    obs_px = np.array([[320.0, 240.0]])
    _, valid = ds.deproject_corners(obs_px, depth, _K, z_max=2.0)
    assert valid[0] == False
