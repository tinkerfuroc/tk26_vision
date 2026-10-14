"""Numerical localization tests, without ROS or OpenCV."""
import numpy as np
import pytest

from tk_vision_specialized.scene_geometry import estimate_position, matched_depth_index


K = [100., 0., 50., 0., 100., 40., 0., 0., 1.]


def test_depth_selection_uses_capture_stamp_not_arrival_order():
    assert matched_depth_index([1_980_000_000, 2_070_000_000], [9., 9.5],
                               2_000_000_000, 10., 2., 0.1) == 0


@pytest.mark.parametrize('stamps,received', [([], []), ([0], [9.]),
    ([1_000_000_000], [9.]), ([2_000_000_000], [1.])])
def test_missing_unsynchronized_or_stale_depth_rejected(stamps, received):
    with pytest.raises(ValueError, match='No fresh depth'):
        matched_depth_index(stamps, received, 2_000_000_000, 10., 2., 0.1)


def test_known_plane_reprojection():
    xyz = estimate_position(np.full((80, 100), 2.), K, (30, 20, 70, 60))
    np.testing.assert_allclose(xyz, [-0.01, -0.01, 2.])


def test_border_background_and_sparse_outliers_do_not_shift_position():
    depth = np.full((80, 100), 7.)
    depth[30:50, 40:60] = 2.
    depth[30:32, 40:42] = 6.
    assert estimate_position(depth, K, (30, 20, 70, 60))[2] == 2.


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf, 9.])
def test_no_usable_depth_is_failure(value):
    with pytest.raises(ValueError, match='Insufficient'):
        estimate_position(np.full((80, 100), value), K, (30, 20, 70, 60))


def test_too_few_pixels_fail_instead_of_returning_zero():
    depth = np.zeros((80, 100))
    depth[40:42, 50:52] = 2.
    with pytest.raises(ValueError, match='Insufficient'):
        estimate_position(depth, K, (30, 20, 70, 60))


@pytest.mark.parametrize('box', [(-1, 0, 20, 20), (0, 0, 101, 80), (10, 10, 10, 20)])
def test_invalid_boxes_fail(box):
    with pytest.raises(ValueError):
        estimate_position(np.ones((80, 100)), K, box)


@pytest.mark.parametrize('focal', [0., -1., np.nan, np.inf])
def test_invalid_calibration_fails(focal):
    k = list(K)
    k[0] = focal
    with pytest.raises(ValueError):
        estimate_position(np.ones((80, 100)), k, (30, 20, 70, 60))
