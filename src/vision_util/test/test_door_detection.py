import numpy as np
import pytest

from vision_util._door_logic import DoorResult, depth_to_meters, evaluate_door

PARAMS = dict(open_threshold_m=1.5, center_patch_px=30, min_valid_px=50)


def _img(fill_m, h=576, w=640):
    return np.full((h, w), fill_m, dtype=np.float32)


def test_closed_near_surface():
    r = evaluate_door(_img(1.0), **PARAMS)
    assert r.is_open == 0
    assert r.valid_count == 900          # 30x30 center patch
    assert r.median_m == pytest.approx(1.0)


def test_open_far_surface():
    r = evaluate_door(_img(3.0), **PARAMS)
    assert r.is_open == 1
    assert r.median_m == pytest.approx(3.0)


def test_open_when_center_all_invalid():
    # open door beyond range -> center returns 0 (invalid)
    r = evaluate_door(_img(0.0), **PARAMS)
    assert r.is_open == 1
    assert r.valid_count == 0
    assert r.median_m == 0.0


def test_boundary_valid_count():
    img = _img(0.0)
    cy, cx = 576 // 2, 640 // 2
    img[cy:cy + 5, cx:cx + 10] = 1.0     # 50 near pixels inside the 30x30 patch
    r = evaluate_door(img, **PARAMS)
    assert r.valid_count == 50
    assert r.is_open == 0                 # 50 >= 50 and median 1.0 < 1.5
    img[cy, cx] = 0.0                     # one fewer valid -> open
    r2 = evaluate_door(img, **PARAMS)
    assert r2.valid_count == 49
    assert r2.is_open == 1


def test_boundary_threshold():
    assert evaluate_door(_img(1.49), **PARAMS).is_open == 0
    assert evaluate_door(_img(1.51), **PARAMS).is_open == 1


def test_non_finite_excluded():
    img = _img(np.nan)
    cy, cx = 576 // 2, 640 // 2
    img[cy:cy + 8, cx:cx + 8] = 1.0      # 64 finite near pixels
    r = evaluate_door(img, **PARAMS)
    assert r.valid_count == 64
    assert r.is_open == 0
    assert np.isfinite(r.median_m)


def test_small_image_clamps():
    r = evaluate_door(_img(1.0, h=10, w=10), **PARAMS)
    assert r.valid_count == 100           # whole 10x10 clamped patch, valid & near
    assert r.is_open == 0


def test_depth_to_meters_16uc1():
    arr = np.array([[1500, 0], [800, 2000]], dtype=np.uint16)
    out = depth_to_meters(arr, '16UC1')
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[1.5, 0.0], [0.8, 2.0]])


def test_depth_to_meters_32fc1_passthrough():
    out = depth_to_meters(np.array([[1.5, 0.0]], dtype=np.float32), '32FC1')
    np.testing.assert_allclose(out, [[1.5, 0.0]])


def test_depth_to_meters_unsupported_raises():
    with pytest.raises(ValueError):
        depth_to_meters(np.zeros((2, 2), np.uint8), 'rgb8')
