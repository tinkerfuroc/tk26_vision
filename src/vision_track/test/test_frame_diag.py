"""Unit tests for the ROS-free per-frame perf/quality diagnostic."""
import numpy as np

from vision_track.core.frame_diag import compute_frame_diag


def _points(h, w, z):
    # Mirror the node's deprojection: x varies with column, y with row, z = depth
    # (person_track_node._depth_image_to_points lines 331-333). A flat x=0 plane
    # would make mask vs bbox centroids indistinguishable laterally.
    pts = np.zeros((h, w, 3), dtype=np.float32)
    cols = np.arange(w, dtype=np.float32)[None, :]
    rows = np.arange(h, dtype=np.float32)[:, None]
    pts[:, :, 0] = np.broadcast_to(cols, (h, w))
    pts[:, :, 1] = np.broadcast_to(rows, (h, w))
    pts[:, :, 2] = z
    return pts


def test_mask_and_valid_counts():
    h, w = 100, 100
    pts = _points(h, w, 2.0)
    valid = np.ones((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:30, 10:30] = 1  # 400 px
    bbox = (0, 0, 100, 100)
    diag = compute_frame_diag(pts, mask, valid, bbox)
    assert diag["mask_pixel_count"] == 400
    assert diag["valid_pixel_count"] == 10000
    assert diag["used_mask"] is True  # mask has >=10 px in bbox


def test_used_mask_false_when_mask_too_sparse():
    h, w = 100, 100
    pts = _points(h, w, 2.0)
    valid = np.ones((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[0:1, 0:3] = 1  # 3 px < 10 → fallback fires
    bbox = (0, 0, 100, 100)
    diag = compute_frame_diag(pts, mask, valid, bbox)
    assert diag["used_mask"] is False


def test_depth_z_iqr_computed():
    h, w = 100, 100
    pts = np.zeros((h, w, 3), dtype=np.float32)
    # Half at z=2.0, half at z=4.0 → IQR spans 2.0.
    pts[:h // 2, :, 2] = 2.0
    pts[h // 2:, :, 2] = 4.0
    valid = np.ones((h, w), dtype=bool)
    mask = np.ones((h, w), dtype=np.uint8)
    bbox = (0, 0, 100, 100)
    diag = compute_frame_diag(pts, mask, valid, bbox)
    assert diag["depth_z_iqr"] > 1.5


def test_both_centroids_present():
    h, w = 100, 100
    pts = _points(h, w, 2.0)
    valid = np.ones((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:90, 10:50] = 1  # left-biased mask
    bbox = (0, 0, 100, 100)
    diag = compute_frame_diag(pts, mask, valid, bbox)
    assert diag["mask_centroid"] is not None
    assert diag["bbox_centroid"] is not None
    # mask is left-biased → mask centroid x < bbox centroid x
    assert diag["mask_centroid"][0] < diag["bbox_centroid"][0]


def test_no_valid_points_marks_no_centroid():
    h, w = 100, 100
    pts = _points(h, w, 2.0)
    valid = np.zeros((h, w), dtype=bool)  # nothing valid
    mask = np.ones((h, w), dtype=np.uint8)
    bbox = (0, 0, 100, 100)
    diag = compute_frame_diag(pts, mask, valid, bbox)
    assert diag["bbox_centroid"] is None
    assert diag["no_centroid"] is True
