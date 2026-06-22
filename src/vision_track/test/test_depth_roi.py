"""Unit tests for the depth-unproject ROI-window helper."""
from vision_track.core.depth_roi import roi_window


def test_window_clamped_and_padded():
    # bbox near the top-left, pad 16, image 640x480.
    x0, y0, x1, y1 = roi_window((10, 5, 100, 200), w=640, h=480, pad=16)
    assert x0 == 0          # 10-16 clamped to 0
    assert y0 == 0          # 5-16 clamped to 0
    assert x1 == 116        # 100+16
    assert y1 == 216        # 200+16


def test_window_clamped_to_image_max():
    x0, y0, x1, y1 = roi_window((600, 460, 700, 500), w=640, h=480, pad=16)
    assert x1 == 640
    assert y1 == 480
    assert x0 == 584
    assert y0 == 444


def test_none_bbox_returns_full_frame():
    assert roi_window(None, w=640, h=480, pad=16) == (0, 0, 640, 480)


def test_degenerate_bbox_returns_full_frame():
    # x2<=x1 after clamp → full frame fallback (caller unprojects everything).
    assert roi_window((300, 300, 300, 300), w=640, h=480, pad=0) == (0, 0, 640, 480)
