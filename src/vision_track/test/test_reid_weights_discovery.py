"""Pure unit tests for discover_cached_reid_weights.

These exercise only the cache-discovery logic (os/pathlib + the filename
constant) against a temp cache dir — no model build, no network, no torch
inference. The helper itself imports nothing heavy.
"""
from vision_track.reid.reid_backbone import (
    MSMT17_OSNET_AIN_X1_0,
    discover_cached_reid_weights,
)


def _make_checkpoint(cache_dir, filename=MSMT17_OSNET_AIN_X1_0):
    path = cache_dir / filename
    path.write_bytes(b"not-a-real-checkpoint")  # presence is all the helper checks
    return path


def test_returns_path_when_msmt17_file_present(tmp_path):
    expected = _make_checkpoint(tmp_path)
    found = discover_cached_reid_weights("osnet_ain_x1_0", cache_dir=str(tmp_path))
    assert found == str(expected)


def test_returns_empty_when_file_absent(tmp_path):
    # Empty cache dir -> no checkpoint -> "".
    assert discover_cached_reid_weights("osnet_ain_x1_0", cache_dir=str(tmp_path)) == ""


def test_returns_empty_for_non_osnet_backbone(tmp_path):
    # Even with the file present, a non-osnet_ain_x1_0 backbone gets "".
    _make_checkpoint(tmp_path)
    assert discover_cached_reid_weights("osnet_x0_25", cache_dir=str(tmp_path)) == ""
    assert discover_cached_reid_weights("resnet50", cache_dir=str(tmp_path)) == ""


def test_constant_is_the_msmt17_filename():
    assert MSMT17_OSNET_AIN_X1_0.startswith("osnet_ain_x1_0_msmt17_")
    assert MSMT17_OSNET_AIN_X1_0.endswith(".pth")
