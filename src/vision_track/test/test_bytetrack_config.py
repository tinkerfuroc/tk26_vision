import os
import pytest

yaml = pytest.importorskip("yaml")

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


def _load(name):
    with open(os.path.join(CFG_DIR, name)) as f:
        return yaml.safe_load(f)


def test_bytetrack_yaml_exists_and_is_bytetrack():
    cfg = _load("bytetrack.yaml")
    assert cfg["tracker_type"] == "bytetrack"


def test_bytetrack_low_conf_recovery_enabled():
    cfg = _load("bytetrack.yaml")
    # The low bin must sit below the detection conf we pass to model.track (0.15),
    # otherwise the two-stage recovery has nothing to recover.
    assert cfg["track_low_thresh"] <= 0.15
    assert cfg["track_high_thresh"] >= 0.2
    assert cfg["new_track_thresh"] >= cfg["track_high_thresh"]
    assert cfg["track_buffer"] >= 30


def test_default_yaml_has_phase1_params():
    cfg = _load("default.yaml")["/**"]["ros__parameters"]
    assert cfg["yolo_track_conf"] <= 0.2
    assert cfg["reid_backbone"] in ("osnet_ain_x1_0", "osnet_x0_25")
