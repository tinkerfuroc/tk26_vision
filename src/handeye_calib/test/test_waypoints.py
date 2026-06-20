from pathlib import Path
import pytest
import yaml
from handeye_calib import waypoints as wp


def test_store_starts_empty():
    s = wp.WaypointStore()
    assert s.list() == []


def test_store_add_validates_length():
    s = wp.WaypointStore()
    s.add([0, 0, 0, 0, 0, 0, 0])
    assert len(s.list()) == 1
    with pytest.raises(ValueError):
        s.add([0, 0, 0])  # too short


def test_store_delete_returns_true_on_hit_false_on_miss():
    s = wp.WaypointStore()
    s.add([0.1] * 7)
    s.add([0.2] * 7)
    assert s.delete(0) is True
    assert s.list() == [[0.2] * 7]
    assert s.delete(99) is False


def test_store_yaml_roundtrip(tmp_path):
    s = wp.WaypointStore()
    s.add([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    s.add([-0.1] * 7)
    p = tmp_path / "handeye_waypoints.yaml"
    s.save_yaml(p, recorded_for_robot="tinker2")
    s2 = wp.WaypointStore()
    n = s2.load_yaml(p)
    assert n == 2
    assert s2.list() == s.list()
    # schema version + robot name persisted in the file
    on_disk = yaml.safe_load(p.read_text())
    assert on_disk["schema_version"] == wp.YAML_SCHEMA_VERSION
    assert on_disk["recorded_for_robot"] == "tinker2"


def test_resolve_waypoints_path(tmp_path):
    p = wp.resolve_waypoints_path("tinker2", tmp_path)
    assert p == tmp_path / "src/tinker_robot_config/robots/tinker2/handeye_waypoints.yaml"
    assert wp.resolve_waypoints_path(None, tmp_path) is None
    assert wp.resolve_waypoints_path("", tmp_path) is None
