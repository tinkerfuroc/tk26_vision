"""Unit tests for CalibWebNode's custom-dataset store + YAML serialization.

The store/serialization methods touch only a handful of attributes, so we
exercise them on a bare instance (object.__new__) rather than standing up a
full rclpy node + camera subscriptions. Covers:
  * add/remove/set_park lifecycle + duplicate rejection
  * dynamic phase-key delegation in list_waypoints/set_waypoints
  * dedupe across custom datasets
  * _serialize_waypoints_yaml emits phase1_custom_datasets and drops legacy keys
  * migration of a legacy single-custom config into the named list
"""

from __future__ import annotations

import threading

import pytest
import yaml

from pan_tilt.calib_web import CalibWebNode
from pan_tilt.calibration.custom_naming import migrate_custom_datasets


def _make_node(loaded_cfg: dict) -> CalibWebNode:
    """A CalibWebNode with just the attributes the store methods need."""
    node = object.__new__(CalibWebNode)
    node.lock = threading.Lock()
    node._loaded_cfg = dict(loaded_cfg)
    node._waypoints = {
        "phase1_waypoints": list(loaded_cfg.get("phase1_waypoints", []) or []),
        "phase2_waypoints": list(loaded_cfg.get("phase2_waypoints", []) or []),
        "sanity_xarm_angles_rad": list(loaded_cfg.get("sanity_xarm_angles_rad", []) or []),
    }
    node._custom_datasets = migrate_custom_datasets(loaded_cfg)
    for legacy in ("phase1_waypoints_custom",
                   "phase1_custom_park_pan_deg",
                   "phase1_custom_park_tilt_deg"):
        node._loaded_cfg.pop(legacy, None)
    return node


WP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def test_legacy_config_migrates_to_one_named_dataset():
    node = _make_node({
        "phase1_waypoints": [WP],
        "phase1_waypoints_custom": [WP, WP],
        "phase1_custom_park_pan_deg": 5.0,
        "phase1_custom_park_tilt_deg": 20.0,
    })
    ds = node.list_custom_datasets()
    assert [d["name"] for d in ds] == ["custom"]
    assert ds[0]["park_pan_deg"] == 5.0
    assert ds[0]["park_tilt_deg"] == 20.0
    assert len(ds[0]["waypoints"]) == 2


def test_add_remove_and_duplicate_rejection():
    node = _make_node({})
    node.add_custom_dataset("High Shelf")  # sanitized -> high_shelf
    assert [d["name"] for d in node.list_custom_datasets()] == ["high_shelf"]
    with pytest.raises(ValueError):
        node.add_custom_dataset("high_shelf")
    assert node.remove_custom_dataset("high_shelf") is True
    assert node.list_custom_datasets() == []
    assert node.remove_custom_dataset("high_shelf") is False


def test_dynamic_phase_key_get_set():
    node = _make_node({})
    node.add_custom_dataset("seat")
    key = "phase1_waypoints_custom:seat"
    assert node.list_waypoints(key) == []
    node.set_waypoints(key, [WP, WP])
    assert len(node.list_waypoints(key)) == 2
    # and it landed in the dataset store, not _waypoints
    assert node._custom_datasets[0]["waypoints"] == [WP, WP]
    assert key not in node._waypoints


def test_set_park():
    node = _make_node({})
    node.add_custom_dataset("seat")
    assert node.set_custom_park("seat", 12.0, 25.0) is True
    d = node.list_custom_datasets()[0]
    assert (d["park_pan_deg"], d["park_tilt_deg"]) == (12.0, 25.0)
    assert node.set_custom_park("nope", 1.0, 1.0) is False


def test_dedupe_walks_custom_datasets():
    node = _make_node({})
    node.add_custom_dataset("seat")
    node.set_waypoints("phase1_waypoints_custom:seat", [WP, list(WP), [9.0] * 7])
    removed = node.dedupe_waypoints()
    assert removed.get("phase1_waypoints_custom:seat") == 1
    assert len(node.list_waypoints("phase1_waypoints_custom:seat")) == 2


def test_serialize_emits_named_list_and_drops_legacy():
    node = _make_node({
        "phase1_waypoints": [WP],
        "phase1_waypoints_custom": [WP],
        "phase1_custom_park_pan_deg": 5.0,
        "phase1_custom_park_tilt_deg": 20.0,
        "__passthrough__": {"board_section": {"squares_x": 5}},
    })
    node.add_custom_dataset("high_shelf")
    node.set_custom_park("high_shelf", 15.0, 30.0)
    node.set_waypoints("phase1_waypoints_custom:high_shelf", [WP])

    out = yaml.safe_load(node._serialize_waypoints_yaml())
    coll = out["collector"]
    # legacy keys gone
    assert "phase1_waypoints_custom" not in coll
    assert "phase1_custom_park_pan_deg" not in coll
    # named list present with both the migrated 'custom' and the new entry
    names = [d["name"] for d in coll["phase1_custom_datasets"]]
    assert names == ["custom", "high_shelf"]
    hs = coll["phase1_custom_datasets"][1]
    assert hs["park_pan_deg"] == 15.0 and hs["park_tilt_deg"] == 30.0
    assert hs["waypoints"] == [WP]
    # passthrough board section round-trips
    assert out["board"] == {"squares_x": 5}


def test_serialize_then_migrate_roundtrips():
    node = _make_node({})
    node.add_custom_dataset("a")
    node.set_custom_park("a", 1.0, 2.0)
    node.set_waypoints("phase1_waypoints_custom:a", [WP])
    coll = yaml.safe_load(node._serialize_waypoints_yaml())["collector"]
    again = migrate_custom_datasets(coll)
    assert again[0]["name"] == "a"
    assert again[0]["park_pan_deg"] == 1.0
    assert again[0]["waypoints"] == [WP]
