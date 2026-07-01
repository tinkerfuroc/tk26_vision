"""Unit tests for the multi-custom-dataset naming + migration helpers.

Pure logic, no hardware / no ROS. Covers:
  - sanitize_custom_name: lowercasing, separator folding, illegal-char rejection.
  - custom_dataset_filenames: legacy bare filenames for 'custom', suffixed otherwise.
  - migrate_custom_datasets: legacy flat keys -> list, new-list passthrough, empty.
"""

import pytest

from pan_tilt.calibration.custom_naming import (
    custom_dataset_filenames,
    migrate_custom_datasets,
    sanitize_custom_name,
)


# ---- sanitize_custom_name ---------------------------------------------------

def test_sanitize_passthrough_lowercase():
    assert sanitize_custom_name("high_shelf") == "high_shelf"


def test_sanitize_folds_spaces_and_dashes_and_case():
    assert sanitize_custom_name("High Shelf-2") == "high_shelf_2"


def test_sanitize_rejects_leading_digit():
    with pytest.raises(ValueError):
        sanitize_custom_name("2nd")


def test_sanitize_rejects_empty():
    with pytest.raises(ValueError):
        sanitize_custom_name("   ")


def test_sanitize_rejects_too_long():
    with pytest.raises(ValueError):
        sanitize_custom_name("a" * 25)


def test_sanitize_strips_illegal_chars():
    # punctuation dropped, result still valid
    assert sanitize_custom_name("seat!!") == "seat"


# ---- custom_dataset_filenames ----------------------------------------------

def test_filenames_legacy_custom_is_bare():
    collect, solve = custom_dataset_filenames("custom")
    assert collect == "phase1_handeye_custom.json"
    assert solve == "handeye_custom.json"


def test_filenames_named_is_suffixed():
    collect, solve = custom_dataset_filenames("high_shelf")
    assert collect == "phase1_handeye_custom_high_shelf.json"
    assert solve == "handeye_custom_high_shelf.json"


# ---- migrate_custom_datasets ------------------------------------------------

def test_migrate_new_list_passthrough():
    coll = {
        "phase1_custom_datasets": [
            {"name": "high_shelf", "park_pan_deg": 15.0, "park_tilt_deg": 30.0,
             "waypoints": [[0.0] * 7]},
        ]
    }
    out = migrate_custom_datasets(coll)
    assert len(out) == 1
    assert out[0]["name"] == "high_shelf"
    assert out[0]["park_pan_deg"] == 15.0
    assert out[0]["waypoints"] == [[0.0] * 7]


def test_migrate_legacy_flat_keys():
    coll = {
        "phase1_waypoints_custom": [[0.1] * 7, [0.2] * 7],
        "phase1_custom_park_pan_deg": 5.0,
        "phase1_custom_park_tilt_deg": 20.0,
    }
    out = migrate_custom_datasets(coll)
    assert len(out) == 1
    assert out[0]["name"] == "custom"
    assert out[0]["park_pan_deg"] == 5.0
    assert out[0]["park_tilt_deg"] == 20.0
    assert len(out[0]["waypoints"]) == 2


def test_migrate_legacy_defaults_park_to_zero():
    coll = {"phase1_waypoints_custom": [[0.1] * 7]}
    out = migrate_custom_datasets(coll)
    assert out[0]["park_pan_deg"] == 0.0
    assert out[0]["park_tilt_deg"] == 0.0


def test_migrate_empty_when_nothing_present():
    assert migrate_custom_datasets({}) == []


def test_migrate_empty_when_legacy_list_empty():
    assert migrate_custom_datasets({"phase1_waypoints_custom": []}) == []


def test_migrate_prefers_new_list_over_legacy():
    coll = {
        "phase1_custom_datasets": [
            {"name": "a", "park_pan_deg": 1.0, "park_tilt_deg": 2.0, "waypoints": []},
        ],
        "phase1_waypoints_custom": [[0.1] * 7],
        "phase1_custom_park_pan_deg": 9.0,
    }
    out = migrate_custom_datasets(coll)
    assert [d["name"] for d in out] == ["a"]


def test_migrate_does_not_mutate_input():
    coll = {"phase1_waypoints_custom": [[0.1] * 7], "phase1_custom_park_pan_deg": 5.0}
    migrate_custom_datasets(coll)
    assert "phase1_waypoints_custom" in coll  # untouched
