"""Tests for ptbench.common.schema — GT annotation dataclasses + JSON IO."""
import json
import math
import tempfile
from pathlib import Path

import pytest

from ptbench.common.schema import (
    GtClip,
    GtFrame,
    GtSchemaError,
    load_gt,
    save_gt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_clip(**overrides) -> GtClip:
    """Return a minimal valid GtClip, overrides applied to constructor."""
    frames = overrides.pop(
        "frames",
        [
            GtFrame(t_ns=1000, present=True, bbox=(10, 20, 110, 220),
                    centroid_field=(0.1, 0.0, 2.5), centroid_track=(0.1, 0.0, 2.5)),
            GtFrame(t_ns=2000, present=False, bbox=None,
                    centroid_field=None, centroid_track=None),
        ],
    )
    return GtClip(
        schema_version=overrides.pop("schema_version", "1.1"),
        clip_id=overrides.pop("clip_id", "test_clip"),
        bag_path=overrides.pop("bag_path", "bags/test"),
        scenario=overrides.pop("scenario", "test_scenario"),
        frames=frames,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_roundtrip_basic(self, tmp_path):
        clip = make_clip()
        p = tmp_path / "gt.json"
        save_gt(clip, p)
        loaded = load_gt(p)
        assert loaded.schema_version == clip.schema_version
        assert loaded.clip_id == clip.clip_id
        assert loaded.bag_path == clip.bag_path
        assert loaded.scenario == clip.scenario
        assert loaded.color_topic == clip.color_topic
        assert loaded.depth_topic == clip.depth_topic
        assert loaded.camera_info_topic == clip.camera_info_topic
        assert loaded.fps_hint == clip.fps_hint
        assert loaded.notes == clip.notes
        assert len(loaded.frames) == len(clip.frames)

    def test_roundtrip_frame_values(self, tmp_path):
        clip = make_clip()
        p = tmp_path / "gt.json"
        save_gt(clip, p)
        loaded = load_gt(p)

        f0 = loaded.frames[0]
        assert f0.t_ns == 1000
        assert f0.present is True
        assert f0.bbox == (10, 20, 110, 220)
        assert f0.centroid_field == (0.1, 0.0, 2.5)
        assert f0.centroid_track == (0.1, 0.0, 2.5)

        f1 = loaded.frames[1]
        assert f1.t_ns == 2000
        assert f1.present is False
        assert f1.bbox is None
        assert f1.centroid_field is None
        assert f1.centroid_track is None

    def test_canonical_json_shape(self, tmp_path):
        """JSON on disk must use arrays (not tuples), bbox/centroid null when None."""
        clip = make_clip()
        p = tmp_path / "gt.json"
        save_gt(clip, p)
        raw = json.loads(p.read_text())

        assert raw["schema_version"] == "1.1"
        assert raw["frames"][0]["bbox"] == [10, 20, 110, 220]
        assert raw["frames"][0]["centroid_field"] == [0.1, 0.0, 2.5]
        assert raw["frames"][0]["centroid_track"] == [0.1, 0.0, 2.5]
        assert raw["frames"][1]["bbox"] is None
        assert raw["frames"][1]["centroid_field"] is None
        assert raw["frames"][1]["centroid_track"] is None

    def test_roundtrip_full_spec_example(self, tmp_path):
        """Matches the canonical JSON shape from the spec verbatim."""
        clip = GtClip(
            schema_version="1.1",
            clip_id="cml_crossing_01",
            bag_path="bags/cml_crossing_01",
            scenario="cml_crossing",
            frames=[
                GtFrame(t_ns=1000, present=True, bbox=(10, 20, 110, 220),
                        centroid_field=(0.1, 0.0, 2.5), centroid_track=(0.1, 0.0, 2.5)),
                GtFrame(t_ns=2000, present=False, bbox=None,
                        centroid_field=None, centroid_track=None),
            ],
        )
        p = tmp_path / "canonical.json"
        save_gt(clip, p)
        loaded = load_gt(p)
        assert loaded.clip_id == "cml_crossing_01"
        assert loaded.frames[0].centroid_field == (0.1, 0.0, 2.5)

    def test_roundtrip_preserves_defaults(self, tmp_path):
        clip = make_clip()
        p = tmp_path / "gt.json"
        save_gt(clip, p)
        loaded = load_gt(p)
        assert loaded.color_topic == "/camera/color/image_raw"
        assert loaded.depth_topic == "/camera/depth/image_raw"
        assert loaded.camera_info_topic == "/camera/color/camera_info"
        assert loaded.fps_hint == 30.0
        assert loaded.notes == ""

    def test_roundtrip_lossless_float_centroid(self, tmp_path):
        clip = make_clip(
            frames=[GtFrame(t_ns=100, present=True, bbox=(0, 0, 10, 10),
                            centroid_field=(1.23456, -0.5, 3.14159),
                            centroid_track=(1.23456, -0.5, 3.14159))]
        )
        p = tmp_path / "gt.json"
        save_gt(clip, p)
        loaded = load_gt(p)
        cx, cy, cz = loaded.frames[0].centroid_field
        assert abs(cx - 1.23456) < 1e-6
        assert abs(cy - (-0.5)) < 1e-9
        assert abs(cz - 3.14159) < 1e-5


# ---------------------------------------------------------------------------
# Validation failures — each must raise GtSchemaError
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_bad_schema_version(self, tmp_path):
        clip = make_clip(schema_version="2.0")
        p = tmp_path / "gt.json"
        save_gt(clip, p)
        # Directly corrupt the file to bypass save-time validation
        raw = json.loads(p.read_text())
        raw["schema_version"] = "2.0"
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="schema_version"):
            load_gt(p)

    def test_empty_frames(self, tmp_path):
        clip = make_clip(frames=[])
        p = tmp_path / "gt.json"
        # Save without validate (raw write)
        raw = {
            "schema_version": "1.0",
            "clip_id": clip.clip_id,
            "bag_path": clip.bag_path,
            "scenario": clip.scenario,
            "color_topic": clip.color_topic,
            "depth_topic": clip.depth_topic,
            "camera_info_topic": clip.camera_info_topic,
            "fps_hint": clip.fps_hint,
            "notes": clip.notes,
            "frames": [],
        }
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="frames"):
            load_gt(p)

    def test_non_monotonic_t_ns(self, tmp_path):
        frames = [
            GtFrame(t_ns=2000, present=False, bbox=None,
                    centroid_field=None, centroid_track=None),
            GtFrame(t_ns=1000, present=False, bbox=None,
                    centroid_field=None, centroid_track=None),  # goes backward
        ]
        clip = make_clip(frames=frames)
        p = tmp_path / "gt.json"
        raw = {
            "schema_version": "1.0",
            "clip_id": clip.clip_id,
            "bag_path": clip.bag_path,
            "scenario": clip.scenario,
            "color_topic": clip.color_topic,
            "depth_topic": clip.depth_topic,
            "camera_info_topic": clip.camera_info_topic,
            "fps_hint": clip.fps_hint,
            "notes": clip.notes,
            "frames": [
                {"t_ns": 2000, "present": False, "bbox": None, "centroid_3d": None},
                {"t_ns": 1000, "present": False, "bbox": None, "centroid_3d": None},
            ],
        }
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="t_ns"):
            load_gt(p)

    def test_equal_t_ns_non_monotonic(self, tmp_path):
        """Equal timestamps are also non-monotonic (must be strictly increasing)."""
        raw = {
            "schema_version": "1.0",
            "clip_id": "x", "bag_path": "b", "scenario": "s",
            "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
            "fps_hint": 30.0, "notes": "",
            "frames": [
                {"t_ns": 1000, "present": False, "bbox": None, "centroid_3d": None},
                {"t_ns": 1000, "present": False, "bbox": None, "centroid_3d": None},
            ],
        }
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="t_ns"):
            load_gt(p)

    def test_present_true_without_bbox(self, tmp_path):
        raw = {
            "schema_version": "1.0",
            "clip_id": "x", "bag_path": "b", "scenario": "s",
            "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
            "fps_hint": 30.0, "notes": "",
            "frames": [
                {"t_ns": 1000, "present": True, "bbox": None, "centroid_3d": None},
            ],
        }
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="bbox"):
            load_gt(p)

    def test_bad_bbox_wrong_count(self, tmp_path):
        raw = {
            "schema_version": "1.0",
            "clip_id": "x", "bag_path": "b", "scenario": "s",
            "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
            "fps_hint": 30.0, "notes": "",
            "frames": [
                {"t_ns": 1000, "present": True, "bbox": [10, 20, 110], "centroid_3d": None},
            ],
        }
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="bbox"):
            load_gt(p)

    def test_bad_bbox_x2_leq_x1(self, tmp_path):
        raw = {
            "schema_version": "1.0",
            "clip_id": "x", "bag_path": "b", "scenario": "s",
            "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
            "fps_hint": 30.0, "notes": "",
            "frames": [
                {"t_ns": 1000, "present": True, "bbox": [100, 20, 50, 220], "centroid_3d": None},
            ],
        }
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="bbox"):
            load_gt(p)

    def test_bad_bbox_y2_leq_y1(self, tmp_path):
        raw = {
            "schema_version": "1.0",
            "clip_id": "x", "bag_path": "b", "scenario": "s",
            "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
            "fps_hint": 30.0, "notes": "",
            "frames": [
                {"t_ns": 1000, "present": True, "bbox": [10, 220, 110, 20], "centroid_3d": None},
            ],
        }
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="bbox"):
            load_gt(p)

    def test_bad_centroid_wrong_count(self, tmp_path):
        raw = {
            "schema_version": "1.0",
            "clip_id": "x", "bag_path": "b", "scenario": "s",
            "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
            "fps_hint": 30.0, "notes": "",
            "frames": [
                {"t_ns": 1000, "present": True, "bbox": [10, 20, 110, 220], "centroid_3d": [0.1, 0.0]},
            ],
        }
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="centroid_field"):
            load_gt(p)

    def test_bad_centroid_non_finite(self, tmp_path):
        raw = {
            "schema_version": "1.0",
            "clip_id": "x", "bag_path": "b", "scenario": "s",
            "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
            "fps_hint": 30.0, "notes": "",
            "frames": [
                {"t_ns": 1000, "present": True, "bbox": [10, 20, 110, 220],
                 "centroid_3d": [float("inf"), 0.0, 2.5]},
            ],
        }
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="centroid_field"):
            load_gt(p)

    def test_bad_bbox_non_finite(self, tmp_path):
        raw = {
            "schema_version": "1.0",
            "clip_id": "x", "bag_path": "b", "scenario": "s",
            "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
            "fps_hint": 30.0, "notes": "",
            "frames": [
                {"t_ns": 1000, "present": True, "bbox": [10, 20, float("nan"), 220], "centroid_3d": None},
            ],
        }
        p = tmp_path / "gt.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="bbox"):
            load_gt(p)
