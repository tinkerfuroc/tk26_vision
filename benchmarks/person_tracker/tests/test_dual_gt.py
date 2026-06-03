"""Schema 1.1 dual-centroid round-trip + 1.0 back-compat + build_gt_clip duals."""
import json

import numpy as np
import pytest

from ptbench.common.schema import (
    SCHEMA_VERSION,
    GtClip,
    GtFrame,
    GtSchemaError,
    load_gt,
    save_gt,
)


def make_clip_11(**overrides) -> GtClip:
    frames = overrides.pop(
        "frames",
        [
            GtFrame(
                t_ns=1000,
                present=True,
                bbox=(10, 20, 110, 220),
                centroid_field=(0.10, 0.0, 2.5),
                centroid_track=(0.12, 0.01, 2.5),
            ),
            GtFrame(
                t_ns=2000,
                present=False,
                bbox=None,
                centroid_field=None,
                centroid_track=None,
            ),
        ],
    )
    return GtClip(
        schema_version=overrides.pop("schema_version", SCHEMA_VERSION),
        clip_id=overrides.pop("clip_id", "dual_clip"),
        bag_path=overrides.pop("bag_path", "bags/dual"),
        scenario=overrides.pop("scenario", "test"),
        frames=frames,
        **overrides,
    )


class TestSchema11RoundTrip:
    def test_version_is_11(self):
        assert SCHEMA_VERSION == "1.1"

    def test_roundtrip_both_centroids(self, tmp_path):
        clip = make_clip_11()
        p = tmp_path / "gt.json"
        save_gt(clip, p)
        loaded = load_gt(p)
        f0 = loaded.frames[0]
        assert f0.centroid_field == (0.10, 0.0, 2.5)
        assert f0.centroid_track == (0.12, 0.01, 2.5)

    def test_canonical_json_has_both_fields(self, tmp_path):
        clip = make_clip_11()
        p = tmp_path / "gt.json"
        save_gt(clip, p)
        raw = json.loads(p.read_text())
        assert raw["schema_version"] == "1.1"
        assert raw["frames"][0]["centroid_field"] == [0.10, 0.0, 2.5]
        assert raw["frames"][0]["centroid_track"] == [0.12, 0.01, 2.5]
        assert raw["frames"][1]["centroid_field"] is None
        assert raw["frames"][1]["centroid_track"] is None


class TestSchema10BackCompat:
    def test_10_centroid_3d_maps_to_both(self, tmp_path):
        raw = {
            "schema_version": "1.0",
            "clip_id": "x", "bag_path": "b", "scenario": "s",
            "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
            "fps_hint": 30.0, "notes": "",
            "frames": [
                {"t_ns": 1000, "present": True, "bbox": [10, 20, 110, 220],
                 "centroid_3d": [0.1, 0.0, 2.5]},
                {"t_ns": 2000, "present": False, "bbox": None, "centroid_3d": None},
            ],
        }
        p = tmp_path / "gt10.json"
        p.write_text(json.dumps(raw))
        loaded = load_gt(p)
        f0 = loaded.frames[0]
        assert f0.centroid_field == (0.1, 0.0, 2.5)
        assert f0.centroid_track == (0.1, 0.0, 2.5)
        assert loaded.frames[1].centroid_field is None
        assert loaded.frames[1].centroid_track is None

    def test_unsupported_version_still_rejected(self, tmp_path):
        raw = {
            "schema_version": "2.0",
            "clip_id": "x", "bag_path": "b", "scenario": "s",
            "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
            "fps_hint": 30.0, "notes": "",
            "frames": [
                {"t_ns": 1000, "present": True, "bbox": [10, 20, 110, 220],
                 "centroid_field": [0.1, 0.0, 2.5], "centroid_track": [0.1, 0.0, 2.5]},
            ],
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="schema_version"):
            load_gt(p)


class TestSchema11Validation:
    def test_bad_centroid_field_count(self, tmp_path):
        raw = {
            "schema_version": "1.1",
            "clip_id": "x", "bag_path": "b", "scenario": "s",
            "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
            "fps_hint": 30.0, "notes": "",
            "frames": [
                {"t_ns": 1000, "present": True, "bbox": [10, 20, 110, 220],
                 "centroid_field": [0.1, 0.0], "centroid_track": None},
            ],
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="centroid_field"):
            load_gt(p)

    def test_bad_centroid_track_non_finite(self, tmp_path):
        raw = {
            "schema_version": "1.1",
            "clip_id": "x", "bag_path": "b", "scenario": "s",
            "color_topic": "/c", "depth_topic": "/d", "camera_info_topic": "/i",
            "fps_hint": 30.0, "notes": "",
            "frames": [
                {"t_ns": 1000, "present": True, "bbox": [10, 20, 110, 220],
                 "centroid_field": None, "centroid_track": [float("nan"), 0.0, 2.5]},
            ],
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(GtSchemaError, match="centroid_track"):
            load_gt(p)


from ptbench.labeler.label_io import FrameAnnotation, build_gt_clip


def _pinhole_K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


class TestBuildGtClipDual:
    def test_present_frame_gets_both_centroids(self):
        H, W = 480, 640
        K = _pinhole_K()
        bbox = (100, 100, 200, 200)
        depth = np.full((H, W), 2500, dtype=np.uint16)  # 2.5 m
        ann = [FrameAnnotation(t_ns=1000, present=True, bbox=bbox, mask=None)]
        clip = build_gt_clip(
            ann, [(1000, depth)], K,
            clip_id="c", bag_path="b", scenario="s",
            color_topic="/c", depth_topic="/d", camera_info_topic="/i",
        )
        assert clip.schema_version == "1.1"
        f0 = clip.frames[0]
        assert f0.centroid_field is not None
        assert f0.centroid_track is not None

    def test_field_uses_mask_track_ignores_it(self):
        # A mask covering only the left half shifts centroid_field left of the
        # bbox-only centroid_track — proving field != track when a mask exists.
        H, W = 480, 640
        K = _pinhole_K()
        bbox = (100, 100, 300, 200)  # 200 wide
        depth = np.full((H, W), 2000, dtype=np.uint16)
        mask = np.zeros((H, W), dtype=np.float32)
        mask[100:200, 100:200] = 1.0  # left half only
        ann = [FrameAnnotation(t_ns=1000, present=True, bbox=bbox, mask=mask)]
        clip = build_gt_clip(
            ann, [(1000, depth)], K,
            clip_id="c", bag_path="b", scenario="s",
            color_topic="/c", depth_topic="/d", camera_info_topic="/i",
        )
        f0 = clip.frames[0]
        xf = f0.centroid_field[0]
        xt = f0.centroid_track[0]
        assert xf < xt  # masked (left half) is left of bbox-only

    def test_absent_frame_has_no_centroids(self):
        ann = [
            FrameAnnotation(t_ns=1000, present=True, bbox=(10, 10, 50, 50), mask=None),
            FrameAnnotation(t_ns=2000, present=False, bbox=None, mask=None),
        ]
        depth = np.full((480, 640), 3000, dtype=np.uint16)
        clip = build_gt_clip(
            ann, [(1000, depth), (2000, depth)], _pinhole_K(),
            clip_id="c", bag_path="b", scenario="s",
            color_topic="/c", depth_topic="/d", camera_info_topic="/i",
        )
        assert clip.frames[1].centroid_field is None
        assert clip.frames[1].centroid_track is None

    def test_divergence_is_measurable(self):
        # The synthetic fixture from the spec's "Testing (now)" section: a
        # tracker matching centroid_track EXACTLY still shows nonzero field
        # error, so the gate (on field) is not fooled by node-identical math.
        from ptbench.common.geometry import dist3d
        H, W = 480, 640
        K = _pinhole_K()
        bbox = (100, 100, 300, 200)
        depth = np.full((H, W), 2000, dtype=np.uint16)
        mask = np.zeros((H, W), dtype=np.float32)
        mask[100:200, 100:200] = 1.0
        ann = [FrameAnnotation(t_ns=1000, present=True, bbox=bbox, mask=mask)]
        clip = build_gt_clip(
            ann, [(1000, depth)], K,
            clip_id="c", bag_path="b", scenario="s",
            color_topic="/c", depth_topic="/d", camera_info_topic="/i",
        )
        f0 = clip.frames[0]
        assert dist3d(f0.centroid_track, f0.centroid_field) > 0.05
