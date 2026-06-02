"""Tests for ptbench.labeler.label_io — PURE labeling logic.

The cv2 UI loop in ``label_cli`` is intentionally not unit-tested (thin shell);
we only assert here that it imports headlessly and exposes a CLI parser.
"""
import numpy as np
import pytest

from ptbench.common.schema import GtSchemaError, load_gt, save_gt
from ptbench.labeler.label_io import (
    FrameAnnotation,
    build_gt_clip,
    nearest_depth,
    propagate_default,
)


# ---------------------------------------------------------------------------
# propagate_default
# ---------------------------------------------------------------------------

class TestPropagateDefault:
    def test_none_prev_returns_none(self):
        assert propagate_default(None) is None

    def test_present_prev_copies_box(self):
        prev = FrameAnnotation(t_ns=10, present=True, bbox=(1.0, 2.0, 3.0, 4.0))
        assert propagate_default(prev) == (1.0, 2.0, 3.0, 4.0)

    def test_absent_prev_returns_none(self):
        prev = FrameAnnotation(t_ns=10, present=False, bbox=(1.0, 2.0, 3.0, 4.0))
        assert propagate_default(prev) is None

    def test_present_prev_without_box_returns_none(self):
        prev = FrameAnnotation(t_ns=10, present=True, bbox=None)
        assert propagate_default(prev) is None


# ---------------------------------------------------------------------------
# nearest_depth
# ---------------------------------------------------------------------------

class TestNearestDepth:
    def _depth(self, val):
        return np.full((4, 4), val, dtype=np.uint16)

    def test_empty_list_returns_none(self):
        assert nearest_depth([], 100) is None

    def test_picks_nearest_below(self):
        d100, d200, d300 = self._depth(1), self._depth(2), self._depth(3)
        depth_list = [(100, d100), (200, d200), (300, d300)]
        out = nearest_depth(depth_list, 240)
        assert out is d200  # |240-200|=40 < |240-300|=60

    def test_picks_nearest_above(self):
        d100, d200, d300 = self._depth(1), self._depth(2), self._depth(3)
        depth_list = [(100, d100), (200, d200), (300, d300)]
        out = nearest_depth(depth_list, 260)
        assert out is d300  # |260-300|=40 < |260-200|=60

    def test_exact_match(self):
        d100, d200 = self._depth(1), self._depth(2)
        out = nearest_depth([(100, d100), (200, d200)], 200)
        assert out is d200

    def test_before_first(self):
        d100, d200 = self._depth(1), self._depth(2)
        out = nearest_depth([(100, d100), (200, d200)], 10)
        assert out is d100


# ---------------------------------------------------------------------------
# build_gt_clip — end-to-end with synthetic depth
# ---------------------------------------------------------------------------

# A simple pinhole K (len-9 row-major): fx=fy=100, cx=cy=50.
K = [100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0]


def _constant_depth_image(h=100, w=100, depth_mm=2000):
    """Constant-depth (2.0 m) image so the sampled centroid is predictable."""
    return np.full((h, w), depth_mm, dtype=np.uint16)


class TestBuildGtClip:
    def test_present_frame_samples_centroid(self):
        depth = _constant_depth_image(depth_mm=2000)  # 2.0 m everywhere
        depth_list = [(1000, depth)]
        # Box centered on cx,cy so x,y ~ 0; z = 2.0 m.
        bbox = (40.0, 40.0, 60.0, 60.0)
        anns = [FrameAnnotation(t_ns=1000, present=True, bbox=bbox)]
        clip = build_gt_clip(
            anns, depth_list, K,
            clip_id="c", bag_path="b", scenario="s",
            color_topic="/c", depth_topic="/d", camera_info_topic="/i",
        )
        assert len(clip.frames) == 1
        f = clip.frames[0]
        assert f.present is True
        assert f.bbox == bbox
        assert f.centroid_3d is not None
        cx, cy, cz = f.centroid_3d
        # ROI samples u,v in [40,60); mean center = 49.5, so
        # x = (49.5-50)*2/100 = -0.01 m (and likewise y); z = median = 2.0.
        assert abs(cx - (-0.01)) < 1e-6
        assert abs(cy - (-0.01)) < 1e-6
        assert abs(cz - 2.0) < 1e-6

    def test_present_frame_offset_box_centroid(self):
        depth = _constant_depth_image(depth_mm=3000)  # 3.0 m
        depth_list = [(500, depth)]
        # Box 60..80 (center u=70): x = (70-50)*3/100 = 0.6 m
        # Box 50..70 (center v=60): y = (60-50)*3/100 = 0.3 m
        bbox = (60.0, 50.0, 80.0, 70.0)
        anns = [FrameAnnotation(t_ns=500, present=True, bbox=bbox)]
        clip = build_gt_clip(
            anns, depth_list, K,
            clip_id="c", bag_path="b", scenario="s",
            color_topic="/c", depth_topic="/d", camera_info_topic="/i",
        )
        cx, cy, cz = clip.frames[0].centroid_3d
        # Mean over u in [60,80): center 69.5 -> (69.5-50)*3/100 = 0.585
        # Mean over v in [50,70): center 59.5 -> (59.5-50)*3/100 = 0.285
        assert abs(cx - 0.585) < 1e-3
        assert abs(cy - 0.285) < 1e-3
        assert abs(cz - 3.0) < 1e-6

    def test_absent_frame_has_no_bbox_or_centroid(self):
        depth_list = [(1000, _constant_depth_image())]
        anns = [FrameAnnotation(t_ns=1000, present=False, bbox=None)]
        clip = build_gt_clip(
            anns, depth_list, K,
            clip_id="c", bag_path="b", scenario="s",
            color_topic="/c", depth_topic="/d", camera_info_topic="/i",
        )
        f = clip.frames[0]
        assert f.present is False
        assert f.bbox is None
        assert f.centroid_3d is None

    def test_present_without_bbox_downgraded(self):
        """present=True + bbox=None must be downgraded so schema stays valid."""
        depth_list = [(1000, _constant_depth_image())]
        anns = [FrameAnnotation(t_ns=1000, present=True, bbox=None)]
        clip = build_gt_clip(
            anns, depth_list, K,
            clip_id="c", bag_path="b", scenario="s",
            color_topic="/c", depth_topic="/d", camera_info_topic="/i",
        )
        f = clip.frames[0]
        assert f.present is False
        assert f.bbox is None
        assert f.centroid_3d is None

    def test_sparse_depth_present_with_none_centroid(self):
        """All-invalid depth → centroid None but present stays True with bbox."""
        # Depth 0 everywhere is < min_depth → no valid points.
        zero_depth = np.zeros((100, 100), dtype=np.uint16)
        depth_list = [(1000, zero_depth)]
        bbox = (40.0, 40.0, 60.0, 60.0)
        anns = [FrameAnnotation(t_ns=1000, present=True, bbox=bbox)]
        clip = build_gt_clip(
            anns, depth_list, K,
            clip_id="c", bag_path="b", scenario="s",
            color_topic="/c", depth_topic="/d", camera_info_topic="/i",
        )
        f = clip.frames[0]
        assert f.present is True
        assert f.bbox == bbox
        assert f.centroid_3d is None  # depth too sparse to sample

    def test_no_depth_frames_centroid_none(self):
        """Empty depth_list → present frame keeps bbox, centroid None."""
        bbox = (40.0, 40.0, 60.0, 60.0)
        anns = [FrameAnnotation(t_ns=1000, present=True, bbox=bbox)]
        clip = build_gt_clip(
            anns, [], K,
            clip_id="c", bag_path="b", scenario="s",
            color_topic="/c", depth_topic="/d", camera_info_topic="/i",
        )
        f = clip.frames[0]
        assert f.present is True
        assert f.bbox == bbox
        assert f.centroid_3d is None

    def test_frames_sorted_by_t_ns(self):
        depth_list = [(1000, _constant_depth_image())]
        anns = [
            FrameAnnotation(t_ns=3000, present=False, bbox=None),
            FrameAnnotation(t_ns=1000, present=False, bbox=None),
            FrameAnnotation(t_ns=2000, present=False, bbox=None),
        ]
        clip = build_gt_clip(
            anns, depth_list, K,
            clip_id="c", bag_path="b", scenario="s",
            color_topic="/c", depth_topic="/d", camera_info_topic="/i",
        )
        assert [f.t_ns for f in clip.frames] == [1000, 2000, 3000]

    def test_duplicate_t_ns_dropped(self):
        depth_list = [(1000, _constant_depth_image())]
        anns = [
            FrameAnnotation(t_ns=1000, present=False, bbox=None),
            FrameAnnotation(t_ns=1000, present=False, bbox=None),
        ]
        clip = build_gt_clip(
            anns, depth_list, K,
            clip_id="c", bag_path="b", scenario="s",
            color_topic="/c", depth_topic="/d", camera_info_topic="/i",
        )
        assert len(clip.frames) == 1

    def test_roundtrip_through_save_load(self, tmp_path):
        """build_gt_clip output must pass save_gt → load_gt without GtSchemaError."""
        depth = _constant_depth_image(depth_mm=2000)
        depth_list = [(1000, depth), (2000, depth), (3000, depth)]
        anns = [
            FrameAnnotation(t_ns=1000, present=True, bbox=(40.0, 40.0, 60.0, 60.0)),
            FrameAnnotation(t_ns=2000, present=False, bbox=None),
            FrameAnnotation(t_ns=3000, present=True, bbox=(45.0, 45.0, 65.0, 65.0)),
        ]
        clip = build_gt_clip(
            anns, depth_list, K,
            clip_id="rt", bag_path="bags/rt", scenario="cml_crossing",
            color_topic="/camera/color/image_raw",
            depth_topic="/camera/depth/image_raw",
            camera_info_topic="/camera/color/camera_info",
            notes="roundtrip test",
        )
        out = tmp_path / "gt.json"
        save_gt(clip, out)
        loaded = load_gt(out)  # would raise GtSchemaError if invalid
        assert loaded.clip_id == "rt"
        assert loaded.scenario == "cml_crossing"
        assert loaded.notes == "roundtrip test"
        assert [f.t_ns for f in loaded.frames] == [1000, 2000, 3000]
        assert loaded.frames[0].present is True
        assert loaded.frames[0].bbox == (40.0, 40.0, 60.0, 60.0)
        assert loaded.frames[0].centroid_3d is not None
        assert loaded.frames[1].present is False
        assert loaded.frames[1].bbox is None
        assert loaded.frames[2].present is True

    def test_roundtrip_with_sparse_centroid_still_valid(self, tmp_path):
        """A present frame with None centroid (sparse depth) is schema-valid."""
        zero_depth = np.zeros((100, 100), dtype=np.uint16)
        anns = [
            FrameAnnotation(t_ns=1000, present=True, bbox=(40.0, 40.0, 60.0, 60.0)),
        ]
        clip = build_gt_clip(
            anns, [(1000, zero_depth)], K,
            clip_id="rt2", bag_path="b", scenario="s",
            color_topic="/c", depth_topic="/d", camera_info_topic="/i",
        )
        out = tmp_path / "gt.json"
        save_gt(clip, out)
        loaded = load_gt(out)  # must not raise despite centroid None
        assert loaded.frames[0].present is True
        assert loaded.frames[0].centroid_3d is None


# ---------------------------------------------------------------------------
# label_cli — headless import + CLI parser
# ---------------------------------------------------------------------------

class TestCliImportHeadless:
    def test_module_imports_without_cv2(self):
        # Should import cleanly even with no display (cv2 imported lazily).
        import ptbench.labeler.label_cli as cli
        assert hasattr(cli, "main")

    def test_arg_parser_defaults(self):
        from ptbench.labeler.label_cli import _parse_args

        ns = _parse_args(["--bag", "/tmp/somebag"])
        assert ns.bag == "/tmp/somebag"
        assert ns.out is None
        assert ns.scenario == "unlabeled"
        assert ns.color_topic == "/camera/color/image_raw"
        assert ns.depth_topic == "/camera/depth/image_raw"
        assert ns.camera_info_topic == "/camera/color/camera_info"

    def test_arg_parser_requires_bag(self):
        from ptbench.labeler.label_cli import _parse_args

        with pytest.raises(SystemExit):
            _parse_args([])
