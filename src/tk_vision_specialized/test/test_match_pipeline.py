"""Unit tests for match_pipeline.py.

The pipeline is pure-Python (no ROS, no network, no GPU). Tests drive it
with fake clients to cover the full failure matrix."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pytest

from tk_vision_specialized.match_pipeline import (
    MatchPipeline,
    PipelineParams,
    FinalRow,
)
from tk_vision_specialized.nms import MatchRow

# Re-export FinalRow so flake8 doesn't flag the import as unused
# (FinalRow is the documented return-row shape; tests check it via field
# access through `final[0]`, not by direct reference to the type).
__all__ = ['FinalRow']


@dataclass
class FakeMatchClient:
    """Returns a deterministic per-batch result keyed by the labels in the
    batch. Raises if a batch is configured with `raise_on_call=True`."""
    per_batch: dict[frozenset[str], list[MatchRow]] = field(default_factory=dict)
    raise_for: set[frozenset[str]] = field(default_factory=set)
    sleep_for: dict[frozenset[str], float] = field(default_factory=dict)
    calls: list[frozenset[str]] = field(default_factory=list)

    def match_batch(
        self, scene_bgr, refs, *, timeout_s, max_retries, logger=None,
    ):
        key = frozenset(label for label, _url in refs)
        self.calls.append(key)
        if key in self.sleep_for:
            time.sleep(self.sleep_for[key])
        if key in self.raise_for:
            raise RuntimeError(f'fake match failure for {sorted(key)}')
        return list(self.per_batch.get(key, []))


@dataclass
class FakeJudgeClient:
    """Returns a deterministic choice per competing-label-set."""
    choices: dict[frozenset[str], object] = field(default_factory=dict)
    raise_for: set[frozenset[str]] = field(default_factory=set)
    calls: list[frozenset[str]] = field(default_factory=list)

    def choose(
        self, crop_bgr, competing, *, timeout_s, max_retries, logger=None,
    ):
        key = frozenset(label for label, _url in competing)
        self.calls.append(key)
        if key in self.raise_for:
            raise RuntimeError(f'fake judge failure for {sorted(key)}')
        return self.choices.get(key)


class FakeSam:
    """Returns one mask per bbox, drawn as the rectangle itself."""

    def __init__(self):
        self.calls = []

    def segment(self, rgb_bgr, bboxes):
        self.calls.append(list(bboxes))
        h, w = rgb_bgr.shape[:2]
        masks = []
        for x1, y1, x2, y2 in bboxes:
            m = np.zeros((h, w), dtype=bool)
            m[y1:y2, x1:x2] = True
            masks.append(m)
        return masks, 0.001


@dataclass
class FakeCameraData:
    """Surfaces only the centroid + TF lookup methods the pipeline calls."""
    centroid_value: object = None
    tf_value: object = None
    tf_support: bool = True

    def centroid_for(self, points, mask, valid_mask, bbox, camera):
        return self.centroid_value

    def transform_point(self, point, target, source, stamp):
        return self.tf_value

    def frame_supports_tf_transform(self, camera):
        return self.tf_support


def _items_map():
    return {
        'milk':  'data:image/jpeg;base64,M',
        'cola':  'data:image/jpeg;base64,C',
        'bread': 'data:image/jpeg;base64,B',
    }


def _make_scene():
    return np.zeros((400, 400, 3), dtype=np.uint8)


def _params(**overrides):
    base = dict(
        batch_size=2,
        max_workers=4,
        vlm_per_call_timeout_s=5.0,
        vlm_max_retries=1,
        stage1_timeout_s=10.0,
        stage2_timeout_s=10.0,
        nms_within_category_iou=0.5,
        cluster_iou=0.5,
        judge_crop_margin_px=10,
        min_valid_centroid_pixels=8,
    )
    base.update(overrides)
    return PipelineParams(**base)


class _Point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def test_empty_scene_returns_no_rows():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [],
        frozenset({'bread'}): [],
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=FakeJudgeClient(),
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0, 0, 1)),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert final == []
    assert counters['batches_ok'] == 2
    assert counters['rows_in'] == 0


def test_single_hit_passes_through():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.9),
        ],
        frozenset({'bread'}): [],
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=FakeJudgeClient(),
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0.1, 0.2, 0.5)),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert len(final) == 1
    assert final[0].row.label == 'milk'


def test_conflict_resolved_by_judge():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.6),
            MatchRow(label='cola', bbox=(12, 12, 52, 52), conf=0.85),
        ],
        frozenset({'bread'}): [],
    })
    judge_client = FakeJudgeClient(choices={
        frozenset({'milk', 'cola'}):
            type('JC', (), {'label': 'milk', 'conf': 0.95})(),
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=judge_client,
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0.1, 0.2, 0.5)),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert len(final) == 1
    assert final[0].row.label == 'milk'
    assert final[0].row.conf == pytest.approx(0.95)
    assert counters['judge_ok'] == 1


def test_judge_abstain_drops_cluster():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.6),
            MatchRow(label='cola', bbox=(12, 12, 52, 52), conf=0.65),
        ],
        frozenset({'bread'}): [],
    })
    judge_client = FakeJudgeClient(choices={
        frozenset({'milk', 'cola'}):
            type('JC', (), {'label': '', 'conf': 0.0})(),
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=judge_client,
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0.1, 0.2, 0.5)),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert final == []
    assert counters['judge_abstain'] == 1


def test_judge_failure_falls_back_to_top_conf():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.6),
            MatchRow(label='cola', bbox=(12, 12, 52, 52), conf=0.85),
        ],
        frozenset({'bread'}): [],
    })
    judge_client = FakeJudgeClient(
        raise_for={frozenset({'milk', 'cola'})},
    )
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=judge_client,
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0.1, 0.2, 0.5)),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert len(final) == 1
    assert final[0].row.label == 'cola'
    assert counters['judge_fail'] == 1


def test_one_batch_fails_others_survive():
    match_client = FakeMatchClient(
        per_batch={
            frozenset({'milk', 'cola'}): [
                MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.9),
            ],
            frozenset({'bread'}): [
                MatchRow(
                    label='bread', bbox=(200, 200, 250, 250), conf=0.8,
                ),
            ],
        },
        raise_for={frozenset({'milk', 'cola'})},
    )
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=FakeJudgeClient(),
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0.1, 0.2, 0.5)),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert len(final) == 1
    assert final[0].row.label == 'bread'
    assert counters['batches_fail'] == 1
    assert counters['batches_ok'] == 1


def test_detection_dropped_when_no_valid_depth():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.9),
        ],
        frozenset({'bread'}): [],
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=FakeJudgeClient(),
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=None),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='',
    )
    assert final == []
    assert counters['detections_dropped_no_depth'] == 1


def test_tf_failure_clears_results():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk', 'cola'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.9),
        ],
        frozenset({'bread'}): [],
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=FakeJudgeClient(),
        sam=FakeSam(),
        camera=FakeCameraData(
            centroid_value=_Point(0.1, 0.2, 0.5),
            tf_value=None,
            tf_support=True,
        ),
        items=_items_map(),
        params=_params(),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=[],
        target_frame='base_link',
    )
    assert final == []
    assert counters['tf_failed'] == 1


def test_category_filter_restricts_scan():
    match_client = FakeMatchClient(per_batch={
        frozenset({'milk'}): [
            MatchRow(label='milk', bbox=(10, 10, 50, 50), conf=0.9),
        ],
    })
    pipeline = MatchPipeline(
        match_client=match_client,
        judge_client=FakeJudgeClient(),
        sam=FakeSam(),
        camera=FakeCameraData(centroid_value=_Point(0.1, 0.2, 0.5)),
        items=_items_map(),
        params=_params(batch_size=2),
    )
    final, counters = pipeline.run(
        scene_bgr=_make_scene(),
        points_xyz=np.zeros((400, 400, 3), dtype=np.float32),
        valid_mask=np.ones((400, 400), dtype=bool),
        camera='realsense',
        category_filter=['milk'],
        target_frame='',
    )
    assert len(final) == 1
    assert match_client.calls == [frozenset({'milk'})]
