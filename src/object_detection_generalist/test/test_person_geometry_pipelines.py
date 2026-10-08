"""Both open-vocab pipelines must route SAM output through person geometry.

Fake ``self`` with recording stubs; ``request_bboxes`` is monkeypatched for
the VLM path, so no network, model, or ROS is touched.
"""
import threading
import time
from types import SimpleNamespace

import numpy as np

from object_detection_generalist import generalist_node as gn
from object_detection_generalist.generalist_node import (
    GeneralistDetectionNode as Node,
)

H, W = 100, 200
RGB = np.zeros((H, W, 3), np.uint8)
PROMPT_BOX = (150, 0, 200, 100)
PERSON_BOX = (152, 20, 200, 100)


def _mask(box):
    m = np.zeros((H, W), dtype=bool)
    x1, y1, x2, y2 = box
    m[y1:y2, x1:x2] = True
    return m


WALL = _mask((150, 0, 200, 40))
PERSON_MASK = _mask(PERSON_BOX)

CTX = dict(
    rgb_img=RGB, points=np.zeros((H, W, 3)),
    valid_mask=np.ones((H, W), bool),
    prompt='person pointing to the left', camera='orbbec',
    header=SimpleNamespace(frame_id='cam', stamp=None),
    sort_mode='closest', return_segments=False, target_frame='map',
)


class _Logger:
    def info(self, *_):
        pass

    warn = error = exception = info


class _World:
    last_labels = ['person pointing to the left']

    def detect(self, rgb, prompt):
        return [PROMPT_BOX], [0.09], 0.01


class _Sam:
    def __init__(self, on_segment=None):
        self.on_segment = on_segment

    def segment(self, rgb, bboxes):
        if self.on_segment is not None:
            self.on_segment()
        return [WALL.copy() for _ in bboxes], 0.1


def _swap(bboxes, masks):
    return [PERSON_BOX], [PERSON_MASK], [{'box': 0, 'action': 'yolo_person'}]


def _drop(bboxes, masks):
    return list(bboxes), [None], [{'box': 0, 'action': 'dropped'}]


def _node(geometry, sam=None):
    rec = {}

    def apply_geom(rgb, bboxes, masks, labels, prompt):
        rec['geom_args'] = (list(bboxes), list(masks), labels, prompt)
        return geometry(bboxes, masks)

    def build(prompt, bboxes, masks, points, valid_mask, camera, **kw):
        rec['build_bboxes'] = list(bboxes)
        rec['build_masks'] = list(masks)
        objs = [
            SimpleNamespace(centroid=SimpleNamespace(x=1.0, y=0.0, z=0.0))
            for m in masks if m is not None
        ]
        return objs, [], [1.0] * len(objs)

    return SimpleNamespace(
        _ensure_world=lambda: _World(),
        _sam=sam or _Sam(),
        _sam_lock=threading.Lock(),
        _apply_person_geometry=apply_geom,
        _build_fallback_objects=build,
        _apply_realsense_range_gate=lambda o, s, c, d=None: (o, s, d),
        _sort_objects_and_segments=lambda o, s, *a, **k: (o, s),
        _empty_result=Node._empty_result,
        _person_geometry_error=Node._person_geometry_error,
        _parse_prompt_classes=Node._parse_prompt_classes,
        get_logger=lambda: _Logger(),
        vlm_model='m', vlm_fallback_models=[], vlm_fallback_on_empty=False,
        vlm_max_retries=1, vlm_timeout_s=1.0, vlm_per_attempt_timeout_s=1.0,
        vlm_stream=False,
        rec=rec,
    )


def _fake_vlm(monkeypatch, label='person'):
    monkeypatch.setattr(
        gn, 'request_bboxes',
        lambda *a, **k: ([PROMPT_BOX], [label], 0.5, {'model_used': 'm'}),
    )


def test_world_pipeline_uses_person_geometry_output():
    node = _node(_swap)
    res = Node._world_pipeline(node, **CTX)
    assert node.rec['geom_args'][2] == ['person pointing to the left']
    assert node.rec['geom_args'][1][0].sum() == WALL.sum()  # SAM went in
    assert node.rec['build_bboxes'] == [PERSON_BOX]
    assert node.rec['build_masks'][0] is PERSON_MASK
    assert res['bboxes'] == [PROMPT_BOX]
    assert res['masks'][0] is PERSON_MASK
    assert res['person_geometry'] == [{'box': 0, 'action': 'yolo_person'}]
    assert res['error'] is None
    assert len(res['objects']) == 1


def test_world_pipeline_all_dropped_reports_error():
    node = _node(_drop)
    res = Node._world_pipeline(node, **CTX)
    assert res['objects'] == []
    assert 'no matching YOLO person mask' in res['error']


def test_vlm_pipeline_uses_person_geometry_output(monkeypatch):
    _fake_vlm(monkeypatch)
    node = _node(_swap)
    res = Node._vlm_pipeline(node, **CTX)
    assert res['source'] == 'vlm_sam'
    assert node.rec['geom_args'][2] == ['person']
    assert node.rec['build_bboxes'] == [PERSON_BOX]
    assert node.rec['build_masks'][0] is PERSON_MASK
    assert res['bboxes'] == [PROMPT_BOX]
    assert res['masks'][0] is PERSON_MASK
    assert res['person_geometry'] == [{'box': 0, 'action': 'yolo_person'}]
    assert res['error'] is None


def test_vlm_pipeline_all_dropped_reports_error(monkeypatch):
    _fake_vlm(monkeypatch)
    node = _node(_drop)
    res = Node._vlm_pipeline(node, **CTX)
    assert res['objects'] == []
    assert 'no matching YOLO person mask' in res['error']


def test_vlm_pipeline_abandoned_during_sam_skips_person_pass(monkeypatch):
    _fake_vlm(monkeypatch)
    ev = threading.Event()
    node = _node(_swap, sam=_Sam(on_segment=ev.set))
    res = Node._vlm_pipeline(node, **CTX, abandon_event=ev)
    assert res['error'] == 'abandoned'
    assert 'geom_args' not in node.rec


def test_log_debug_records_person_geometry():
    captured = {}
    node = SimpleNamespace(
        enable_vlm=False, vlm_model='m', vlm_fallback_models=[],
        vlm_fallback_on_empty=False,
        _write_debug_artifacts=lambda rgb, dets, **kw: captured.update(
            kw, dets=dets),
    )
    request = SimpleNamespace(
        camera='orbbec', target_frame='map', sort_closest=True,
        sort_highest=False, force_vlm_sam=False, use_vlm_sam_fallback=True,
    )
    result = Node._empty_result('vlm_sam')
    result.update(
        objects=[object()], bboxes=[PROMPT_BOX], masks=[PERSON_MASK],
        person_geometry=[{'box': 0, 'action': 'yolo_person'}],
    )
    Node._log_debug(node, time.perf_counter(), request, 'person', RGB, result)
    assert captured['request_ctx']['person_geometry'] == [
        {'box': 0, 'action': 'yolo_person'}
    ]
    assert captured['dets'][0]['mask'] is PERSON_MASK
