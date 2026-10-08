"""Node-method tests for person geometry, using a fake ``self``.

The methods only touch the attributes set on the SimpleNamespace below, so
no rclpy.init, camera, or model is needed (same approach as
test_generalist_intake_adoption.py).
"""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from object_detection_generalist.generalist_node import (
    GeneralistDetectionNode as Node,
)
from object_detection_generalist.person_geometry import (
    PersonInstance,
    parse_unmatched_policy,
)

H, W = 100, 200
RGB = np.zeros((H, W, 3), np.uint8)


def _mask(box):
    m = np.zeros((H, W), dtype=bool)
    x1, y1, x2, y2 = box
    m[y1:y2, x1:x2] = True
    return m


PROMPT_BOX = (150, 0, 200, 100)
WALL = _mask((150, 0, 200, 40))  # what SAM returned: the wall at the box top
PERSON_BOX = (152, 20, 200, 100)
PERSON = PersonInstance(bbox=PERSON_BOX, mask=_mask(PERSON_BOX), conf=0.8)
CHAIR_BOX = (10, 50, 60, 100)
CHAIR = _mask(CHAIR_BOX)


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(('info', m))

    def warn(self, m):
        self.lines.append(('warn', m))

    def error(self, m):
        self.lines.append(('error', m))


def _node(instances=(), raises=None, **overrides):
    logger = _Logger()
    calls = []

    def detect(rgb):
        calls.append(rgb.shape)
        if raises is not None:
            raise raises
        return list(instances)

    node = SimpleNamespace(
        person_seg_geometry=True,
        person_match_min_iou=0.3,
        person_match_min_containment=0.8,
        person_seg_unmatched='drop',
        _person_cls_id=0,
        _detect_person_instances=detect,
        _box_classes=Node._box_classes,
        get_logger=lambda: logger,
        logger=logger,
        detect_calls=calls,
    )
    for k, v in overrides.items():
        setattr(node, k, v)
    return node


def _apply(node, bboxes, masks, labels, prompt):
    return Node._apply_person_geometry(node, RGB, bboxes, masks, labels, prompt)


def test_person_box_takes_yolo_person_mask_and_box():
    node = _node([PERSON])
    boxes, masks, info = _apply(
        node, [PROMPT_BOX], [WALL], None, 'person pointing to the left')
    assert boxes == [PERSON_BOX]
    assert masks[0] is PERSON.mask
    assert info[0]['box'] == 0
    assert info[0]['action'] == 'yolo_person'
    assert info[0]['person_bbox'] == list(PERSON_BOX)
    assert info[0]['iou'] == pytest.approx(0.768, abs=1e-3)


def test_object_prompt_untouched_and_yolo_not_run():
    node = _node([PERSON])
    boxes, masks, info = _apply(node, [PROMPT_BOX], [WALL], None, 'red bowl')
    assert boxes == [PROMPT_BOX]
    assert masks[0] is WALL
    assert info == []
    assert node.detect_calls == []


def test_vlm_label_mentioning_person_does_not_make_object_box_a_person():
    node = _node([PERSON])
    boxes, masks, info = _apply(
        node, [PROMPT_BOX], [WALL], ['woman holding a bowl'], 'bowl')
    assert masks[0] is WALL
    assert info == []
    assert node.detect_calls == []


def test_multiclass_only_person_boxes_swapped():
    node = _node([PERSON])
    boxes, masks, info = _apply(
        node, [CHAIR_BOX, PROMPT_BOX], [CHAIR, WALL], ['chair', 'person'],
        'person . chair')
    assert masks[0] is CHAIR
    assert masks[1] is PERSON.mask
    assert boxes == [CHAIR_BOX, PERSON_BOX]
    assert [e['box'] for e in info] == [1]


def test_multiclass_unlabelled_box_untouched():
    node = _node([PERSON])
    boxes, masks, info = _apply(
        node, [PROMPT_BOX], [WALL], None, 'person . chair')
    assert masks[0] is WALL
    assert node.detect_calls == []


def test_unmatched_person_box_dropped_by_default():
    node = _node([])
    boxes, masks, info = _apply(node, [PROMPT_BOX], [WALL], None, 'person')
    assert masks[0] is None
    assert boxes == [PROMPT_BOX]
    assert info[0]['action'] == 'dropped'


def test_unmatched_person_box_keeps_sam_mask_with_sam_policy():
    node = _node([], person_seg_unmatched='sam')
    boxes, masks, info = _apply(node, [PROMPT_BOX], [WALL], None, 'person')
    assert masks[0] is WALL
    assert info[0]['action'] == 'sam_unmatched'


def test_yolo_failure_keeps_sam_masks():
    node = _node(raises=RuntimeError('CUDA out of memory'))
    boxes, masks, info = _apply(node, [PROMPT_BOX], [WALL], None, 'person')
    assert masks[0] is WALL
    assert boxes == [PROMPT_BOX]
    assert info == [{'box': 0, 'action': 'sam_yolo_error'}]
    assert any(level == 'error' for level, _ in node.logger.lines)


def test_disabled_feature_is_a_noop():
    node = _node([PERSON], person_seg_geometry=False)
    boxes, masks, info = _apply(node, [PROMPT_BOX], [WALL], None, 'person')
    assert masks[0] is WALL
    assert node.detect_calls == []


def test_model_without_person_class_is_a_noop():
    node = _node([PERSON], _person_cls_id=None)
    boxes, masks, info = _apply(node, [PROMPT_BOX], [WALL], None, 'person')
    assert masks[0] is WALL
    assert node.detect_calls == []


def test_inputs_not_mutated():
    node = _node([PERSON])
    in_boxes, in_masks = [PROMPT_BOX], [WALL]
    _apply(node, in_boxes, in_masks, None, 'person')
    assert in_boxes == [PROMPT_BOX]
    assert in_masks[0] is WALL


def test_person_geometry_error_only_when_everything_dropped():
    err = Node._person_geometry_error
    assert err([object()], [{'action': 'dropped'}]) is None
    assert err([], [{'action': 'yolo_person'}]) is None
    assert err([], []) is None
    msg = err([], [{'action': 'dropped'}, {'action': 'dropped'}])
    assert '2 person box(es)' in msg
    assert 'no matching YOLO person mask' in msg


def test_detect_person_instances_never_passes_classes_and_filters():
    seen = {}
    m_person = np.zeros((96, 64), np.float32)
    m_person[10:80, 5:30] = 1.0
    m_chair = np.zeros((96, 64), np.float32)
    m_chair[10:80, 35:60] = 1.0
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.array([[5, 10, 30, 80], [35, 10, 60, 80]], float),
            conf=np.array([0.9, 0.9]),
            cls=np.array([0.0, 56.0]),
        ),
        masks=SimpleNamespace(data=np.stack([m_person, m_chair])),
    )

    def model(img, **kwargs):
        seen['shape'] = img.shape
        seen['kwargs'] = kwargs
        return [result]

    node = SimpleNamespace(
        model=model, _yolo_model_lock=threading.Lock(),
        person_seg_conf=0.25, _person_cls_id=0,
    )
    out = Node._detect_person_instances(node, np.zeros((90, 64, 3), np.uint8))
    assert 'classes' not in seen['kwargs']
    assert seen['shape'] == (96, 64, 3)
    assert seen['kwargs']['imgsz'] == (96, 64)
    assert [i.bbox for i in out] == [(5, 10, 30, 80)]


@pytest.mark.parametrize('value, expected', [
    ('drop', 'drop'), (' SAM ', 'sam'), ('keep', None), ('', None), (None, None),
])
def test_parse_unmatched_policy(value, expected):
    assert parse_unmatched_policy(value) == expected


# --- final-review fixes -----------------------------------------------------

def _futile_node(**overrides):
    node = SimpleNamespace(
        person_seg_geometry=True, _person_cls_id=0,
        person_seg_unmatched='drop', person_seg_conf=0.25,
        conf_threshold=0.0, excluded_classes=set(),
    )
    for k, v in overrides.items():
        setattr(node, k, v)
    return node


def test_person_fallback_futile_for_person_prompt_by_default():
    assert Node._person_fallback_is_futile(_futile_node(), 'person')


@pytest.mark.parametrize('overrides, prompt', [
    ({}, 'cup'),
    ({'person_seg_unmatched': 'sam'}, 'person'),
    ({'person_seg_geometry': False}, 'person'),
    ({'_person_cls_id': None}, 'person'),
    # Parent filters harder than the person pass: the pass can find more.
    ({'conf_threshold': 0.5}, 'person'),
    # Parent never returns persons, so the pass may find what it skipped.
    ({'excluded_classes': {'person'}}, 'person'),
])
def test_person_fallback_not_futile(overrides, prompt):
    assert not Node._person_fallback_is_futile(_futile_node(**overrides), prompt)


def _callback_node(futile):
    from std_msgs.msg import Header

    calls = []

    def record(name, source):
        def run(**ctx):
            calls.append(name)
            return Node._empty_result(source)
        return run

    return SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: Header().stamp)),
        _select_camera=lambda c: 'orbbec',
        _wait_for_recent_frame=lambda c: (object(), object()),
        _get_intrinsic=lambda c: object(),
        _process_orbbec_data=lambda rgb, depth, intr: (
            RGB, np.zeros((H, W, 3)), np.ones((H, W), bool), Header()),
        _select_sort_mode=Node._select_sort_mode,
        _vision_logger=SimpleNamespace(enabled=False),
        _yolo_class_names={'person'},
        _yolo_pipeline=lambda **ctx: Node._empty_result('yolo'),
        _race_world_vlm=record('race', 'vlm_sam'),
        _world_pipeline=record('world', 'yolo_world'),
        _vlm_pipeline=record('vlm', 'vlm_sam'),
        _person_fallback_is_futile=lambda prompt: futile,
        _frame_supports_tf_transform=lambda c: True,
        allow_auto_fallback=True, enable_vlm=False,
        get_logger=lambda: _Logger(), calls=calls,
    )


@pytest.mark.parametrize('use_race', [True, False])
def test_callback_skips_futile_person_fallback(use_race):
    from tinker_vision_msgs_26.srv import ObjectDetectionGeneralist as Srv
    node = _callback_node(futile=True)
    req = Srv.Request(prompt='person', camera='orbbec',
                      use_vlm_sam_fallback=use_race)
    res = Node._generalist_service_callback(node, req, Srv.Response())
    assert node.calls == []
    assert res.status == 1


@pytest.mark.parametrize('use_race, expected', [(True, ['race']),
                                                (False, ['world'])])
def test_callback_keeps_person_fallback_when_not_futile(use_race, expected):
    from tinker_vision_msgs_26.srv import ObjectDetectionGeneralist as Srv
    node = _callback_node(futile=False)
    req = Srv.Request(prompt='person', camera='orbbec',
                      use_vlm_sam_fallback=use_race)
    Node._generalist_service_callback(node, req, Srv.Response())
    assert node.calls == expected


def test_parent_yolo_pass_holds_the_shared_model_lock(monkeypatch):
    # A late race leg can still be in _detect_person_instances when the next
    # call's YOLO pipeline runs; both use the same Ultralytics predictor.
    from object_detection_new.object_seg_yolo import YOLOSegmentationNode
    seen = {}

    def parent(self, *args, **kwargs):
        seen['locked'] = self._yolo_model_lock.locked()
        return 'objects', ['segments']

    monkeypatch.setattr(YOLOSegmentationNode, '_detect_objects', parent)
    node = Node.__new__(Node)
    node._yolo_model_lock = threading.Lock()
    assert node._detect_objects(RGB, None, 'person', None, None) == (
        'objects', ['segments'])
    assert seen['locked'] is True
