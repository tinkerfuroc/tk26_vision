"""Unit tests for person_geometry (pure numpy; no ROS node, no model).

Run:
    cd /home/tinker/tk25_ws && bash -c 'source /opt/ros/humble/setup.bash && \
        source install/setup.bash && \
        source src/tk26_vision/.venv-vision-main/bin/activate && \
        python -m pytest src/tk26_vision/src/object_detection_generalist/test/test_person_geometry.py -v'
"""
from types import SimpleNamespace

import numpy as np
import pytest

from object_detection_generalist.person_geometry import (
    PersonInstance,
    box_containment,
    box_iou,
    instances_from_yolo_result,
    is_person_phrase,
    match_person_instances,
    pad_to_multiple,
    person_class_id,
)


@pytest.mark.parametrize('text', [
    'person',
    'Person',
    'PERSON WAVING',
    'person pointing to the left',
    'the person pointing to the left',
    'standing person',
    'person standing',
    'waving person',
    'a man in a red shirt',
    'woman',
    'people',
    'guest',
    'child',
    'person holding a cup',
    'person sitting on the sofa',
])
def test_person_phrases(text):
    assert is_person_phrase(text)


@pytest.mark.parametrize('text', [
    '',
    '   ',
    'red bowl',
    'chair',
    'cup next to the person',
    'cup held by the person',
    "the person's bag",
    "person's bag",
    'bag of the person',
    'bowl on the table near a man',
    'mannequin',
    'manipulator',
    'humanoid robot',
])
def test_non_person_phrases(text):
    assert not is_person_phrase(text)


def test_person_class_id_dict_names():
    assert person_class_id({0: 'person', 1: 'bicycle'}) == 0


def test_person_class_id_list_names():
    assert person_class_id(['cup', 'person']) == 1


def test_person_class_id_missing():
    assert person_class_id({0: 'cup'}) is None

MATCH = dict(min_iou=0.3, min_containment=0.8)


def test_box_iou_identical():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_box_iou_disjoint():
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_box_iou_half_overlap():
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_box_iou_degenerate_box():
    assert box_iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


def test_box_containment_fully_inside():
    assert box_containment((2, 2, 4, 4), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_box_containment_partial():
    assert box_containment((5, 0, 15, 10), (0, 0, 10, 10)) == pytest.approx(0.5)


def test_box_containment_degenerate_inner():
    assert box_containment((3, 3, 3, 9), (0, 0, 10, 10)) == 0.0


def test_pad_to_multiple_pads_bottom_right_with_zeros():
    img = np.ones((720, 1280, 3), np.uint8)
    out = pad_to_multiple(img, 32)
    assert out.shape == (736, 1280, 3)
    assert out[720:].sum() == 0
    assert out[:720].min() == 1


def test_pad_to_multiple_noop_returns_same_array():
    img = np.ones((64, 64, 3), np.uint8)
    assert pad_to_multiple(img, 32) is img


def _mask(h, w, box):
    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = box
    m[y1:y2, x1:x2] = True
    return m


def _inst(box, conf=0.9, h=100, w=200):
    return PersonInstance(bbox=box, mask=_mask(h, w, box), conf=conf)


def test_match_high_iou():
    ms = match_person_instances(
        [(10, 10, 50, 90)], [_inst((12, 12, 50, 88))], **MATCH)
    assert ms[0].instance == 0
    assert ms[0].iou > 0.85


def test_match_by_containment_when_iou_low():
    # GPSR rerun57 205905: only the feet are visible; the YOLO person box is
    # a short strip at the bottom of a tall prompt box.
    prompt, feet = (153, 0, 200, 100), (153, 85, 200, 100)
    ms = match_person_instances([prompt], [_inst(feet)], **MATCH)
    assert ms[0].iou < 0.3
    assert ms[0].containment == pytest.approx(1.0)
    assert ms[0].instance == 0


def test_no_match_when_disjoint_reports_zero_overlap():
    # e.g. the robot arm detected as 'person' elsewhere in the frame
    ms = match_person_instances(
        [(150, 0, 200, 60)], [_inst((0, 60, 100, 100))], **MATCH)
    assert ms[0].instance is None
    assert ms[0].iou == 0.0
    assert ms[0].containment == 0.0


def test_one_to_one_duplicate_prompt_boxes():
    ms = match_person_instances(
        [(10, 10, 50, 90), (11, 11, 50, 90)], [_inst((10, 10, 50, 90))],
        **MATCH)
    assert [m.instance for m in ms] == [0, None]


def test_two_people_each_box_gets_its_own_instance():
    ms = match_person_instances(
        [(10, 10, 50, 90), (120, 10, 160, 90)],
        [_inst((118, 12, 160, 90)), _inst((10, 12, 52, 90))],
        **MATCH)
    assert [m.instance for m in ms] == [1, 0]


def test_prefers_full_body_over_partial_instance():
    ms = match_person_instances(
        [(10, 0, 50, 100)],
        [_inst((10, 80, 50, 100)), _inst((10, 2, 50, 100))],
        **MATCH)
    assert ms[0].instance == 1


def test_match_empty_inputs():
    assert match_person_instances([], [_inst((0, 0, 5, 5))], **MATCH) == []
    ms = match_person_instances([(0, 0, 5, 5)], [], **MATCH)
    assert ms[0].instance is None


def _result(xyxy, conf, cls, masks):
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.asarray(xyxy, float),
            conf=np.asarray(conf, float),
            cls=np.asarray(cls, float),
        ),
        masks=SimpleNamespace(data=np.asarray(masks, np.float32)),
    )


def test_instances_filter_class_and_conf_and_crop_padding():
    h, w = 90, 64  # model input padded to 96x64
    m0 = np.zeros((96, 64), np.float32)
    m0[10:80, 5:30] = 1.0  # person
    m1 = np.zeros((96, 64), np.float32)
    m1[10:80, 35:60] = 1.0  # chair (COCO 56)
    m2 = np.zeros((96, 64), np.float32)
    m2[0:20, 0:10] = 1.0  # low-confidence person
    r = _result(
        [[5, 10, 30, 80], [35, 10, 60, 80], [0, 0, 10, 20]],
        [0.9, 0.9, 0.1], [0, 56, 0], [m0, m1, m2])
    out = instances_from_yolo_result(r, h, w, conf_floor=0.25, cls_id=0)
    assert len(out) == 1
    inst = out[0]
    assert inst.bbox == (5, 10, 30, 80)
    assert inst.mask.shape == (h, w) and inst.mask.dtype == bool
    assert inst.mask.sum() == 70 * 25
    assert inst.conf == pytest.approx(0.9)


def test_instances_keep_largest_component_inside_box():
    m = np.zeros((64, 64), np.float32)
    m[10:40, 10:30] = 1.0
    m[50:52, 50:52] = 1.0  # stray blob outside the box
    r = _result([[10, 10, 30, 40]], [0.8], [0], [m])
    out = instances_from_yolo_result(r, 64, 64, conf_floor=0.25, cls_id=0)
    assert out[0].mask.sum() == 30 * 20


def test_instances_none_when_no_boxes_or_masks():
    r = SimpleNamespace(boxes=None, masks=None)
    assert instances_from_yolo_result(
        r, 10, 10, conf_floor=0.25, cls_id=0) == []


def test_instances_skip_mask_smaller_than_image():
    r = _result([[0, 0, 10, 10]], [0.9], [0], [np.ones((32, 32))])
    assert instances_from_yolo_result(
        r, 64, 64, conf_floor=0.25, cls_id=0) == []
