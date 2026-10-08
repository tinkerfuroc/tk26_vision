"""Replay the GPSR sim frames where SAM segmented the wall behind a person.

Report: tinker-sim/6.0.1/docs/issue_reports/2026-10-08-gpsr-nav-bug-report.md
section 4. Runs the real yolo11m-seg model through the node's person pass
and matching, and checks the matched mask's centroid falls in
``person_region`` — read by hand from the frame as where a person-mask
centroid lands — and that the old SAM wall-mask centroid does not. Skipped when the weights are not in the local cache —
tests never download.
"""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision_util.weights_cache import find_cached

from object_detection_generalist.generalist_node import (
    GeneralistDetectionNode as Node,
)
from object_detection_generalist.person_geometry import (
    DEFAULT_MIN_CONTAINMENT,
    DEFAULT_MIN_IOU,
    DEFAULT_PERSON_CONF,
    match_person_instances,
    person_class_id,
)

FIX = Path(__file__).parent / 'fixtures' / 'person_geometry'
CASES = json.loads((FIX / 'cases.json').read_text())
WEIGHTS = find_cached('yolo11m-seg.pt')

pytestmark = pytest.mark.skipif(
    WEIGHTS is None, reason='yolo11m-seg.pt not in the local weight cache')


@pytest.fixture(scope='module')
def node():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    return SimpleNamespace(
        model=model, _yolo_model_lock=threading.Lock(),
        person_seg_conf=DEFAULT_PERSON_CONF,
        _person_cls_id=person_class_id(model.names),
    )


def _match(node, case):
    img = cv2.imread(str(FIX / case['frame']))
    assert img is not None, case['frame']
    instances = Node._detect_person_instances(node, img)
    match = match_person_instances(
        [tuple(case['prompt_box'])], instances,
        min_iou=DEFAULT_MIN_IOU, min_containment=DEFAULT_MIN_CONTAINMENT,
    )[0]
    return instances, match


@pytest.mark.parametrize(
    'case', [c for c in CASES if c['expect'] == 'person'],
    ids=lambda c: f"{c['source']}-{c['frame']}")
def test_person_box_matches_mask_on_the_person(node, case):
    instances, match = _match(node, case)
    assert match.instance is not None, (
        f"no YOLO person matched {case['prompt_box']} "
        f"(best iou={match.iou:.2f}, cont={match.containment:.2f})")
    ys, xs = np.nonzero(instances[match.instance].mask)
    x1, y1, x2, y2 = case['person_region']
    assert x1 <= xs.mean() <= x2 and y1 <= ys.mean() <= y2, (
        f'mask centroid ({xs.mean():.0f}, {ys.mean():.0f}) outside '
        f"person region {case['person_region']}")


@pytest.mark.xfail(
    strict=True,
    reason='known gap: YOLO-seg labels the robot arm "person"; needs a '
           'self-body filter (follow-up, out of scope)')
@pytest.mark.parametrize(
    'case', [c for c in CASES if c['expect'] == 'robot_arm'],
    ids=lambda c: c['frame'])
def test_robot_arm_box_is_not_matched(node, case):
    _, match = _match(node, case)
    assert match.instance is None


@pytest.mark.parametrize(
    'case', [c for c in CASES if 'sam_wall_centroid' in c],
    ids=lambda c: c['frame'])
def test_person_region_excludes_old_sam_wall_centroid(case):
    # The region check above only proves the fix if the wall-mask centroid
    # the old SAM path produced for this frame would have failed it.
    x, y = case['sam_wall_centroid']
    x1, y1, x2, y2 = case['person_region']
    assert not (x1 <= x <= x2 and y1 <= y <= y2)
