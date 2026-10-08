# Generalist Person Seg-Mask Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For person boxes from the generalist's open-vocab paths (VLM+SAM, which is used most, and YOLO-World+SAM), take the 3D centroid from the matching COCO `person` instance mask of the pretrained YOLO-seg model instead of a box-prompted MobileSAM mask.

**Architecture:** A new pure module `person_geometry.py` (phrase classification, box overlap, instance extraction, one-to-one box↔instance matching) plus three node methods on `GeneralistDetectionNode` (`_detect_person_instances`, `_apply_person_geometry`, `_person_geometry_error`). Both `_world_pipeline` and `_vlm_pipeline` call `_apply_person_geometry` between SAM and `_build_fallback_objects`. Person boxes get the YOLO person mask, and the YOLO person bbox as the centroid ROI. Person boxes with no matching YOLO person are dropped by default. Non-person boxes are untouched.

**Tech Stack:** ROS 2 Humble, Python 3.10 (`.venv-vision-main`), Ultralytics 8.4.33 (YOLO11m-seg, already loaded as `self.model`), numpy, OpenCV, pytest.

**Spec:** No separate design doc. The requirement comes from `~/tinker-sim/6.0.1/docs/issue_reports/2026-10-08-gpsr-nav-bug-report.md` §4 plus the root-cause analysis summarized under *Background* below. The user's decisions: use the seg mask, and cover both the VLM and YOLO-World paths.

## Background (root cause — read once)

GPSR sim living-room trials (batches `t2-20-live-rerun57/59/64/68/79/92`) picked the person at the south-west wall, ~1 m behind the real person at (−3.8, −3.95), and once outside the arena at (−2.89, −5.80). The generalist itself emitted those points. Vision-log overlays in `~/tinker-sim/6.0.1/vision_log/20260822_081720/` show the mechanism:

1. The person is clipped by the right image edge. YOLO-World returns a weak box (conf 0.05–0.2).
2. MobileSAM, prompted with that box, segments the largest smooth region, which is **the wall** (or the wall/floor seam). `sam_mask.py` keeps the largest connected component in the box, so the wall wins.
3. `_calculate_centroid` (`object_detection_new/object_seg_yolo.py:884`) averages x/y and takes the median depth over the mask, so the centroid is on the wall. The camera distance jumps from ~2.3 m to ~3.3 m along the same ray.

Offline check with `yolo11m-seg.pt` on the saved frames (done while planning):

| Frame | Path | Prompt box | YOLO person found | IoU | Containment |
|---|---|---|---|---|---|
| `20261005_005738_508` (rerun64 wall) | yolo_world | [1121,46,1279,567] | [1122,124,1279,529] conf .71 | .77 | — |
| `20261005_090529_466` (rerun92 wall) | yolo_world | [1127,38,1279,719] | [1126,140,1279,532] conf .78 | .57 | — |
| `20261004_205905_361` (rerun57 outside arena, feet only) | yolo_world | [1153,99,1279,719] | [1153,646,1279,718] conf .53 | .12 | 1.00 |
| `20261005_015548_606` (rerun68 wall, feet only) | yolo_world | [1200,185,1279,695] | [1201,483,1279,627] conf .27 | .28 | 1.00 |
| `20261003_080553_756` | vlm_sam (qwen) | [1183,382,1279,564] | [1178,381,1279,560] conf .84 | .93 | .95 |
| `20260822_114230_129` | vlm_sam (gemini) | [650,182,796,582] | [650,187,791,581] conf .94 | .95 | 1.00 |
| `20261004_205435_943` (**robot arm**) | yolo_world | [0,444,672,719] | [219,447,669,716] conf .75 | .66 | 1.00 |

So matching needs **IoU ≥ 0.3 OR containment ≥ 0.8**, because "feet only" frames have low IoU but full containment. VLM boxes are tight (IoU > 0.9). The last row is a **known gap that this plan does not fix**: YOLO-seg labels the robot's own arm `person` (conf ~0.75 in every downward-looking frame). It is pinned by a strict-xfail test (Task 5) and listed under *Out of scope*.

## Prerequisites

- The working tree has the user's **uncommitted** multi-class YOLO-World work in `generalist_node.py`, `world_bbox.py`, and `test/test_world_bbox_multiclass.py`, which provides `world.last_labels`. Task 4 edits `generalist_node.py`. **The user must commit (or explicitly hand over) that work before Task 3 starts. Do not revert, stash, or re-commit it yourself.** `config/fastdds_shm.xml` and `vision_bringup.launch.py` are also modified and unrelated: never `git add` them.
- Create a feature branch from the current branch (`tinker2-net`): `git switch -c person-seg-geometry`.

## Global Constraints

- Python 3.10, venv `src/tk26_vision/.venv-vision-main`; **no new dependencies**.
- **Never pass `classes=` (or any kwarg other than `imgsz`/`verbose`) to `self.model(...)` in the person pass.** Ultralytics 8.4.33 `engine/model.py:531` merges predict kwargs into the cached predictor args (`self.predictor.args = get_cfg(self.predictor.args, args)`), so `classes=[0]` would persist and silently make the parent YOLO path (`_detect_objects`) person-only. Filter on `boxes.cls` in Python instead.
- Ultralytics applies `conf=0.25` on every predict call, so `person_seg_conf` values below 0.25 have no effect. Document this; don't work around it.
- Do not modify `object_detection_new` (shared with the specialist node).
- The new params and their defaults, verbatim: `person_seg_geometry=True`, `person_seg_conf=0.25`, `person_match_min_iou=0.3`, `person_match_min_containment=0.8`, `person_seg_unmatched='drop'` (allowed: `'drop'`, `'sam'`).
- Test command (zsh can't source the ROS setup files, so wrap in bash):
  ```bash
  cd /home/tinker/tk25_ws && bash -c 'source /opt/ros/humble/setup.bash && source install/setup.bash && source src/tk26_vision/.venv-vision-main/bin/activate && ROS2_PTH_WARNED=1 python -m pytest src/tk26_vision/src/object_detection_generalist/test -q'
  ```
  Baseline before Task 1: `39 passed`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Object prompts that mention a person** ("cup held by the person", "the person's bag"). Expected: treated as objects, SAM mask kept. Pinned in Task 1 (`test_non_person_phrases`).
2. **A VLM label that mentions a person on a non-person single-class prompt** (prompt `bowl`, label `woman holding a bowl`). Expected: still a bowl, no YOLO pass. Pinned in Task 3 (`test_vlm_label_mentioning_person_does_not_make_object_box_a_person`).
3. **Multi-class prompts mixing people and objects** (`person . chair`). Expected: only boxes whose label normalizes to the person class are swapped. Pinned in Task 3 (`test_multiclass_only_person_boxes_swapped`, `test_multiclass_unlabelled_box_untouched`).
4. **Shared YOLO model state.** Expected: the person pass leaves the parent `/object_detection_generalist` YOLO path unchanged (no sticky `classes`). Pinned in Task 3 (`test_detect_person_instances_never_passes_classes_and_filters`).
5. **The YOLO person pass throws** (CUDA OOM, model error). Expected: the call still succeeds with the old SAM masks and an error log, never a crash or an empty answer. Pinned in Task 3 (`test_yolo_failure_keeps_sam_masks`).

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/object_detection_generalist/object_detection_generalist/person_geometry.py` | Create | Pure helpers: `is_person_phrase`, `person_class_id`, `parse_unmatched_policy`, `box_iou`, `box_containment`, `pad_to_multiple`, `PersonInstance`, `instances_from_yolo_result`, `PersonMatch`, `match_person_instances`, default constants |
| `src/object_detection_generalist/object_detection_generalist/generalist_node.py` | Modify | Params; `_person_cls_id`; `_detect_person_instances`; `_box_classes`; `_apply_person_geometry`; `_person_geometry_error`; wiring in `_world_pipeline` / `_vlm_pipeline`; `person_geometry` in `_log_debug`; clearer skip warning in `_build_fallback_objects` |
| `src/object_detection_generalist/test/test_person_geometry.py` | Create | Unit tests for the pure module |
| `src/object_detection_generalist/test/test_person_geometry_node.py` | Create | Node-method tests with a fake `self` |
| `src/object_detection_generalist/test/test_person_geometry_pipelines.py` | Create | Pipeline-wiring tests (both paths) + debug-log test |
| `src/object_detection_generalist/test/test_person_geometry_replay.py` | Create | Real-model replay on the GPSR frames |
| `src/object_detection_generalist/test/fixtures/person_geometry/` | Create | 7 JPEG frames + `cases.json` |
| `CLAUDE.md`, `DEV_NOTES.md` | Modify | Param docs + dated entry |

---

### Task 1: Person-phrase classification

**Files:**
- Create: `src/object_detection_generalist/object_detection_generalist/person_geometry.py`
- Test: `src/object_detection_generalist/test/test_person_geometry.py`

**Interfaces:**
- Produces: `is_person_phrase(text: str) -> bool`, `person_class_id(names) -> int | None` (`names` is a dict `{id: name}` or a list).

- [ ] **Step 1: Write the failing test**

Create `test/test_person_geometry.py`:

```python
"""Unit tests for person_geometry (pure numpy; no ROS node, no model).

Run:
    cd /home/tinker/tk25_ws && bash -c 'source /opt/ros/humble/setup.bash && \
        source install/setup.bash && \
        source src/tk26_vision/.venv-vision-main/bin/activate && \
        python -m pytest src/tk26_vision/src/object_detection_generalist/test/test_person_geometry.py -v'
"""
import pytest

from object_detection_generalist.person_geometry import (
    is_person_phrase,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run (from Global Constraints, narrowed): `... python -m pytest src/tk26_vision/src/object_detection_generalist/test/test_person_geometry.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'object_detection_generalist.person_geometry'`.

- [ ] **Step 3: Write minimal implementation**

Create `object_detection_generalist/person_geometry.py`:

```python
"""Person geometry for the generalist's open-vocab paths.

YOLO-World and VLM boxes are segmented by box-prompted MobileSAM. On person
boxes clipped by the image edge, SAM often segments the wall behind the
person, so the 3D centroid lands on the wall (GPSR sim 2026-10-04/05, report
``tinker-sim/6.0.1/docs/issue_reports/2026-10-08-gpsr-nav-bug-report.md``
section 4). For person boxes the generalist instead takes the mask of the
COCO ``person`` instance that the pretrained YOLO-seg model finds in the
same box. Everything here is pure numpy so it can be tested without a node
or a model.
"""

from __future__ import annotations

import re

PERSON_WORDS = frozenset({
    'person', 'persons', 'people', 'man', 'men', 'woman', 'women',
    'human', 'humans', 'guest', 'guests', 'child', 'children', 'kid',
    'kids', 'boy', 'boys', 'girl', 'girls', 'lady', 'ladies',
    'gentleman', 'gentlemen', 'operator', 'individual',
})

# A person word names the target only when it comes before any of these:
# "person next to the cup" is a person, "cup next to the person" is a cup.
RELATION_WORDS = frozenset({
    'with', 'next', 'near', 'beside', 'by', 'on', 'in', 'at', 'of',
    'behind', 'under', 'above', 'below', 'from', 'for', 'to', 'belonging',
    'between', 'beneath', 'inside', 'outside', 'over', 'against', 'around',
    'along', 'across',
})


def is_person_phrase(text: str) -> bool:
    """True when ``text`` names a person rather than an object near one."""
    tokens = re.findall(r'[a-z]+', (text or '').lower())
    for i, tok in enumerate(tokens):
        if tok in RELATION_WORDS:
            return False
        if tok in PERSON_WORDS:
            # "person's bag": the possessive person word modifies the next noun.
            return not (i + 1 < len(tokens) and tokens[i + 1] == 's')
    return False


def person_class_id(names) -> int | None:
    """Index of the ``person`` class in an Ultralytics ``model.names``."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for cls_id, name in items:
        if name == 'person':
            return int(cls_id)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... python -m pytest src/tk26_vision/src/object_detection_generalist/test/test_person_geometry.py -v`
Expected: all PASS (31 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/object_detection_generalist/object_detection_generalist/person_geometry.py src/object_detection_generalist/test/test_person_geometry.py
git commit -m "feat(generalist): person-phrase classification for person geometry

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Box overlap, YOLO instance extraction, and matching

**Files:**
- Modify: `src/object_detection_generalist/object_detection_generalist/person_geometry.py`
- Test: `src/object_detection_generalist/test/test_person_geometry.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1 beyond the module.
- Produces:
  - `DEFAULT_PERSON_CONF = 0.25`, `DEFAULT_MIN_IOU = 0.3`, `DEFAULT_MIN_CONTAINMENT = 0.8`
  - `box_iou(a, b) -> float`, `box_containment(inner, outer) -> float` (boxes are `(x1, y1, x2, y2)` ints)
  - `pad_to_multiple(img: np.ndarray, k: int = 32) -> np.ndarray` (pads bottom/right with zeros; returns `img` itself when already aligned)
  - `@dataclass(frozen=True) PersonInstance(bbox: tuple[int,int,int,int], mask: np.ndarray (bool HxW), conf: float)`
  - `instances_from_yolo_result(result, h: int, w: int, *, conf_floor: float, cls_id: int) -> list[PersonInstance]`
  - `@dataclass(frozen=True) PersonMatch(instance: int | None, iou: float, containment: float)`
  - `match_person_instances(prompt_bboxes, instances, *, min_iou: float, min_containment: float) -> list[PersonMatch]` (one entry per prompt box, same order)

- [ ] **Step 1: Write the failing tests**

Append to `test/test_person_geometry.py`:

```python
from types import SimpleNamespace

import numpy as np

from object_detection_generalist.person_geometry import (
    PersonInstance,
    box_containment,
    box_iou,
    instances_from_yolo_result,
    match_person_instances,
    pad_to_multiple,
)

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... python -m pytest src/tk26_vision/src/object_detection_generalist/test/test_person_geometry.py -v`
Expected: collection error `ImportError: cannot import name 'PersonInstance'`.

- [ ] **Step 3: Write minimal implementation**

In `person_geometry.py`, extend the imports at the top to:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from vision_util.mask_utils import largest_connected_component_in_bbox

Bbox = tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords

DEFAULT_PERSON_CONF = 0.25
DEFAULT_MIN_IOU = 0.3
# A feet-only person (edge-clipped) has a small box inside a tall prompt box:
# low IoU, but fully contained.
DEFAULT_MIN_CONTAINMENT = 0.8
```

Append to the end of the file:

```python
def _area(b: Bbox) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _intersection(a: Bbox, b: Bbox) -> int:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return ix * iy if ix > 0 and iy > 0 else 0


def box_iou(a: Bbox, b: Bbox) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def box_containment(inner: Bbox, outer: Bbox) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    area = _area(inner)
    return _intersection(inner, outer) / area if area > 0 else 0.0


def pad_to_multiple(img: np.ndarray, k: int = 32) -> np.ndarray:
    """Zero-pad bottom/right so both sides are multiples of ``k``.

    Same padding the parent's ``_detect_objects`` uses, so the model sees
    the image without letterboxing and its masks crop back with [:h, :w].
    """
    h, w = img.shape[:2]
    hp, wp = -(-h // k) * k, -(-w // k) * k
    if (hp, wp) == (h, w):
        return img
    return cv2.copyMakeBorder(
        img, 0, hp - h, 0, wp - w, cv2.BORDER_CONSTANT, value=0)


@dataclass(frozen=True)
class PersonInstance:
    bbox: Bbox
    mask: np.ndarray  # bool HxW at the original image size
    conf: float


def _to_numpy(x) -> np.ndarray:
    """Tensor (any device) or array-like to numpy."""
    if hasattr(x, 'detach'):
        x = x.detach()
    if hasattr(x, 'cpu'):
        x = x.cpu()
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.asarray(x)


def instances_from_yolo_result(result, h: int, w: int, *,
                               conf_floor: float,
                               cls_id: int) -> list[PersonInstance]:
    """Person instances from one Ultralytics seg result.

    ``h``/``w`` are the original (unpadded) image size; masks come back at
    the padded model-input size and are cropped to it.
    """
    boxes = getattr(result, 'boxes', None)
    masks = getattr(result, 'masks', None)
    if boxes is None or masks is None or getattr(masks, 'data', None) is None:
        return []
    xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
    confs = _to_numpy(boxes.conf).reshape(-1)
    classes = _to_numpy(boxes.cls).reshape(-1)
    data = _to_numpy(masks.data)
    out: list[PersonInstance] = []
    for i in range(min(len(xyxy), len(confs), len(classes), len(data))):
        if int(classes[i]) != cls_id or float(confs[i]) < conf_floor:
            continue
        mask = data[i][:h, :w] > 0.5
        if mask.shape != (h, w):
            continue
        x1, y1, x2, y2 = (int(v) for v in xyxy[i])
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        mask = largest_connected_component_in_bbox(mask, (x1, y1, x2, y2))
        if not mask.any():
            continue
        out.append(PersonInstance(
            bbox=(x1, y1, x2, y2), mask=mask, conf=float(confs[i])))
    return out


@dataclass(frozen=True)
class PersonMatch:
    instance: int | None  # index into instances; None when unmatched
    iou: float            # matched pair's IoU, or best seen when unmatched
    containment: float    # same, for |person ∩ prompt| / |person|


def match_person_instances(prompt_bboxes: Sequence[Bbox],
                           instances: Sequence[PersonInstance], *,
                           min_iou: float,
                           min_containment: float) -> list[PersonMatch]:
    """One-to-one greedy assignment of prompt boxes to person instances.

    A pair qualifies when IoU >= ``min_iou`` or the instance box lies at
    least ``min_containment`` inside the prompt box. Pairs are taken in
    descending IoU (then containment) order, so a full-body instance beats a
    feet-only fragment and a duplicate prompt box does not reuse a person.
    """
    n = len(prompt_bboxes)
    best_iou = [0.0] * n
    best_cont = [0.0] * n
    candidates = []
    for p, pb in enumerate(prompt_bboxes):
        for k, inst in enumerate(instances):
            iou = box_iou(pb, inst.bbox)
            cont = box_containment(inst.bbox, pb)
            best_iou[p] = max(best_iou[p], iou)
            best_cont[p] = max(best_cont[p], cont)
            if iou >= min_iou or cont >= min_containment:
                candidates.append((iou, cont, p, k))
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    assigned: dict[int, PersonMatch] = {}
    used: set[int] = set()
    for iou, cont, p, k in candidates:
        if p in assigned or k in used:
            continue
        assigned[p] = PersonMatch(k, iou, cont)
        used.add(k)
    return [
        assigned.get(p, PersonMatch(None, best_iou[p], best_cont[p]))
        for p in range(n)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `... python -m pytest src/tk26_vision/src/object_detection_generalist/test/test_person_geometry.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/object_detection_generalist/object_detection_generalist/person_geometry.py src/object_detection_generalist/test/test_person_geometry.py
git commit -m "feat(generalist): person instance extraction and box matching

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Node methods — params, YOLO person pass, mask swap

**Files:**
- Modify: `src/object_detection_generalist/object_detection_generalist/person_geometry.py` (add `parse_unmatched_policy`)
- Modify: `src/object_detection_generalist/object_detection_generalist/generalist_node.py` (imports, `_declare_parameters`, `_load_parameters`, `__init__`, new methods)
- Test: `src/object_detection_generalist/test/test_person_geometry_node.py`

**Interfaces:**
- Consumes (Task 2): `PersonInstance`, `instances_from_yolo_result`, `match_person_instances`, `pad_to_multiple`, `person_class_id`, `is_person_phrase`, `DEFAULT_*`.
- Produces (used by Task 4):
  - `GeneralistDetectionNode._apply_person_geometry(self, rgb_img, bboxes, masks, labels, prompt) -> tuple[list[Bbox], list[np.ndarray | None], list[dict]]`. It returns `(geom_bboxes, masks, info)`; the lists are new, the inputs are not mutated. `info` has one dict per person box with keys `box` (int index), `action` (`'yolo_person' | 'dropped' | 'sam_unmatched' | 'sam_yolo_error'`), `iou`, `containment` (floats; absent for `'sam_yolo_error'`), plus `person_bbox` (list) and `person_conf` (float) for `'yolo_person'`.
  - `GeneralistDetectionNode._person_geometry_error(objects, person_info) -> str | None` (staticmethod).
  - `GeneralistDetectionNode._detect_person_instances(self, rgb_img) -> list[PersonInstance]`.
  - `GeneralistDetectionNode._box_classes(n_boxes, labels, prompt) -> list[str]` (staticmethod).
  - `person_geometry.parse_unmatched_policy(value) -> str | None`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_person_geometry_node.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... python -m pytest src/tk26_vision/src/object_detection_generalist/test/test_person_geometry_node.py -v`
Expected: collection error `ImportError: cannot import name 'parse_unmatched_policy'`.

- [ ] **Step 3a: Add `parse_unmatched_policy` to `person_geometry.py`**

Append:

```python
UNMATCHED_POLICIES = ('drop', 'sam')


def parse_unmatched_policy(value) -> str | None:
    """Normalize the ``person_seg_unmatched`` param; None when invalid."""
    v = str(value or '').strip().lower()
    return v if v in UNMATCHED_POLICIES else None
```

- [ ] **Step 3b: Imports in `generalist_node.py`**

After `from .sam_mask import SamPredictor` add:

```python
from .person_geometry import (
    DEFAULT_MIN_CONTAINMENT,
    DEFAULT_MIN_IOU,
    DEFAULT_PERSON_CONF,
    PersonInstance,
    instances_from_yolo_result,
    is_person_phrase,
    match_person_instances,
    pad_to_multiple,
    parse_unmatched_policy,
    person_class_id,
)
```

- [ ] **Step 3c: Declare params**

In `_declare_parameters`, directly after `self.declare_parameter('world_iou_threshold', 0.5)` add:

```python
        # Person geometry (GPSR sim report 2026-10-08 §4): for person boxes
        # from YOLO-World or the VLM, take the 3D centroid from the matching
        # COCO 'person' instance mask of the pretrained YOLO-seg model rather
        # than a box-prompted SAM mask, which on edge-clipped boxes segments
        # the wall behind the person.
        self.declare_parameter('person_seg_geometry', True)
        # Instance floor. Ultralytics applies its own conf=0.25 on every
        # predict call, so values below 0.25 have no effect.
        self.declare_parameter('person_seg_conf', DEFAULT_PERSON_CONF)
        # A person box matches a YOLO person when box IoU >= min_iou OR the
        # YOLO person box lies >= min_containment inside it (feet-only
        # edge-clipped persons have low IoU but full containment).
        self.declare_parameter('person_match_min_iou', DEFAULT_MIN_IOU)
        self.declare_parameter(
            'person_match_min_containment', DEFAULT_MIN_CONTAINMENT)
        # 'drop': a person box with no matching YOLO person is removed.
        # 'sam': keep its SAM mask (the pre-2026-10 behaviour).
        self.declare_parameter('person_seg_unmatched', 'drop')
```

- [ ] **Step 3d: Load params**

In `_load_parameters`, after the `self.world_iou_threshold = ...` assignment add:

```python
        self.person_seg_geometry = bool(
            self.get_parameter('person_seg_geometry').value
        )
        self.person_seg_conf = float(
            self.get_parameter('person_seg_conf').value
        )
        self.person_match_min_iou = float(
            self.get_parameter('person_match_min_iou').value
        )
        self.person_match_min_containment = float(
            self.get_parameter('person_match_min_containment').value
        )
        raw_policy = self.get_parameter('person_seg_unmatched').value
        policy = parse_unmatched_policy(raw_policy)
        if policy is None:
            self.get_logger().warn(
                f'person_seg_unmatched={raw_policy!r} is not drop|sam; '
                'using drop'
            )
            policy = 'drop'
        self.person_seg_unmatched = policy
```

- [ ] **Step 3e: Resolve the person class id at startup**

In `__init__`, directly after `self._yolo_class_names = set(self.model.names.values())` add:

```python
        self._person_cls_id = person_class_id(self.model.names)
        if self.person_seg_geometry and self._person_cls_id is None:
            self.get_logger().warn(
                "person_seg_geometry: YOLO model has no 'person' class; "
                'person boxes keep their SAM masks'
            )
```

- [ ] **Step 3f: Add the methods**

Add this block right after `_cancel_vlm` (before `_log_debug`):

```python
    # --- person geometry ---------------------------------------------------

    def _detect_person_instances(self, rgb_img) -> list[PersonInstance]:
        """Run the pretrained YOLO-seg model; return its person instances.

        Never pass ``classes=`` here: Ultralytics merges predict kwargs into
        the cached predictor args (engine/model.py, ``get_cfg(
        self.predictor.args, args)``), so a class filter would stick and make
        the parent YOLO path person-only. Filter on ``boxes.cls`` instead.
        The lock serializes the race path's two legs on the shared model.
        """
        h, w = rgb_img.shape[:2]
        padded = pad_to_multiple(rgb_img, 32)
        with self._yolo_model_lock:
            results = self.model(
                padded, imgsz=padded.shape[:2], verbose=False,
            )
        instances: list[PersonInstance] = []
        for result in results:
            instances.extend(instances_from_yolo_result(
                result, h, w,
                conf_floor=self.person_seg_conf,
                cls_id=self._person_cls_id,
            ))
        return instances

    @staticmethod
    def _box_classes(n_boxes: int, labels, prompt: str) -> list[str]:
        """Prompt class each box answers; '' when it cannot be told.

        Single-class prompt: every box is that class and the label is
        ignored, so a VLM label like 'woman holding a bowl' on prompt 'bowl'
        stays a bowl. Multi-class prompt: the box label normalized onto the
        prompt classes; unlabelled or unmatched boxes get ''.
        """
        classes = GeneralistDetectionNode._parse_prompt_classes(prompt)
        if len(classes) == 1:
            return [classes[0]] * n_boxes
        out = []
        for i in range(n_boxes):
            label = labels[i] if labels and i < len(labels) else ''
            cls = GeneralistDetectionNode._normalize_vlm_label(
                label, classes, prompt)
            out.append(cls if cls in classes else '')
        return out

    def _apply_person_geometry(self, rgb_img, bboxes, masks, labels,
                               prompt: str):
        """Swap SAM masks for YOLO person masks on person boxes.

        Returns ``(geom_bboxes, masks, info)`` as new lists aligned 1:1 with
        the input. A matched person box gets the YOLO instance mask and the
        instance bbox (the centroid ROI); an unmatched one gets mask None
        (dropped) or keeps its SAM mask, per ``person_seg_unmatched``.
        Non-person boxes pass through unchanged. ``info`` has one entry per
        person box for the vision log.
        """
        geom_bboxes = [tuple(int(v) for v in b) for b in bboxes]
        masks = list(masks)
        if (not self.person_seg_geometry or self._person_cls_id is None
                or not bboxes):
            return geom_bboxes, masks, []
        person_idx = [
            i for i, cls in enumerate(
                self._box_classes(len(bboxes), labels, prompt))
            if is_person_phrase(cls)
        ]
        if not person_idx:
            return geom_bboxes, masks, []
        try:
            instances = self._detect_person_instances(rgb_img)
        except Exception as exc:  # noqa: BLE001 — degrade to SAM masks
            self.get_logger().error(
                f'person geometry: YOLO person pass failed ({exc}); '
                'keeping SAM masks'
            )
            return geom_bboxes, masks, [
                {'box': i, 'action': 'sam_yolo_error'} for i in person_idx
            ]
        matches = match_person_instances(
            [geom_bboxes[i] for i in person_idx], instances,
            min_iou=self.person_match_min_iou,
            min_containment=self.person_match_min_containment,
        )
        info = []
        for i, match in zip(person_idx, matches):
            entry = {
                'box': i,
                'iou': round(match.iou, 3),
                'containment': round(match.containment, 3),
            }
            if match.instance is not None:
                inst = instances[match.instance]
                masks[i] = inst.mask
                geom_bboxes[i] = inst.bbox
                entry.update(
                    action='yolo_person',
                    person_bbox=list(inst.bbox),
                    person_conf=round(inst.conf, 3),
                )
            elif self.person_seg_unmatched == 'sam':
                entry['action'] = 'sam_unmatched'
            else:
                masks[i] = None
                entry['action'] = 'dropped'
            info.append(entry)
        self.get_logger().info(
            f'person geometry: {len(person_idx)} person box(es), '
            f'{len(instances)} YOLO person(s): '
            + ', '.join(
                f"box{e['box']}={e['action']}"
                f"(iou={e['iou']:.2f},cont={e['containment']:.2f})"
                for e in info
            )
        )
        return geom_bboxes, masks, info

    @staticmethod
    def _person_geometry_error(objects, person_info) -> str | None:
        """Error text when person geometry dropped every person box."""
        dropped = sum(
            1 for e in person_info if e.get('action') == 'dropped')
        if objects or not dropped:
            return None
        return (
            f'person geometry: {dropped} person box(es) had no matching '
            'YOLO person mask (dropped)'
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `... python -m pytest src/tk26_vision/src/object_detection_generalist/test -q`
Expected: all PASS (39 baseline + Task 1–3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/object_detection_generalist/object_detection_generalist/person_geometry.py src/object_detection_generalist/object_detection_generalist/generalist_node.py src/object_detection_generalist/test/test_person_geometry_node.py
git commit -m "feat(generalist): YOLO person pass and SAM-mask swap for person boxes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Wire into the VLM and YOLO-World pipelines + vision log

**Files:**
- Modify: `src/object_detection_generalist/object_detection_generalist/generalist_node.py` (`_world_pipeline`, `_vlm_pipeline`, `_log_debug`, `_build_fallback_objects`)
- Test: `src/object_detection_generalist/test/test_person_geometry_pipelines.py`

**Interfaces:**
- Consumes (Task 3): `_apply_person_geometry(rgb_img, bboxes, masks, labels, prompt) -> (geom_bboxes, masks, info)`, `_person_geometry_error(objects, info)`.
- Produces: both pipeline result dicts gain `'person_geometry': list[dict]`. `'masks'` now holds the masks actually used (None for dropped boxes). `'bboxes'` stays the detector's boxes. `'error'` is set when every person box was dropped. The vision-log request JSON gains `person_geometry`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_person_geometry_pipelines.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... python -m pytest src/tk26_vision/src/object_detection_generalist/test/test_person_geometry_pipelines.py -v`
Expected: FAIL. `test_world_pipeline_uses_person_geometry_output` raises `KeyError: 'geom_args'` (the pipeline never calls `_apply_person_geometry`), and `test_log_debug_records_person_geometry` raises `KeyError: 'person_geometry'`.

- [ ] **Step 3a: `_world_pipeline`**

Replace the block from `with self._sam_lock:` down to the end of the method's `return {...}` with:

```python
        with self._sam_lock:
            masks, sam_elapsed = self._sam.segment(rgb_img, bboxes)
        labels = world_labels if len(world_labels) == len(bboxes) else None
        geom_bboxes, masks, person_info = self._apply_person_geometry(
            rgb_img, bboxes, masks, labels, prompt,
        )
        objects, segments, closest_distances = self._build_fallback_objects(
            prompt, geom_bboxes, masks, points, valid_mask, camera,
            return_segments=return_segments, confs=confs,
            labels=labels,
            header=header, target_frame=target_frame,
        )
        objects, segments, closest_distances = self._apply_realsense_range_gate(
            objects, segments, camera, closest_distances,
        )
        if objects:
            objects, segments = self._sort_objects_and_segments(
                objects, segments, sort_mode,
                camera=camera, source_frame=header.frame_id, header=header,
                closest_distances=closest_distances,
            )
        return {
            'source': 'yolo_world',
            'objects': objects, 'segments': segments,
            'bboxes': bboxes, 'masks': masks, 'confs': confs,
            'world_elapsed': world_elapsed, 'vlm_elapsed': 0.0,
            'sam_elapsed': sam_elapsed,
            'person_geometry': person_info,
            'error': self._person_geometry_error(objects, person_info),
        }
```

- [ ] **Step 3b: `_vlm_pipeline`**

Replace the block from `with self._sam_lock:` down to the end of the method's `return {...}` with:

```python
        with self._sam_lock:
            # Re-check after acquiring the lock — caller may have abandoned
            # while we were queued behind another SAM call.
            if abandon_event is not None and abandon_event.is_set():
                return self._empty_result(
                    'vlm_sam', vlm_elapsed=vlm_elapsed, error='abandoned',
                )
            masks, sam_elapsed = self._sam.segment(rgb_img, bboxes)
        # Abandoned while SAM ran: skip the YOLO person pass too.
        if abandon_event is not None and abandon_event.is_set():
            return self._empty_result(
                'vlm_sam', vlm_elapsed=vlm_elapsed, error='abandoned',
            )
        geom_bboxes, masks, person_info = self._apply_person_geometry(
            rgb_img, bboxes, masks, cls_per_box, prompt,
        )
        objects, segments, closest_distances = self._build_fallback_objects(
            prompt, geom_bboxes, masks, points, valid_mask, camera,
            return_segments=return_segments,
            labels=cls_per_box,
            header=header, target_frame=target_frame,
        )
        objects, segments, closest_distances = self._apply_realsense_range_gate(
            objects, segments, camera, closest_distances,
        )
        if objects:
            objects, segments = self._sort_objects_and_segments(
                objects, segments, sort_mode,
                camera=camera, source_frame=header.frame_id, header=header,
                closest_distances=closest_distances,
            )
        return {
            'source': 'vlm_sam',
            'objects': objects, 'segments': segments,
            'bboxes': bboxes, 'masks': masks, 'confs': [],
            'world_elapsed': 0.0, 'vlm_elapsed': vlm_elapsed,
            'sam_elapsed': sam_elapsed,
            'vlm_labels': cls_per_box,
            'vlm_raw_labels': list(raw_labels),
            'vlm_meta': vlm_meta,
            'person_geometry': person_info,
            'error': self._person_geometry_error(objects, person_info),
        }
```

- [ ] **Step 3c: `_log_debug`**

In the `request_ctx = {...}` literal, after the `'depth_source': ...` entry add:

```python
            # Per person box: yolo_person | dropped | sam_unmatched |
            # sam_yolo_error, with match IoU/containment.
            'person_geometry': result.get('person_geometry') or [],
```

- [ ] **Step 3d: `_build_fallback_objects` warning**

Replace the warning text so a person-geometry drop isn't reported as a SAM failure:

```python
                self.get_logger().warn(
                    f'fallback object {i}: empty or rejected mask for '
                    f'bbox={bbox}; skipping'
                )
```

(It replaces `f'fallback object {i}: empty mask post-CC for bbox={bbox}; '` / `'skipping'`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `... python -m pytest src/tk26_vision/src/object_detection_generalist/test -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/object_detection_generalist/object_detection_generalist/generalist_node.py src/object_detection_generalist/test/test_person_geometry_pipelines.py
git commit -m "fix(generalist): person boxes take YOLO person masks on VLM and YOLO-World paths

SAM on edge-clipped person boxes segmented the wall behind the person and
the centroid landed on the wall (GPSR sim report 2026-10-08 section 4).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Real-frame replay regression + docs

**Files:**
- Create: `src/object_detection_generalist/test/fixtures/person_geometry/*.jpg` (7 frames) and `cases.json`
- Create: `src/object_detection_generalist/test/test_person_geometry_replay.py`
- Modify: `CLAUDE.md` (Configuration list), `DEV_NOTES.md` (new top entry)

**Interfaces:**
- Consumes: `GeneralistDetectionNode._detect_person_instances`, `match_person_instances`, `person_class_id`, `DEFAULT_MIN_IOU`, `DEFAULT_MIN_CONTAINMENT`, `DEFAULT_PERSON_CONF`, `vision_util.weights_cache.find_cached`.

- [ ] **Step 1: Stage fixtures**

```bash
V=/home/tinker/tinker-sim/6.0.1/vision_log/20260822_081720
F=/home/tinker/tk25_ws/src/tk26_vision/src/object_detection_generalist/test/fixtures/person_geometry
mkdir -p $F
for ts in 20261005_005738_508 20261005_090529_466 20261004_205905_361 20261005_015548_606 20261004_205435_943; do
  cp $V/generalist_detection_node_yolo_world_orig_$ts.jpg $F/$ts.jpg
done
for ts in 20261003_080553_756 20260822_114230_129; do
  cp $V/generalist_detection_node_vlm_sam_orig_$ts.jpg $F/$ts.jpg
done
ls -la $F
```

Expected: 7 JPEGs, each roughly 50–250 KB.

Create `$F/cases.json`. The `person_region` values are hand-read from the frames (x1, y1, x2, y2 that the person's body occupies), not taken from model output:

```json
[
  {"frame": "20261005_005738_508.jpg", "source": "yolo_world",
   "prompt": "person pointing to the left", "prompt_box": [1121, 46, 1279, 567],
   "expect": "person", "person_region": [1120, 120, 1280, 540],
   "note": "rerun64 confirm scan; SAM masked the wall -> (-4.39, -5.22)"},
  {"frame": "20261005_090529_466.jpg", "source": "yolo_world",
   "prompt": "person pointing to the left", "prompt_box": [1127, 38, 1279, 719],
   "expect": "person", "person_region": [1120, 130, 1280, 540],
   "note": "rerun92; SAM masked the wall -> (-4.40, -5.22)"},
  {"frame": "20261004_205905_361.jpg", "source": "yolo_world",
   "prompt": "person pointing to the left", "prompt_box": [1153, 99, 1279, 719],
   "expect": "person", "person_region": [1150, 630, 1280, 720],
   "note": "rerun57; feet only; SAM masked wall/floor seam -> (-2.89, -5.80) outside arena"},
  {"frame": "20261005_015548_606.jpg", "source": "yolo_world",
   "prompt": "person pointing to the left", "prompt_box": [1200, 185, 1279, 695],
   "expect": "person", "person_region": [1195, 470, 1280, 640],
   "note": "rerun68; feet only; SAM masked the wall -> (-4.43, -5.16)"},
  {"frame": "20261003_080553_756.jpg", "source": "vlm_sam",
   "prompt": "standing person", "prompt_box": [1183, 382, 1279, 564],
   "expect": "person", "person_region": [1180, 380, 1280, 570],
   "note": "VLM (qwen3-vl) box on legs clipped at the right edge"},
  {"frame": "20260822_114230_129.jpg", "source": "vlm_sam",
   "prompt": "person", "prompt_box": [650, 182, 796, 582],
   "expect": "person", "person_region": [650, 180, 800, 585],
   "note": "VLM (gemini-2.5-flash) full-body box"},
  {"frame": "20261004_205435_943.jpg", "source": "yolo_world",
   "prompt": "person pointing to the left", "prompt_box": [0, 444, 672, 719],
   "expect": "robot_arm", "person_region": null,
   "note": "rerun57; YOLO-World boxed the robot's own arm; YOLO-seg also calls it person (known gap)"}
]
```

- [ ] **Step 2: Write the replay test**

Create `test/test_person_geometry_replay.py`:

```python
"""Replay the GPSR sim frames where SAM segmented the wall behind a person.

Report: tinker-sim/6.0.1/docs/issue_reports/2026-10-08-gpsr-nav-bug-report.md
section 4. Runs the real yolo11m-seg model through the node's person pass
and matching, and checks the matched mask sits on the person (hand-read
region), not the wall. Skipped when the weights are not in the local cache —
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
```

- [ ] **Step 3: Run the replay test**

Run: `... python -m pytest src/tk26_vision/src/object_detection_generalist/test/test_person_geometry_replay.py -v`
Expected: 6 PASS + 1 XFAIL. During planning, the same model on these frames matched every person case (IoU .57–.95 or containment 1.0), and the matched centroids fell inside the regions. If a case fails, **don't loosen the region**: open the frame and the matched instance, and report the result.

- [ ] **Step 4: Docs**

`CLAUDE.md`, *Configuration* section: after the `object_detection_new:` bullet, add:

```markdown
- `object_detection_generalist` person geometry: for person boxes from the VLM+SAM or YOLO-World+SAM paths (a box is a person when its prompt class names a person — `is_person_phrase` in `person_geometry.py`; "cup next to the person" is not), the 3D centroid comes from the matching COCO `person` instance mask of the pretrained YOLO-seg model (`model_path`), not the box-prompted SAM mask — SAM segmented the wall behind edge-clipped persons (GPSR sim report 2026-10-08 §4). Params: `person_seg_geometry` (default `true`), `person_seg_conf` (`0.25`; Ultralytics' own 0.25 floor makes lower values moot), `person_match_min_iou` (`0.3`) OR `person_match_min_containment` (`0.8`, catches feet-only edge-clipped persons), `person_seg_unmatched` (`'drop'` default — a person box with no YOLO person is removed; `'sam'` keeps the SAM mask). Per-box outcome is logged (`person geometry: …`) and written to the vision-log JSON as `person_geometry`. Known gap: YOLO-seg labels the robot's own arm `person` in downward-looking head frames; a box on the arm still matches (strict-xfail in `test_person_geometry_replay.py`).
```

`DEV_NOTES.md`: insert a new entry directly above `## 2026-08-22 — .env-driven VLM model ids…`:

```markdown
## 2026-10-08 — generalist: person boxes take YOLO person masks (GPSR wall-centroid fix)

**Symptom.** GPSR sim living-room trials (batches `t2-20-live-rerun57/59/64/68/79/92`) picked the person ~1 m behind the real one, on the south-west wall, and once outside the arena (report `tinker-sim/6.0.1/docs/issue_reports/2026-10-08-gpsr-nav-bug-report.md` §4).

**Root cause.** The generalist emitted those points. With the person clipped by the image edge, YOLO-World returned a weak box (conf 0.05–0.2), box-prompted MobileSAM segmented the wall inside it, and `_calculate_centroid` put the centroid on the wall (camera distance ~2.3 → ~3.3 m).

**Fix.** `person_geometry.py` + `_apply_person_geometry`: on both the VLM and YOLO-World paths, person boxes take the matching YOLO-seg `person` instance mask and box (IoU ≥ 0.3 or containment ≥ 0.8). Unmatched person boxes are dropped (`person_seg_unmatched`). Replay: `test/test_person_geometry_replay.py` on 6 saved frames (4 YOLO-World wall/outside cases, 2 VLM).

**Open.** YOLO-seg calls the robot's own arm `person` (conf ~0.75 in downward head frames). A YOLO-World/VLM box on the arm still matches, which is status quo, pinned as a strict xfail. It needs a self-body filter, e.g. reject centroids inside the robot footprint in `base_link`.
```

- [ ] **Step 5: Run the full suite**

Run: `... python -m pytest src/tk26_vision/src/object_detection_generalist/test -q`
Expected: all PASS, 1 xfailed.

- [ ] **Step 6: Commit**

```bash
git add src/object_detection_generalist/test/fixtures/person_geometry src/object_detection_generalist/test/test_person_geometry_replay.py CLAUDE.md DEV_NOTES.md
git commit -m "test(generalist): replay GPSR wall-centroid frames through person geometry; docs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Build, static smoke, and live sim confirmation

No code. This verifies the deployed node. **The sim run takes the GPU and the single-tenant stack for ~25 min, so ask the user before Step 3.**

- [ ] **Step 1: Build**

```bash
cd /home/tinker/tk25_ws && ./src/tk26_vision/scripts/build.sh --packages-select object_detection_generalist
```

Expected: `Finished <<< object_detection_generalist`, no errors.

- [ ] **Step 2: T0 static smoke**

```bash
cd /home/tinker/tk25_ws && bash src/tk26_vision/scripts/tests/t0_static.sh
```

Expected: PASS (entry-point import of `generalist_node` included).

- [ ] **Step 3 (after the user approves): rerun the rerun92 living-room trial**

```bash
B=/home/tinker/tk25_ws/src/tk25_decision/src/behavior_tree/behavior_tree/GPSR/gpsr_runs/bench
N=$B/t2-20-live-persongeom1
mkdir -p $N
cp $B/t2-20-live-rerun92/battery-natural.sh $N/
sed -e 's/t2-20-live-rerun92/t2-20-live-persongeom1/' -e 's/^GPSR_ROS_DOMAIN_ID=92$/GPSR_ROS_DOMAIN_ID=93/' \
  $B/t2-20-live-rerun92/battery.conf > $N/battery.conf
cat $N/battery.conf   # OUT must point at persongeom1, STARTS="2"
bash $N/battery-natural.sh
```

- [ ] **Step 4: Check the outcome**

```bash
S=/home/tinker/tinker-sim/6.0.1/$(grep -ho "gpsr_stack_logs/[0-9T]*" $N/gpsr-stack-up.log | head -1)
grep -h "person geometry:\|Sorted by closest (orbbec)" $S/03-vision-0.log
grep -h "Picked detection at" $N/runs/*/orchestrator.log | sort | uniq -c
```

Pass criteria:
- Every `person geometry:` line for the living-room calls shows `yolo_person` or `dropped`, never `sam_*`.
- Every `Sorted by closest (orbbec)` point and every `Picked detection at` for the pointing-left person is within 0.5 m of (−3.8, −3.95), and none falls at y < −5.0 (the wall band).
- If a point is still off, check the matching `vision_log` overlay before changing anything. The overlay now shows the mask actually used.

Report the run verdict from `$N/runs/*/run.json` regardless of the outcome. The delivery FAIL also has non-vision causes (report §2–§3).

---

## Out of scope (follow-ups to raise with the user)

- **Robot arm detected as `person`.** YOLO-seg (and YOLO-World) label the robot's own gripper arm `person` in downward-looking head frames. This affects the `yolo_known` "person" path too (`pick closest person` → the arm at ~1 m, as in rerun57/68/79). The fix needs a self-body filter, such as rejecting centroids inside the robot footprint in `base_link`.
- **BT confirm-scan override** (`tk25_decision`): the confirm scan's detection replaces the scan's detection even when the two are >1 m apart.
- **Pan-sweep stops at an edge-clipped detection** (`tk25_decision`): the BT accepts a person at the image border instead of panning toward them.
- **Downstream static-map guard** in the approach planner (report §4 "Fix").
- Skipping SAM entirely for boxes that person geometry will replace (saves ~130 ms on person prompts). Only worth doing if latency matters.
