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


UNMATCHED_POLICIES = ('drop', 'sam')


def parse_unmatched_policy(value) -> str | None:
    """Normalize the ``person_seg_unmatched`` param; None when invalid."""
    v = str(value or '').strip().lower()
    return v if v in UNMATCHED_POLICIES else None
