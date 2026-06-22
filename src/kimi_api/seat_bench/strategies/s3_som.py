"""S3 set-of-mark: detect seat candidates with YOLO-World (open vocab:
chair/sofa/stool/bench/couch/armchair), draw numbered boxes on the image,
and ask the VLM only to PICK a number + occupancy. Removes coordinate
regression from the final decision. Recommendation point = center of the
picked candidate box.

Degraded fallback: if YOLO-World yields < 2 candidates, fall back to S1's
VLM boxes as the marks (logged via res.raw['som_source']='s1_fallback').
"""

from __future__ import annotations

import cv2
import numpy as np

from ..geometry import box_center
from ..paths import ensure_kimi_api_importable
from ..providers import call_vlm
from ..world_candidates import detect_seat_candidates
from .base import Result, build_request_text
from .s1_bbox_select import run as s1_run

ensure_kimi_api_importable()
from kimi_api._image_utils import encode_to_data_url  # noqa: E402

_SYSTEM = (
    "The image has numbered boxes drawn over candidate seats. Return JSON "
    '{"choice": <int or -1>} where choice is the number of the best EMPTY '
    "seat for a new guest, or -1 if every numbered seat is occupied. A seat "
    "is occupied if a person sits on it or a large object rests on the "
    "cushion."
)
_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": "integer"}},
    "required": ["choice"], "additionalProperties": False,
}


def _draw_marks(img_bgr: np.ndarray, boxes: list) -> np.ndarray:
    out = img_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(out, str(i), (x1 + 4, max(20, y1 + 26)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    return out


def run(img_bgr: np.ndarray, req: dict, provider: str, logger=None) -> Result:
    res = Result(strategy="s3", provider=provider)
    boxes, det_elapsed = detect_seat_candidates(img_bgr, logger=logger)
    som_source = "yolo_world"

    if len(boxes) < 2:
        # Degrade to S1's VLM boxes as marks.
        s1 = s1_run(img_bgr, req, provider, logger=logger)
        res.elapsed_s += s1.elapsed_s
        res.n_calls += s1.n_calls
        boxes = [tuple(s["box_2d_px"]) for s in _s1_boxes_px(s1, img_bgr)]
        som_source = "s1_fallback"
        if len(boxes) < 1:
            res.error = s1.error or "no candidates from yolo or s1"
            res.raw = {"som_source": som_source}
            return res

    res.elapsed_s += det_elapsed
    marked = _draw_marks(img_bgr, boxes)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": encode_to_data_url(marked)}},
            {"type": "text", "text": build_request_text(req)
                + " Pick the numbered box that is the best empty seat."},
        ]},
    ]
    try:
        parsed, elapsed = call_vlm(provider, messages, schema=_SCHEMA,
                                   schema_name="seat_som", logger=logger)
    except Exception as exc:  # noqa: BLE001
        res.error = str(exc)
        res.raw = {"som_source": som_source, "n_candidates": len(boxes)}
        return res

    res.elapsed_s += elapsed
    res.n_calls += 1
    res.raw = {"som_source": som_source, "n_candidates": len(boxes), "vlm": parsed}
    choice = int(parsed.get("choice", -1))
    if choice < 0 or choice >= len(boxes):
        res.chosen_label = "none"
        return res
    box = boxes[choice]
    res.box_xyxy = list(box)
    res.point_xy = list(box_center(box))
    res.chosen_label = f"candidate_{choice}"
    return res


def _s1_boxes_px(s1: Result, img_bgr: np.ndarray) -> list:
    """Decode S1's normalized seat boxes to pixel boxes for SoM fallback."""
    from ..geometry import decode_box_xyxy
    h, w = img_bgr.shape[:2]
    out = []
    for s in (s1.visible_seats or []):
        box = decode_box_xyxy(s.get("box_2d"), w, h)
        if box is not None:
            out.append({"box_2d_px": list(box)})
    return out
