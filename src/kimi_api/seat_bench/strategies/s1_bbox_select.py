"""S1: one call returns a box + occupancy for every visible seat AND the
chosen empty seat's label. Recommendation point = center of the chosen
seat's box. Tests whether box regression localizes truer than pointing.
"""

from __future__ import annotations

import numpy as np

from ..geometry import box_center, decode_box_xyxy
from ..paths import ensure_kimi_api_importable
from ..providers import call_vlm
from .base import Result, build_request_text

ensure_kimi_api_importable()
from kimi_api._image_utils import encode_to_data_url  # noqa: E402

_SYSTEM = (
    "You help a robot seat a new guest. Look at the image and return JSON "
    "with fields: seats, choice.\n"
    "seats — array, one entry per sittable cushion (a 2-cushion sofa = 2 "
    "entries, a single armchair/stool = 1). Each entry: "
    '{"label": "<short identifier with a visual anchor>", '
    '"box_2d": [x1,y1,x2,y2], "occupied": true|false}. '
    "box_2d is the tight bounding box of the SEAT CUSHION (the flat surface "
    "a person sits on, NOT the backrest), normalized 0-1000 over the image "
    "where (0,0) is top-left and (1000,1000) is bottom-right.\n"
    "A cushion is OCCUPIED if a person sits on it or a large object rests on "
    "the cushion fabric; objects on a table/floor/armrest do not occupy it.\n"
    "choice — the label of one entry whose occupied is false (your "
    'recommendation), or "none" if every seat is occupied or none are visible.'
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "seats": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "box_2d": {"type": "array", "items": {"type": "integer"},
                           "minItems": 4, "maxItems": 4},
                "occupied": {"type": "boolean"},
            },
            "required": ["label", "box_2d", "occupied"],
            "additionalProperties": False,
        }},
        "choice": {"type": "string"},
    },
    "required": ["seats", "choice"],
    "additionalProperties": False,
}


def run(img_bgr: np.ndarray, req: dict, provider: str, logger=None) -> Result:
    h, w = img_bgr.shape[:2]
    res = Result(strategy="s1", provider=provider)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": encode_to_data_url(img_bgr)}},
            {"type": "text", "text": build_request_text(req)},
        ]},
    ]
    try:
        parsed, elapsed = call_vlm(provider, messages, schema=_SCHEMA,
                                   schema_name="seat_bbox_select", logger=logger)
    except Exception as exc:  # noqa: BLE001
        res.error = str(exc)
        return res
    res.elapsed_s, res.n_calls, res.raw = elapsed, 1, parsed
    seats = parsed.get("seats", []) or []
    res.visible_seats = seats
    choice = str(parsed.get("choice", "none") or "none")
    res.chosen_label = choice
    if choice.strip().lower() == "none":
        return res
    chosen = next((s for s in seats
                   if str(s.get("label", "")).strip().lower() == choice.strip().lower()), None)
    if chosen is None:
        res.error = f"choice {choice!r} not in seats list"
        return res
    box = decode_box_xyxy(chosen.get("box_2d"), w, h)
    if box is None:
        res.error = "chosen box failed to decode"
        return res
    res.box_xyxy = list(box)
    res.point_xy = list(box_center(box))
    return res
