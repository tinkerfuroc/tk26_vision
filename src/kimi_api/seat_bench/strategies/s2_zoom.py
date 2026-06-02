"""S2: coarse boxes + select (call 1, reusing S1's contract), then crop the
chosen seat's box with margin and ask a SECOND call to place a precise
point on the cushion within the high-res crop. Crop-space point is mapped
back to full-image coordinates. Targets the 'point on wrong object' error
by giving call 2 far more pixels on the actual seat.
"""

from __future__ import annotations

import numpy as np

from ..geometry import decode_point_yx
from ..paths import ensure_kimi_api_importable
from ..providers import call_vlm
from .base import Result
from .s1_bbox_select import run as s1_run

ensure_kimi_api_importable()
from kimi_api._image_utils import encode_to_data_url  # noqa: E402

_CROP_SYSTEM = (
    "This image is a close-up crop of ONE seat. Return JSON {\"point\": [y, x]} "
    "where the point lands on the cushion fabric (the flat surface a person "
    "sits on, not the backrest, armrest, floor, or any person/object on it). "
    "y and x are integers 0-1000 normalized to THIS crop's dimensions "
    "(y=0 top, x=0 left)."
)
_CROP_SCHEMA = {
    "type": "object",
    "properties": {"point": {"type": "array", "items": {"type": "integer"},
                             "minItems": 2, "maxItems": 2}},
    "required": ["point"], "additionalProperties": False,
}

_MARGIN_FRAC = 0.25  # expand the coarse box by 25% per side before cropping


def run(img_bgr: np.ndarray, req: dict, provider: str, logger=None) -> Result:
    h, w = img_bgr.shape[:2]
    coarse = s1_run(img_bgr, req, provider, logger=logger)
    res = Result(strategy="s2", provider=provider)
    res.visible_seats = coarse.visible_seats
    res.chosen_label = coarse.chosen_label
    res.n_calls = coarse.n_calls
    res.elapsed_s = coarse.elapsed_s
    res.raw = {"coarse": coarse.raw}
    if coarse.error:
        res.error = f"coarse: {coarse.error}"
        return res
    if coarse.box_xyxy is None:        # chose "none" or no decodable box
        return res

    x1, y1, x2, y2 = coarse.box_xyxy
    mx = int((x2 - x1) * _MARGIN_FRAC)
    my = int((y2 - y1) * _MARGIN_FRAC)
    cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
    cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        res.error = "empty crop"
        res.box_xyxy = list(coarse.box_xyxy)
        return res
    ch, cw = crop.shape[:2]

    messages = [
        {"role": "system", "content": _CROP_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": encode_to_data_url(crop)}},
            {"type": "text", "text": "Place the point on the seat cushion."},
        ]},
    ]
    try:
        parsed, elapsed2 = call_vlm(provider, messages, schema=_CROP_SCHEMA,
                                    schema_name="seat_crop_point", logger=logger)
    except Exception as exc:  # noqa: BLE001
        res.error = f"refine: {exc}"
        res.box_xyxy = list(coarse.box_xyxy)
        res.point_xy = list(((x1 + x2) // 2, (y1 + y2) // 2))  # fall back to coarse center
        return res

    res.n_calls += 1
    res.elapsed_s += elapsed2
    res.raw["refine"] = parsed
    pt_crop = decode_point_yx(parsed.get("point"), cw, ch)
    res.box_xyxy = list(coarse.box_xyxy)
    if pt_crop is None:
        res.point_xy = list(((x1 + x2) // 2, (y1 + y2) // 2))
        return res
    res.point_xy = [cx1 + pt_crop[0], cy1 + pt_crop[1]]  # map crop -> full image
    return res
