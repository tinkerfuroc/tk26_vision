"""Waving-person VLM client for the detect_waving fallback.

Mirrors the control flow of kimi_api/_seat_bbox_vlm.py (single call -> provider
chain, strict json_schema -> json_object fallback, errors-only fallthrough) but
stays kimi_api-free: it uses the in-package _vlm_common encoder, resolves keys
straight from os.environ, and hard-codes the provider base URLs as constants —
the same decoupled convention vlm_match_client.py / qwen_match_vlm.py use.

The VLM is asked for the whole-person box of every visibly-waving person so the
boxes overlap YOLO person masks; the server turns each box into a 3D centroid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class WavingVlmError(RuntimeError):
    """Hard failure: missing key, exhausted retries, or unparseable response."""


@dataclass
class WavingVlmResult:
    """Outcome of a waving VLM call.

    boxes are whole-person xyxy pixel boxes for people the model judged waving.
    error is set only on soft failures that should trigger provider fallback; a
    clean empty result (boxes == [] with error is None) is a terminal answer.
    """

    boxes: list = field(default_factory=list)
    provider: str = ''
    elapsed_s: float = 0.0
    error: Optional[str] = None


def decode_box_xyxy(box_2d, w: int, h: int):
    """Decode a [x1,y1,x2,y2] 0-1000 box to clamped xyxy pixels, or None.

    Returns None for malformed input or a zero-area box. x scales by width, y by
    height; corners are swapped if inverted; result is clamped to [0, w-1]/[0,
    h-1] because the box drives depth sampling on the image grid.
    """
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(box_2d[i]) for i in range(4))
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    px1 = max(0, min(int(round(x1 * w / 1000.0)), w - 1))
    py1 = max(0, min(int(round(y1 * h / 1000.0)), h - 1))
    px2 = max(0, min(int(round(x2 * w / 1000.0)), w - 1))
    py2 = max(0, min(int(round(y2 * h / 1000.0)), h - 1))
    if px2 <= px1 or py2 <= py1:
        return None
    return (px1, py1, px2, py2)


def select_boxes(parsed: dict, w: int, h: int) -> WavingVlmResult:
    """Pure: turn a parsed VLM response into a WavingVlmResult.

    Keeps entries whose waving flag is true and whose box decodes to a non-empty
    pixel box. Never sets .error — malformed individual entries are skipped, and
    an all-skipped response is a clean empty result.
    """
    res = WavingVlmResult()
    persons = parsed.get('persons', []) or []
    if not isinstance(persons, list):
        return res
    for entry in persons:
        if not isinstance(entry, dict) or not entry.get('waving'):
            continue
        box = decode_box_xyxy(entry.get('box_2d'), w, h)
        if box is not None:
            res.boxes.append(box)
    return res
