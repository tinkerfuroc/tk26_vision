"""S0 control: replicate the production pointing prompt across providers.

Uses the same system prompt as kimi_api._seat_vlm so S0 numbers are
directly comparable to what currently ships. Calls go through the
benchmark provider layer (not _seat_vlm.request_seat) so gemini and qwen
share one code path.
"""

from __future__ import annotations

import numpy as np

from ..geometry import decode_point_yx
from ..paths import ensure_kimi_api_importable
from ..providers import call_vlm
from .base import Result, build_request_text

ensure_kimi_api_importable()
from kimi_api._seat_vlm import _SYSTEM_PROMPT, _RESPONSE_SCHEMA  # noqa: E402
from kimi_api._image_utils import encode_to_data_url  # noqa: E402


def run(img_bgr: np.ndarray, req: dict, provider: str, logger=None) -> Result:
    h, w = img_bgr.shape[:2]
    res = Result(strategy="s0", provider=provider)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": encode_to_data_url(img_bgr)}},
            {"type": "text", "text": build_request_text(req)},
        ]},
    ]
    try:
        parsed, elapsed = call_vlm(provider, messages, schema=_RESPONSE_SCHEMA,
                                   schema_name="seat_pointing", logger=logger)
    except Exception as exc:  # noqa: BLE001
        res.error = str(exc)
        return res
    res.elapsed_s = elapsed
    res.n_calls = 1
    res.raw = parsed
    res.chosen_label = str(parsed.get("label", "none") or "none")
    res.visible_seats = parsed.get("visible_seats", []) or []
    pt = decode_point_yx(parsed.get("point"), w, h)
    if res.chosen_label.strip().lower() == "none":
        pt = None
    res.point_xy = list(pt) if pt else None
    return res
