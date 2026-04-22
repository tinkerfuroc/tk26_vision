"""Gemini 2.5 Pro open-vocabulary bounding-box client.

Takes an RGB (BGR numpy) frame plus a natural-language target class and
returns pixel-space xyxy bounding boxes for every matching instance.
Used only on the VLM+SAM fallback path of the generalist detection service.

Reuses kimi_api._env (OpenRouter key + base URL discovery) and the base64
data-URL encoding pattern from kimi_api.feature_matching. The OpenAI client
is constructed lazily inside `request_bboxes` so the owning ROS node can
start cleanly even when `OPENROUTER_API_KEY` is unset (T1 invariant).
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from typing import List, Tuple

import cv2
import numpy as np

from kimi_api._env import base_url, load_env, require_api_key


Bbox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords


_SYSTEM_PROMPT = (
    "You detect instances of the user's target class in the image. "
    "Respond ONLY as strict JSON of the form "
    "{\"detections\": [{\"label\": \"<class>\", "
    "\"box_2d\": [y0, x0, y1, x1]}, ...]}. "
    "Coordinates are normalized integers in [0, 1000] where (0,0) is "
    "the top-left of the image and (1000,1000) is the bottom-right. "
    "Return an empty detections list if no matches. "
    "No markdown, no prose, no explanation."
)


def _encode_data_url(rgb_bgr: np.ndarray) -> str:
    """Encode a BGR image as a JPEG data URL."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, rgb_bgr)
        with open(tmp_path, 'rb') as f:
            data = f.read()
    finally:
        os.unlink(tmp_path)
    return f'data:image/jpeg;base64,{base64.b64encode(data).decode("utf-8")}'


def _decode_bbox(box_2d, w: int, h: int) -> Bbox | None:
    """Decode a Gemini [y0, x0, y1, x1] 0-1000 normalized box to xyxy pixels."""
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        return None
    try:
        y0, x0, y1, x1 = (float(v) for v in box_2d)
    except (TypeError, ValueError):
        return None

    # Guard against models that occasionally flip the axis order.
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 < x0:
        x0, x1 = x1, x0

    px1 = int(round(x0 * w / 1000.0))
    py1 = int(round(y0 * h / 1000.0))
    px2 = int(round(x1 * w / 1000.0))
    py2 = int(round(y1 * h / 1000.0))

    px1 = max(0, min(px1, w - 1))
    px2 = max(0, min(px2, w - 1))
    py1 = max(0, min(py1, h - 1))
    py2 = max(0, min(py2, h - 1))

    if px2 <= px1 or py2 <= py1:
        return None
    return (px1, py1, px2, py2)


class VlmBboxError(RuntimeError):
    """Raised on non-recoverable VLM call failures (e.g. missing API key).

    The service callback converts this to `status=1` with a human message so
    lazy-init stays compatible with node startup when no key is available.
    """


def request_bboxes(
    rgb_bgr: np.ndarray,
    prompt: str,
    *,
    model: str,
    max_retries: int = 3,
    timeout_s: float = 20.0,
    logger=None,
) -> List[Bbox]:
    """Ask Gemini for every xyxy bounding box matching `prompt` in the image.

    Returns an empty list on parse exhaustion or an explicit "no matches"
    response. Raises `VlmBboxError` only for configuration problems we want
    the caller to surface (missing API key).
    """

    load_env()
    try:
        api_key = require_api_key()
    except RuntimeError as exc:
        raise VlmBboxError(str(exc)) from exc

    # Imported lazily so the node can start without openai installed on the
    # default path (openai is in kimi_api's requirements, and the generalist
    # package depends on kimi_api at runtime, so this should succeed).
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url())
    data_url = _encode_data_url(rgb_bgr)
    h, w = rgb_bgr.shape[:2]

    messages = [
        {'role': 'system', 'content': _SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': data_url}},
                {'type': 'text', 'text': f'Target class: {prompt}'},
            ],
        },
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.with_options(timeout=timeout_s).chat.completions.create(
                model=model,
                messages=messages,
                response_format={'type': 'json_object'},
            )
            raw = completion.choices[0].message.content or ''
            parsed = json.loads(raw)
            detections = parsed.get('detections', []) or []

            boxes: List[Bbox] = []
            for det in detections:
                box_2d = det.get('box_2d') if isinstance(det, dict) else None
                decoded = _decode_bbox(box_2d, w, h)
                if decoded is not None:
                    boxes.append(decoded)

            if logger is not None:
                logger.info(
                    f'VLM returned {len(detections)} raw detection(s), '
                    f'{len(boxes)} valid box(es) after decode (attempt '
                    f'{attempt}/{max_retries}).'
                )
            return boxes
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            last_error = exc
            if logger is not None:
                logger.warning(
                    f'VLM JSON parse failed (attempt {attempt}/{max_retries}): {exc}'
                )
        except Exception as exc:  # noqa: BLE001 — network/API layer is broad
            last_error = exc
            if logger is not None:
                logger.warning(
                    f'VLM call failed (attempt {attempt}/{max_retries}): {exc}'
                )

    if logger is not None:
        logger.error(
            f'VLM bbox request exhausted {max_retries} retries; '
            f'last error: {last_error}'
        )
    return []
