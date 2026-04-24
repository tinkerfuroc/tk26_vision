"""Gemini 2.5 Flash seat-recommendation client.

Single structured-output call that returns both a recommendation sentence
and a bounding box for the recommended empty seat. Structurally modelled
on `object_detection_generalist.vlm_bbox.request_bboxes` (retry loop,
strict -> loose response_format fallback, lazy OpenAI client, close in
finally), but task-specific: one bbox + one sentence, not N labelled
bboxes.

Does not import from `object_detection_generalist` to keep `kimi_api` a
dependency leaf. `_decode_bbox` is inlined verbatim from that module.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from typing import Sequence, Tuple

import cv2
import numpy as np

from ._env import base_url, load_env, require_api_key


Bbox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords


_SYSTEM_PROMPT = (
    'You will recommend a seat for a new guest based on the image and '
    'optional descriptions of existing occupants, and identify its '
    'bounding box.\n'
    '- The `recommendation` field: begin with "Please sit at ..." and '
    'describe what furniture they should sit on and its position '
    'relative to people in the picture (to the right or left hand of '
    'named occupants). 1-2 sentences, no explanations.\n'
    '- The `box_2d` field: exactly four integers [ymin, xmin, ymax, xmax] '
    'normalized to 0-1000, tightly enclosing the empty seat referenced '
    'in the recommendation. (0,0) is the top-left of the image. Do NOT '
    'return 3D coordinates, depth, rotation, or any extra values.\n'
    'If no empty seat is visible, set recommendation to a brief '
    'explanation and box_2d to [0, 0, 0, 0].'
)


_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'recommendation': {'type': 'string'},
        'box_2d': {
            'type': 'array',
            'items': {'type': 'integer'},
            'minItems': 4,
            'maxItems': 4,
        },
    },
    'required': ['recommendation', 'box_2d'],
    'additionalProperties': False,
}


def _encode_data_url(rgb_bgr: np.ndarray) -> str:
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
    """Decode a Gemini [y0, x0, y1, x1] 0-1000 normalized box to xyxy pixels.

    Copied verbatim from `object_detection_generalist.vlm_bbox._decode_bbox`
    to keep `kimi_api` a dependency leaf. Tolerant of >4-element payloads
    (takes first 4 when the model slips into 3D-bbox mode) and of flipped
    axis order.
    """
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) < 4:
        return None
    try:
        y0, x0, y1, x1 = (float(box_2d[i]) for i in range(4))
    except (TypeError, ValueError):
        return None

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


class VlmSeatError(RuntimeError):
    """Raised on non-recoverable VLM config failures (e.g. missing API key)."""


def _build_text_prompt(names: Sequence[str], features: Sequence[str]) -> str:
    # Same builder shape as feature_recognition.py:189-191.
    text = 'Recommend a seat for a new guest.'
    for name, feature in zip(names, features):
        text += f' The person matching description: {feature} is called {name}.'
    return text


def request_seat(
    rgb_bgr: np.ndarray,
    names: Sequence[str],
    features: Sequence[str],
    *,
    model: str,
    max_retries: int = 3,
    timeout_s: float = 20.0,
    logger=None,
) -> tuple[str, Bbox | None, float]:
    """Ask Gemini for a recommendation sentence + single bbox.

    Returns ``(recommendation_text, bbox_xyxy_or_None, elapsed_s)``.
    ``bbox_xyxy`` is ``None`` if the model reported no empty seat
    (explicit [0,0,0,0]) or if decoding failed on every retry. The
    caller decides whether to surface the sentence anyway.

    Raises ``VlmSeatError`` only on configuration problems the caller
    should propagate as `status=1` (missing API key).
    """

    load_env()
    try:
        api_key = require_api_key()
    except RuntimeError as exc:
        raise VlmSeatError(str(exc)) from exc

    from openai import OpenAI  # lazy, see vlm_bbox.py

    client = OpenAI(api_key=api_key, base_url=base_url())

    t0 = time.perf_counter()
    try:
        data_url = _encode_data_url(rgb_bgr)
        h, w = rgb_bgr.shape[:2]

        text_prompt = _build_text_prompt(names, features)

        messages = [
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': data_url}},
                    {'type': 'text', 'text': text_prompt},
                ],
            },
        ]

        response_format_strict = {
            'type': 'json_schema',
            'json_schema': {
                'name': 'seat_recommendation',
                'strict': True,
                'schema': _RESPONSE_SCHEMA,
            },
        }
        response_format_loose = {'type': 'json_object'}
        use_strict = True

        last_error: Exception | None = None
        last_text = ''
        for attempt in range(1, max_retries + 1):
            response_format = (
                response_format_strict if use_strict else response_format_loose
            )
            try:
                completion = client.with_options(timeout=timeout_s).chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                )
                raw = completion.choices[0].message.content or ''
                parsed = json.loads(raw)
                rec_text = str(parsed.get('recommendation', '') or '')
                box_2d = parsed.get('box_2d')
                bbox = _decode_bbox(box_2d, w, h)

                if logger is not None:
                    logger.info(
                        f'VLM seat call returned bbox={bbox}, '
                        f'recommendation_len={len(rec_text)} '
                        f'(attempt {attempt}/{max_retries}).'
                    )
                return rec_text, bbox, time.perf_counter() - t0
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                last_error = exc
                last_text = ''
                if logger is not None:
                    logger.warning(
                        f'VLM seat JSON parse failed (attempt {attempt}/{max_retries}): {exc}'
                    )
            except Exception as exc:  # noqa: BLE001
                exc_text = str(exc).lower()
                if use_strict and (
                    'json_schema' in exc_text
                    or 'response_format' in exc_text
                    or 'schema' in exc_text
                ):
                    if logger is not None:
                        logger.warning(
                            'VLM route rejected json_schema response_format '
                            f'({exc}); falling back to json_object.'
                        )
                    use_strict = False
                last_error = exc
                if logger is not None:
                    logger.warning(
                        f'VLM seat call failed (attempt {attempt}/{max_retries}): {exc}'
                    )

        if logger is not None:
            logger.error(
                f'VLM seat request exhausted {max_retries} retries; '
                f'last error: {last_error}'
            )
        return last_text, None, time.perf_counter() - t0
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
