"""Gemini 2.5 seat-pointing client.

Single structured-output call that returns a short seat label and a
pointing pixel [y, x] normalized 0-1000. The caller samples depth at
that pixel to get the 3D sit-on point.

Why pointing instead of a bounding box: Google's Gemini spatial-
understanding cookbook documents that the pointing mode is trained for
affordances ("where to sit/grasp"), whereas bbox mode targets the
object outline — so the model picks the sittable surface directly,
without having to decide where the cushion edge is. That eliminates
the "bbox spans the gap between two cushions" and "bbox biased to the
thin visible top edge" failure modes. The cookbook also warns that
extra reasoning/CoT fields next to the coordinates hurt accuracy, so
we ship a tiny `label` (per Google's "unique characteristic" rule) and
nothing else.

Structurally modelled on `object_detection_generalist.vlm_bbox.request_bboxes`
(retry loop, strict -> loose response_format fallback, lazy OpenAI
client, client.close in finally). Kept dependency-leaf: no import from
`object_detection_generalist`.
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


Point = Tuple[int, int]  # (x, y) in pixel coords


_SYSTEM_PROMPT = (
    'You are helping a robot place a new guest on an empty seat. Fill '
    'the JSON fields in this exact order: `visible_seats`, then '
    '`label`, then `point`.\n'
    '\n'
    '- The `visible_seats` field (emitted FIRST): an array covering '
    'EVERY sittable spot visible in the image. For each spot, include '
    '  {"label": "<unique short identifier>", "occupied": true|false, '
    '"reason": "<why occupied / why empty>"}.\n'
    '  Count the cushions accurately — if a sofa has two visible '
    'cushions, list two entries, not three.\n'
    '  Occupancy rules — read carefully:\n'
    '    (i) A spot is OCCUPIED only if a person sits on it, OR a '
    'substantial object rests DIRECTLY on the cushion surface '
    '(laptop, bag, folded clothes, cushion-pillow etc.).\n'
    '    (ii) Objects on a nearby COFFEE TABLE, SIDE TABLE, FLOOR, '
    'or other furniture do NOT occupy the seat. Cables / wires '
    'draped loosely DO NOT occupy the seat. A water bottle on a '
    'coffee table in front of the sofa does NOT occupy the sofa. Be '
    'strict about physical contact with the cushion.\n'
    '    (iii) If in doubt about whether an item is on the cushion '
    'vs on a table in front of it, mark the seat as empty.\n'
    '  If NO sittable furniture is visible at all, return an empty '
    'array [].\n'
    '\n'
    '- The `label` field: copy the label of one entry from '
    '`visible_seats` that has occupied=false. This is the seat you '
    'recommend. If every entry in `visible_seats` has occupied=true, '
    'or the array is empty, set label to exactly "none".\n'
    '\n'
    '- The `point` field: exactly two integers [y, x] normalized to '
    '0-1000, pointing at the sitting surface of the seat named by '
    '`label` — the exact spot a guest would land on when they sit '
    'down. Axis convention: y=0 is the TOP of the image, y=1000 is '
    'the BOTTOM; x=0 is the LEFT, x=1000 is the RIGHT.\n'
    '  Hard constraints on where the point may land:\n'
    '    (a) It MUST be on visible seating furniture (cushion / seat '
    'pan). Never on a wall, poster, floor, air, backrest, armrest, '
    'another occupant, or the gap between two cushions.\n'
    '    (b) The sitting surface you target must belong to the seat '
    'named by `label`. If that seat is partially occluded, point at '
    'a visibly-empty portion of its cushion.\n'
    '    (c) For any seat that has a visible backrest, the pointing '
    'pixel MUST lie BELOW the bottom edge of that backrest — i.e., on '
    'the horizontal seat cushion, not on the vertical backrest. The '
    'cushion typically sits in the lower half of a chair\'s visible '
    'bounding extent.\n'
    '    (d) Before committing, mentally check: is the pixel at [y, '
    'x] the same colour/material as the cushion? If it matches a '
    'wall / poster / different colour behind the sofa, SHIFT y '
    'downward (larger y value) until the pixel is on the cushion.\n'
    '  If `label` is "none", set point to [0, 0].\n'
    '\n'
    'Do not hallucinate a seat that is not visibly present. It is '
    'correct and expected to return label="none" when every seat in '
    'view is occupied — the robot will handle that case.'
)


_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        # Declared first so Gemini emits an enumeration of all visible
        # seats + occupancy status BEFORE committing to a point. Not a
        # free-form CoT — structured enumeration (Set-of-Mark style) is
        # what actually helps spatial disambiguation per the research.
        'visible_seats': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'label': {'type': 'string'},
                    'occupied': {'type': 'boolean'},
                    'reason': {'type': 'string'},
                },
                'required': ['label', 'occupied', 'reason'],
                'additionalProperties': False,
            },
        },
        'label': {'type': 'string'},
        'point': {
            'type': 'array',
            'items': {'type': 'integer'},
            'minItems': 2,
            'maxItems': 2,
        },
    },
    'required': ['visible_seats', 'label', 'point'],
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


def _decode_point(point_yx, w: int, h: int) -> Point | None:
    """Decode Gemini's [y, x] 0-1000 normalized point to (x, y) pixels.

    Returns None if the payload is malformed or is the explicit
    "no seat" sentinel [0, 0]. Tolerant of >2-element payloads in case
    the model slips into bbox / 3D mode (take first 2).
    """
    if not isinstance(point_yx, (list, tuple)) or len(point_yx) < 2:
        return None
    try:
        y0, x0 = float(point_yx[0]), float(point_yx[1])
    except (TypeError, ValueError):
        return None
    if y0 == 0.0 and x0 == 0.0:
        return None

    px = int(round(x0 * w / 1000.0))
    py = int(round(y0 * h / 1000.0))
    px = max(0, min(px, w - 1))
    py = max(0, min(py, h - 1))
    return (px, py)


class VlmSeatError(RuntimeError):
    """Raised on non-recoverable VLM config failures (e.g. missing API key)."""


def _build_text_prompt(names: Sequence[str], features: Sequence[str]) -> str:
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
) -> tuple[str, Point | None, list, float]:
    """Ask Gemini for a single pointing pixel + short label.

    Returns ``(label, point_xy_or_None, visible_seats, elapsed_s)``.
    ``point_xy`` is ``None`` if the model reported no empty seat (label
    "none" or explicit [0, 0]) or decoding failed on every retry.
    ``visible_seats`` is the model's enumeration of all seats visible
    in the image with occupancy status (useful for logging / debugging
    the pointing decision).

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
                'name': 'seat_pointing',
                'strict': True,
                'schema': _RESPONSE_SCHEMA,
            },
        }
        response_format_loose = {'type': 'json_object'}
        use_strict = True

        last_error: Exception | None = None
        last_label = ''
        for attempt in range(1, max_retries + 1):
            response_format = (
                response_format_strict if use_strict else response_format_loose
            )
            try:
                completion = client.with_options(timeout=timeout_s).chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    # Google cookbook recommends ~0.5 for spatial output;
                    # fully-deterministic 0 can cause looping.
                    temperature=0.5,
                )
                raw = completion.choices[0].message.content or ''
                parsed = json.loads(raw)
                label = str(parsed.get('label', '') or '')
                point_yx = parsed.get('point')
                point_xy = _decode_point(point_yx, w, h)
                visible_seats = parsed.get('visible_seats') or []
                if not isinstance(visible_seats, list):
                    visible_seats = []
                # "none" label is a redundant no-seat signal the prompt asks for.
                if label.strip().lower() == 'none':
                    point_xy = None

                if logger is not None:
                    logger.info(
                        f'VLM seat call returned point={point_xy}, '
                        f'label={label!r}, '
                        f'visible_seats={len(visible_seats)} entries '
                        f'(attempt {attempt}/{max_retries}).'
                    )
                return label, point_xy, visible_seats, time.perf_counter() - t0
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                last_error = exc
                last_label = ''
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
        return last_label, None, [], time.perf_counter() - t0
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
