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

import json
import time
from typing import Sequence, Tuple

import numpy as np

from ._env import base_url, load_env, require_api_key
from ._image_utils import encode_to_data_url


Point = Tuple[int, int]  # (x, y) in pixel coords


_SYSTEM_PROMPT = (
    'You are helping a robot place a new guest on an empty seat.\n'
    '\n'
    'Look at the image and produce a JSON object with three fields, in '
    'this order: visible_seats, label, point.\n'
    '\n'
    'visible_seats — array of every sittable spot visible in the image. '
    'One entry per cushion or per single-person seat (a 2-cushion sofa = '
    '2 entries; a 3-cushion sofa = 3 entries; a continuous-cushion '
    'loveseat = 1 entry). Each entry:\n'
    '{"label": "<short identifier with a visual anchor>", "occupied": '
    'true|false, "reason": "<one short clause>"}\n'
    '\n'
    'The label must include a visual anchor that uniquely locates this '
    'cushion — describe what is on it, next to it, or behind it. Good '
    'labels: "left cushion of the gray sofa, next to the white pillow", '
    '"armchair under the window", "middle seat between the two people '
    'on the couch". Avoid bare numbering like "seat 1" or "cushion 2".\n'
    '\n'
    'A spot is OCCUPIED when a person is sitting on it, or a large object is '
    'resting directly on the cushion fabric (backpack, laptop).\n'
    '\n'
    'A spot is EMPTY when the cushion fabric is visible and clear. '
    'Objects on a coffee table, side table, floor, or armrest do not '
    'occupy the cushion. Loose cables or wires do not occupy the '
    'cushion. When uncertain whether an item rests on the cushion or on '
    'a table in front of it, mark EMPTY.\n'
    '\n'
    'Only include seats that are actually present in the image. If no '
    'sittable furniture is visible, return [].\n'
    '\n'
    'label — copy the label of one entry from visible_seats whose '
    'occupied is false. This is your recommendation. If every entry is '
    'occupied, or visible_seats is empty, set label to "none".\n'
    '\n'
    'point — [y, x] with each value an integer in 0–1000, normalized to '
    'image dimensions. y=0 is the top of the image, y=1000 is the '
    'bottom; x=0 is the left, x=1000 is the right. The point must land '
    'on the visible cushion fabric of the seat named by label — the '
    'flat horizontal surface a person\'s seat would touch.\n'
    '\n'
    'The point must satisfy all of:\n'
    '- on cushion fabric, not on a backrest, armrest, wall, floor, or '
    'the gap between cushions\n'
    '- on the seat named by label, not a neighboring seat\n'
    '- not on a person, laptop, bag, bottle, food, blanket, or the '
    'coffee table in front of the sofa (coffee tables look horizontal '
    'too, so verify the y value lands on the seat itself, not on a '
    'table surface in front of it)\n'
    '- if the seat has a backrest, below the bottom edge of that '
    'backrest\n'
    '- the same color and material as the cushion (if the pixel matches '
    'the wall behind the sofa, increase y until you are on the '
    'cushion)\n'
    '\n'
    'If label is "none", set point to [0, 0].\n'
    '\n'
    'It is correct to return label="none" when every visible seat is '
    'occupied. The robot handles that case.'
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
    fewshots: Sequence[object] | None = None,
) -> tuple[str, Point | None, list, float]:
    """Ask Gemini for a single pointing pixel + short label.

    Returns ``(label, point_xy_or_None, visible_seats, elapsed_s)``.
    ``point_xy`` is ``None`` if the model reported no empty seat (label
    "none" or explicit [0, 0]) or decoding failed on every retry.
    ``visible_seats`` is the model's enumeration of all seats visible
    in the image with occupancy status (useful for logging / debugging
    the pointing decision).

    ``fewshots`` is an optional iterable of ``_seat_fewshot.FewshotExample``
    (kept loosely typed here to avoid an import cycle). When non-empty,
    each example is prepended as a ``user(image+generic-prompt) /
    assistant(json.dumps(answer))`` turn before the live request — the
    model mimics the form and judgement, not the content (the few-shot
    user prompt deliberately omits per-call names/features).

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
        data_url = encode_to_data_url(rgb_bgr)
        h, w = rgb_bgr.shape[:2]

        text_prompt = _build_text_prompt(names, features)

        messages: list = [{'role': 'system', 'content': _SYSTEM_PROMPT}]
        if fewshots:
            for ex in fewshots:
                ex_url = encode_to_data_url(ex.image_bgr)
                messages.append({
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': ex_url}},
                        {
                            'type': 'text',
                            'text': 'Recommend a seat for a new guest.',
                        },
                    ],
                })
                messages.append({
                    'role': 'assistant',
                    'content': json.dumps(ex.answer),
                })
        messages.append({
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': data_url}},
                {'type': 'text', 'text': text_prompt},
            ],
        })

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

        # Force thinking on for Gemini 2.5 Pro: by default Pro auto-budgets
        # thinking tokens, which OpenRouter sometimes routes through with
        # zero budget on cached prompts — silently falling back to non-
        # thinking accuracy. Setting an explicit budget via OpenRouter's
        # `reasoning` extra_body passes through to Gemini's thinkingConfig.
        # Flash 2.5 also accepts the field but doesn't benefit; we gate on
        # 'pro' to avoid request rejections from Flash-only routes.
        extra_body: dict | None = None
        if 'pro' in model.lower():
            extra_body = {'reasoning': {'enabled': True, 'max_tokens': 2048}}

        last_error: Exception | None = None
        last_label = ''
        for attempt in range(1, max_retries + 1):
            response_format = (
                response_format_strict if use_strict else response_format_loose
            )
            try:
                create_kwargs = dict(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    # 0.2 — Google cookbook caps spatial at 0.5 and notes
                    # 0 can cause looping; 0.2 keeps just enough variance
                    # to avoid loops while reducing per-call jitter on the
                    # pointing pixel.
                    temperature=0.2,
                )
                if extra_body is not None:
                    create_kwargs['extra_body'] = extra_body
                completion = client.with_options(timeout=timeout_s).chat.completions.create(
                    **create_kwargs,
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
