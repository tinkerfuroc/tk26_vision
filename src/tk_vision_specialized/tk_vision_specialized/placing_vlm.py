"""Gemini/Qwen structured-output client for tabletop placing-location
proposals.

Sister of `object_detection_generalist.vlm_bbox` but tuned for *empty*-region
enumeration instead of object detection: the system prompt asks the VLM to
identify clear flat regions on a desktop large enough to fit a user-described
item, ranked best-to-worst. Returns a parallel list of pixel-space bounding
boxes plus the rank labels the model emitted (typically "rank1", "rank2", ...).

Reuses `kimi_api._env` for OpenRouter/DashScope key + base URL discovery so a
missing `OPENROUTER_API_KEY`/`DASHSCOPE_API_KEY` raises a single
`VlmPlacingError` the service callback can convert to status=-1 without
crashing node startup. `request_placing_bboxes_chain` tries providers in
order (Gemini primary, Qwen fallback by default at the node layer) so the
service degrades to a second provider instead of a bare failure.
"""

from __future__ import annotations

import json
import time
from typing import List, Sequence, Tuple

import numpy as np

from kimi_api._env import (
    base_url,
    dashscope_base_url,
    load_env,
    require_api_key,
    require_dashscope_api_key,
)
from kimi_api._image_utils import encode_to_data_url


Bbox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords


def _system_prompt(max_candidates: int) -> str:
    return (
        "You are helping a service robot place an item on a desktop or "
        "tabletop visible in the image. Identify clear, flat, unoccupied "
        "regions on the desktop large enough to fit the item described by "
        "the user. Rank them best (#1) to worst. Best = (a) clearly empty, "
        "(b) flat, (c) safely away from edges and existing objects, "
        "(d) large enough for the item with margin. "
        f"Return up to {max_candidates} regions as bounding boxes "
        "[ymin, xmin, ymax, xmax] normalized to 0-1000, where (0,0) is the "
        "top-left of the image. Each box must enclose only empty surface, "
        "not surrounding objects. The label field must be 'rank1', 'rank2', "
        "... matching the rank position. If no suitable region exists, "
        "return an empty detections list. Never return masks, depth, or 3D "
        "coordinates."
    )


def _response_schema(max_candidates: int) -> dict:
    return {
        'type': 'object',
        'properties': {
            'detections': {
                'type': 'array',
                'maxItems': max_candidates,
                'items': {
                    'type': 'object',
                    'properties': {
                        'label': {'type': 'string'},
                        'box_2d': {
                            'type': 'array',
                            'items': {'type': 'integer'},
                            'minItems': 4,
                            'maxItems': 4,
                        },
                    },
                    'required': ['label', 'box_2d'],
                    'additionalProperties': False,
                },
            },
        },
        'required': ['detections'],
        'additionalProperties': False,
    }


def _decode_bbox(box_2d, w: int, h: int) -> Bbox | None:
    """Decode a Gemini [y0, x0, y1, x1] 0-1000 normalized box to xyxy pixels."""
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


class VlmPlacingError(RuntimeError):
    """Raised on non-recoverable VLM call failures (e.g. missing API key)."""


def request_placing_bboxes(
    rgb_bgr: np.ndarray,
    *,
    item_description: str,
    max_candidates: int = 5,
    model: str,
    provider: str = 'gemini',
    max_retries: int = 1,
    timeout_s: float = 8.0,
    logger=None,
) -> tuple[List[Bbox], List[str], float]:
    """Ask Gemini (or, with ``provider='qwen'``, DashScope) for ranked empty
    placing regions on the visible desktop.

    Returns ``(boxes, ranks, elapsed_s)`` — parallel arrays ordered as the
    model returned them (already best-to-worst); ``ranks[i]`` is the raw
    label string (typically ``'rank1'``...). Empty lists are a legitimate
    "no suitable region" answer. Raises `VlmPlacingError` on missing
    credentials, an unknown provider, or retry exhaustion — so "the VLM
    never answered" is never conflated with "the desk is full".
    """
    load_env()
    if provider == 'qwen':
        try:
            api_key = require_dashscope_api_key()
        except RuntimeError as exc:
            raise VlmPlacingError(str(exc)) from exc
        b_url = dashscope_base_url()
    elif provider == 'gemini':
        try:
            api_key = require_api_key()
        except RuntimeError as exc:
            raise VlmPlacingError(str(exc)) from exc
        b_url = base_url()
    else:
        raise VlmPlacingError(f"unknown provider {provider!r} (expected qwen|gemini)")

    from openai import OpenAI

    max_candidates = max(1, min(int(max_candidates), 10))
    client = OpenAI(api_key=api_key, base_url=b_url)

    _t0 = time.perf_counter()
    try:
        data_url = encode_to_data_url(rgb_bgr)
        h, w = rgb_bgr.shape[:2]

        messages = [
            {'role': 'system', 'content': _system_prompt(max_candidates)},
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': data_url}},
                    {
                        'type': 'text',
                        'text': f'Item to place: {item_description}',
                    },
                ],
            },
        ]

        response_format_strict = {
            'type': 'json_schema',
            'json_schema': {
                'name': 'placing_regions',
                'strict': True,
                'schema': _response_schema(max_candidates),
            },
        }
        response_format_loose = {'type': 'json_object'}
        use_strict = True

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            response_format = (
                response_format_strict if use_strict
                else response_format_loose
            )
            try:
                completion = client.with_options(
                    timeout=timeout_s,
                ).chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                )
                raw = completion.choices[0].message.content or ''
                parsed = json.loads(raw)
                detections = parsed.get('detections', []) or []

                boxes: List[Bbox] = []
                ranks: List[str] = []
                for det in detections:
                    if not isinstance(det, dict):
                        continue
                    decoded = _decode_bbox(det.get('box_2d'), w, h)
                    if decoded is None:
                        continue
                    boxes.append(decoded)
                    rank = det.get('label')
                    ranks.append(rank if isinstance(rank, str) else '')

                if logger is not None:
                    logger.info(
                        f'VLM placing returned {len(detections)} region(s); '
                        f'{len(boxes)} valid after decode '
                        f'(attempt {attempt}/{max_retries}).'
                    )
                if detections and not boxes:
                    # The model asserted regions exist but every box failed to
                    # decode (wrong coordinate convention, malformed payload)
                    # — unusable output, not a legitimate "no suitable
                    # region"; retry so the fallback provider can fire.
                    raise ValueError(
                        f'all {len(detections)} detection(s) failed bbox decode')
                return boxes, ranks, time.perf_counter() - _t0
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                last_error = exc
                if logger is not None:
                    logger.warning(
                        f'VLM placing JSON parse failed '
                        f'(attempt {attempt}/{max_retries}): {exc}'
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
                            'VLM placing route rejected json_schema '
                            f'response_format ({exc}); falling back to '
                            'json_object for the rest of this call.'
                        )
                    use_strict = False
                last_error = exc
                if logger is not None:
                    logger.warning(
                        f'VLM placing call failed '
                        f'(attempt {attempt}/{max_retries}): {exc}'
                    )

        raise VlmPlacingError(
            f'[{provider}] exhausted {max_retries} retries; last={last_error}')
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def request_placing_bboxes_chain(
    rgb_bgr: np.ndarray,
    *,
    item_description: str,
    max_candidates: int = 5,
    provider_models: Sequence[tuple],
    max_retries: int = 1,
    timeout_s: float = 8.0,
    logger=None,
) -> tuple[List[Bbox], List[str], float, str]:
    """Try (provider, model) pairs in order for ranked placing regions.

    Returns ``(boxes, ranks, elapsed_s, provider_used)``. A
    ``VlmPlacingError`` (missing key or retry exhaustion) falls through to
    the next provider. A legitimate result -- including no suitable region
    found (empty lists) -- is returned immediately without trying further
    providers. Raises ``VlmPlacingError`` if every provider fails (or none
    are configured), so the caller can report "VLM unavailable" rather
    than misreading total failure as an empty desktop.
    """
    errors = []
    for provider, model in provider_models:
        try:
            boxes, ranks, elapsed_s = request_placing_bboxes(
                rgb_bgr, item_description=item_description,
                max_candidates=max_candidates,
                provider=provider, model=model, max_retries=max_retries,
                timeout_s=timeout_s, logger=logger,
            )
        except VlmPlacingError as exc:
            errors.append(f'{provider}: {exc}')
            if logger is not None:
                logger.warning(
                    f'placing VLM provider {provider} failed: {exc}; trying next.')
            continue
        return boxes, ranks, elapsed_s, provider
    raise VlmPlacingError(
        'all providers failed: ' + (' | '.join(errors) or 'no providers configured'))
