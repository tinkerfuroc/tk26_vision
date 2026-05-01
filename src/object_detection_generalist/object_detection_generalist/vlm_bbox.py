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

import json
import time
from typing import List, Tuple

import numpy as np

from kimi_api._env import base_url, load_env, require_api_key
from kimi_api._image_utils import encode_to_data_url


Bbox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords


_SYSTEM_PROMPT = (
    # Canonical Gemini spatial-understanding phrasing — Google's training
    # docs warn that even small wording changes (e.g. flipping the axis
    # order to xmin,ymin,xmax,ymax) measurably degrade accuracy on the
    # 2.x family. Keep this string close to the official cookbook.
    "Return bounding boxes as an array of objects with labels. "
    "Never return masks. Limit to 25 objects. "
    "If an object is present multiple times, give each object a unique "
    "label according to its distinct characteristics "
    "(colors, size, position, etc.). "
    "The box_2d MUST be exactly four integers [ymin, xmin, ymax, xmax] "
    "normalized to 0-1000, where (0,0) is the top-left of the image. "
    "Do NOT return 3D coordinates, depth, rotation, or any extra values. "
    "If no instances match the target class, return an empty detections list."
)


# JSON Schema for the structured-output response_format. Critical: by
# pinning `box_2d` to exactly 4 integer items, the model is structurally
# prevented from emitting 9-value 3D bbox payloads (a known Gemini 2.5
# failure mode where the prompt mentions a 3D-like object — see
# generalist_node smoke-test notes). OpenRouter forwards json_schema to
# Gemini's native structured output mode.
_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'detections': {
            'type': 'array',
            'maxItems': 25,
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
    """Decode a Gemini [y0, x0, y1, x1] 0-1000 normalized box to xyxy pixels.

    Tolerant of >4-element payloads: when Gemini slips into 3D-bbox mode
    despite the schema (e.g. `[y0, x0, y1, x1, depth, dx, dy, dz, yaw]`),
    take the first 4 values which are still y0/x0/y1/x1 in the 2.x family.
    Reject anything shorter than 4.
    """
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) < 4:
        return None
    try:
        y0, x0, y1, x1 = (float(box_2d[i]) for i in range(4))
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
    abandon_event=None,
    client_holder: dict | None = None,
) -> tuple[List[Bbox], List[str], float]:
    """Ask Gemini for every xyxy bounding box matching `prompt` in the image.

    Returns ``(boxes, labels, elapsed_s)`` where ``labels[i]`` is Gemini's
    free-form distinguishing label for ``boxes[i]`` (parallel arrays; empty
    string if Gemini omitted the field for that detection). ``elapsed_s`` is
    wall-clock seconds for the entire VLM call (including retries). Returns
    empty lists on parse exhaustion or an explicit "no matches" response.
    Raises `VlmBboxError` only for configuration problems we want the caller
    to surface (missing API key).

    Cancellation
    ------------
    If ``abandon_event`` (a ``threading.Event``) is set, the loop returns
    early on the next retry boundary OR when an exception is raised by the
    in-flight HTTP call (typical case: the caller closed ``client_holder['client']``
    from another thread, which interrupts httpx and surfaces a ReadError /
    ConnectError here). This lets the race coordinator abandon a VLM call
    without waiting for it to time out naturally.

    ``client_holder`` is an optional dict (passed by the caller) into which
    this function publishes its OpenAI client so the caller can close it
    cross-thread to force-cancel the HTTP. The function still owns the
    client's lifetime (closes it in `finally`); the caller's close() just
    accelerates termination.
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
    if client_holder is not None:
        client_holder['client'] = client

    _t0 = time.perf_counter()
    try:
        data_url = encode_to_data_url(rgb_bgr)
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

        # Prefer JSON-Schema structured output (enforces box_2d shape at
        # the model level). Fall back to plain json_object on the first
        # call that errors with response_format mismatch — some
        # OpenRouter routes / older Gemini variants don't accept the
        # json_schema form.
        response_format_strict = {
            'type': 'json_schema',
            'json_schema': {
                'name': 'detections_response',
                'strict': True,
                'schema': _RESPONSE_SCHEMA,
            },
        }
        response_format_loose = {'type': 'json_object'}
        use_strict = True

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            if abandon_event is not None and abandon_event.is_set():
                if logger is not None:
                    logger.info(
                        f'VLM call abandoned before attempt '
                        f'{attempt}/{max_retries}'
                    )
                return [], [], time.perf_counter() - _t0
            response_format = (
                response_format_strict if use_strict
                else response_format_loose
            )
            try:
                completion = client.with_options(timeout=timeout_s).chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                )
                raw = completion.choices[0].message.content or ''
                parsed = json.loads(raw)
                detections = parsed.get('detections', []) or []

                boxes: List[Bbox] = []
                labels: List[str] = []
                for det in detections:
                    if not isinstance(det, dict):
                        continue
                    decoded = _decode_bbox(det.get('box_2d'), w, h)
                    if decoded is None:
                        continue
                    boxes.append(decoded)
                    raw_label = det.get('label')
                    labels.append(raw_label if isinstance(raw_label, str) else '')

                if logger is not None:
                    logger.info(
                        f'VLM returned {len(detections)} raw detection(s), '
                        f'{len(boxes)} valid box(es) after decode (attempt '
                        f'{attempt}/{max_retries}).'
                    )
                return boxes, labels, time.perf_counter() - _t0
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                last_error = exc
                if logger is not None:
                    logger.warning(
                        f'VLM JSON parse failed (attempt {attempt}/{max_retries}): {exc}'
                    )
            except Exception as exc:  # noqa: BLE001 — network/API layer is broad
                # If the caller closed our client to abandon the call, httpx
                # raises an error here. Detect that and exit cleanly instead
                # of treating it as a retryable failure.
                if abandon_event is not None and abandon_event.is_set():
                    if logger is not None:
                        logger.info(
                            f'VLM call abandoned mid-flight '
                            f'({type(exc).__name__}: {exc}); exiting'
                        )
                    return [], [], time.perf_counter() - _t0
                # If json_schema isn't supported by this route (typical
                # signature: HTTP 400 with response_format / schema in the
                # message), drop to plain json_object for remaining
                # attempts. Heuristic match on the exception text.
                exc_text = str(exc).lower()
                if use_strict and (
                    'json_schema' in exc_text
                    or 'response_format' in exc_text
                    or 'schema' in exc_text
                ):
                    if logger is not None:
                        logger.warning(
                            'VLM route rejected json_schema response_format '
                            f'({exc}); falling back to json_object for the '
                            'rest of this call.'
                        )
                    use_strict = False
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
        return [], [], time.perf_counter() - _t0
    finally:
        # Always close the client we own. Idempotent — caller may have
        # already closed it as part of the abandon path.
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
