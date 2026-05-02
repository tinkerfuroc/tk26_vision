"""OpenRouter VLM open-vocabulary bounding-box client.

Takes an RGB (BGR numpy) frame plus a natural-language target class and
returns pixel-space xyxy bounding boxes for every matching instance. Used only
on the VLM+SAM fallback path of the generalist detection service.

Gemini remains the primary model. If it fails with API/network/timeout/parse
or malformed-box errors, the caller can provide fallback model names (Qwen by
default at the node layer). A clean empty detections response is authoritative
unless ``fallback_on_empty=True`` is explicitly requested.

Reuses kimi_api._env (OpenRouter key + base URL discovery) and the base64
data-URL encoding pattern from kimi_api.feature_matching. The OpenAI client is
constructed lazily inside `request_bboxes` so the owning ROS node can start
cleanly even when `OPENROUTER_API_KEY` is unset (T1 invariant).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Callable, List, Tuple

import numpy as np

from kimi_api._env import base_url, load_env, require_api_key
from kimi_api._image_utils import encode_to_data_url


Bbox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords


_GEMINI_SYSTEM_PROMPT = (
    # Canonical Gemini spatial-understanding phrasing. Google's training docs
    # warn that even small wording changes (e.g. flipping the axis order to
    # xmin,ymin,xmax,ymax) measurably degrade accuracy on the 2.x family.
    "Return bounding boxes as an array of objects with labels. "
    "Target objects will be provided with `.` separation. "
    "You should look for all of them. "
    "Never return masks. Limit to 25 objects. "
    "If an object is present multiple times, give each object a unique "
    "label according to its distinct characteristics "
    "(colors, size, position, etc.). "
    "The box_2d MUST be exactly four integers [ymin, xmin, ymax, xmax] "
    "normalized to 0-1000, where (0,0) is the top-left of the image. "
    "Do NOT return 3D coordinates, depth, rotation, or any extra values. "
    "If no instances match the target class, return an empty detections list."
)


_QWEN_SYSTEM_PROMPT = (
    "You are a precise visual grounding model. Return only JSON matching the "
    "requested schema. Detect every visible instance of the target classes. "
    "For each detection, output label and box_2d. The box_2d MUST be exactly "
    "four integers [x1, y1, x2, y2] in pixel coordinates in the original image "
    "size, where (0,0) is the top-left corner. Never return masks, points, "
    "3D coordinates, depth, rotation, markdown, or explanatory text. Limit to "
    "25 objects. If no instances match, return an empty detections list."
)


# JSON Schema for the structured-output response_format. Critical: by pinning
# `box_2d` to exactly 4 integer items, the model is structurally prevented from
# emitting 9-value 3D bbox payloads (a known Gemini 2.5 failure mode where the
# prompt mentions a 3D-like object). OpenRouter forwards json_schema to routes
# that support native structured output.
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


@dataclass(frozen=True)
class _ProviderProfile:
    name: str
    system_prompt: str
    bbox_format: str


_GEMINI_PROFILE = _ProviderProfile(
    name='gemini',
    system_prompt=_GEMINI_SYSTEM_PROMPT,
    bbox_format='normalized_yxyx',
)
_QWEN_PROFILE = _ProviderProfile(
    name='qwen',
    system_prompt=_QWEN_SYSTEM_PROMPT,
    bbox_format='pixel_xyxy',
)


def _provider_profile_for_model(model: str) -> _ProviderProfile:
    """Pick the prompt/decoder profile for an OpenRouter model tag."""
    model_l = (model or '').lower()
    if 'qwen' in model_l:
        return _QWEN_PROFILE
    return _GEMINI_PROFILE


def _decode_gemini_bbox(box_2d, w: int, h: int) -> Bbox | None:
    """Decode Gemini [y0, x0, y1, x1] 0-1000 normalized bbox to pixels."""
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
    return _clip_pixel_bbox(px1, py1, px2, py2, w, h)


def _decode_qwen_bbox(box_2d, w: int, h: int) -> Bbox | None:
    """Decode Qwen [x1, y1, x2, y2] pixel bbox to clipped xyxy pixels."""
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

    return _clip_pixel_bbox(
        int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), w, h,
    )


def _clip_pixel_bbox(px1: int, py1: int, px2: int, py2: int,
                     w: int, h: int) -> Bbox | None:
    px1 = max(0, min(px1, w - 1))
    px2 = max(0, min(px2, w - 1))
    py1 = max(0, min(py1, h - 1))
    py2 = max(0, min(py2, h - 1))

    if px2 <= px1 or py2 <= py1:
        return None
    return (px1, py1, px2, py2)


def _decode_bbox(box_2d, w: int, h: int) -> Bbox | None:
    """Backward-compatible Gemini decoder used by existing tests/imports."""
    return _decode_gemini_bbox(box_2d, w, h)


def _decode_bbox_for_profile(box_2d, w: int, h: int,
                             profile: _ProviderProfile) -> Bbox | None:
    if profile.bbox_format == 'pixel_xyxy':
        return _decode_qwen_bbox(box_2d, w, h)
    return _decode_gemini_bbox(box_2d, w, h)


class VlmBboxError(RuntimeError):
    """Raised on non-recoverable VLM call failures (e.g. missing API key).

    The service callback converts this to `status=1` with a human message so
    lazy-init stays compatible with node startup when no key is available.
    """


class _VlmModelFailure(RuntimeError):
    """Internal failure for one model after exhausting its retry loop."""

    def __init__(self, message: str, attempts: list[dict] | None = None):
        super().__init__(message)
        self.attempts = attempts or []


def _normalise_fallback_models(fallback_models) -> list[str]:
    if not fallback_models:
        return []
    if isinstance(fallback_models, str):
        return [m.strip() for m in fallback_models.split(',') if m.strip()]
    return [str(m).strip() for m in fallback_models if str(m).strip()]


def _build_model_chain(model: str, fallback_models) -> list[str]:
    seen = set()
    models = []
    for model_name in [model, *_normalise_fallback_models(fallback_models)]:
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        models.append(model_name)
    return models


def request_bboxes(
    rgb_bgr: np.ndarray,
    prompt: str,
    *,
    model: str,
    fallback_models=None,
    fallback_on_empty: bool = False,
    max_retries: int = 3,
    timeout_s: float = 30.0,
    per_attempt_timeout_s: float = 10.0,
    logger=None,
    abandon_event=None,
    client_holder: dict | None = None,
    stream: bool = True,
) -> tuple[List[Bbox], List[str], float, dict]:
    """Ask VLM model(s) for every xyxy bounding box matching `prompt`.

    Returns ``(boxes, labels, elapsed_s, metadata)`` where ``labels[i]`` is the
    model's free-form distinguishing label for ``boxes[i]``. ``metadata``
    includes the primary model, fallback list, selected model, per-attempt
    status, and an error string when every model failed.

    Fallback behavior
    -----------------
    The primary model runs first. Fallback models run only after
    API/network/timeout/stream/JSON/decode failure. A clean empty detections
    response stops the chain unless ``fallback_on_empty=True``.

    Cancellation
    ------------
    If ``abandon_event`` is set, the loop returns early before each model
    attempt, at every retry boundary, on streamed-chunk boundaries, or when a
    cross-thread client close raises from the HTTP layer. This preserves the
    YOLO-World race contract: an abandoned VLM leg never starts later fallback
    model calls or FastSAM work.

    Timeout discipline
    ------------------
    Two budgets:

    * ``per_attempt_timeout_s`` — hard cap for **one** retry attempt.
      Forwarded to ``client.with_options(timeout=...)`` so httpx raises
      ``ReadTimeout`` if the stream stalls past this many seconds. Catches the
      "occasional 40 s outlier" failure mode: when the first attempt's read
      hangs, we abandon and retry instead of waiting out the overall budget.
    * ``timeout_s`` — overall wall-clock budget across **all** attempts and
      fallback models. Existing in-loop hard-deadline check (per-chunk) ensures
      an in-flight stream is aborted if it overruns this cap.

    Per-attempt budget passed to httpx is
    ``min(per_attempt_timeout_s, remaining_overall_budget)``.
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

    # Keep retry ownership here, not inside the SDK. The OpenAI client retries
    # timeouts twice by default; we also enforce timeout_s as one hard
    # wall-clock budget across our retry/fallback loop below.
    client = OpenAI(api_key=api_key, base_url=base_url(), max_retries=0)
    if client_holder is not None:
        client_holder['client'] = client

    _t0 = time.perf_counter()
    hard_deadline = _t0 + float(timeout_s)
    try:
        data_url = encode_to_data_url(rgb_bgr)
        h, w = rgb_bgr.shape[:2]
        model_chain = _build_model_chain(model, fallback_models)

        def single_model_fn(model_name: str):
            return _request_bboxes_single_model(
                client=client,
                data_url=data_url,
                image_shape=(h, w),
                prompt=prompt,
                model=model_name,
                max_retries=max_retries,
                timeout_s=timeout_s,
                per_attempt_timeout_s=per_attempt_timeout_s,
                hard_deadline=hard_deadline,
                logger=logger,
                abandon_event=abandon_event,
                stream=stream,
            )

        boxes, labels, _, metadata = _run_model_chain(
            model_chain=model_chain,
            fallback_on_empty=fallback_on_empty,
            single_model_fn=single_model_fn,
            abandon_event=abandon_event,
            logger=logger,
            started_at=_t0,
            hard_deadline=hard_deadline,
            timeout_s=timeout_s,
        )
        return boxes, labels, time.perf_counter() - _t0, metadata
    finally:
        # Always close the client we own. Idempotent: caller may have already
        # closed it as part of the abandon path.
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def _run_model_chain(
    *,
    model_chain: list[str],
    fallback_on_empty: bool,
    single_model_fn: Callable[[str], tuple[List[Bbox], List[str], list[dict]]],
    abandon_event=None,
    logger=None,
    started_at: float | None = None,
    hard_deadline: float | None = None,
    timeout_s: float | None = None,
) -> tuple[List[Bbox], List[str], float, dict]:
    """Pure fallback orchestration helper, unit-testable without OpenAI."""
    _t0 = started_at if started_at is not None else time.perf_counter()
    attempts: list[dict] = []
    primary_model = model_chain[0] if model_chain else ''
    fallback_models = model_chain[1:]
    last_error: str | None = None
    last_empty_model: str | None = None

    for model_index, model_name in enumerate(model_chain):
        if hard_deadline is not None and time.perf_counter() >= hard_deadline:
            error = (
                f'VLM hard timeout after {float(timeout_s):.1f}s'
                if timeout_s is not None else
                'VLM hard timeout'
            )
            metadata = _metadata(
                primary_model, fallback_models, attempts,
                model_used=None, error=error,
                fallback_on_empty=fallback_on_empty,
            )
            if logger is not None:
                logger.error(error)
            return [], [], time.perf_counter() - _t0, metadata
        if abandon_event is not None and abandon_event.is_set():
            metadata = _metadata(
                primary_model, fallback_models, attempts,
                model_used=None, error='abandoned',
                fallback_on_empty=fallback_on_empty, abandoned=True,
            )
            return [], [], time.perf_counter() - _t0, metadata

        try:
            boxes, labels, model_attempts = single_model_fn(model_name)
            attempts.extend(model_attempts)
        except _VlmModelFailure as exc:
            attempts.extend(exc.attempts)
            last_error = str(exc)
            if hard_deadline is not None and time.perf_counter() >= hard_deadline:
                metadata = _metadata(
                    primary_model, fallback_models, attempts,
                    model_used=None, error=last_error,
                    fallback_on_empty=fallback_on_empty,
                )
                if logger is not None:
                    logger.error(last_error)
                return [], [], time.perf_counter() - _t0, metadata
            if logger is not None and model_index + 1 < len(model_chain):
                next_model = model_chain[model_index + 1]
                logger.warning(
                    f'VLM model {model_name} failed ({exc}); '
                    f'trying fallback model {next_model}'
                )
            continue

        if boxes:
            metadata = _metadata(
                primary_model, fallback_models, attempts,
                model_used=model_name, error=None,
                fallback_on_empty=fallback_on_empty,
            )
            return boxes, labels, time.perf_counter() - _t0, metadata

        last_empty_model = model_name
        if not fallback_on_empty or model_index + 1 == len(model_chain):
            metadata = _metadata(
                primary_model, fallback_models, attempts,
                model_used=model_name, error=None,
                fallback_on_empty=fallback_on_empty,
            )
            return [], [], time.perf_counter() - _t0, metadata

        if logger is not None:
            next_model = model_chain[model_index + 1]
            logger.info(
                f'VLM model {model_name} returned no detections; '
                f'trying fallback model {next_model}'
            )

    error = last_error or (
        'VLM model chain is empty'
        if not model_chain else
        f'VLM model chain exhausted after empty response from {last_empty_model}'
    )
    metadata = _metadata(
        primary_model, fallback_models, attempts,
        model_used=None, error=error,
        fallback_on_empty=fallback_on_empty,
    )
    return [], [], time.perf_counter() - _t0, metadata


def _metadata(
    primary_model: str,
    fallback_models: list[str],
    attempts: list[dict],
    *,
    model_used: str | None,
    error: str | None,
    fallback_on_empty: bool,
    abandoned: bool = False,
) -> dict:
    return {
        'primary_model': primary_model,
        'fallback_models': list(fallback_models),
        'model_used': model_used,
        'attempts': list(attempts),
        'error': error,
        'fallback_on_empty': bool(fallback_on_empty),
        'abandoned': bool(abandoned),
    }


def _request_bboxes_single_model(
    *,
    client,
    data_url: str,
    image_shape: tuple[int, int],
    prompt: str,
    model: str,
    max_retries: int,
    timeout_s: float,
    per_attempt_timeout_s: float,
    hard_deadline: float,
    logger=None,
    abandon_event=None,
    stream: bool,
) -> tuple[List[Bbox], List[str], list[dict]]:
    """Run one model with its own retry loop.

    Returns boxes/labels plus per-attempt metadata on success or clean empty.
    Raises ``_VlmModelFailure`` when every retry failed.
    """
    h, w = image_shape
    profile = _provider_profile_for_model(model)
    messages = _build_messages(data_url, prompt, profile, image_shape)
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
    attempts: list[dict] = []
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        attempt_t0 = time.perf_counter()
        remaining_s = hard_deadline - attempt_t0
        if remaining_s <= 0.0:
            message = f'VLM hard timeout after {timeout_s:.1f}s'
            attempts.append(_attempt_record(
                model, profile, attempt, 'error', attempt_t0,
                error=message,
            ))
            if logger is not None:
                logger.error(message)
            raise _VlmModelFailure(message, attempts)
        if abandon_event is not None and abandon_event.is_set():
            attempts.append(_attempt_record(
                model, profile, attempt, 'abandoned', attempt_t0,
                error='abandoned before attempt',
            ))
            return [], [], attempts

        response_format = (
            response_format_strict if use_strict else response_format_loose
        )
        try:
            raw = _request_raw_completion(
                client=client,
                model=model,
                messages=messages,
                response_format=response_format,
                timeout_s=remaining_s,
                per_attempt_timeout_s=per_attempt_timeout_s,
                hard_deadline=hard_deadline,
                hard_timeout_s=timeout_s,
                stream=stream,
                logger=logger,
                abandon_event=abandon_event,
                attempt=attempt,
                max_retries=max_retries,
                attempt_t0=attempt_t0,
            )
            boxes, labels, raw_count = _parse_detections(raw, w, h, profile)
            status = 'success' if boxes else 'empty'
            attempts.append(_attempt_record(
                model, profile, attempt, status, attempt_t0,
                raw_detections=raw_count, valid_boxes=len(boxes),
            ))
            if logger is not None:
                logger.info(
                    f'VLM {model} returned {raw_count} raw detection(s), '
                    f'{len(boxes)} valid box(es) after decode (attempt '
                    f'{attempt}/{max_retries}).'
                )
            return boxes, labels, attempts
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            last_error = exc
            attempts.append(_attempt_record(
                model, profile, attempt, 'error', attempt_t0,
                error=f'{type(exc).__name__}: {exc}',
            ))
            if logger is not None:
                logger.warning(
                    f'VLM JSON/decode failed for {model} '
                    f'(attempt {attempt}/{max_retries}): {exc}'
                )
            if time.perf_counter() >= hard_deadline:
                break
        except Exception as exc:  # noqa: BLE001 - network/API layer is broad
            if abandon_event is not None and abandon_event.is_set():
                attempts.append(_attempt_record(
                    model, profile, attempt, 'abandoned', attempt_t0,
                    error=f'{type(exc).__name__}: {exc}',
                ))
                if logger is not None:
                    logger.info(
                        f'VLM call abandoned mid-flight for {model} '
                        f'({type(exc).__name__}: {exc}); exiting'
                    )
                return [], [], attempts

            exc_text = str(exc).lower()
            if use_strict and (
                'json_schema' in exc_text
                or 'response_format' in exc_text
                or 'schema' in exc_text
            ):
                if logger is not None:
                    logger.warning(
                        f'VLM route {model} rejected json_schema '
                        f'response_format ({exc}); falling back to '
                        'json_object for the rest of this model call.'
                    )
                use_strict = False
            last_error = exc
            attempts.append(_attempt_record(
                model, profile, attempt, 'error', attempt_t0,
                error=f'{type(exc).__name__}: {exc}',
            ))
            if logger is not None:
                logger.warning(
                    f'VLM call failed for {model} '
                    f'(attempt {attempt}/{max_retries}, '
                    f'{time.perf_counter() - attempt_t0:.2f}s, '
                    f'{type(exc).__name__}): {exc}'
                )
            if time.perf_counter() >= hard_deadline:
                break

    message = (
        f'VLM hard timeout after {timeout_s:.1f}s; last error: {last_error}'
        if time.perf_counter() >= hard_deadline else
        f'VLM model {model} exhausted {max_retries} retries; '
        f'last error: {last_error}'
    )
    if logger is not None:
        logger.error(message)
    raise _VlmModelFailure(message, attempts)


def _build_messages(data_url: str, prompt: str, profile: _ProviderProfile,
                    image_shape: tuple[int, int]) -> list[dict]:
    h, w = image_shape
    user_text = (
        f'Target classes: {prompt}. '
        f'Original image size: width={w}, height={h}.'
    )
    return [
        {'role': 'system', 'content': profile.system_prompt},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': user_text},
                {'type': 'image_url', 'image_url': {'url': data_url}},
            ],
        },
    ]


def _request_raw_completion(
    *,
    client,
    model: str,
    messages: list[dict],
    response_format: dict,
    timeout_s: float,
    per_attempt_timeout_s: float,
    hard_deadline: float,
    hard_timeout_s: float,
    stream: bool,
    logger=None,
    abandon_event=None,
    attempt: int,
    max_retries: int,
    attempt_t0: float,
) -> str:
    if stream:
        raw_parts: list[str] = []
        chunk_count = 0
        remaining_s = hard_deadline - time.perf_counter()
        if remaining_s <= 0.0:
            raise TimeoutError(
                f'VLM hard timeout after {hard_timeout_s:.1f}s'
            )
        response_stream = client.with_options(
            timeout=max(0.1, min(per_attempt_timeout_s, remaining_s))
        ).chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
            stream=True,
        )
        try:
            for chunk in response_stream:
                chunk_count += 1
                if time.perf_counter() >= hard_deadline:
                    raise TimeoutError(
                        f'VLM hard timeout after {hard_timeout_s:.1f}s'
                    )
                stream_error = getattr(chunk, 'error', None)
                if stream_error:
                    raise RuntimeError(
                        f'OpenRouter stream error: {stream_error}'
                    )
                if abandon_event is not None and abandon_event.is_set():
                    if logger is not None:
                        logger.info(
                            f'VLM stream abandoned mid-stream for {model} '
                            f'(attempt {attempt}/{max_retries}, '
                            f'{chunk_count} chunk(s) received)'
                        )
                    return ''
                choices = getattr(chunk, 'choices', None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], 'delta', None)
                piece = getattr(delta, 'content', None) if delta else None
                if piece:
                    raw_parts.append(piece)
        finally:
            try:
                response_stream.close()
            except Exception:  # noqa: BLE001
                pass
        raw = ''.join(raw_parts)
        if logger is not None:
            logger.info(
                f'VLM stream {model} attempt {attempt}/{max_retries}: '
                f'{chunk_count} chunk(s), {len(raw)} char(s), '
                f'{time.perf_counter() - attempt_t0:.2f}s.'
            )
        return raw

    remaining_s = hard_deadline - time.perf_counter()
    if remaining_s <= 0.0:
        raise TimeoutError(
            f'VLM hard timeout after {hard_timeout_s:.1f}s'
        )
    completion = client.with_options(
        timeout=max(0.1, min(per_attempt_timeout_s, remaining_s))
    ).chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
    )
    return completion.choices[0].message.content or ''


def _parse_detections(raw: str, w: int, h: int,
                      profile: _ProviderProfile) -> tuple[List[Bbox], List[str], int]:
    parsed = json.loads(raw)
    detections = parsed.get('detections', []) or []
    if not isinstance(detections, list):
        raise ValueError('detections must be an array')

    boxes: List[Bbox] = []
    labels: List[str] = []
    invalid_boxes = 0
    for det in detections:
        if not isinstance(det, dict):
            invalid_boxes += 1
            continue
        decoded = _decode_bbox_for_profile(det.get('box_2d'), w, h, profile)
        if decoded is None:
            invalid_boxes += 1
            continue
        boxes.append(decoded)
        raw_label = det.get('label')
        labels.append(raw_label if isinstance(raw_label, str) else '')

    if detections and not boxes:
        raise ValueError(
            f'all {len(detections)} detection(s) had invalid box_2d values'
        )
    if invalid_boxes and not boxes:
        raise ValueError(
            f'{invalid_boxes} detection(s) had invalid box_2d values'
        )
    return boxes, labels, len(detections)


def _attempt_record(
    model: str,
    profile: _ProviderProfile,
    attempt: int,
    status: str,
    started_at: float,
    *,
    error: str | None = None,
    raw_detections: int | None = None,
    valid_boxes: int | None = None,
) -> dict:
    record = {
        'model': model,
        'provider': profile.name,
        'attempt': int(attempt),
        'status': status,
        'elapsed': time.perf_counter() - started_at,
    }
    if error:
        record['error'] = error
    if raw_detections is not None:
        record['raw_detections'] = int(raw_detections)
    if valid_boxes is not None:
        record['valid_boxes'] = int(valid_boxes)
    return record
