"""Qwen3-VL grounding client for the object_match service.

Sister of ``object_detection_generalist.vlm_bbox`` but tuned for *visual*
grounding rather than text grounding: the user sends two images (a SCENE
captured by the robot's camera and a REFERENCE photo of one target item)
and asks Qwen3-VL to locate the reference item in the scene.

Reasons for picking Qwen3-VL specifically (over Qwen2.5-VL or qwen-vl-max):

* Output coordinates are normalised to ``0..1000`` over the original scene
  image dimensions. Qwen2.5-VL emits absolute pixels of the *resized*
  image, which forces the caller to replicate the resize bookkeeping.
* Multi-image input is officially supported on the OpenAI-compatible
  endpoint — ``[image, image, text]`` parts in one user turn.

Qwen-only, no provider chain: unlike kimi_api's Gemini-primary/Qwen-fallback
nodes, this client is called unconditionally by ``object_match_server.py``.
The ``qwen_api_backend`` argument selects DashScope (default) or OpenRouter
as the Qwen host; base URL, API key, and model resolution for both is
centralized in ``kimi_api._env.resolve_qwen_target`` (see that module for
the exact env-var / model-id rules per backend). The ``base_url`` argument
here, when non-empty, always overrides the backend's own default URL.
"""

from __future__ import annotations

import base64
import json
import re
import time
from typing import List, Tuple

import cv2
import numpy as np

from kimi_api._env import load_env, resolve_qwen_target


Bbox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords


class QwenMatchError(RuntimeError):
    """Raised on non-recoverable Qwen call failures (e.g. missing API key)."""


def _system_prompt(top_k: int) -> str:
    return (
        "You are a visual-grounding assistant for a service robot. The user "
        "provides two images. Image 1 is a SCENE captured by the robot's "
        "camera. Image 2 is a REFERENCE photo of one target item. Find every "
        "instance of the reference item visible in the scene and return up "
        f"to {top_k} bounding boxes ranked best-to-worst by match "
        "confidence. Coordinates 'box_2d' are [x1, y1, x2, y2] normalised "
        "to 0-1000 over the SCENE (image 1) dimensions, where (0, 0) is the "
        "top-left corner and (1000, 1000) is the bottom-right. Confidence "
        "is your subjective match score in [0.0, 1.0]. If no instance is "
        "visible in the scene, return an empty detections list. Output "
        "JSON only, with no commentary or markdown fences."
    )


def _response_schema(top_k: int) -> dict:
    return {
        'type': 'object',
        'properties': {
            'detections': {
                'type': 'array',
                'maxItems': top_k,
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
                        'confidence': {
                            'type': 'number',
                            'minimum': 0.0,
                            'maximum': 1.0,
                        },
                    },
                    'required': ['label', 'box_2d', 'confidence'],
                    'additionalProperties': False,
                },
            },
        },
        'required': ['detections'],
        'additionalProperties': False,
    }


_FENCE_RE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.MULTILINE)


def _strip_fences(text: str) -> str:
    """Drop ```json ... ``` fences that some Qwen revisions emit despite the
    explicit "no markdown" instruction."""
    if '```' not in text:
        return text
    return _FENCE_RE.sub('', text).strip()


def _encode_data_url(rgb_bgr: np.ndarray) -> str:
    """Encode a BGR image as a base64 JPEG data URL."""
    ok, buf = cv2.imencode('.jpg', rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise QwenMatchError('cv2.imencode failed for scene image')
    return (
        'data:image/jpeg;base64,'
        + base64.b64encode(buf.tobytes()).decode('utf-8')
    )


def _decode_bbox(box_2d, w: int, h: int) -> Bbox | None:
    """Decode Qwen3-VL [x1, y1, x2, y2] 0-1000 normalised box to xyxy pixels."""
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
    px1 = int(round(x1 * w / 1000.0))
    py1 = int(round(y1 * h / 1000.0))
    px2 = int(round(x2 * w / 1000.0))
    py2 = int(round(y2 * h / 1000.0))
    px1 = max(0, min(px1, w - 1))
    px2 = max(0, min(px2, w - 1))
    py1 = max(0, min(py1, h - 1))
    py2 = max(0, min(py2, h - 1))
    if px2 <= px1 or py2 <= py1:
        return None
    return (px1, py1, px2, py2)


def request_match_bboxes(
    scene_bgr: np.ndarray,
    ref_data_url: str,
    *,
    item_name: str,
    top_k: int = 3,
    model: str = '',
    base_url: str = '',
    qwen_api_backend: str = 'dashscope',
    max_retries: int = 1,
    timeout_s: float = 12.0,
    logger=None,
) -> tuple[List[Bbox], List[float], List[str], float]:
    """Ask Qwen3-VL to locate the REFERENCE item inside the SCENE.

    Returns ``(boxes, confidences, labels, elapsed_s)`` — parallel lists in
    the order Qwen returned them (already best-to-worst per the prompt
    contract). Empty lists on parse exhaustion or "no instance visible"
    responses. Raises ``QwenMatchError`` for missing credentials, an
    invalid ``qwen_api_backend``, or a model-id shape mismatch (see
    ``kimi_api._env.resolve_qwen_target``) -- these are the only failure
    modes that reach the caller as an exception; everything else degrades
    to an empty result.
    """
    load_env()
    try:
        base_url, api_key, model = resolve_qwen_target(
            qwen_api_backend, model, base_url)
    except RuntimeError as exc:
        raise QwenMatchError(str(exc)) from exc

    from openai import OpenAI

    top_k = max(1, min(int(top_k), 10))
    client = OpenAI(api_key=api_key, base_url=base_url)

    _t0 = time.perf_counter()
    try:
        scene_url = _encode_data_url(scene_bgr)
        h, w = scene_bgr.shape[:2]

        messages = [
            {'role': 'system', 'content': _system_prompt(top_k)},
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': scene_url}},
                    {'type': 'image_url', 'image_url': {'url': ref_data_url}},
                    {
                        'type': 'text',
                        'text': (
                            f'Image 1 is the scene. Image 2 is the reference '
                            f'photo of the item "{item_name}". Find every '
                            f'instance of "{item_name}" in image 1 and return '
                            f'up to {top_k} candidates ranked by confidence.'
                        ),
                    },
                ],
            },
        ]

        response_format_strict = {
            'type': 'json_schema',
            'json_schema': {
                'name': 'object_match',
                'strict': True,
                'schema': _response_schema(top_k),
            },
        }
        response_format_loose = {'type': 'json_object'}
        use_strict = True

        last_error: Exception | None = None
        attempts_used = 0
        while attempts_used < max_retries:
            attempts_used += 1
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
                parsed = json.loads(_strip_fences(raw))
                detections = parsed.get('detections', []) or []

                boxes: List[Bbox] = []
                confs: List[float] = []
                labels: List[str] = []
                for det in detections:
                    if not isinstance(det, dict):
                        continue
                    decoded = _decode_bbox(det.get('box_2d'), w, h)
                    if decoded is None:
                        continue
                    try:
                        conf = float(det.get('confidence', 0.0))
                    except (TypeError, ValueError):
                        conf = 0.0
                    conf = max(0.0, min(conf, 1.0))
                    label = det.get('label')
                    boxes.append(decoded)
                    confs.append(conf)
                    labels.append(label if isinstance(label, str) else '')

                if logger is not None:
                    logger.info(
                        f'Qwen3-VL returned {len(detections)} candidate(s); '
                        f'{len(boxes)} valid after decode '
                        f'(attempt {attempts_used}/{max_retries}).'
                    )
                return boxes, confs, labels, time.perf_counter() - _t0
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                last_error = exc
                if logger is not None:
                    logger.warning(
                        f'Qwen3-VL JSON parse failed '
                        f'(attempt {attempts_used}/{max_retries}): {exc}'
                    )
            except Exception as exc:  # noqa: BLE001
                exc_text = str(exc).lower()
                schema_rejected = use_strict and (
                    'json_schema' in exc_text
                    or 'response_format' in exc_text
                    or 'schema' in exc_text
                )
                if schema_rejected:
                    # Schema rejection is deterministic (model doesn't support
                    # strict output). Flip to json_object and retry without
                    # consuming the transient-failure budget.
                    use_strict = False
                    attempts_used -= 1
                    if logger is not None:
                        logger.warning(
                            'Qwen3-VL rejected json_schema response_format '
                            f'({exc}); retrying with json_object.'
                        )
                    continue
                last_error = exc
                if logger is not None:
                    logger.warning(
                        f'Qwen3-VL call failed '
                        f'(attempt {attempts_used}/{max_retries}): {exc}'
                    )

        if logger is not None:
            logger.error(
                f'Qwen3-VL request exhausted {max_retries} retries; '
                f'last error: {last_error}'
            )
        return [], [], [], time.perf_counter() - _t0
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
