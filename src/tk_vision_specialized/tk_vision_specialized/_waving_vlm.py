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

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import openai
from dotenv import load_dotenv

from ._vlm_common import encode_data_url

# Populate os.environ from .env files up the CWD tree at import time.
load_dotenv(override=False)

_QWEN_KEY_NAMES = ('DASHCOPE_API_KEY', 'DASHSCOPE_API_KEY')

_QWEN_DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
_GEMINI_DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'
_FENCE_RE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.MULTILINE)

_SYSTEM_PROMPT = (
    'You watch a scene for a service robot and find people who are WAVING. '
    'A person is waving if a hand or arm is raised at or above shoulder/head '
    'height to get attention (an open raised hand counts). Do NOT count arms '
    'resting down, crossed, or held at the waist. Return JSON {"persons": '
    '[{"box_2d": [x1,y1,x2,y2], "waving": true}, ...]}. box_2d is the tight '
    'box around the WHOLE PERSON (head to feet if visible), normalized 0-1000 '
    'over the image where (0,0) is top-left and (1000,1000) is bottom-right. '
    'Only include people who are actually waving; return an empty list if none.'
)

_SCHEMA = {
    'type': 'object',
    'properties': {
        'persons': {'type': 'array', 'items': {
            'type': 'object',
            'properties': {
                'box_2d': {'type': 'array', 'items': {'type': 'integer'},
                           'minItems': 4, 'maxItems': 4},
                'waving': {'type': 'boolean'},
            },
            'required': ['box_2d', 'waving'],
            'additionalProperties': False,
        }},
    },
    'required': ['persons'],
    'additionalProperties': False,
}


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


def _resolve_key(provider: str) -> Optional[str]:
    """Resolve a provider's API key from os.environ only, or None."""
    if provider == 'qwen':
        for name in _QWEN_KEY_NAMES:
            val = os.environ.get(name)
            if val:
                return val
        return None
    if provider == 'gemini':
        return os.environ.get('OPENROUTER_API_KEY') or None
    return None


def has_provider_key(provider: str) -> bool:
    """True if the provider's API key is present in the environment."""
    return _resolve_key(provider) is not None


def build_provider_models(primary: str, fallback: str, *, has_key, model_for,
                          logger=None) -> list:
    """Ordered (provider, model) chain, dropping providers with no key.

    Non-fatal: returns [] if no provider has a key (caller treats an empty chain
    as 'VLM fallback unavailable' rather than crashing). A blank or duplicate
    fallback is ignored.
    """
    chain = []
    if has_key(primary):
        chain.append((primary, model_for(primary)))
    elif logger is not None:
        logger.warn(f'Waving VLM primary provider {primary!r} key missing.')
    if fallback and fallback != primary:
        if has_key(fallback):
            chain.append((fallback, model_for(fallback)))
        elif logger is not None:
            logger.warn(f'Waving VLM fallback provider {fallback!r} key missing.')
    return chain


def _strip_fences(text: str) -> str:
    """Drop ```json ... ``` fences some models emit despite instructions."""
    return _FENCE_RE.sub('', text).strip() if '```' in text else text


def request_waving_persons(rgb_bgr: np.ndarray, *, provider: str, model: str,
                           timeout_s: float = 20.0, max_retries: int = 3,
                           logger=None) -> WavingVlmResult:
    """Single-provider waving detection.

    Raises WavingVlmError on hard failure (missing key, exhausted retries).
    Returns a WavingVlmResult; .boxes may be empty (a clean 'nobody waving').
    """
    key = _resolve_key(provider)
    if not key:
        raise WavingVlmError(
            f'{provider} API key not set (qwen: {_QWEN_KEY_NAMES}, '
            f'gemini: OPENROUTER_API_KEY).')
    if provider == 'qwen':
        b_url, reasoning = _QWEN_DEFAULT_BASE_URL, False
    elif provider == 'gemini':
        b_url, reasoning = _GEMINI_DEFAULT_BASE_URL, True
    else:
        raise WavingVlmError(f'unknown provider {provider!r} (expected qwen|gemini)')

    client = openai.OpenAI(api_key=key, base_url=b_url)
    h, w = rgb_bgr.shape[:2]
    messages = [
        {'role': 'system', 'content': _SYSTEM_PROMPT},
        {'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': encode_data_url(rgb_bgr)}},
            {'type': 'text', 'text': 'Find every person waving in this image.'},
        ]},
    ]
    rf_strict = {'type': 'json_schema',
                 'json_schema': {'name': 'waving_persons', 'strict': True,
                                 'schema': _SCHEMA}}
    rf_loose = {'type': 'json_object'}
    use_strict = True
    extra_body = ({'reasoning': {'enabled': True, 'max_tokens': 1024}}
                  if reasoning else None)

    t0 = time.perf_counter()
    last_error: Optional[Exception] = None
    try:
        for attempt in range(1, max_retries + 1):
            rf = rf_strict if use_strict else rf_loose
            try:
                kwargs = dict(model=model, messages=messages,
                              response_format=rf, temperature=0.2)
                if extra_body is not None:
                    kwargs['extra_body'] = extra_body
                completion = client.with_options(
                    timeout=timeout_s).chat.completions.create(**kwargs)
                raw = completion.choices[0].message.content or ''
                parsed = json.loads(_strip_fences(raw))
                res = select_boxes(parsed, w, h)
                res.provider = provider
                res.elapsed_s = time.perf_counter() - t0
                if logger is not None:
                    logger.info(
                        f'[{provider}] waving VLM -> {len(res.boxes)} box(es) '
                        f'(attempt {attempt}/{max_retries})')
                return res
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                last_error = exc
                if logger is not None:
                    logger.warning(f'[{provider}] parse failed '
                                   f'(attempt {attempt}/{max_retries}): {exc}')
            except Exception as exc:  # noqa: BLE001
                txt = str(exc).lower()
                if use_strict and any(k in txt for k in
                                      ('json_schema', 'response_format', 'schema')):
                    use_strict = False
                    if logger is not None:
                        logger.warning(f'[{provider}] schema rejected; using '
                                       f'json_object: {exc}')
                last_error = exc
                if logger is not None:
                    logger.warning(f'[{provider}] call failed '
                                   f'(attempt {attempt}/{max_retries}): {exc}')
        raise WavingVlmError(
            f'[{provider}] exhausted {max_retries} retries; last={last_error}')
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
