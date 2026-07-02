"""Gemini/Qwen shelf-layer VLM client for grocery_categorize's Categorize
action.

Mirrors the provider-chain conventions of _feature_vlm.py / _match_vlm.py.
The payload here is a JSON object {object_description, shelf_description,
reason, layer} parsed with json.loads and validated for the three keys
('layer', 'shelf_description', 'reason') the caller depends on. A parse/
validation failure retries within the same provider budget; only
exhausting a provider's retries moves to the next provider.

Unlike the other kimi_api nodes, grocery_categorize previously had no
retry loop at all -- adding this provider chain also adds the missing
single-provider retry.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import openai

from ._env import (
    base_url,
    dashscope_base_url,
    require_api_key,
    require_dashscope_api_key,
)
from ._vlm_text import strip_fences


def _coerce_layer(response: dict) -> bool:
    """Coerce response['layer'] to int in place; False if not coercible.

    The caller indexes clusters[int(layer)], so a non-integer layer must
    consume a retry rather than crash the action callback after acceptance.
    bool is rejected explicitly (int(True) == 1 would silently pass).
    """
    layer = response.get('layer')
    if isinstance(layer, bool):
        return False
    try:
        response['layer'] = int(layer)
    except (TypeError, ValueError):
        return False
    return True


class ShelfVlmError(RuntimeError):
    """Hard failure: missing API key, unknown provider, or every attempt
    (API call, parse, or field validation) failed for a provider."""


@dataclass
class ShelfVlmResult:
    response: dict = field(default_factory=dict)
    provider: str = ''
    elapsed_s: float = 0.0


def request_shelf_layer(
    sys_prompt: str,
    shelf_img_url: str,
    obj_seg_url: str,
    *,
    provider: str,
    model: str,
    timeout_s: float = 20.0,
    max_retries: int = 3,
    logger=None,
) -> ShelfVlmResult:
    """Single-provider shelf-layer call with parse/validate retry.

    Each attempt calls the model, parses the response with json.loads, and
    checks for the 'layer', 'shelf_description', and 'reason' keys the
    caller depends on. An API exception, parse failure, or missing field
    all consume one of max_retries attempts on this provider. Raises
    ShelfVlmError on missing key, unknown provider, or exhausting every
    attempt.
    """
    if provider == 'qwen':
        try:
            api_key = require_dashscope_api_key()
        except RuntimeError as exc:
            raise ShelfVlmError(str(exc)) from exc
        b_url = dashscope_base_url()
    elif provider == 'gemini':
        try:
            api_key = require_api_key()
        except RuntimeError as exc:
            raise ShelfVlmError(str(exc)) from exc
        b_url = base_url()
    else:
        raise ShelfVlmError(f"unknown provider {provider!r} (expected qwen|gemini)")

    client = openai.OpenAI(api_key=api_key, base_url=b_url)
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'picture of shelf'},
                {'type': 'image_url', 'image_url': {'url': shelf_img_url}},
                {'type': 'text', 'text': 'picture of new object.'},
                {'type': 'image_url', 'image_url': {'url': obj_seg_url}},
            ],
        },
    ]

    t0 = time.perf_counter()
    last_error: Optional[str] = None
    try:
        for attempt in range(1, max_retries + 1):
            try:
                completion = client.with_options(
                    timeout=timeout_s).chat.completions.create(
                    model=model, messages=messages)
            except Exception as exc:  # noqa: BLE001
                last_error = f'API call failed: {exc}'
                if logger is not None:
                    logger.warning(
                        f'[{provider}] VLM call failed '
                        f'(attempt {attempt}/{max_retries}): {exc}')
                if attempt < max_retries:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                continue

            try:
                response = json.loads(
                    strip_fences(completion.choices[0].message.content or ''))
            except Exception as exc:  # noqa: BLE001
                last_error = f'parse failed: {exc}'
                if logger is not None:
                    logger.info(
                        f'[{provider}] parse failed '
                        f'(attempt {attempt}/{max_retries}): {exc}')
                continue

            if not isinstance(response, dict):
                last_error = f'response is not a JSON object: {response!r}'
            elif 'layer' not in response:
                last_error = "response missing 'layer' field"
            elif 'shelf_description' not in response:
                last_error = "response missing 'shelf_description' field"
            elif 'reason' not in response:
                last_error = "response missing 'reason' field"
            elif not _coerce_layer(response):
                last_error = f"'layer' is not an integer: {response['layer']!r}"
            else:
                if logger is not None:
                    logger.info(
                        f'[{provider}] shelf layer accepted '
                        f'(attempt {attempt}/{max_retries}).')
                return ShelfVlmResult(
                    response=response, provider=provider,
                    elapsed_s=time.perf_counter() - t0)
            if logger is not None:
                logger.info(
                    f'[{provider}] validate failed '
                    f'(attempt {attempt}/{max_retries}): {last_error}')
        raise ShelfVlmError(
            f'[{provider}] exhausted {max_retries} attempts; last={last_error}')
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def request_shelf_layer_chain(
    sys_prompt: str,
    shelf_img_url: str,
    obj_seg_url: str,
    *,
    provider_models: Sequence[tuple],
    timeout_s: float = 20.0,
    max_retries: int = 3,
    logger=None,
) -> ShelfVlmResult:
    """Try (provider, model) pairs in order; return the first success.

    Errors-only fallthrough: a ShelfVlmError (exhausted attempts, missing
    key) falls through to the next provider. Raises ShelfVlmError if every
    provider fails, or if provider_models is empty.
    """
    errors = []
    for provider, model in provider_models:
        try:
            return request_shelf_layer(
                sys_prompt, shelf_img_url, obj_seg_url,
                provider=provider, model=model,
                timeout_s=timeout_s, max_retries=max_retries, logger=logger,
            )
        except ShelfVlmError as exc:
            errors.append(f'{provider}: {exc}')
            if logger is not None:
                logger.warning(
                    f'shelf VLM provider {provider} failed: {exc}; trying next.')
    raise ShelfVlmError(
        'all providers failed: '
        + (' | '.join(errors) or 'no providers configured'))
