"""Gemini/Qwen plain-text VLM client for feature_extraction_service.

Mirrors the control flow of _seat_bbox_vlm.py / tk_vision_specialized's
_waving_vlm.py (single call -> provider chain, errors-only fallthrough)
but for a plain free-text completion -- feature_extraction has no JSON
schema to parse, so this client is smaller than the schema-based ones.
request_feature_description_chain() tries providers in order so the node
can do Gemini -> Qwen fallback when Gemini exhausts its retries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import openai

from ._env import (
    base_url,
    dashscope_base_url,
    require_api_key,
    require_dashscope_api_key,
)


class FeatureVlmError(RuntimeError):
    """Hard failure: missing API key, unknown provider, or exhausted retries."""


@dataclass
class FeatureVlmResult:
    text: str = ''
    provider: str = ''
    elapsed_s: float = 0.0


def request_feature_description(
    image_url: str,
    sys_prompt: str,
    user_text: str,
    *,
    provider: str,
    model: str,
    timeout_s: float = 20.0,
    max_retries: int = 3,
    logger=None,
) -> FeatureVlmResult:
    """Single-provider plain-text VLM call with exponential-backoff retries.

    Raises FeatureVlmError on hard failure (missing key, unknown provider,
    or every attempt failing).
    """
    if provider == 'qwen':
        try:
            api_key = require_dashscope_api_key()
        except RuntimeError as exc:
            raise FeatureVlmError(str(exc)) from exc
        b_url = dashscope_base_url()
    elif provider == 'gemini':
        try:
            api_key = require_api_key()
        except RuntimeError as exc:
            raise FeatureVlmError(str(exc)) from exc
        b_url = base_url()
    else:
        raise FeatureVlmError(f"unknown provider {provider!r} (expected qwen|gemini)")

    client = openai.OpenAI(api_key=api_key, base_url=b_url)
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': image_url}},
            {'type': 'text', 'text': user_text},
        ]},
    ]

    t0 = time.perf_counter()
    last_error: Optional[Exception] = None
    try:
        for attempt in range(1, max_retries + 1):
            try:
                completion = client.with_options(
                    timeout=timeout_s).chat.completions.create(
                    model=model, messages=messages)
                text = completion.choices[0].message.content or ''
                # None/empty content (safety block, empty candidate) is a
                # failed attempt, not an answer — retrying here is what lets
                # the chain's fallback provider fire on a blocked response.
                if not text.strip():
                    raise ValueError('completion content is empty')
                if logger is not None:
                    logger.info(
                        f'[{provider}] feature VLM call succeeded '
                        f'(attempt {attempt}/{max_retries})')
                return FeatureVlmResult(
                    text=text, provider=provider,
                    elapsed_s=time.perf_counter() - t0)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if logger is not None:
                    logger.warning(
                        f'[{provider}] VLM call failed '
                        f'(attempt {attempt}/{max_retries}): {exc}')
                if attempt < max_retries:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
        raise FeatureVlmError(
            f'[{provider}] exhausted {max_retries} retries; last={last_error}')
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def request_feature_description_chain(
    image_url: str,
    sys_prompt: str,
    user_text: str,
    *,
    provider_models: Sequence[tuple],
    timeout_s: float = 20.0,
    max_retries: int = 3,
    logger=None,
) -> FeatureVlmResult:
    """Try (provider, model) pairs in order; return the first success.

    Errors-only fallthrough: a FeatureVlmError (exhausted retries, missing
    key) falls through to the next provider. Raises FeatureVlmError if
    every provider fails, or if provider_models is empty.
    """
    errors = []
    for provider, model in provider_models:
        try:
            return request_feature_description(
                image_url, sys_prompt, user_text,
                provider=provider, model=model,
                timeout_s=timeout_s, max_retries=max_retries, logger=logger,
            )
        except FeatureVlmError as exc:
            errors.append(f'{provider}: {exc}')
            if logger is not None:
                logger.warning(
                    f'feature VLM provider {provider} failed: {exc}; '
                    f'trying next.')
    raise FeatureVlmError(
        'all providers failed: '
        + (' | '.join(errors) or 'no providers configured'))
