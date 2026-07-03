"""Gemini/Qwen labels-only scan VLM client for object_scan.

Given ONE scene image and a candidate vocabulary (one batch), asks the model
which candidates are visible and returns the validated subset (drops
hallucinations / anything not in the vocabulary). Mirrors the provider-chain
conventions of _match_vlm.py: single-provider call with parse retry, then a
chain that falls through to the next provider only on error. An empty list is a
legitimate terminal answer ("none of these here") and does NOT trigger
fallback.

Ported from scripts/object_scan_webui/scan_core.py (the tuning harness), which
is the reference implementation this validates against.
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import openai

from ._env import base_url, require_api_key, resolve_qwen_target
from ._vlm_text import strip_fences

_SYS_PROMPT = (
    "You are a precise visual object detector. You are given ONE photo of a "
    "scene and a list of candidate object names. Return a JSON array "
    "containing exactly the candidate names -- copied verbatim from the list "
    "-- that are clearly visible in the photo. Only include a name if you are "
    "confident the object is present. Never include a name that is not in the "
    "list. If none are present, return []. Output ONLY the JSON array, nothing "
    "else."
)


class ScanVlmError(RuntimeError):
    """Hard failure: missing key, unknown provider, or every attempt failed."""


@dataclass
class ScanVlmResult:
    labels: list = field(default_factory=list)
    provider: str = ''
    elapsed_s: float = 0.0


def validate_labels(raw_text: str, candidates: list) -> list:
    """Parse a JSON array; keep only entries matching a candidate.

    Case-insensitive match; returns the candidate's original casing, deduped.
    Raises ValueError if the payload is not a list (unparseable -> retry).
    """
    parsed = None
    cleaned = strip_fences(raw_text or '')
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(cleaned)
            break
        except Exception:  # noqa: BLE001
            continue
    if not isinstance(parsed, list):
        raise ValueError(f'not a JSON list: {raw_text!r}')
    lut = {c.lower(): c for c in candidates}
    out, seen = [], set()
    for item in parsed:
        key = str(item).strip().lower()
        if key in lut and lut[key] not in seen:
            out.append(lut[key])
            seen.add(lut[key])
    return out


def request_scan_labels(
    image_url: str,
    candidates: list,
    *,
    provider: str,
    model: str,
    qwen_api_backend: str = 'dashscope',
    timeout_s: float = 20.0,
    max_retries: int = 3,
    logger=None,
) -> ScanVlmResult:
    """Single-provider labels-scan call with parse retry.

    An API exception or unparseable (non-list) response consumes one of
    max_retries attempts. Raises ScanVlmError on missing key, unknown provider,
    or exhausting every attempt. An empty (but valid) list is a success.
    """
    if provider == 'qwen':
        try:
            b_url, api_key, model = resolve_qwen_target(qwen_api_backend, model)
        except RuntimeError as exc:
            raise ScanVlmError(str(exc)) from exc
    elif provider == 'gemini':
        try:
            api_key = require_api_key()
        except RuntimeError as exc:
            raise ScanVlmError(str(exc)) from exc
        b_url = base_url()
    else:
        raise ScanVlmError(f"unknown provider {provider!r} (expected qwen|gemini)")

    client = openai.OpenAI(api_key=api_key, base_url=b_url)
    messages = [
        {'role': 'system', 'content': _SYS_PROMPT},
        {'role': 'user', 'content': [
            {'type': 'text',
             'text': f'Candidate object names: {json.dumps(candidates)}. '
                     'Which of these are visible in the photo?'},
            {'type': 'image_url', 'image_url': {'url': image_url}},
        ]},
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
                        f'[{provider}] scan VLM call failed '
                        f'(attempt {attempt}/{max_retries}): {exc}')
                if attempt < max_retries:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                continue

            raw = completion.choices[0].message.content
            if logger is not None:
                logger.info(f'[{provider}] scan response: {raw}')
            try:
                labels = validate_labels(raw, candidates)
            except ValueError as exc:
                last_error = f'parse failed: {exc}'
                if logger is not None:
                    logger.info(
                        f'[{provider}] parse failed '
                        f'(attempt {attempt}/{max_retries}): {exc}')
                continue
            return ScanVlmResult(
                labels=labels, provider=provider,
                elapsed_s=time.perf_counter() - t0)
        raise ScanVlmError(
            f'[{provider}] exhausted {max_retries} attempts; last={last_error}')
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def request_scan_labels_chain(
    image_url: str,
    candidates: list,
    *,
    provider_models: Sequence[tuple],
    qwen_api_backend: str = 'dashscope',
    timeout_s: float = 20.0,
    max_retries: int = 3,
    logger=None,
) -> ScanVlmResult:
    """Try (provider, model) pairs in order; return the first success.

    Errors-only fallthrough: a ScanVlmError (exhausted attempts / missing key)
    falls through to the next provider. Raises ScanVlmError if every provider
    fails, or if provider_models is empty.
    """
    errors = []
    for provider, model in provider_models:
        try:
            return request_scan_labels(
                image_url, candidates, provider=provider, model=model,
                qwen_api_backend=qwen_api_backend, timeout_s=timeout_s,
                max_retries=max_retries, logger=logger,
            )
        except ScanVlmError as exc:
            errors.append(f'{provider}: {exc}')
            if logger is not None:
                logger.warning(
                    f'scan VLM provider {provider} failed: {exc}; trying next.')
    raise ScanVlmError(
        'all providers failed: '
        + (' | '.join(errors) or 'no providers configured'))
