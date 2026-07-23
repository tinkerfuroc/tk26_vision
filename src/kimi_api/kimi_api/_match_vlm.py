"""Gemini/Qwen JSON-list VLM client for the feature-matching action.

Mirrors the provider-chain conventions of _feature_vlm.py / _seat_bbox_vlm.py,
but the payload here is a JSON list of candidate indices (one per reference/
description) rather than free text or a bbox+select schema, parsed with
ast.literal_eval and validated/patched via patch_result. A parse/validation
failure retries within the same provider (unchanged from before this
fallback was added); only exhausting a provider's full retry budget moves
to the next provider.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import openai

from ._env import (
    base_url,
    require_api_key,
    resolve_qwen_target,
)
from ._vlm_text import strip_fences


class MatchVlmError(RuntimeError):
    """Hard failure: missing API key, unknown provider, or every attempt
    (API call, parse, or validation) failed for a provider."""


def _raise_if_aborted(should_abort: Optional[Callable[[], bool]]) -> None:
    if should_abort is not None and should_abort():
        raise MatchVlmError('match VLM request aborted')


@dataclass
class MatchVlmResult:
    indices: list = field(default_factory=list)
    provider: str = ''
    elapsed_s: float = 0.0


def patch_result(raw, n_targets, n_cand):
    """Coerce a VLM result list into a valid [0, n_cand) assignment.

    Contract: callers (the HRI BT in particular) require one centroid per
    requested feature so they can index without having to track which
    references the VLM hedged on. This function enforces that contract --
    every cell is patched to a valid candidate index, even when the VLM
    emits None / -1 / out-of-range / a non-int. Only structurally
    unsalvageable input (not a list, or empty list when targets exist)
    returns None to trigger a retry.

    Returns (patched_list, msg). patched_list is None if the input is
    structurally unsalvageable. Otherwise patched_list is length-n_targets
    with every entry in [0, n_cand).

    Per-cell rules (cyclic `i % n_cand` is the tk23-legacy fallback):
      * missing  (i >= len(raw))                       -> i % n_cand
      * None / non-int + non-coercible / bool          -> i % n_cand
      * out-of-range int (incl. -1, the old sentinel)  -> i % n_cand
      * numeric string ("3")                           -> int("3")

    A result in which EVERY cell required the cyclic fallback carries zero
    usable VLM signal; with the provider chain available it is treated as
    unsalvageable (retry / fall back) rather than fabricating a full
    assignment — except when n_cand == 1, where every assignment is [0, ...]
    regardless and retrying gains nothing.
    """
    if not isinstance(raw, list):
        return None, f'not a list: {raw!r}'
    if len(raw) == 0 and n_targets > 0:
        return None, 'empty list'
    patched = []
    n_fallback = 0
    for i in range(n_targets):
        v = raw[i] if i < len(raw) else None
        fabricated = False
        if isinstance(v, bool):
            v, fabricated = i % n_cand, True
        elif isinstance(v, int):
            pass
        elif v is None:
            v, fabricated = i % n_cand, True
        else:
            try:
                v = int(v)
            except (TypeError, ValueError):
                v, fabricated = i % n_cand, True
        if not fabricated and (v < 0 or v >= n_cand):
            v, fabricated = i % n_cand, True
        n_fallback += fabricated
        patched.append(v)
    if n_cand > 1 and n_targets > 0 and n_fallback == n_targets:
        return None, 'every cell required cyclic fallback (no usable signal)'
    return patched, ''


def request_match_indices(
    sys_prompt: str,
    user_content: list,
    *,
    n_feats: int,
    n_cand: int,
    provider: str,
    model: str,
    qwen_api_backend: str = 'dashscope',
    timeout_s: float = 20.0,
    max_retries: int = 3,
    logger=None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> MatchVlmResult:
    """Single-provider match-index call with parse/validate retry.

    Each attempt calls the model, parses the response with
    ast.literal_eval, and validates/patches it via patch_result. An API
    exception, parse failure, or unsalvageable result all consume one of
    max_retries attempts on this provider. Raises MatchVlmError on missing
    key, unknown provider, or exhausting every attempt.
    """
    if provider == 'qwen':
        try:
            b_url, api_key, model = resolve_qwen_target(qwen_api_backend, model)
        except RuntimeError as exc:
            raise MatchVlmError(str(exc)) from exc
    elif provider == 'gemini':
        try:
            api_key = require_api_key()
        except RuntimeError as exc:
            raise MatchVlmError(str(exc)) from exc
        b_url = base_url()
    else:
        raise MatchVlmError(f"unknown provider {provider!r} (expected qwen|gemini)")

    client = openai.OpenAI(api_key=api_key, base_url=b_url, max_retries=0)
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_content},
    ]

    t0 = time.perf_counter()
    last_error: Optional[str] = None
    try:
        for attempt in range(1, max_retries + 1):
            _raise_if_aborted(should_abort)
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
                _raise_if_aborted(should_abort)
                if attempt < max_retries:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                continue

            try:
                raw = completion.choices[0].message.content
                if logger is not None:
                    logger.info(f'[{provider}] LLM response: {raw}')
                parsed = ast.literal_eval(strip_fences(raw or ''))
            except Exception as exc:  # noqa: BLE001
                last_error = f'parse failed: {exc}'
                if logger is not None:
                    logger.info(
                        f'[{provider}] parse failed '
                        f'(attempt {attempt}/{max_retries}): {exc}')
                _raise_if_aborted(should_abort)
                continue

            patched, msg = patch_result(parsed, n_feats, n_cand)
            if patched is not None:
                if logger is not None:
                    if patched != parsed:
                        logger.info(
                            f'[{provider}] patched VLM result: '
                            f'{parsed} -> {patched}')
                    logger.info(
                        f'[{provider}] match accepted '
                        f'(attempt {attempt}/{max_retries}).')
                return MatchVlmResult(
                    indices=patched, provider=provider,
                    elapsed_s=time.perf_counter() - t0)
            last_error = f'unsalvageable: {msg}'
            if logger is not None:
                logger.info(
                    f'[{provider}] validate failed '
                    f'(attempt {attempt}/{max_retries}): {msg}')
            _raise_if_aborted(should_abort)
        raise MatchVlmError(
            f'[{provider}] exhausted {max_retries} attempts; last={last_error}')
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def request_match_indices_chain(
    sys_prompt: str,
    user_content: list,
    *,
    n_feats: int,
    n_cand: int,
    provider_models: Sequence[tuple],
    qwen_api_backend: str = 'dashscope',
    timeout_s: float = 20.0,
    max_retries: int = 3,
    logger=None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> MatchVlmResult:
    """Try (provider, model) pairs in order; return the first success.

    Errors-only fallthrough: a MatchVlmError (exhausted attempts, missing
    key) falls through to the next provider. Raises MatchVlmError if every
    provider fails, or if provider_models is empty.
    """
    errors = []
    for provider, model in provider_models:
        _raise_if_aborted(should_abort)
        try:
            return request_match_indices(
                sys_prompt, user_content,
                n_feats=n_feats, n_cand=n_cand,
                provider=provider, model=model,
                qwen_api_backend=qwen_api_backend,
                timeout_s=timeout_s, max_retries=max_retries, logger=logger,
                should_abort=should_abort,
            )
        except MatchVlmError as exc:
            _raise_if_aborted(should_abort)
            errors.append(f'{provider}: {exc}')
            if logger is not None:
                logger.warning(
                    f'match VLM provider {provider} failed: {exc}; trying next.')
    raise MatchVlmError(
        'all providers failed: '
        + (' | '.join(errors) or 'no providers configured'))
