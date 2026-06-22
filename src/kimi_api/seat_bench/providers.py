"""Unified OpenAI-compatible VLM caller for the benchmark.

Two providers:
  - "gemini": OpenRouter, google/gemini-2.5-pro, reasoning enabled.
  - "qwen":   DashScope OpenAI-compatible, qwen3-vl-plus.

call_vlm() returns the parsed JSON dict (or raises). Strict json_schema is
tried first, falling back to json_object if the route rejects the schema
(same pattern as kimi_api._seat_vlm / qwen_match_vlm).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

from .paths import ensure_kimi_api_importable

ensure_kimi_api_importable()
from kimi_api import _env  # noqa: E402


GEMINI_MODEL = "google/gemini-2.5-pro"
QWEN_MODEL = "qwen3-vl-plus"


@dataclass
class ProviderCfg:
    name: str
    model: str
    api_key: str
    base_url: str
    reasoning: bool


def _provider_cfg(provider: str) -> ProviderCfg:
    _env.load_env()
    if provider == "gemini":
        return ProviderCfg("gemini", GEMINI_MODEL, _env.require_api_key(),
                           _env.base_url(), reasoning=True)
    if provider == "qwen":
        return ProviderCfg("qwen", QWEN_MODEL, _env.require_dashscope_api_key(),
                           _env.dashscope_base_url(), reasoning=False)
    raise ValueError(f"unknown provider {provider!r} (expected gemini|qwen)")


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text).strip() if "```" in text else text


def call_vlm(
    provider: str,
    messages: list,
    *,
    schema: Optional[dict] = None,
    schema_name: str = "seat_bench",
    timeout_s: float = 30.0,
    max_retries: int = 3,
    temperature: float = 0.2,
    logger=None,
) -> tuple[dict, float]:
    """Return (parsed_json, elapsed_s). Raises RuntimeError on exhaustion."""
    cfg = _provider_cfg(provider)
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    rf_strict = (
        {"type": "json_schema",
         "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
        if schema else {"type": "json_object"}
    )
    rf_loose = {"type": "json_object"}
    use_strict = schema is not None
    extra_body = {"reasoning": {"enabled": True, "max_tokens": 2048}} if cfg.reasoning else None

    t0 = time.perf_counter()
    last_error: Optional[Exception] = None
    try:
        for attempt in range(1, max_retries + 1):
            rf = rf_strict if use_strict else rf_loose
            try:
                kwargs = dict(model=cfg.model, messages=messages,
                              response_format=rf, temperature=temperature)
                if extra_body is not None:
                    kwargs["extra_body"] = extra_body
                completion = client.with_options(timeout=timeout_s).chat.completions.create(**kwargs)
                raw = completion.choices[0].message.content or ""
                return json.loads(_strip_fences(raw)), time.perf_counter() - t0
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                last_error = exc
                if logger:
                    logger(f"[{provider}] JSON parse failed attempt {attempt}/{max_retries}: {exc}")
            except Exception as exc:  # noqa: BLE001
                txt = str(exc).lower()
                if use_strict and any(k in txt for k in ("json_schema", "response_format", "schema")):
                    use_strict = False
                    if logger:
                        logger(f"[{provider}] schema rejected, falling back to json_object: {exc}")
                last_error = exc
                if logger:
                    logger(f"[{provider}] call failed attempt {attempt}/{max_retries}: {exc}")
        raise RuntimeError(f"[{provider}] exhausted {max_retries} retries; last={last_error}")
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
