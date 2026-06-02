"""Qwen3-VL / Gemini bbox+select seat client for seat_recommend_bbox.

Production port of the 'S1' strategy that won the 2026-06-02 seat-bench
benchmark (docs/superpowers/plans/2026-06-02-seat-recommend-strategy-benchmark.md;
results in src/kimi_api/seat_bench/report.md). One structured call returns a
cushion box + occupancy for every visible seat plus the chosen empty seat's
label; the caller takes the chosen box's centre as the depth-sampling pixel.

Provider-agnostic: 'qwen' -> DashScope (qwen3-vl-plus), 'gemini' -> OpenRouter
(google/gemini-2.5-pro, reasoning enabled). Keys/base-urls via kimi_api._env.
request_seat_bbox_chain() tries providers in order so the node can do
Qwen -> Gemini fallback. Kept independent of seat_bench/ (eval scaffolding).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ._env import (
    base_url,
    dashscope_base_url,
    load_env,
    require_api_key,
    require_dashscope_api_key,
)
from ._image_utils import encode_to_data_url

Box = tuple[int, int, int, int]


class VlmSeatBboxError(RuntimeError):
    """Hard failure: missing API key, exhausted retries, or unparseable response."""


@dataclass
class SeatBboxResult:
    label: str = "none"
    box_xyxy: Optional[list] = None          # [x1,y1,x2,y2] pixels, or None for "none"
    seats: list = field(default_factory=list)
    provider: str = ""
    elapsed_s: float = 0.0
    error: Optional[str] = None              # soft selection error -> triggers fallback


_SYSTEM = (
    "You help a robot seat a new guest. Look at the image and return JSON "
    "with fields: seats, choice.\n"
    "seats — array, one entry per sittable cushion (a 2-cushion sofa = 2 "
    "entries, a single armchair/stool = 1). Each entry: "
    '{"label": "<short identifier with a visual anchor>", '
    '"box_2d": [x1,y1,x2,y2], "occupied": true|false}. '
    "box_2d is the tight bounding box of the SEAT CUSHION (the flat surface "
    "a person sits on, NOT the backrest), normalized 0-1000 over the image "
    "where (0,0) is top-left and (1000,1000) is bottom-right.\n"
    "A cushion is OCCUPIED if a person sits on it or a large object rests on "
    "the cushion fabric; objects on a table/floor/armrest do not occupy it.\n"
    "choice — the label of one entry whose occupied is false (your "
    'recommendation), or "none" if every seat is occupied or none are visible.'
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "seats": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "box_2d": {"type": "array", "items": {"type": "integer"},
                           "minItems": 4, "maxItems": 4},
                "occupied": {"type": "boolean"},
            },
            "required": ["label", "box_2d", "occupied"],
            "additionalProperties": False,
        }},
        "choice": {"type": "string"},
    },
    "required": ["seats", "choice"],
    "additionalProperties": False,
}

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text).strip() if "```" in text else text


def decode_box_xyxy(box_2d, w: int, h: int) -> Optional[Box]:
    """Decode a [x1,y1,x2,y2] 0-1000 normalized box to clamped xyxy pixels."""
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


def _build_text_prompt(
    names: Sequence[str],
    features: Sequence[str],
    known_seats: Optional[Sequence[str]] = None,
) -> str:
    """Mirror _seat_vlm._build_text_prompt so requests read identically."""
    text = "Recommend a seat for a new guest."
    for name, feature in zip(names or [], features or []):
        text += f" The person matching description: {feature} is called {name}."
    if known_seats:
        lines = "\n".join(f'  - "{s}"' for s in known_seats)
        text += (
            "\n\nThe seats in this room are pre-catalogued. The recommendation "
            "label (choice) MUST be exactly one of these strings, character-for-"
            'character, or "none" if every catalogued seat is occupied or not '
            "visible:\n" + lines + "\nFor seats, only include catalogued seats "
            "actually visible; do not invent or rename seats."
        )
    return text


def select_box(
    parsed: dict,
    w: int,
    h: int,
    known_seats: Optional[Sequence[str]],
) -> SeatBboxResult:
    """Pure: turn a parsed VLM response into a SeatBboxResult.

    .error is set (and box None) for soft failures that should trigger
    provider fallback: out-of-catalog choice, choice-not-in-seats,
    undecodable box. A "none" choice is a clean terminal answer (no error).
    """
    res = SeatBboxResult()
    seats = parsed.get("seats", []) or []
    res.seats = seats if isinstance(seats, list) else []
    choice = str(parsed.get("choice", "none") or "none")
    res.label = choice
    if choice.strip().lower() == "none":
        res.box_xyxy = None
        return res
    if known_seats and choice not in known_seats:
        res.error = f"out-of-catalog choice {choice!r}; catalog={list(known_seats)}"
        return res
    chosen = next(
        (s for s in res.seats
         if str(s.get("label", "")).strip().lower() == choice.strip().lower()),
        None,
    )
    if chosen is None:
        res.error = f"choice {choice!r} not in seats list"
        return res
    box = decode_box_xyxy(chosen.get("box_2d"), w, h)
    if box is None:
        res.error = f"chosen box for {choice!r} failed to decode"
        return res
    res.box_xyxy = list(box)
    return res


def request_seat_bbox(
    rgb_bgr: np.ndarray,
    names: Sequence[str],
    features: Sequence[str],
    *,
    provider: str,
    model: str,
    known_seats: Optional[Sequence[str]] = None,
    timeout_s: float = 20.0,
    max_retries: int = 3,
    logger=None,
) -> SeatBboxResult:
    """Single-provider bbox+select. Raises VlmSeatBboxError on hard failure;
    returns a SeatBboxResult whose .error may be set on soft selection issues."""
    load_env()
    if provider == "qwen":
        try:
            api_key = require_dashscope_api_key()
        except RuntimeError as exc:
            raise VlmSeatBboxError(str(exc)) from exc
        b_url, reasoning = dashscope_base_url(), False
    elif provider == "gemini":
        try:
            api_key = require_api_key()
        except RuntimeError as exc:
            raise VlmSeatBboxError(str(exc)) from exc
        b_url, reasoning = base_url(), True
    else:
        raise VlmSeatBboxError(f"unknown provider {provider!r} (expected qwen|gemini)")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=b_url)
    h, w = rgb_bgr.shape[:2]
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": encode_to_data_url(rgb_bgr)}},
            {"type": "text", "text": _build_text_prompt(names, features, known_seats)},
        ]},
    ]
    rf_strict = {"type": "json_schema",
                 "json_schema": {"name": "seat_bbox_select", "strict": True, "schema": _SCHEMA}}
    rf_loose = {"type": "json_object"}
    use_strict = True
    extra_body = {"reasoning": {"enabled": True, "max_tokens": 2048}} if reasoning else None

    t0 = time.perf_counter()
    last_error: Optional[Exception] = None
    try:
        for attempt in range(1, max_retries + 1):
            rf = rf_strict if use_strict else rf_loose
            try:
                kwargs = dict(model=model, messages=messages,
                              response_format=rf, temperature=0.2)
                if extra_body is not None:
                    kwargs["extra_body"] = extra_body
                completion = client.with_options(timeout=timeout_s).chat.completions.create(**kwargs)
                raw = completion.choices[0].message.content or ""
                parsed = json.loads(_strip_fences(raw))
                res = select_box(parsed, w, h, known_seats)
                res.provider = provider
                res.elapsed_s = time.perf_counter() - t0
                if logger is not None:
                    logger.info(
                        f"[{provider}] bbox+select choice={res.label!r} "
                        f"box={res.box_xyxy} seats={len(res.seats)} "
                        f"err={res.error} (attempt {attempt}/{max_retries})"
                    )
                return res
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                last_error = exc
                if logger is not None:
                    logger.warning(f"[{provider}] parse failed "
                                   f"(attempt {attempt}/{max_retries}): {exc}")
            except Exception as exc:  # noqa: BLE001
                txt = str(exc).lower()
                if use_strict and any(k in txt for k in
                                      ("json_schema", "response_format", "schema")):
                    use_strict = False
                    if logger is not None:
                        logger.warning(f"[{provider}] schema rejected; "
                                       f"falling back to json_object: {exc}")
                last_error = exc
                if logger is not None:
                    logger.warning(f"[{provider}] call failed "
                                   f"(attempt {attempt}/{max_retries}): {exc}")
        raise VlmSeatBboxError(
            f"[{provider}] exhausted {max_retries} retries; last={last_error}")
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def request_seat_bbox_chain(
    rgb_bgr: np.ndarray,
    names: Sequence[str],
    features: Sequence[str],
    *,
    provider_models: Sequence[tuple],
    known_seats: Optional[Sequence[str]] = None,
    timeout_s: float = 20.0,
    max_retries: int = 3,
    logger=None,
) -> SeatBboxResult:
    """Try (provider, model) pairs in order. Return the first CLEAN result —
    a decodable box, or a legitimate "none" (no error). Hard errors
    (VlmSeatBboxError) and soft selection errors fall through to the next
    provider. Raises VlmSeatBboxError if every provider fails."""
    errors = []
    for provider, model in provider_models:
        try:
            res = request_seat_bbox(
                rgb_bgr, names, features,
                provider=provider, model=model, known_seats=known_seats,
                timeout_s=timeout_s, max_retries=max_retries, logger=logger,
            )
        except VlmSeatBboxError as exc:
            errors.append(f"{provider}: {exc}")
            if logger is not None:
                logger.warning(f"bbox+select provider {provider} failed: {exc}; trying next.")
            continue
        if res.error:
            errors.append(f"{provider}: {res.error}")
            if logger is not None:
                logger.warning(f"bbox+select provider {provider} soft-failed: "
                               f"{res.error}; trying next.")
            continue
        return res
    raise VlmSeatBboxError("all providers failed: " + " | ".join(errors))
