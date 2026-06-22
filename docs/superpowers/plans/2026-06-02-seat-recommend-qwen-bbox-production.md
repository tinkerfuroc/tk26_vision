# Production Seat-Recommend: Qwen3-VL bbox+select Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the production seat-recommendation node from Gemini *pointing* to **Qwen3-VL bbox+select (S1)** — the winning strategy from the 2026-06-02 benchmark — with a Qwen→Gemini fallback chain, deriving the depth pixel from the chosen cushion box's center.

**Architecture:** A new provider-agnostic VLM module `kimi_api/_seat_bbox_vlm.py` ports the benchmark's S1 prompt/schema/decode into production form (`request_seat_bbox` per-provider + `request_seat_bbox_chain` for Qwen→Gemini fallback). The node `seat_recommend_bbox.py` gains a `vlm_strategy` param (`bbox_select` default, `point` for rollback), provider params, and a strategy-dispatch branch that feeds the chosen box's center into the existing snap/robust-depth/TF pipeline. Snap-to-horizontal flips to **default off** (the box already localizes the cushion). No change to `_seat_vlm.py` (the legacy point path stays intact for rollback).

**Tech Stack:** Python 3.10 (`.venv-vision-main`), rclpy, OpenAI SDK (DashScope + OpenRouter OpenAI-compatible endpoints), OpenCV, NumPy, pytest. ROS2 Humble.

**Provenance:** Benchmark + decision in `docs/superpowers/plans/2026-06-02-seat-recommend-strategy-benchmark.md` and `src/kimi_api/seat_bench/report.md`. Winner: s0_qwen 91% / s1_qwen 94% (tie at top); user chose **S1 bbox+select** for the richer per-seat output. The benchmark's reference implementation is `src/kimi_api/seat_bench/strategies/s1_bbox_select.py` + `providers.py` — **do not import from `seat_bench/`** (it is eval scaffolding); port the logic into production `kimi_api/`.

---

## Design decisions (locked with the user, 2026-06-02)

1. **Fallback chain:** Qwen S1 → Gemini S1 → fail (`status=1`). A *legitimate* "no empty seat" (`choice="none"`) from the primary provider is a valid terminal answer and does **not** trigger fallback; only hard errors (missing key, exhausted retries, parse failure) and soft selection errors (out-of-catalog choice, choice-not-in-seats, undecodable box) fall through to the next provider.
2. **Depth pixel from box:** use the chosen box's **center**; feed it into the existing 5-tier robust-depth resolver. Snap-to-horizontal becomes a ROS param **defaulting to off**.
3. **Rollback:** `vlm_strategy` param (`bbox_select` default | `point`) keeps the legacy Gemini-pointing path (`_seat_vlm.request_seat`) runnable via `-p`.

---

## Background: how the current node works (read before Task 2/3)

`src/kimi_api/kimi_api/seat_recommend_bbox.py` (763 lines), service `/seat_recommend_bbox_service`, srv `SeatRecommendBbox`. Current callback flow (`seat_recommend_bbox_callback`, ~line 443):
1. Grab synced Orbbec color+depth + intrinsics; look up TF `target_frame ← depth frame` up front.
2. Call `request_seat(...)` (`_seat_vlm.py`) → `(label, point_xy, visible_seats, vlm_elapsed)` — Gemini pointing, `[y,x]` 0-1000.
3. `point_xy is None` → fail "No empty seat".
4. Catalog guard: if `known_seats` and `label not in known_seats` → fail.
5. `vlm_px = point`; optional `_snap_to_horizontal` → `cx,cy`.
6. Synthesize `response.bbox` as a ±`point_bbox_halfsize_px` box around `cx,cy`.
7. `_resolve_depth_robust(depth, cx, cy, bbox_xyxy)` → `(uu,vv,z,tier)`; unproject; TF to `target_frame`; return.

We change steps 2, 3, 5, 6 (and add params/init). Steps 1, 4, 7 stay. `_seat_vlm.py` is untouched.

The `SeatBboxResult` produced by the new module carries: `label: str`, `box_xyxy: list|None` (pixels), `seats: list`, `provider: str`, `elapsed_s: float`, `error: str|None`.

---

## File structure

- **Create:** `src/kimi_api/kimi_api/_seat_bbox_vlm.py` — provider-agnostic bbox+select client (prompt, schema, `decode_box_xyxy`, `select_box`, `SeatBboxResult`, `VlmSeatBboxError`, `request_seat_bbox`, `request_seat_bbox_chain`).
- **Create:** `src/kimi_api/test/test_seat_bbox_vlm.py` — unit tests for the pure logic (`decode_box_xyxy`, `select_box`, `request_seat_bbox_chain` with a monkeypatched provider call).
- **Modify:** `src/kimi_api/kimi_api/seat_recommend_bbox.py` — params + init key-check (Task 2); callback strategy dispatch + box handling + snap default (Task 3).
- **Modify:** `src/tk25_ws/CLAUDE.md` *(actually `/home/tinker/tk25_ws/CLAUDE.md` is the workspace file; the vision params are documented there)* — add the new params (Task 4).

Run context for all tasks:
```bash
source /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/activate   # harmless ROS2 stderr warning — ignore
cd /home/tinker/tk25_ws/src/tk26_vision/src/kimi_api
# git repo root for commits: /home/tinker/tk25_ws/src/tk26_vision  (branch: dev — do not switch)
```

---

## Task 1: `_seat_bbox_vlm.py` — provider-agnostic bbox+select client (TDD)

**Files:**
- Create: `src/kimi_api/kimi_api/_seat_bbox_vlm.py`
- Test: `src/kimi_api/test/test_seat_bbox_vlm.py`

- [ ] **Step 1: Write the failing tests**

`src/kimi_api/test/test_seat_bbox_vlm.py`:
```python
"""Unit tests for the production bbox+select seat client (no network)."""
import pytest
from kimi_api import _seat_bbox_vlm as m
from kimi_api._seat_bbox_vlm import SeatBboxResult, VlmSeatBboxError


# --- decode_box_xyxy ---
def test_decode_box_scales_and_orders():
    assert m.decode_box_xyxy([500, 500, 250, 250], 1000, 1000) == (250, 250, 500, 500)


def test_decode_box_degenerate_is_none():
    assert m.decode_box_xyxy([100, 100, 100, 100], 1000, 1000) is None


def test_decode_box_malformed_is_none():
    assert m.decode_box_xyxy("nope", 640, 480) is None
    assert m.decode_box_xyxy([1, 2, 3], 640, 480) is None


# --- select_box ---
_SEATS = [
    {"label": "left chair", "box_2d": [100, 100, 200, 300], "occupied": False},
    {"label": "right chair", "box_2d": [600, 100, 700, 300], "occupied": True},
]


def test_select_box_valid_choice_returns_box():
    res = m.select_box({"seats": _SEATS, "choice": "left chair"}, 1000, 1000, None)
    assert res.error is None
    assert res.box_xyxy == [100, 100, 200, 300]
    assert res.label == "left chair"


def test_select_box_none_is_clean_no_error():
    res = m.select_box({"seats": _SEATS, "choice": "none"}, 1000, 1000, None)
    assert res.error is None
    assert res.box_xyxy is None
    assert res.label == "none"


def test_select_box_choice_not_in_seats_is_error():
    res = m.select_box({"seats": _SEATS, "choice": "sofa"}, 1000, 1000, None)
    assert res.error is not None
    assert res.box_xyxy is None


def test_select_box_out_of_catalog_is_error():
    res = m.select_box({"seats": _SEATS, "choice": "left chair"}, 1000, 1000,
                       ["only this seat"])
    assert res.error is not None


def test_select_box_undecodable_box_is_error():
    seats = [{"label": "x", "box_2d": [5, 5, 5, 5], "occupied": False}]
    res = m.select_box({"seats": seats, "choice": "x"}, 1000, 1000, None)
    assert res.error is not None


# --- request_seat_bbox_chain (monkeypatch the per-provider call) ---
def _fake(monkeypatch, by_provider):
    """by_provider: dict provider -> (result_or_exc)."""
    def fake_request(rgb, names, features, *, provider, model, **kw):
        v = by_provider[provider]
        if isinstance(v, Exception):
            raise v
        v.provider = provider
        return v
    monkeypatch.setattr(m, "request_seat_bbox", fake_request)


def test_chain_first_success_short_circuits(monkeypatch):
    good = SeatBboxResult(label="left", box_xyxy=[1, 2, 3, 4])
    _fake(monkeypatch, {"qwen": good, "gemini": RuntimeError("should not call")})
    res = m.request_seat_bbox_chain(None, [], [],
                                    provider_models=[("qwen", "q"), ("gemini", "g")])
    assert res.box_xyxy == [1, 2, 3, 4] and res.provider == "qwen"


def test_chain_hard_error_falls_back(monkeypatch):
    good = SeatBboxResult(label="r", box_xyxy=[5, 6, 7, 8])
    _fake(monkeypatch, {"qwen": VlmSeatBboxError("boom"), "gemini": good})
    res = m.request_seat_bbox_chain(None, [], [],
                                    provider_models=[("qwen", "q"), ("gemini", "g")])
    assert res.provider == "gemini" and res.box_xyxy == [5, 6, 7, 8]


def test_chain_soft_error_falls_back(monkeypatch):
    soft = SeatBboxResult(label="bad", error="out-of-catalog")
    good = SeatBboxResult(label="r", box_xyxy=[5, 6, 7, 8])
    _fake(monkeypatch, {"qwen": soft, "gemini": good})
    res = m.request_seat_bbox_chain(None, [], [],
                                    provider_models=[("qwen", "q"), ("gemini", "g")])
    assert res.provider == "gemini"


def test_chain_legit_none_does_not_fall_back(monkeypatch):
    none_res = SeatBboxResult(label="none", box_xyxy=None, error=None)
    _fake(monkeypatch, {"qwen": none_res, "gemini": RuntimeError("should not call")})
    res = m.request_seat_bbox_chain(None, [], [],
                                    provider_models=[("qwen", "q"), ("gemini", "g")])
    assert res.label == "none" and res.box_xyxy is None and res.provider == "qwen"


def test_chain_all_fail_raises(monkeypatch):
    _fake(monkeypatch, {"qwen": VlmSeatBboxError("a"), "gemini": VlmSeatBboxError("b")})
    with pytest.raises(VlmSeatBboxError):
        m.request_seat_bbox_chain(None, [], [],
                                  provider_models=[("qwen", "q"), ("gemini", "g")])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision/src/kimi_api && pytest test/test_seat_bbox_vlm.py -v`
Expected: collection/import error — `No module named 'kimi_api._seat_bbox_vlm'`.

- [ ] **Step 3: Implement `src/kimi_api/kimi_api/_seat_bbox_vlm.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision/src/kimi_api && pytest test/test_seat_bbox_vlm.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/kimi_api/kimi_api/_seat_bbox_vlm.py src/kimi_api/test/test_seat_bbox_vlm.py
git commit -m "feat(kimi_api): bbox+select seat VLM client with Qwen->Gemini fallback"
```

---

## Task 2: node params + strategy-aware init key check

**Files:**
- Modify: `src/kimi_api/kimi_api/seat_recommend_bbox.py`

> All edits use unique anchor strings (line numbers drift). Make them with the Edit tool.

- [ ] **Step 1: Add the new parameter declarations**

Find this anchor (the end of the declare block):
```python
        self.declare_parameter('fewshot_enabled', False)
        self.declare_parameter('max_fewshots', 3)
```
Replace with:
```python
        self.declare_parameter('fewshot_enabled', False)
        self.declare_parameter('max_fewshots', 3)
        # --- VLM strategy / provider (2026-06-02: switch to Qwen bbox+select) ---
        # 'bbox_select' (default) = one structured call returns a cushion box +
        # occupancy per seat + the chosen empty seat (benchmark winner, S1).
        # 'point' = legacy Gemini pointing via _seat_vlm.request_seat (rollback).
        self.declare_parameter('vlm_strategy', 'bbox_select')
        # Primary provider for bbox_select, then fallback. 'qwen' = DashScope
        # qwen3-vl-plus (benchmark best); 'gemini' = OpenRouter gemini-2.5-pro.
        self.declare_parameter('vlm_provider', 'qwen')
        self.declare_parameter('vlm_fallback_provider', 'gemini')  # '' to disable
        self.declare_parameter('bbox_model_qwen', 'qwen3-vl-plus')
        self.declare_parameter('bbox_model_gemini', 'google/gemini-2.5-pro')
```

- [ ] **Step 2: Flip the snap default to off**

Find this anchor:
```python
        self.declare_parameter('snap_enabled', True)
```
Replace with:
```python
        # Default OFF as of 2026-06-02: bbox_select localizes the cushion via the
        # chosen box, so the box centre is already on the seat — snap-to-horizontal
        # adds latency and can wander. Re-enable with -p snap_enabled:=true for the
        # legacy point path or noisy depth.
        self.declare_parameter('snap_enabled', False)
```

- [ ] **Step 3: Read the new parameters**

Find this anchor (end of the param-read block):
```python
        self.max_fewshots = int(
            self.get_parameter('max_fewshots').get_parameter_value().integer_value
        )
```
Replace with:
```python
        self.max_fewshots = int(
            self.get_parameter('max_fewshots').get_parameter_value().integer_value
        )
        self.vlm_strategy = (
            self.get_parameter('vlm_strategy').get_parameter_value().string_value
        )
        self.vlm_provider = (
            self.get_parameter('vlm_provider').get_parameter_value().string_value
        )
        self.vlm_fallback_provider = (
            self.get_parameter('vlm_fallback_provider').get_parameter_value().string_value
        )
        self.bbox_model_qwen = (
            self.get_parameter('bbox_model_qwen').get_parameter_value().string_value
        )
        self.bbox_model_gemini = (
            self.get_parameter('bbox_model_gemini').get_parameter_value().string_value
        )
```

- [ ] **Step 4: Add the import + strategy-aware key check**

Find this anchor (top-of-file import):
```python
from ._env import load_env, require_api_key
```
Replace with:
```python
from ._env import load_env, require_api_key, require_dashscope_api_key
from ._seat_bbox_vlm import request_seat_bbox_chain, VlmSeatBboxError
```

Find this anchor (the existing init key check):
```python
        # Fail-fast on missing key — matches feature_recognition pattern so
        # the T1 negative test (no .env) surfaces at node init.
        require_api_key()
```
Replace with:
```python
        # Fail-fast on the API key(s) the configured strategy/providers need —
        # matches feature_recognition pattern so the T1 negative test (no .env)
        # surfaces at node init. bbox_select builds an ordered provider chain
        # (primary required, fallback dropped-with-warning if its key is absent);
        # the legacy point path needs OpenRouter.
        if self.vlm_strategy == 'bbox_select':
            self._provider_models = self._resolve_provider_chain()
        else:
            require_api_key()
            self._provider_models = []
```

- [ ] **Step 5: Add the `_model_for` + `_resolve_provider_chain` helpers**

Find this anchor (an existing method header to anchor before):
```python
    def camera_info_orbbec_callback(self, info):
```
Insert ABOVE it:
```python
    def _model_for(self, provider: str) -> str:
        return self.bbox_model_qwen if provider == 'qwen' else self.bbox_model_gemini

    def _has_provider_key(self, provider: str) -> bool:
        try:
            (require_dashscope_api_key if provider == 'qwen' else require_api_key)()
            return True
        except RuntimeError:
            return False

    def _resolve_provider_chain(self) -> list:
        """Ordered (provider, model) chain for bbox_select. Primary key is
        required (raises at init if missing); a fallback whose key is absent is
        dropped with a warning."""
        primary = self.vlm_provider
        if not self._has_provider_key(primary):
            # Re-call to raise the descriptive RuntimeError for the missing key.
            (require_dashscope_api_key if primary == 'qwen' else require_api_key)()
        chain = [(primary, self._model_for(primary))]
        fb = self.vlm_fallback_provider
        if fb and fb != primary:
            if self._has_provider_key(fb):
                chain.append((fb, self._model_for(fb)))
            else:
                self.get_logger().warn(
                    f'Fallback provider {fb!r} key missing; fallback disabled.'
                )
        self.get_logger().info(
            f'bbox+select provider chain: {[p for p, _ in chain]}'
        )
        return chain

    def camera_info_orbbec_callback(self, info):
```

- [ ] **Step 6: Verify the node still imports and constructs (no cameras needed)**

Run:
```bash
source /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/activate
cd /home/tinker/tk25_ws/src/tk26_vision/src/kimi_api
python -c "
import rclpy
from kimi_api.seat_recommend_bbox import SeatRecommendBboxService
rclpy.init()
n = SeatRecommendBboxService()
print('strategy=', n.vlm_strategy, 'chain=', getattr(n, '_provider_models', None))
n.destroy_node(); rclpy.shutdown()
print('OK')
"
```
Expected: prints `strategy= bbox_select chain= [('qwen', 'qwen3-vl-plus'), ('gemini', 'google/gemini-2.5-pro')]` (both keys present in `.env`) then `OK`. If a key is missing the chain shortens or it raises a descriptive RuntimeError naming the missing key — that is correct fail-fast behavior.

- [ ] **Step 7: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/kimi_api/kimi_api/seat_recommend_bbox.py
git commit -m "feat(seat_recommend_bbox): vlm_strategy/provider params + provider-chain init"
```

---

## Task 3: node callback — strategy dispatch + box-driven depth pixel

**Files:**
- Modify: `src/kimi_api/kimi_api/seat_recommend_bbox.py`

> Depends on Task 2. All edits anchor-based.

- [ ] **Step 1: Replace the few-shot load + `request_seat` call with strategy dispatch**

Find this anchor (few-shot block + the point VLM call + its error handling):
```python
        # 2. Gemini call — returns a pointing pixel + short label.
        fewshots = None
        if self.fewshot_enabled:
            fewshots = load_fewshots(self.max_fewshots, logger=self.get_logger())
            self.get_logger().info(
                f'Few-shot enabled: applying {len(fewshots)} example(s) '
                f'(max_fewshots={self.max_fewshots}).'
            )
        try:
            label, point_xy, visible_seats, vlm_elapsed = request_seat(
                color_img,
                request.names,
                request.features,
                model=self.llm_model,
                timeout_s=self.vlm_timeout_s,
                max_retries=self.vlm_max_retries,
                logger=self.get_logger(),
                fewshots=fewshots,
                known_seats=known_seats,
            )
        except VlmSeatError as exc:
            return self._fail(response, f'VLM unavailable: {exc}')
```
Replace with:
```python
        # 2. VLM call. bbox_select (default) returns a cushion box + chosen seat
        # across a Qwen->Gemini provider chain; point is the legacy Gemini path.
        # `box_px` is the chosen cushion box in pixels (None for the point path
        # or a "none" result); `point_xy` is the legacy pointing pixel.
        box_px = None
        provider_used = ''
        fewshots = None
        if self.vlm_strategy == 'bbox_select':
            try:
                sel = request_seat_bbox_chain(
                    color_img,
                    request.names,
                    request.features,
                    provider_models=self._provider_models,
                    known_seats=known_seats or None,
                    timeout_s=self.vlm_timeout_s,
                    max_retries=self.vlm_max_retries,
                    logger=self.get_logger(),
                )
            except VlmSeatBboxError as exc:
                return self._fail(response, f'VLM bbox+select unavailable: {exc}')
            label = sel.label
            visible_seats = sel.seats
            vlm_elapsed = sel.elapsed_s
            provider_used = sel.provider
            box_px = tuple(sel.box_xyxy) if sel.box_xyxy else None
            point_xy = None
        else:
            if self.fewshot_enabled:
                fewshots = load_fewshots(self.max_fewshots, logger=self.get_logger())
                self.get_logger().info(
                    f'Few-shot enabled: applying {len(fewshots)} example(s) '
                    f'(max_fewshots={self.max_fewshots}).'
                )
            try:
                label, point_xy, visible_seats, vlm_elapsed = request_seat(
                    color_img,
                    request.names,
                    request.features,
                    model=self.llm_model,
                    timeout_s=self.vlm_timeout_s,
                    max_retries=self.vlm_max_retries,
                    logger=self.get_logger(),
                    fewshots=fewshots,
                    known_seats=known_seats,
                )
            except VlmSeatError as exc:
                return self._fail(response, f'VLM unavailable: {exc}')
            provider_used = 'gemini'
```

- [ ] **Step 2: Record strategy/provider in the log context**

Find this anchor:
```python
            'fewshot_enabled': bool(self.fewshot_enabled),
            'n_fewshots': int(len(fewshots)) if fewshots is not None else 0,
        }
```
Replace with:
```python
            'fewshot_enabled': bool(self.fewshot_enabled),
            'n_fewshots': int(len(fewshots)) if fewshots is not None else 0,
            'vlm_strategy': self.vlm_strategy,
            'vlm_provider': provider_used,
        }
```

- [ ] **Step 3: Make the no-empty-seat check cover both strategies**

Find this anchor:
```python
        if point_xy is None:
            log_extras['event'] = 'no_empty_seat'
            _write_log(None)
            return self._fail(response, 'No empty seat detected by VLM.')
```
Replace with:
```python
        # No empty seat: point path yields point_xy is None; bbox_select yields a
        # "none" choice (box_px is None) with no error.
        if point_xy is None and box_px is None:
            log_extras['event'] = 'no_empty_seat'
            _write_log(None)
            return self._fail(response, 'No empty seat detected by VLM.')
```

- [ ] **Step 4: Seed the working pixel from the box centre (bbox_select) or the point**

Find this anchor:
```python
        vlm_px = (int(point_xy[0]), int(point_xy[1]))
        log_extras['vlm_point'] = [vlm_px[0], vlm_px[1]]
```
Replace with:
```python
        # Working pixel: chosen box centre for bbox_select, else the VLM point.
        if box_px is not None:
            bx0, by0, bx1, by1 = box_px
            vlm_px = ((bx0 + bx1) // 2, (by0 + by1) // 2)
            log_extras['vlm_box'] = [int(bx0), int(by0), int(bx1), int(by1)]
        else:
            vlm_px = (int(point_xy[0]), int(point_xy[1]))
        log_extras['vlm_point'] = [vlm_px[0], vlm_px[1]]
```

- [ ] **Step 5: Use the real box for the response (bbox_select) instead of synthesizing**

Find this anchor:
```python
        # Synthesize a small bbox around the (possibly snapped) point for the
        # response's bbox field (used by callers for overlay and pan-tilt aiming).
        h_img, w_img = color_img.shape[:2]
        r = max(1, int(self.point_bbox_halfsize_px))
        bbox_xyxy = (
            max(0, cx - r),
            max(0, cy - r),
            min(w_img - 1, cx + r),
            min(h_img - 1, cy + r),
        )
```
Replace with:
```python
        # bbox field: use the VLM's actual cushion box (bbox_select); else
        # synthesize a small box around the (possibly snapped) point.
        h_img, w_img = color_img.shape[:2]
        if box_px is not None:
            bbox_xyxy = (
                max(0, min(int(box_px[0]), w_img - 1)),
                max(0, min(int(box_px[1]), h_img - 1)),
                max(0, min(int(box_px[2]), w_img - 1)),
                max(0, min(int(box_px[3]), h_img - 1)),
            )
        else:
            r = max(1, int(self.point_bbox_halfsize_px))
            bbox_xyxy = (
                max(0, cx - r),
                max(0, cy - r),
                min(w_img - 1, cx + r),
                min(h_img - 1, cy + r),
            )
```

- [ ] **Step 6: Verify the node imports + constructs after the callback edits**

Run:
```bash
source /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/activate
cd /home/tinker/tk25_ws/src/tk26_vision/src/kimi_api
python -c "
import rclpy
from kimi_api.seat_recommend_bbox import SeatRecommendBboxService
rclpy.init(); n = SeatRecommendBboxService(); n.destroy_node(); rclpy.shutdown()
print('construct OK')
"
python -m pyflakes kimi_api/seat_recommend_bbox.py || echo "(pyflakes not installed — skip)"
```
Expected: `construct OK`. pyflakes (if present) reports no undefined names. A warning that `request_seat`/`VlmSeatError`/`load_fewshots` are still imported is fine — the point path still uses them.

- [ ] **Step 7: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/kimi_api/kimi_api/seat_recommend_bbox.py
git commit -m "feat(seat_recommend_bbox): dispatch bbox_select vs point; box-centre depth pixel"
```

---

## Task 4: build, smoke-test, and document

**Files:**
- Modify: `/home/tinker/tk25_ws/CLAUDE.md` (vision param docs)

- [ ] **Step 1: Build the package with the venv-aware wrapper**

Run:
```bash
cd /home/tinker/tk25_ws
./src/tk26_vision/scripts/build.sh --packages-select kimi_api 2>&1 | tail -15
```
Expected: `Summary: 1 package finished`. If it errors on stale symlinks: `rm -rf build/kimi_api install/kimi_api` and rebuild.

- [ ] **Step 2: Run the kimi_api unit tests (lint + new module)**

Run:
```bash
source /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/activate
cd /home/tinker/tk25_ws/src/tk26_vision/src/kimi_api
pytest test/test_seat_bbox_vlm.py -q
```
Expected: 13 passed.

- [ ] **Step 3: Live single-call smoke against a benchmark scene (no cameras, exercises the real provider chain)**

This drives the production chain end-to-end on a saved image, confirming Qwen is hit first and returns a real box:
```bash
source /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/activate
cd /home/tinker/tk25_ws/src/tk26_vision/src/kimi_api
python -c "
import cv2, json
from kimi_api._seat_bbox_vlm import request_seat_bbox_chain
img = cv2.imread('seat_bench/dataset/scene_017.jpg')
req = json.load(open('seat_bench/dataset/scene_017.req.json'))
res = request_seat_bbox_chain(img, req['names'], req['features'],
    provider_models=[('qwen','qwen3-vl-plus'),('gemini','google/gemini-2.5-pro')],
    known_seats=req.get('known_seats') or None, logger=None)
print('provider=', res.provider, 'choice=', res.label, 'box=', res.box_xyxy, 'seats=', len(res.seats), 'err=', res.error)
"
```
Expected: `provider= qwen choice= <an empty seat label> box= [x1,y1,x2,y2] seats= 3 err= None`. (scene_017: left occupied, middle+right empty.) A transient 429 → retry once.

- [ ] **Step 4: Node-level service smoke IF cameras are up (operator step; skip if no Orbbec)**

With the Orbbec camera running (see `CAMERA_BRINGUP.md`) and the node started (`ros2 run kimi_api seat_recommend_bbox`):
```bash
ros2 service call /seat_recommend_bbox_service tinker_vision_msgs_26/srv/SeatRecommendBbox \
  "{camera: 'orbbec', names: [], features: [], target_frame: 'base_link', known_seats: []}"
```
Expected on a scene with an empty seat: `status: 0`, a non-empty `recommendation`, a `bbox` matching the seat, and a `centroid` in `base_link`. `status: 1` with `error_msg` "No empty seat..." is the correct empty-scene response, not a failure. If no cameras are available, record this step as skipped (covered by the existing T2 live tier in `scripts/tests/`).

- [ ] **Step 5: Document the new params in `/home/tinker/tk25_ws/CLAUDE.md`**

Find this anchor in `/home/tinker/tk25_ws/CLAUDE.md` (the kimi_api config bullet):
```
- `kimi_api/*`: `llm_model`, `detection_service`, `log_prompts`
```
Replace with:
```
- `kimi_api/*`: `llm_model`, `detection_service`, `log_prompts`
- `kimi_api/seat_recommend_bbox`: `vlm_strategy` (default `'bbox_select'` — one structured Qwen3-VL call returns a cushion box + occupancy per seat + the chosen empty seat, the 2026-06-02 benchmark winner; set `'point'` for the legacy Gemini-pointing path), `vlm_provider` (default `'qwen'`) + `vlm_fallback_provider` (default `'gemini'`, `''` to disable) define the bbox_select fallback chain, `bbox_model_qwen` (default `'qwen3-vl-plus'`) / `bbox_model_gemini` (default `'google/gemini-2.5-pro'`). `snap_enabled` now defaults to **`false`** (the chosen box already localizes the cushion; the box centre seeds the robust-depth resolver). bbox_select needs `DASHSCOPE_API_KEY` (or the typo'd `DASHCOPE_API_KEY`) for Qwen and `OPENROUTER_API_KEY` for the Gemini fallback; the point path needs only `OPENROUTER_API_KEY`. Benchmark + rationale: `src/tk26_vision/src/kimi_api/seat_bench/report.md`.
```

- [ ] **Step 6: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add /home/tinker/tk25_ws/CLAUDE.md
git commit -m "docs(vision): document seat_recommend_bbox vlm_strategy/provider params"
```
(Note: `CLAUDE.md` is in the outer `tk25_ws` repo root, which is a different git working tree than `tk26_vision`. If `git add` reports the file is outside the repo, commit it from its own repo: `cd /home/tinker/tk25_ws && git add CLAUDE.md && git commit -m "..."`. If `tk25_ws` is not a git repo, leave the doc edit uncommitted and note it in the task report.)

---

## Self-review notes (plan author)

- **Spec coverage:** Qwen S1 primary (Task 1 `request_seat_bbox` qwen + Task 2 `vlm_provider='qwen'`) ✓; Qwen→Gemini→fail fallback (Task 1 `request_seat_bbox_chain` + tests) ✓; box-centre depth pixel (Task 3 Step 4) ✓; snap param default off (Task 2 Step 2) ✓; `vlm_strategy` rollback to point path (Task 2 + Task 3 dispatch keeps `request_seat`) ✓; legit "none" does not fall back (Task 1 `select_box` + chain + `test_chain_legit_none_does_not_fall_back`) ✓; response.bbox uses the real box (Task 3 Step 5) ✓; docs (Task 4 Step 5) ✓.
- **Type consistency:** `SeatBboxResult` fields (`label`, `box_xyxy`, `seats`, `provider`, `elapsed_s`, `error`) are produced by `select_box`/`request_seat_bbox` and consumed identically in the node (`sel.label`, `sel.seats`, `sel.elapsed_s`, `sel.provider`, `sel.box_xyxy`). `request_seat_bbox_chain(provider_models=[(provider, model), ...])` signature matches the node's `self._provider_models` built by `_resolve_provider_chain`. `box_px` is a tuple-or-None throughout the callback.
- **Placeholder scan:** none — every code/command step is concrete.
- **Risk notes for the executor:** (a) the node is 763 lines; anchors are unique strings, but re-read the file region before each Edit to confirm the anchor still matches after prior edits. (b) `_seat_vlm.request_seat` / `VlmSeatError` / `load_fewshots` imports stay — do not remove them (point path). (c) `CLAUDE.md` lives in the outer `tk25_ws` tree (see Task 4 Step 6 fallback). (d) The legacy `llm_model`/`vlm_timeout_s`/`vlm_max_retries` params still feed both paths; the bbox path reuses `vlm_timeout_s`/`vlm_max_retries`.
```
