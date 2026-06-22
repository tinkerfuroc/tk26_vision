# Waving-detection VLM fallback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When `detect_waving_persons` finds fewer wavers than the request's new
`min_waving_persons` threshold, augment the MediaPipe result with a Qwen3-VL→Gemini
VLM pass that recovers the missed wavers and returns depth-derived 3D centroids.

**Architecture:** A new pure VLM client module (`_waving_vlm.py`) mirrors the
*control flow* of `kimi_api/_seat_bbox_vlm.py` (single-call → provider chain,
strict-schema → json_object fallback, errors-only fallthrough) but uses the
package's existing **kimi_api-free** convention (`_vlm_common.encode_data_url`,
`os.environ` keys, base-URL constants). A new pure geometry module
(`_waving_geometry.py`) turns VLM boxes into 3D centroids by reusing an
overlapping YOLO mask or falling back to box-center depth over the back-projected
XYZ grid the server already computes. The server (`waving_person_server.py`)
gains a non-fatal provider-chain resolver, a trigger after the MediaPipe loop,
and an augment step that dedups and appends VLM wavers before the existing sort /
overlay / log / transform stages.

**Tech Stack:** ROS2 Humble (`rclpy`), `ultralytics` YOLO11m-seg, MediaPipe Pose,
OpenAI-compatible SDK (`openai`), DashScope Qwen3-VL + OpenRouter Gemini, numpy,
OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-06-02-waving-vlm-fallback-design.md`

**Conventions (read before starting):**
- `ament_flake8` + `ament_pep257` lint every file in the package. Keep lines
  ≤ 99 cols; every module/class/public function needs a docstring. Broad
  `except Exception` uses `# noqa: BLE001`, matching existing code.
- `test_copyright.py` is `@pytest.mark.skip` — no copyright headers required.
- Tests import `from tk_vision_specialized.<module> import ...` and monkeypatch
  `openai.OpenAI` per-test. No network in unit tests.
- Build vision packages with the wrapper (patches venv shebangs):
  `./src/tk26_vision/scripts/build.sh --packages-select <pkg>`. Interface
  packages can use plain colcon.
- Run package tests from the workspace root:
  `python3 -m pytest src/tk26_vision/src/tk_vision_specialized/test/<file> -v`
  (the shared venv `.venv-vision-main` has pytest, openai, numpy, cv2).

---

## Task 1: Add `min_waving_persons` to the `DetectWaving` service

**Files:**
- Modify: `src/tk26_vision/src/tinker_vision_msgs_26/srv/DetectWaving.srv`

- [ ] **Step 1: Add the request field**

Edit `DetectWaving.srv` so the request block reads exactly:

```
float32 threshold_meters
string target_frame
int32 min_waving_persons
---
int32 status
string error_msg
geometry_msgs/PointStamped[] waving_persons

sensor_msgs/Image rgb_image
sensor_msgs/Image depth_image
sensor_msgs/Image[] segments
```

(Only the new `int32 min_waving_persons` line is added, directly under
`string target_frame`. The response block is unchanged.)

- [ ] **Step 2: Rebuild the interface package**

Run:
```bash
cd /home/tinker/tk25_ws
colcon build --packages-select tinker_vision_msgs_26
source install/setup.bash
```
Expected: build finishes `Finished <<< tinker_vision_msgs_26` with no errors.

- [ ] **Step 3: Verify the generated interface carries the field**

Run:
```bash
ros2 interface show tinker_vision_msgs_26/srv/DetectWaving
```
Expected output includes the line `int32 min_waving_persons` in the request
section (above the `---`).

- [ ] **Step 4: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tinker_vision_msgs_26/srv/DetectWaving.srv
git commit -m "feat(msgs): add min_waving_persons to DetectWaving.srv

Caller-supplied threshold; <=0 disables the VLM fallback (back-compat)."
```

---

## Task 2: `_waving_vlm.py` — result type + pure decoders (`decode_box_xyxy`, `select_boxes`)

**Files:**
- Create: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py`
- Create: `src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_waving_vlm.py`:

```python
"""Unit tests for _waving_vlm.py — pure decoders, key resolution, and the
provider chain. No network: the OpenAI client is monkeypatched per-test."""

from __future__ import annotations

from types import SimpleNamespace

import json

import pytest

from tk_vision_specialized._waving_vlm import (
    WavingVlmResult,
    WavingVlmError,
    decode_box_xyxy,
    select_boxes,
)


def test_decode_box_xyxy_scales_and_clamps():
    # 0-1000 normalized -> pixels on a 1000x500 frame; x by width, y by height.
    assert decode_box_xyxy([100, 200, 300, 400], 1000, 500) == (100, 100, 300, 200)
    # out-of-range clamps to image bounds (w-1, h-1).
    assert decode_box_xyxy([0, 0, 1000, 1000], 1000, 500) == (0, 0, 999, 499)


def test_decode_box_xyxy_swaps_inverted_corners():
    assert decode_box_xyxy([300, 400, 100, 200], 1000, 500) == (100, 100, 300, 200)


def test_decode_box_xyxy_rejects_degenerate_and_malformed():
    assert decode_box_xyxy([500, 500, 500, 500], 1000, 500) is None  # zero area
    assert decode_box_xyxy([10], 1000, 500) is None                  # wrong length
    assert decode_box_xyxy('nope', 1000, 500) is None                # wrong type


def test_select_boxes_keeps_only_waving_with_decodable_box():
    parsed = {'persons': [
        {'box_2d': [100, 200, 300, 400], 'waving': True},
        {'box_2d': [0, 0, 100, 100], 'waving': False},     # dropped: not waving
        {'box_2d': [500, 500, 500, 500], 'waving': True},  # dropped: zero area
    ]}
    res = select_boxes(parsed, 1000, 500)
    assert isinstance(res, WavingVlmResult)
    assert res.boxes == [(100, 100, 300, 200)]
    assert res.error is None


def test_select_boxes_clean_empty_when_no_wavers():
    res = select_boxes({'persons': []}, 1000, 500)
    assert res.boxes == []
    assert res.error is None  # clean empty is terminal, not an error
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
python3 -m pytest src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named
'tk_vision_specialized._waving_vlm'` (module not created yet).

- [ ] **Step 3: Create the module with constants, result type, and decoders**

Create `tk_vision_specialized/_waving_vlm.py`:

```python
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
from typing import Optional, Sequence

import numpy as np
from dotenv import load_dotenv

from ._vlm_common import encode_data_url


_QWEN_DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
_GEMINI_DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'
_QWEN_KEY_NAMES = ('DASHCOPE_API_KEY', 'DASHSCOPE_API_KEY')

# Load `.env` once at import so per-call key resolution reads os.environ only —
# keeps pytest monkeypatch behaviour predictable (mirrors vlm_match_client.py).
load_dotenv(override=False)

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


def _strip_fences(text: str) -> str:
    """Drop ```json ... ``` fences some models emit despite instructions."""
    return _FENCE_RE.sub('', text).strip() if '```' in text else text


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python3 -m pytest src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py \
        src/tk_vision_specialized/test/test_waving_vlm.py
git commit -m "feat(waving): _waving_vlm result type + pure box decoders"
```

---

## Task 3: `_waving_vlm.py` — key resolution + non-fatal provider-chain builder

**Files:**
- Modify: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py`
- Modify: `src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_waving_vlm.py`:

```python
from tk_vision_specialized._waving_vlm import (  # noqa: E402
    has_provider_key,
    build_provider_models,
)


def test_has_provider_key_qwen_accepts_either_spelling(monkeypatch):
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    assert has_provider_key('qwen') is False
    monkeypatch.setenv('DASHCOPE_API_KEY', 'legacy')   # typo'd spelling
    assert has_provider_key('qwen') is True


def test_has_provider_key_gemini_uses_openrouter(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    assert has_provider_key('gemini') is False
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    assert has_provider_key('gemini') is True


def test_build_provider_models_primary_plus_fallback():
    chain = build_provider_models(
        'qwen', 'gemini',
        has_key=lambda p: True,
        model_for=lambda p: f'model-{p}',
    )
    assert chain == [('qwen', 'model-qwen'), ('gemini', 'model-gemini')]


def test_build_provider_models_drops_keyless_providers():
    # Primary key missing -> primary dropped; fallback present -> kept.
    chain = build_provider_models(
        'qwen', 'gemini',
        has_key=lambda p: p == 'gemini',
        model_for=lambda p: f'model-{p}',
    )
    assert chain == [('gemini', 'model-gemini')]


def test_build_provider_models_empty_when_no_keys():
    chain = build_provider_models(
        'qwen', 'gemini',
        has_key=lambda p: False,
        model_for=lambda p: f'model-{p}',
    )
    assert chain == []


def test_build_provider_models_blank_fallback_disabled():
    chain = build_provider_models(
        'qwen', '',
        has_key=lambda p: True,
        model_for=lambda p: f'model-{p}',
    )
    assert chain == [('qwen', 'model-qwen')]
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
python3 -m pytest src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py -k "provider_key or provider_models" -v
```
Expected: FAIL — `ImportError: cannot import name 'has_provider_key'`.

- [ ] **Step 3: Implement the resolver + builder**

Add to `_waving_vlm.py` (after `decode_box_xyxy` / `select_boxes`):

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
python3 -m pytest src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py -v
```
Expected: all tests (Task 2 + Task 3) PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py \
        src/tk_vision_specialized/test/test_waving_vlm.py
git commit -m "feat(waving): non-fatal provider-chain resolution for waving VLM"
```

---

## Task 4: `_waving_vlm.py` — single-provider `request_waving_persons`

**Files:**
- Modify: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py`
- Modify: `src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_waving_vlm.py`:

```python
import numpy as np  # noqa: E402
import openai  # noqa: E402

from tk_vision_specialized._waving_vlm import request_waving_persons  # noqa: E402


def _completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _make_fake_openai(script):
    """Build a fake openai.OpenAI whose .create runs `script(kwargs)`.

    `script` returns the response content string, or raises to simulate an API
    error. Records constructor kwargs and create kwargs on the class for asserts.
    """
    class _Fake:
        last_init = None
        calls = []

        def __init__(self, **kw):
            _Fake.last_init = kw

        def with_options(self, **_kw):
            return self

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kw):
            _Fake.calls.append(kw)
            return _completion(script(kw))

        def close(self):
            pass

    return _Fake


def _img():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_request_waving_persons_returns_boxes(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'k')
    payload = json.dumps({'persons': [{'box_2d': [100, 100, 200, 300],
                                       'waving': True}]})
    fake = _make_fake_openai(lambda kw: payload)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_waving_persons(_img(), provider='qwen', model='qwen3-vl-plus')

    # box_2d [100,100,200,300] on 640x480: x*640/1000, y*480/1000.
    assert res.boxes == [(64, 48, 128, 144)]
    assert res.provider == 'qwen'
    assert res.error is None
    assert fake.last_init['base_url'] == _QWEN_DEFAULT_BASE_URL_VALUE
    assert fake.last_init['api_key'] == 'k'


def test_request_waving_persons_missing_key_raises(monkeypatch):
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    with pytest.raises(WavingVlmError, match='key'):
        request_waving_persons(_img(), provider='qwen', model='m')


def test_request_waving_persons_falls_back_to_json_object_on_schema_reject(
        monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    payload = json.dumps({'persons': []})

    def script(kw):
        rf = kw.get('response_format') or {}
        if rf.get('type') == 'json_schema':
            raise RuntimeError('response_format json_schema not supported')
        return payload

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_waving_persons(_img(), provider='gemini', model='g',
                                 max_retries=3)
    assert res.boxes == []
    assert res.error is None
    # First attempt strict, retried attempt loose.
    assert fake.calls[0]['response_format']['type'] == 'json_schema'
    assert fake.calls[-1]['response_format']['type'] == 'json_object'


# Expose the constant value for the assertion above without re-importing.
from tk_vision_specialized._waving_vlm import (  # noqa: E402
    _QWEN_DEFAULT_BASE_URL as _QWEN_DEFAULT_BASE_URL_VALUE,
)
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
python3 -m pytest src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py -k request_waving_persons -v
```
Expected: FAIL — `ImportError: cannot import name 'request_waving_persons'`.

- [ ] **Step 3: Implement `request_waving_persons`**

Add to `_waving_vlm.py`:

```python
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

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=b_url)
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
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
python3 -m pytest src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py \
        src/tk_vision_specialized/test/test_waving_vlm.py
git commit -m "feat(waving): single-provider request_waving_persons VLM call"
```

---

## Task 5: `_waving_vlm.py` — provider-chain `request_waving_persons_chain`

**Files:**
- Modify: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py`
- Modify: `src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_waving_vlm.py`:

```python
from tk_vision_specialized._waving_vlm import (  # noqa: E402
    request_waving_persons_chain,
)


def test_chain_falls_through_on_error(monkeypatch):
    calls = []

    def fake(rgb, *, provider, model, **kw):
        calls.append(provider)
        if provider == 'qwen':
            raise WavingVlmError('qwen down')
        return WavingVlmResult(boxes=[(1, 2, 3, 4)], provider='gemini')

    monkeypatch.setattr(
        'tk_vision_specialized._waving_vlm.request_waving_persons', fake)
    res = request_waving_persons_chain(
        _img(), provider_models=[('qwen', 'q'), ('gemini', 'g')])
    assert calls == ['qwen', 'gemini']
    assert res.boxes == [(1, 2, 3, 4)]
    assert res.provider == 'gemini'


def test_chain_clean_empty_does_not_fall_through(monkeypatch):
    calls = []

    def fake(rgb, *, provider, model, **kw):
        calls.append(provider)
        return WavingVlmResult(boxes=[], provider=provider)  # clean empty

    monkeypatch.setattr(
        'tk_vision_specialized._waving_vlm.request_waving_persons', fake)
    res = request_waving_persons_chain(
        _img(), provider_models=[('qwen', 'q'), ('gemini', 'g')])
    assert calls == ['qwen']          # gemini never tried
    assert res.boxes == []
    assert res.provider == 'qwen'


def test_chain_all_fail_raises(monkeypatch):
    def fake(rgb, *, provider, model, **kw):
        raise WavingVlmError(f'{provider} down')

    monkeypatch.setattr(
        'tk_vision_specialized._waving_vlm.request_waving_persons', fake)
    with pytest.raises(WavingVlmError, match='all providers failed'):
        request_waving_persons_chain(
            _img(), provider_models=[('qwen', 'q'), ('gemini', 'g')])
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
python3 -m pytest src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py -k chain -v
```
Expected: FAIL — `ImportError: cannot import name 'request_waving_persons_chain'`.

- [ ] **Step 3: Implement the chain**

Add to `_waving_vlm.py`:

```python
def request_waving_persons_chain(rgb_bgr: np.ndarray, *,
                                 provider_models: Sequence[tuple],
                                 timeout_s: float = 20.0, max_retries: int = 3,
                                 logger=None) -> WavingVlmResult:
    """Try (provider, model) pairs in order; return the first CLEAN result.

    Errors-only fallthrough: a hard WavingVlmError or a soft .error falls
    through to the next provider, but a clean result (any boxes, or a legitimate
    empty list with no error) is returned immediately. Raises WavingVlmError if
    every provider fails.
    """
    errors = []
    for provider, model in provider_models:
        try:
            res = request_waving_persons(
                rgb_bgr, provider=provider, model=model,
                timeout_s=timeout_s, max_retries=max_retries, logger=logger)
        except WavingVlmError as exc:
            errors.append(f'{provider}: {exc}')
            if logger is not None:
                logger.warning(f'waving VLM provider {provider} failed: {exc}; '
                               f'trying next.')
            continue
        if res.error:
            errors.append(f'{provider}: {res.error}')
            if logger is not None:
                logger.warning(f'waving VLM provider {provider} soft-failed: '
                               f'{res.error}; trying next.')
            continue
        return res
    raise WavingVlmError('all providers failed: ' + ' | '.join(errors))
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
python3 -m pytest src/tk26_vision/src/tk_vision_specialized/test/test_waving_vlm.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Run flake8 on the new module**

Run:
```bash
python3 -m flake8 --max-line-length=99 \
  src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py
```
Expected: no output (clean). Fix any reported line.

- [ ] **Step 6: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py \
        src/tk_vision_specialized/test/test_waving_vlm.py
git commit -m "feat(waving): qwen->gemini provider chain with errors-only fallthrough"
```

---

## Task 6: `_waving_geometry.py` — `box_iou`, `is_duplicate_box`, `centroid_from_box`

**Files:**
- Create: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/_waving_geometry.py`
- Create: `src/tk26_vision/src/tk_vision_specialized/test/test_waving_geometry.py`

These are pure numpy helpers (no ROS), so they're fully unit-testable. The server
imports them in Task 7.

- [ ] **Step 1: Write the failing tests**

Create `test/test_waving_geometry.py`:

```python
"""Unit tests for _waving_geometry.py — pure box/depth helpers."""

from __future__ import annotations

import numpy as np

from tk_vision_specialized._waving_geometry import (
    box_iou,
    is_duplicate_box,
    centroid_from_box,
)


def test_box_iou_identical_is_one():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_box_iou_disjoint_is_zero():
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_box_iou_half_overlap():
    # 10x10 boxes sharing a 5x10 strip -> inter=50, union=150 -> 1/3.
    iou = box_iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert abs(iou - (50.0 / 150.0)) < 1e-6


def test_is_duplicate_box_by_iou():
    existing = [(0, 0, 10, 10)]
    assert is_duplicate_box((1, 1, 11, 11), existing, iou_thresh=0.3) is True
    assert is_duplicate_box((40, 40, 50, 50), existing, iou_thresh=0.3) is False


def test_is_duplicate_box_by_center_inside():
    # Low IoU but the new box's center sits inside an existing box -> duplicate.
    existing = [(0, 0, 100, 100)]
    assert is_duplicate_box((40, 40, 60, 60), existing, iou_thresh=0.99) is True


def _grid_with_depth(h, w, z_value, valid_region=None):
    """Build a (points, validmask) pair where points[...,2]=z_value.

    XY are arbitrary linear ramps; only Z (and validity) matter for the asserts.
    valid_region = (y0, y1, x0, x1) marks the only valid pixels (else all valid).
    """
    xs = np.tile(np.arange(w, dtype=float), (h, 1))
    ys = np.tile(np.arange(h, dtype=float)[:, None], (1, w))
    zs = np.full((h, w), float(z_value))
    points = np.stack([xs, ys, zs], axis=2)
    validmask = np.zeros((h, w), dtype=bool)
    if valid_region is None:
        validmask[:] = True
    else:
        y0, y1, x0, x1 = valid_region
        validmask[y0:y1, x0:x1] = True
    return points, validmask


def test_centroid_from_box_reuses_overlapping_mask():
    points, validmask = _grid_with_depth(100, 100, z_value=2.0)
    # A YOLO person mask covering a 20x20 patch at known depth 5.0.
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    points[10:30, 10:30, 2] = 5.0
    person_records = [(10, 10, 30, 30, mask)]
    out = centroid_from_box(points, validmask, (12, 12, 28, 28), person_records)
    assert out is not None
    centroid, used_mask = out
    assert abs(centroid[2] - 5.0) < 1e-6     # median Z from the reused mask
    assert used_mask.sum() > 0


def test_centroid_from_box_box_center_fallback_when_no_mask():
    points, validmask = _grid_with_depth(100, 100, z_value=3.0)
    out = centroid_from_box(points, validmask, (40, 40, 60, 60), person_records=[])
    assert out is not None
    centroid, _ = out
    assert abs(centroid[2] - 3.0) < 1e-6


def test_centroid_from_box_none_when_no_valid_depth():
    # Valid pixels only in a far corner; box + its expansion never reach them.
    points, validmask = _grid_with_depth(
        200, 200, z_value=3.0, valid_region=(190, 200, 190, 200))
    out = centroid_from_box(points, validmask, (10, 10, 20, 20), person_records=[])
    assert out is None
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
python3 -m pytest src/tk26_vision/src/tk_vision_specialized/test/test_waving_geometry.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named
'tk_vision_specialized._waving_geometry'`.

- [ ] **Step 3: Implement the module**

Create `tk_vision_specialized/_waving_geometry.py`:

```python
"""Pure box/depth helpers for the waving VLM fallback.

No ROS, no network: these turn a VLM whole-person box into a 3D centroid using
the back-projected XYZ grid the waving server already computes. A box that
overlaps a YOLO person seg-mask reuses that mask (clean silhouette median);
otherwise it falls back to the valid depth inside the box, expanding once if the
box is too sparse.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def box_iou(a, b) -> float:
    """Intersection-over-union of two xyxy boxes. 0.0 if they do not overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_in_box(box, other) -> bool:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return other[0] <= cx <= other[2] and other[1] <= cy <= other[3]


def is_duplicate_box(box, existing_boxes: Sequence, *, iou_thresh: float) -> bool:
    """True if box duplicates any existing box (IoU >= thresh or center inside)."""
    for other in existing_boxes:
        if box_iou(box, other) >= iou_thresh or _center_in_box(box, other):
            return True
    return False


def _centroid_over_mask(points: np.ndarray, mask: np.ndarray):
    """Mean XY + median Z over the True pixels of mask, or None if empty."""
    if not mask.any():
        return None
    pts = points[mask]
    centroid = np.mean(pts, axis=0)
    centroid[2] = np.median(pts[:, 2])
    return centroid


def _expand(box, factor, w, h):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * factor, (y2 - y1) * factor
    nx1 = max(0, int(round(cx - bw / 2.0)))
    ny1 = max(0, int(round(cy - bh / 2.0)))
    nx2 = min(w, int(round(cx + bw / 2.0)))
    ny2 = min(h, int(round(cy + bh / 2.0)))
    return nx1, ny1, nx2, ny2


def centroid_from_box(points: np.ndarray, validmask: np.ndarray, box_xyxy,
                      person_records: Sequence, *, mask_iou_thresh: float = 0.3,
                      min_valid: int = 10):
    """Return (centroid_xyz, used_mask) for a VLM box, or None.

    Tier 1: if box_xyxy overlaps a person_records seg-mask (box-vs-mask-bbox IoU
            >= mask_iou_thresh and that record has a mask), reuse mask & valid.
    Tier 2: else the box rectangle & valid; if < min_valid px, expand once x1.5.
    Returns the XYZ centroid (mean XY, median Z) and the bool mask actually used
    (so the caller can log it), or None when no usable depth exists.
    """
    h, w = validmask.shape

    best_mask = None
    best_iou = mask_iou_thresh
    for rec in person_records:
        rx1, ry1, rx2, ry2, rmask = rec[0], rec[1], rec[2], rec[3], rec[4]
        if rmask is None:
            continue
        iou = box_iou(box_xyxy, (rx1, ry1, rx2, ry2))
        if iou >= best_iou:
            best_iou = iou
            best_mask = rmask
    if best_mask is not None:
        combined = best_mask & validmask
        if combined.sum() >= min_valid:
            centroid = _centroid_over_mask(points, combined)
            if centroid is not None:
                return centroid, combined

    for factor in (1.0, 1.5):
        x1, y1, x2, y2 = _expand(box_xyxy, factor, w, h)
        rect = np.zeros((h, w), dtype=bool)
        rect[y1:y2, x1:x2] = True
        combined = rect & validmask
        if combined.sum() >= min_valid:
            centroid = _centroid_over_mask(points, combined)
            if centroid is not None:
                return centroid, combined
    return None
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
python3 -m pytest src/tk26_vision/src/tk_vision_specialized/test/test_waving_geometry.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Flake8 the new module**

Run:
```bash
python3 -m flake8 --max-line-length=99 \
  src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/_waving_geometry.py
```
Expected: no output.

- [ ] **Step 6: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/_waving_geometry.py \
        src/tk_vision_specialized/test/test_waving_geometry.py
git commit -m "feat(waving): pure box/depth helpers for VLM->3D centroid"
```

---

## Task 7: Wire the fallback into `waving_person_server.py`

**Files:**
- Modify: `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py`

This is integration code (needs ROS + cameras to fully exercise), so verification
is a node-start smoke test plus the existing T-suite. Apply the edits in order.

- [ ] **Step 1: Add imports for the new helpers**

In `waving_person_server.py`, find the existing import block near the top (after
`from vision_util.weights_cache import resolve_weights`) and add:

```python
from ._waving_vlm import (
    request_waving_persons_chain,
    build_provider_models,
    has_provider_key,
    WavingVlmError,
)
from ._waving_geometry import is_duplicate_box, centroid_from_box
```

- [ ] **Step 2: Declare the VLM parameters in `__init__`**

In `DetectWavingPersonsNode.__init__`, find the vision-logging param block:

```python
        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')
```

Immediately **before** that block, insert the VLM param declarations and the
chain resolution:

```python
        self.declare_parameter('enable_vlm_fallback', True)
        self.declare_parameter('vlm_provider', 'qwen')
        self.declare_parameter('vlm_fallback_provider', 'gemini')
        self.declare_parameter('vlm_model_qwen', 'qwen3-vl-plus')
        self.declare_parameter('vlm_model_gemini', 'google/gemini-2.5-pro')
        self.declare_parameter('vlm_timeout_s', 20.0)
        self.declare_parameter('vlm_max_retries', 3)
        self.declare_parameter('vlm_dedup_iou', 0.3)
        self.enable_vlm_fallback = (
            self.get_parameter('enable_vlm_fallback').value)
        self.vlm_provider = self.get_parameter('vlm_provider').value
        self.vlm_fallback_provider = (
            self.get_parameter('vlm_fallback_provider').value)
        self.vlm_model_qwen = self.get_parameter('vlm_model_qwen').value
        self.vlm_model_gemini = self.get_parameter('vlm_model_gemini').value
        self.vlm_timeout_s = float(self.get_parameter('vlm_timeout_s').value)
        self.vlm_max_retries = int(self.get_parameter('vlm_max_retries').value)
        self.vlm_dedup_iou = float(self.get_parameter('vlm_dedup_iou').value)
        self._vlm_chain = self._resolve_provider_chain()
```

- [ ] **Step 3: Add the `_resolve_provider_chain` method**

Add this method to the class (e.g. directly above `is_waving`):

```python
    def _resolve_provider_chain(self):
        """Build the (provider, model) chain; [] if fallback off or no keys.

        Non-fatal: a missing API key disables the VLM fallback (the node still
        serves MediaPipe-only) instead of raising at init.
        """
        if not self.enable_vlm_fallback:
            self.get_logger().info('VLM waving fallback disabled by param.')
            return []

        def model_for(provider):
            return (self.vlm_model_qwen if provider == 'qwen'
                    else self.vlm_model_gemini)

        chain = build_provider_models(
            self.vlm_provider, self.vlm_fallback_provider,
            has_key=has_provider_key, model_for=model_for,
            logger=self.get_logger())
        if not chain:
            self.get_logger().warn(
                'VLM fallback enabled but no provider API key found; '
                'serving MediaPipe-only.')
        else:
            self.get_logger().info(
                f'Waving VLM chain: {[p for p, _ in chain]}')
        return chain
```

- [ ] **Step 4: Retain every person's mask in the detection loop**

In `detect_waving_callback`, find where the per-person lists are initialised:

```python
        waving_persons_centroids = []
        waving_annotations = []
        waving_masks = []
        all_person_annotations = []  # (x1, y1, x2, y2, landmarks, is_wave) for every person
        person_candidates = 0
```

Add two tracking lists:

```python
        waving_persons_centroids = []
        waving_annotations = []
        waving_masks = []
        waving_sources = []  # 'mp' or 'vlm', kept aligned with the lists above
        person_records = []  # (x1, y1, x2, y2, seg_mask_or_None) for every person
        all_person_annotations = []  # (x1, y1, x2, y2, landmarks, is_wave) for every person
        person_candidates = 0
```

Then, inside the `if self.yolo.names[int(box.cls[0])] == 'person':` block, after
`person_roi = rgb_image[y1:y2, x1:x2]` and the empty-ROI guard, compute the
person's seg-mask once and record it. Find:

```python
                    pose_results = self.pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                    is_wave = self.is_waving(pose_results.pose_landmarks, person_roi)
                    all_person_annotations.append(
                        (x1, y1, x2, y2, pose_results.pose_landmarks, is_wave)
                    )
```

and insert, right after `all_person_annotations.append(...)`:

```python
                    rec_mask = None
                    if masks is not None and i < len(masks.data):
                        seg = masks.data[i].cpu().numpy().astype(np.uint8)
                        if seg.shape != rgb_image.shape[:2]:
                            seg = cv2.resize(
                                seg,
                                (rgb_image.shape[1], rgb_image.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        rec_mask = seg.astype(bool)
                    person_records.append((x1, y1, x2, y2, rec_mask))
```

Also, in the existing waving branch where MediaPipe wavers are appended (after
`waving_masks.append(person_mask)`), add the source tag. Find:

```python
                                waving_persons_centroids.append(point_stamped)
                                waving_annotations.append((x1, y1, x2, y2, pose_results.pose_landmarks))
                                waving_masks.append(person_mask)
```

and append after `waving_masks.append(person_mask)`:

```python
                                waving_sources.append('mp')
```

- [ ] **Step 5: Trigger the VLM augment after the loop, before the sort**

Find the block right after the detection loop:

```python
        self.get_logger().info(f'Person candidates checked: {person_candidates}')
        # sort waving person centroids from closest to farthest (keep annotations + masks aligned)
        if waving_persons_centroids:
```

Insert the augment call and the source-aware sort. Replace the sort block:

```python
        if waving_persons_centroids:
            triples = sorted(
                zip(waving_persons_centroids, waving_annotations, waving_masks),
                key=lambda t: t[0].point.z,
            )
            waving_persons_centroids = [p for p, _, _ in triples]
            waving_annotations = [a for _, a, _ in triples]
            waving_masks = [m for _, _, m in triples]
```

with this (adds the VLM augment first, then sorts 4 aligned lists):

```python
        n_vlm_added = 0
        vlm_provider_used = ''
        if (self._vlm_chain
                and request.min_waving_persons > 0
                and len(waving_persons_centroids) < request.min_waving_persons):
            n_vlm_added, vlm_provider_used = self._vlm_augment(
                rgb_image, points, validmask_points, header, request,
                person_records, waving_persons_centroids,
                waving_annotations, waving_masks, waving_sources,
            )
            self.get_logger().info(
                f'VLM fallback added {n_vlm_added} waver(s) '
                f'(provider={vlm_provider_used or "none"}).')

        if waving_persons_centroids:
            quads = sorted(
                zip(waving_persons_centroids, waving_annotations,
                    waving_masks, waving_sources),
                key=lambda t: t[0].point.z,
            )
            waving_persons_centroids = [p for p, _, _, _ in quads]
            waving_annotations = [a for _, a, _, _ in quads]
            waving_masks = [m for _, _, m, _ in quads]
            waving_sources = [s for _, _, _, s in quads]
```

- [ ] **Step 6: Add the `_vlm_augment` method**

Add this method to the class (e.g. directly below `_resolve_provider_chain`):

```python
    def _vlm_augment(self, rgb_image, points, validmask_points, header, request,
                     person_records, waving_persons_centroids,
                     waving_annotations, waving_masks, waving_sources):
        """Call the VLM chain and append the wavers MediaPipe missed.

        Mutates the four aligned waver lists in place. Returns
        (n_added, provider_used). Never raises: any VLM failure logs a warning
        and returns (0, '') so the service still answers with MediaPipe results.
        """
        try:
            result = request_waving_persons_chain(
                rgb_image, provider_models=self._vlm_chain,
                timeout_s=self.vlm_timeout_s, max_retries=self.vlm_max_retries,
                logger=self.get_logger())
        except WavingVlmError as exc:
            self.get_logger().warn(f'VLM waving fallback unavailable: {exc}')
            return 0, ''

        existing_boxes = [(a[0], a[1], a[2], a[3]) for a in waving_annotations]
        n_added = 0
        for box in result.boxes:
            if is_duplicate_box(box, existing_boxes,
                                iou_thresh=self.vlm_dedup_iou):
                continue
            out = centroid_from_box(points, validmask_points, box,
                                    person_records)
            if out is None:
                self.get_logger().info(
                    f'VLM box {box} skipped: no usable depth.')
                continue
            centroid, used_mask = out
            if (request.threshold_meters > 0
                    and centroid[2] > request.threshold_meters):
                self.get_logger().info(
                    f'VLM waver dropped: depth {centroid[2]:.2f}m > threshold '
                    f'{request.threshold_meters:.2f}m')
                continue
            point_stamped = PointStamped()
            point_stamped.header = header
            point_stamped.point.x = float(centroid[0])
            point_stamped.point.y = float(centroid[1])
            point_stamped.point.z = float(centroid[2])
            x1, y1, x2, y2 = box
            waving_persons_centroids.append(point_stamped)
            waving_annotations.append((x1, y1, x2, y2, None))
            waving_masks.append(used_mask)
            waving_sources.append('vlm')
            existing_boxes.append(box)
            n_added += 1
        return n_added, result.provider
```

- [ ] **Step 7: Draw VLM wavers distinctly on the debug overlay**

Find the annotate + publish block:

```python
        annotated = self._annotate_all_persons(rgb_image, all_person_annotations)
        self._publish_debug_image(
```

Between those two lines, add a loop that draws the VLM wavers (source `'vlm'`)
in orange with a `waving (vlm)` label:

```python
        annotated = self._annotate_all_persons(rgb_image, all_person_annotations)
        for (x1, y1, x2, y2, _lm), src in zip(waving_annotations, waving_sources):
            if src != 'vlm':
                continue
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(annotated, 'waving (vlm)', (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        self._publish_debug_image(
```

- [ ] **Step 8: Guard the latent `_frame_queue` AttributeError**

Find:

```python
        if self.show_window and self._frame_queue is not None:
```

Replace with:

```python
        if self.show_window and getattr(self, '_frame_queue', None) is not None:
```

- [ ] **Step 9: Tag VLM wavers in the vision log + record context**

Find the vision-log detection builder:

```python
        if self._vision_logger.enabled:
            detections = []
            for (x1, y1, x2, y2, _lm), pt, person_mask in zip(
                waving_annotations, waving_persons_centroids, waving_masks
            ):
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'mask': person_mask,
                    'cls_name': 'waving_person',
                    'conf': 1.0,
                    'centroid': [(x1 + x2) // 2, (y1 + y2) // 2],
                    'centroid_3d': [float(pt.point.x), float(pt.point.y), float(pt.point.z)],
                })
            self._vision_logger.write(
                rgb_image, detections,
                request_ctx={
                    'target_frame': request.target_frame,
                    'threshold_meters': float(request.threshold_meters),
                },
                branch='detect_waving',
                extras={'n_person_candidates': person_candidates},
                timings={'detect_waving': time.perf_counter() - _t0},
            )
```

Replace it with the source-aware version:

```python
        if self._vision_logger.enabled:
            detections = []
            for (x1, y1, x2, y2, _lm), pt, person_mask, src in zip(
                waving_annotations, waving_persons_centroids,
                waving_masks, waving_sources
            ):
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'mask': person_mask,
                    'cls_name': ('waving_person_vlm' if src == 'vlm'
                                 else 'waving_person'),
                    'conf': 1.0,
                    'centroid': [(x1 + x2) // 2, (y1 + y2) // 2],
                    'centroid_3d': [float(pt.point.x), float(pt.point.y),
                                    float(pt.point.z)],
                })
            self._vision_logger.write(
                rgb_image, detections,
                request_ctx={
                    'target_frame': request.target_frame,
                    'threshold_meters': float(request.threshold_meters),
                    'min_waving_persons': int(request.min_waving_persons),
                },
                branch='detect_waving',
                extras={'n_person_candidates': person_candidates,
                        'n_vlm_added': n_vlm_added,
                        'vlm_provider': vlm_provider_used},
                timings={'detect_waving': time.perf_counter() - _t0},
            )
```

- [ ] **Step 10: Build the package with the venv wrapper**

Run:
```bash
cd /home/tinker/tk25_ws
./src/tk26_vision/scripts/build.sh --packages-select tk_vision_specialized
source install/setup.bash
```
Expected: `Finished <<< tk_vision_specialized` with no errors.

- [ ] **Step 11: Flake8 the modified server**

Run:
```bash
python3 -m flake8 --max-line-length=99 \
  src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py
```
Expected: no output. Fix any long lines / unused imports.

- [ ] **Step 12: Node-start smoke test with NO API keys (must not crash)**

Run (deliberately unset keys so the chain resolves empty):
```bash
cd /home/tinker/tk25_ws
env -u OPENROUTER_API_KEY -u DASHSCOPE_API_KEY -u DASHCOPE_API_KEY \
  timeout 12 ros2 run tk_vision_specialized waving_person_server \
  --ros-args -p show_window:=false 2>&1 | tee /tmp/waving_start.log | head -40
```
Expected: the log shows `Detect Waving Persons node started` and a line
`VLM fallback enabled but no provider API key found; serving MediaPipe-only.`
(or, if keys happen to be set, `Waving VLM chain: ['qwen', ...]`). It must NOT
raise a Python traceback. `timeout` ending the process is success.

- [ ] **Step 13: Run the package unit tests (regression)**

Run:
```bash
python3 -m pytest src/tk26_vision/src/tk_vision_specialized/test/ -q
```
Expected: all tests pass (new waving tests + existing suite; pre-existing
flake8/pep257 lint tests cover the new modules).

- [ ] **Step 14: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py
git commit -m "feat(waving): VLM fallback augments MediaPipe wavers below threshold

Trigger on min_waving_persons; qwen->gemini chain; reuse YOLO mask or
box-center depth for 3D; dedup + add-all; distinct overlay + log tag;
guard latent _frame_queue AttributeError."
```

---

## Task 8: Documentation + final verification

**Files:**
- Modify: `src/tk26_vision/CLAUDE.md`

- [ ] **Step 1: Document the new params + behavior**

In `src/tk26_vision/CLAUDE.md`, under the `## Configuration` list, add a bullet
describing the waving fallback (place it after the `object_match_all_server`
bullet):

```markdown
- `waving_person_server` (`detect_waving_persons`): VLM fallback augments the
  MediaPipe waver list when `min_waving_persons` (new `DetectWaving.srv` request
  field, default `0` = off) exceeds the heuristic's count. `enable_vlm_fallback`
  (default `true`, global kill-switch), `vlm_provider` (`qwen`) →
  `vlm_fallback_provider` (`gemini`, `''` disables) errors-only chain;
  `vlm_model_qwen` (`qwen3-vl-plus`) / `vlm_model_gemini`
  (`google/gemini-2.5-pro`); `vlm_timeout_s` (20.0), `vlm_max_retries` (3),
  `vlm_dedup_iou` (0.3). Keys: `DASHSCOPE_API_KEY` (qwen) / `OPENROUTER_API_KEY`
  (gemini) — resolved via `_waving_vlm.py` (no `kimi_api` import). A missing key
  disables the fallback (no crash). VLM-found wavers reuse an overlapping YOLO
  mask or box-center depth for their 3D centroid and are tagged
  `waving_person_vlm` in the vision log.
```

- [ ] **Step 2: Update the vision-logging node list**

In `src/tk26_vision/CLAUDE.md`, find the `vision_logging_enabled` paragraph that
enumerates logging nodes (`yolo_seg_{node,default_node}`, `generalist_node`,
`person_track_node`, `waving_person_server`, ...). It already lists
`waving_person_server`; no change needed unless absent — if absent, add
`waving_person_server` to that list. (Verify only.)

- [ ] **Step 3: Full test sweep**

Run:
```bash
cd /home/tinker/tk25_ws
python3 -m pytest src/tk26_vision/src/tk_vision_specialized/test/ -q
```
Expected: all pass.

- [ ] **Step 4: T0 static smoke (interfaces + imports)**

Run:
```bash
cd /home/tinker/tk25_ws
bash src/tk26_vision/scripts/tests/t0_static.sh 2>&1 | tail -30
```
Expected: PASS (this checks entry-point imports + ROS interface availability;
confirms `DetectWaving` with the new field and the new modules import cleanly).
If T0 is unavailable in this environment, skip with a note.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add CLAUDE.md
git commit -m "docs(waving): document VLM fallback params + behavior"
```

---

## Self-review notes (for the implementer)

- **Spec coverage:** Task 1 = contract field; Tasks 2–5 = `_waving_vlm.py`
  (decoders, keys, chain, single+chain calls); Task 6 = box→3D + dedup
  (`_waving_geometry.py`); Task 7 = trigger, augment, mask reuse, depth filter,
  transform (inherited), overlay, log tags, non-fatal init, `_frame_queue` fix;
  Task 8 = docs. The transform of VLM centroids is handled by the *existing*
  end-of-callback transform block because VLM `PointStamped`s are created in
  `header.frame_id` and appended before that block runs — no new code needed.
- **Add-all (not cap):** `_vlm_augment` appends every deduped, depth-valid VLM
  waver; `min_waving_persons` only gates whether the VLM runs.
- **Type consistency:** the four aligned lists
  (`waving_persons_centroids` / `waving_annotations` / `waving_masks` /
  `waving_sources`) are appended together in both the MediaPipe branch (Step 4)
  and `_vlm_augment` (Step 6), and sorted together (Step 5). `centroid_from_box`
  returns `(centroid_xyz, used_mask)`; `_vlm_augment` unpacks exactly that.
- **No network in unit tests:** every `_waving_vlm` call test monkeypatches
  `openai.OpenAI` or `request_waving_persons`; geometry tests are pure numpy.
```
