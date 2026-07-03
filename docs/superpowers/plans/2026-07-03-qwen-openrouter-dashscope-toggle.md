# Qwen VLM Backend Toggle (DashScope vs OpenRouter) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `qwen_api_backend` ROS parameter (`'dashscope'` | `'openrouter'`) to all 11 Qwen VLM call sites in `tk26_vision`, so a robot can be launched with its Qwen traffic routed through OpenRouter instead of DashScope for regional-latency reasons, without touching any other behavior.

**Architecture:** One new pure function, `resolve_qwen_target(backend, model_param_value, base_url_override='')`, added to `kimi_api/_env.py` and imported by all 11 call sites (including `tk_vision_specialized`'s modules, which previously stayed `kimi_api`-free — that convention is deliberately overridden here). It centralizes: sentinel-default model substitution, backend validation, key resolution, and base-url resolution. `object_detection_generalist`'s `generalist_node.py` is the one exception — it keeps its existing `dashscope/model`-prefix routing (`vlm_bbox._split_provider`) and gets a separate fail-fast validator instead of the shared resolver, per the design spec.

**Tech Stack:** Python 3.10, ROS2 Humble (rclpy), `openai` SDK (OpenAI-compatible client pointed at either host), `pytest`.

## Global Constraints

- **Spec of record:** `docs/superpowers/specs/2026-07-03-qwen-openrouter-dashscope-toggle-design.md`. This plan implements it exactly; do not re-litigate its decisions (see its "Review history" — three adversarial review rounds already resolved the open questions).
- **Out of scope, do not touch:** the duplicated Gemini-primary → Qwen-fallback chain-*ordering* logic (the `for provider, model in provider_models: try/except/continue` loops) — only the Qwen leg's key/base-url/model *resolution* changes. Do not consolidate that loop logic across files.
- **`qwen_api_backend` has a hardcoded default `'dashscope'` on every node — no environment variable, no `.env` involvement, no shell/robot-profile mechanism.** This was explicitly decided after two rounds of review found env-var approaches broken; do not reintroduce one.
- **Sentinel philosophy:** every per-node Qwen-model param's default changes from `'qwen3-vl-plus'` to `''`. `''` means "use the backend's own default model"; any non-empty value is honored **verbatim on either backend**, never silently rewritten. This same principle applies to the one `vlm_base_url` param whose default isn't already `''` (`object_match_server.py` — see Task 6).
- **Known, accepted side effect:** switching `tk_vision_specialized`'s four `kimi_api`-free modules to the shared resolver changes their `DASHSCOPE_API_KEY`/`DASHCOPE_API_KEY` lookup order (from each module's own typo-first tuple to `kimi_api._env.require_dashscope_api_key()`'s canonical-first order). This is an unavoidable consequence of centralizing onto one resolver — not a separate initiative to "fix the key-order bug." One existing test (`test_qwen_client_resolves_dashcope_typo_first` in `test_vlm_match_client.py`) asserts the old order and must be updated in Task 7; no other existing test asserts key precedence.
- **Provisional OpenRouter default model:** `qwen/qwen3-vl-32b-instruct`, hardcoded as a hardcoded constant with a comment flagging it as unverified pending the benchmark in Task 13. Do not treat this as final — it unblocks coding, it does not clear the model for competition use.
- **4 node executables have no launch-file coverage today** (`grocery_categorize`, `object_match_server`, `object_match_all_server`, `placing_location_server` — only bare `ros2 run`). This plan does **not** add launch files for them; they still get the ROS param and are flipped via `-p qwen_api_backend:=openrouter` on their own `ros2 run` invocation. Do not build launch-file coverage as part of this plan.
- **Test runner:** every touched package's tests run via `pytest src/<package>/test/ -q` from `/home/tinker/tk25_ws/src/tk26_vision` (each package's `test/conftest.py` inserts the package source dir onto `sys.path`, so no ROS build is required to run these tests).
- **Repo / branch:** `/home/tinker/tk25_ws/src/tk26_vision`, git branch `dev`. Commit after every task.

---

### Task 1: Shared resolver — `resolve_qwen_target` in `kimi_api/_env.py`

**Files:**
- Modify: `src/kimi_api/kimi_api/_env.py`
- Test: `src/kimi_api/test/test_env_resolve_qwen_target.py` (create)

**Interfaces:**
- Produces: `resolve_qwen_target(backend: str, model_param_value: str, base_url_override: str = '') -> tuple[str, str, str]` — returns `(base_url, api_key, model)`. Raises `RuntimeError` on: invalid `backend` (not `'dashscope'`/`'openrouter'`), missing required key for the selected backend, or a model-id shape that doesn't match the backend (OpenRouter ids contain `/`, DashScope ids don't) when the caller passed a non-empty `model_param_value`. Every later task in this plan imports and calls this function; treat its name and signature as fixed.

- [ ] **Step 1: Write the failing tests**

Create `src/kimi_api/test/test_env_resolve_qwen_target.py`:

```python
"""Unit tests for kimi_api._env.resolve_qwen_target."""
import pytest

from kimi_api._env import resolve_qwen_target


def test_dashscope_sentinel_model_uses_dashscope_default(monkeypatch):
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    base_url, api_key, model = resolve_qwen_target('dashscope', '')
    assert base_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    assert api_key == 'ds-key'
    assert model == 'qwen3-vl-plus'


def test_dashscope_explicit_model_honored_verbatim(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    _, _, model = resolve_qwen_target('dashscope', 'qwen-vl-max')
    assert model == 'qwen-vl-max'


def test_dashscope_missing_key_raises(monkeypatch):
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='DASHSCOPE_API_KEY'):
        resolve_qwen_target('dashscope', '')


def test_dashscope_rejects_openrouter_shaped_model(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    with pytest.raises(RuntimeError, match='dashscope'):
        resolve_qwen_target('dashscope', 'qwen/qwen3-vl-32b-instruct')


def test_openrouter_sentinel_model_uses_openrouter_default(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    base_url, api_key, model = resolve_qwen_target('openrouter', '')
    assert base_url == 'https://openrouter.ai/api/v1'
    assert api_key == 'or-key'
    assert model == 'qwen/qwen3-vl-32b-instruct'


def test_openrouter_explicit_model_honored_verbatim(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    _, _, model = resolve_qwen_target('openrouter', 'qwen/qwen3.7-plus')
    assert model == 'qwen/qwen3.7-plus'


def test_openrouter_missing_key_raises(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='OPENROUTER_API_KEY'):
        resolve_qwen_target('openrouter', '')


def test_openrouter_rejects_dashscope_shaped_model(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    with pytest.raises(RuntimeError, match='openrouter'):
        resolve_qwen_target('openrouter', 'qwen3-vl-plus')


def test_invalid_backend_raises(monkeypatch):
    with pytest.raises(RuntimeError, match='qwen_api_backend'):
        resolve_qwen_target('bogus', '')


def test_base_url_override_wins_regardless_of_backend(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    base_url, _, _ = resolve_qwen_target(
        'dashscope', '', base_url_override='https://self-hosted.example/v1')
    assert base_url == 'https://self-hosted.example/v1'


def test_openrouter_base_url_override_still_uses_dashscope_key_when_backend_dashscope(monkeypatch):
    # base_url_override does not change which key is required — only which
    # host is called. Confirms the two concerns (key selection vs base URL)
    # are independent.
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    _, api_key, _ = resolve_qwen_target(
        'dashscope', '', base_url_override='https://gateway.example/v1')
    assert api_key == 'ds-key'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/kimi_api/test/test_env_resolve_qwen_target.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_qwen_target'`

- [ ] **Step 3: Implement `resolve_qwen_target`**

Read the current end of `src/kimi_api/kimi_api/_env.py` (it currently ends at `require_dashscope_api_key()`, line 85). Append:

```python
_VALID_QWEN_BACKENDS = ('dashscope', 'openrouter')

# Provisional default for qwen_api_backend='openrouter' — OpenRouter does not
# carry the exact 'qwen3-vl-plus' slug this codebase defaults to on DashScope.
# This is the SAFER open-weight starting bet (same Qwen3-VL family as the
# calibrated bbox decoder in object_detection_generalist/vlm_bbox.py), NOT a
# verified final choice. Do not rely on this for a competition run until the
# benchmark task in docs/superpowers/specs/2026-07-03-qwen-openrouter-
# dashscope-toggle-design.md §"Default OpenRouter model" has actually run
# (image modality + known-position bbox format + regional latency) and this
# constant has been updated accordingly.
_OPENROUTER_QWEN_DEFAULT_MODEL = 'qwen/qwen3-vl-32b-instruct'

_DASHSCOPE_DEFAULT_MODEL = 'qwen3-vl-plus'


def resolve_qwen_target(
    backend: str,
    model_param_value: str,
    base_url_override: str = '',
) -> tuple[str, str, str]:
    """Return (base_url, api_key, model) for a Qwen call on the given backend.

    `backend` is 'dashscope' or 'openrouter' — the caller's qwen_api_backend
    ROS param. `model_param_value` is the caller's own qwen-model ROS param:
    '' means "use this backend's default model"; any non-empty value is
    honored verbatim on either backend (never silently rewritten — an
    explicit value the operator set for the "wrong" backend raises instead
    of being swapped, since OpenRouter ids contain '/' and DashScope ids
    don't, and mixing them up is very likely a config mistake worth
    surfacing loudly). `base_url_override`, if non-empty, always wins over
    the backend's own base URL — it does not change which API key is
    required.

    Raises RuntimeError on: invalid backend, missing required key for the
    selected backend, or a model-id shape mismatch (see above).
    """
    if backend not in _VALID_QWEN_BACKENDS:
        raise RuntimeError(
            f'Invalid qwen_api_backend {backend!r}; expected one of '
            f'{_VALID_QWEN_BACKENDS}.'
        )

    model = model_param_value or ''

    if backend == 'dashscope':
        resolved_model = model or _DASHSCOPE_DEFAULT_MODEL
        if '/' in resolved_model:
            raise RuntimeError(
                f"qwen_api_backend='dashscope' but model {resolved_model!r} "
                "looks like an OpenRouter id (contains '/'). Pass a bare "
                "DashScope model id, or set qwen_api_backend='openrouter'."
            )
        api_key = require_dashscope_api_key()
        resolved_base_url = base_url_override or dashscope_base_url()
        return resolved_base_url, api_key, resolved_model

    # openrouter
    resolved_model = model or _OPENROUTER_QWEN_DEFAULT_MODEL
    if '/' not in resolved_model:
        raise RuntimeError(
            f"qwen_api_backend='openrouter' but model {resolved_model!r} "
            "looks like a bare DashScope id (no '/'). Pass an OpenRouter "
            "'org/name' id, or set qwen_api_backend='dashscope'."
        )
    api_key = require_api_key()
    resolved_base_url = base_url_override or base_url()
    return resolved_base_url, api_key, resolved_model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/kimi_api/test/test_env_resolve_qwen_target.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/kimi_api/kimi_api/_env.py src/kimi_api/test/test_env_resolve_qwen_target.py
git commit -m "feat(kimi_api): add resolve_qwen_target for the DashScope/OpenRouter toggle"
```

---

### Task 2: `feature_recognition.py` + `_feature_vlm.py`

**Files:**
- Modify: `src/kimi_api/kimi_api/_feature_vlm.py`
- Modify: `src/kimi_api/kimi_api/feature_recognition.py`
- Modify: `src/kimi_api/test/test_feature_vlm.py`

**Interfaces:**
- Consumes: `resolve_qwen_target(backend, model_param_value, base_url_override='')` from Task 1.
- Produces: `request_feature_description(..., qwen_api_backend='dashscope')`, `request_feature_description_chain(..., qwen_api_backend='dashscope')` — both gain this keyword-only parameter. `FeatureService.qwen_api_backend` (str, from the new ROS param) is read by `feature_recognition.py`.

- [ ] **Step 1: Update the failing/changing tests first**

In `src/kimi_api/test/test_feature_vlm.py`, find the two tests that currently assert `base_url` for the qwen leg (around line 78) and gemini leg (around line 93), and the missing-key test (around line 100). Update the qwen-leg test to pass the new parameter and add two new tests. The existing test around line 73 currently reads:

```python
    provider='qwen', model='qwen3-vl-plus')
```

Change the call to include `qwen_api_backend='dashscope'` explicitly (so the test is unambiguous about which backend it's exercising), and add:

```python
def test_request_feature_description_qwen_openrouter_backend(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    fake = _make_fake_openai(lambda kw: _SUCCESS_PAYLOAD)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    request_feature_description(
        'data:url', 'sys', 'user',
        provider='qwen', model='', qwen_api_backend='openrouter')

    assert fake.last_init['base_url'] == 'https://openrouter.ai/api/v1'
    assert fake.last_init['api_key'] == 'or-key'


def test_request_feature_description_qwen_openrouter_missing_key_raises(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with pytest.raises(FeatureVlmError, match='OPENROUTER_API_KEY'):
        request_feature_description(
            'data:url', 'sys', 'user',
            provider='qwen', model='', qwen_api_backend='openrouter')
```

(Match the exact fixture names — `_make_fake_openai`, `_SUCCESS_PAYLOAD` or equivalent, `openai` import alias — already used elsewhere in this file; do not invent new ones.)

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/kimi_api/test/test_feature_vlm.py -v`
Expected: the two new tests FAIL with `TypeError: request_feature_description() got an unexpected keyword argument 'qwen_api_backend'`

- [ ] **Step 3: Implement in `_feature_vlm.py`**

Add `resolve_qwen_target` to the imports at the top of the file (alongside the existing `require_dashscope_api_key`, `dashscope_base_url`, `require_api_key`, `base_url` imports — check whether any of those four become unused after this change and remove only the ones with no other call site in this file).

Change `request_feature_description`'s signature to add `qwen_api_backend: str = 'dashscope',` as a keyword-only parameter (alongside the existing `provider`, `model` params), and replace the quoted qwen-branch body:

```python
    if provider == 'qwen':
        try:
            api_key = require_dashscope_api_key()
        except RuntimeError as exc:
            raise FeatureVlmError(str(exc)) from exc
        b_url = dashscope_base_url()
```

with:

```python
    if provider == 'qwen':
        try:
            b_url, api_key, model = resolve_qwen_target(qwen_api_backend, model)
        except RuntimeError as exc:
            raise FeatureVlmError(str(exc)) from exc
```

Add `qwen_api_backend: str = 'dashscope',` to `request_feature_description_chain`'s signature too, and thread it into its call to `request_feature_description(...)`.

- [ ] **Step 4: Wire the ROS param through `feature_recognition.py`**

In `FeatureService.__init__` (around line 181), change:

```python
self.declare_parameter('feature_model_qwen', 'qwen3-vl-plus')
```

to:

```python
self.declare_parameter('feature_model_qwen', '')
self.declare_parameter('qwen_api_backend', 'dashscope')
```

and read it alongside `self.feature_model_qwen` (find that existing `self.feature_model_qwen = self.get_parameter('feature_model_qwen').value` line and add immediately after):

```python
self.qwen_api_backend = self.get_parameter('qwen_api_backend').value
```

In `_resolve_feature_provider_chain` (around line 244), the existing fallback-arming block:

```python
                try:
                    require_dashscope_api_key()
                    chain.append(('qwen', self.feature_model_qwen))
                except RuntimeError:
                    self.get_logger().warn(
                        f'Fallback provider {fb!r} key missing; fallback disabled.'
                    )
```

becomes backend-aware (this is the fallback-arming check the design spec requires — it must test whichever key `self.qwen_api_backend` needs, not always DashScope):

```python
                try:
                    _, _, resolved_model = resolve_qwen_target(
                        self.qwen_api_backend, self.feature_model_qwen)
                    chain.append(('qwen', resolved_model))
                except RuntimeError:
                    self.get_logger().warn(
                        f'Fallback provider {fb!r} key missing; fallback disabled.'
                    )
```

Find the call site that invokes `request_feature_description_chain(...)` (inside the service callback) and add `qwen_api_backend=self.qwen_api_backend` to its kwargs.

This node's existing fail-fast behavior (`require_api_key()` called unguarded at `__init__` for the Gemini primary — unrelated to this toggle) and graceful-disable behavior for the Qwen fallback (unchanged shape, now backend-aware) are both preserved exactly.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/kimi_api/test/test_feature_vlm.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/kimi_api/kimi_api/_feature_vlm.py src/kimi_api/kimi_api/feature_recognition.py src/kimi_api/test/test_feature_vlm.py
git commit -m "feat(kimi_api): wire qwen_api_backend through feature_recognition"
```

---

### Task 3: `feature_matching.py` + `_match_vlm.py`

**Files:**
- Modify: `src/kimi_api/kimi_api/_match_vlm.py`
- Modify: `src/kimi_api/kimi_api/feature_matching.py`
- Modify: `src/kimi_api/test/test_match_vlm.py`

**Interfaces:**
- Consumes: `resolve_qwen_target` (Task 1).
- Produces: `request_match_indices(..., qwen_api_backend='dashscope')`, `request_match_indices_chain(..., qwen_api_backend='dashscope')`.

This task is structurally identical to Task 2 — `_match_vlm.py`'s qwen branch (around line 118) has the exact same shape:

```python
    if provider == 'qwen':
        try:
            api_key = require_dashscope_api_key()
        except RuntimeError as exc:
            raise MatchVlmError(str(exc)) from exc
        b_url = dashscope_base_url()
```

- [ ] **Step 1: Add tests to `test_match_vlm.py`**, mirroring Task 2 Step 1's two new tests (`test_request_match_indices_qwen_openrouter_backend`, `test_request_match_indices_qwen_openrouter_missing_key_raises`), adapted to `request_match_indices`'s actual signature (`n_feats`, `n_cand`, etc. — match the existing call shape at the file's current line ~152) and `MatchVlmError` as the raised type.

- [ ] **Step 2: Run to verify failure.** `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/kimi_api/test/test_match_vlm.py -v`

- [ ] **Step 3: Implement in `_match_vlm.py`** — same substitution as Task 2 Step 3, note this file does `import openai; client = openai.OpenAI(...)` (module-level import, not `from openai import OpenAI`) — keep that style, only replace the qwen-branch body and add the `qwen_api_backend` parameter to both `request_match_indices` and `request_match_indices_chain`.

- [ ] **Step 4: Wire `feature_matching.py`** — same pattern as Task 2 Step 4:
  - `self.declare_parameter('match_model_qwen', 'qwen3-vl-plus')` (line 112) → `''`.
  - Add `self.declare_parameter('qwen_api_backend', 'dashscope')` and `self.qwen_api_backend = self.get_parameter('qwen_api_backend').value`.
  - `_resolve_match_provider_chain` (lines 168-189) — same backend-aware rewrite as Task 2's `_resolve_feature_provider_chain`.
  - Thread `qwen_api_backend=self.qwen_api_backend` into the `request_match_indices_chain(...)` call site.

- [ ] **Step 5: Run to verify pass.** `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/kimi_api/test/test_match_vlm.py -v`

- [ ] **Step 6: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/kimi_api/kimi_api/_match_vlm.py src/kimi_api/kimi_api/feature_matching.py src/kimi_api/test/test_match_vlm.py
git commit -m "feat(kimi_api): wire qwen_api_backend through feature_matching"
```

---

### Task 4: `grocery_categorize.py` + `_categorize_vlm.py`

**Files:**
- Modify: `src/kimi_api/kimi_api/_categorize_vlm.py`
- Modify: `src/kimi_api/kimi_api/grocery_categorize.py`
- Modify: `src/kimi_api/test/test_categorize_vlm.py`

**Interfaces:**
- Consumes: `resolve_qwen_target` (Task 1).
- Produces: `request_shelf_layer(..., qwen_api_backend='dashscope')`, `request_shelf_layer_chain(..., qwen_api_backend='dashscope')`.

Same shape as Tasks 2-3. `_categorize_vlm.py`'s qwen branch (line 83) is the identical pattern; `import openai` module-level style like `_match_vlm.py`.

- [ ] **Step 1: Add tests to `test_categorize_vlm.py`**, mirroring Task 2's pattern, adapted to `request_shelf_layer`'s signature (`sys_prompt`, `shelf_img_url`, `obj_seg_url`) and `ShelfVlmError`.

- [ ] **Step 2: Run to verify failure.** `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/kimi_api/test/test_categorize_vlm.py -v`

- [ ] **Step 3: Implement in `_categorize_vlm.py`** — same substitution pattern, add `qwen_api_backend` to `request_shelf_layer` and `request_shelf_layer_chain`.

- [ ] **Step 4: Wire `grocery_categorize.py`**:
  - `self.declare_parameter('categorize_model_qwen', 'qwen3-vl-plus')` (line 66) → `''`.
  - Add `qwen_api_backend` param + attribute.
  - `_resolve_categorize_provider_chain` (lines 123-144) — same backend-aware rewrite.
  - Thread `qwen_api_backend=self.qwen_api_backend` into the `request_shelf_layer_chain(...)` call site.

- [ ] **Step 5: Run to verify pass.** `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/kimi_api/test/test_categorize_vlm.py -v`

- [ ] **Step 6: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/kimi_api/kimi_api/_categorize_vlm.py src/kimi_api/kimi_api/grocery_categorize.py src/kimi_api/test/test_categorize_vlm.py
git commit -m "feat(kimi_api): wire qwen_api_backend through grocery_categorize"
```

---

### Task 5: `seat_recommend_bbox.py` + `_seat_vlm.py` + `_seat_bbox_vlm.py`

**Files:**
- Modify: `src/kimi_api/kimi_api/_seat_vlm.py`
- Modify: `src/kimi_api/kimi_api/_seat_bbox_vlm.py`
- Modify: `src/kimi_api/kimi_api/seat_recommend_bbox.py`
- Modify: `src/kimi_api/test/test_seat_vlm.py`
- Modify: `src/kimi_api/test/test_seat_bbox_vlm.py` (only if you add a new direct-construction test — see Step 1 note)

**Interfaces:**
- Consumes: `resolve_qwen_target` (Task 1).
- Produces: `request_seat(..., qwen_api_backend='dashscope')`, `request_seat_chain(..., qwen_api_backend='dashscope')`, `request_seat_bbox(..., qwen_api_backend='dashscope')`, `request_seat_bbox_chain(..., qwen_api_backend='dashscope')`.

This node is the outlier among the kimi_api group: it has a real `vlm_provider` param (default `'qwen'` — Qwen is *primary* for the default `bbox_select` strategy) with **no non-VLM fallback** for that strategy, and a distinct `_model_for`/`_has_provider_key` dispatch instead of a hardcoded Gemini-primary chain builder.

- [ ] **Step 1: Add tests to `test_seat_vlm.py`**, mirroring Task 2's pattern for `request_seat` (signature: `request_seat(rgb_bgr, names, features, *, model, provider, ...)`, error type `VlmSeatError`).

`test_seat_bbox_vlm.py` currently never calls `request_seat_bbox` directly (only `request_seat_bbox_chain` with monkeypatched fakes) — per Task grounding research, this is a pre-existing test gap, not something this plan is obligated to fill. Optionally add one direct test of `request_seat_bbox`'s new `qwen_api_backend` parameter for symmetry with the other files; if you skip it, note in the commit message that the gap is pre-existing and not newly introduced.

- [ ] **Step 2: Run to verify failure.** `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/kimi_api/test/test_seat_vlm.py src/kimi_api/test/test_seat_bbox_vlm.py -v`

- [ ] **Step 3: Implement in `_seat_vlm.py`** — `request_seat`'s qwen branch (lines 226-233):

```python
    if provider == 'qwen':
        try:
            api_key = require_dashscope_api_key()
        except RuntimeError as exc:
            raise VlmSeatError(str(exc)) from exc
        b_url = dashscope_base_url()
```

becomes:

```python
    if provider == 'qwen':
        try:
            b_url, api_key, model = resolve_qwen_target(qwen_api_backend, model)
        except RuntimeError as exc:
            raise VlmSeatError(str(exc)) from exc
```

Add `qwen_api_backend: str = 'dashscope',` to `request_seat` and `request_seat_chain`.

- [ ] **Step 4: Implement in `_seat_bbox_vlm.py`** — `request_seat_bbox`'s qwen branch (lines 347-356) is the same shape but also derives `reasoning`:

```python
    if provider == "qwen":
        try:
            api_key = require_dashscope_api_key()
        except RuntimeError as exc:
            raise VlmSeatBboxError(str(exc)) from exc
        b_url, reasoning = dashscope_base_url(), False
```

becomes:

```python
    if provider == "qwen":
        try:
            b_url, api_key, model = resolve_qwen_target(qwen_api_backend, model)
        except RuntimeError as exc:
            raise VlmSeatBboxError(str(exc)) from exc
        reasoning = False
```

Add `qwen_api_backend: str = 'dashscope',` to `request_seat_bbox` and `request_seat_bbox_chain`.

- [ ] **Step 5: Wire `seat_recommend_bbox.py`**:
  - `self.declare_parameter('bbox_model_qwen', 'qwen3-vl-plus')` (line 115) → `''`.
  - Add `self.declare_parameter('qwen_api_backend', 'dashscope')` and read it into `self.qwen_api_backend`.
  - `_model_for` (lines 271-272) is unaffected — it still just picks which raw param value to use, sentinel or not.
  - `_has_provider_key` (lines 274-279) becomes backend-aware for the `'qwen'` case only:

    ```python
    def _has_provider_key(self, provider: str) -> bool:
        try:
            if provider == 'qwen':
                resolve_qwen_target(self.qwen_api_backend, '')
            else:
                require_api_key()
            return True
        except RuntimeError:
            return False
    ```

  - `_resolve_provider_chain` (lines 281-308): the primary-provider check `(require_dashscope_api_key if primary == 'qwen' else require_api_key)()` must also become backend-aware — replace with:

    ```python
        if not self._has_provider_key(primary):
            if primary == 'qwen':
                resolve_qwen_target(self.qwen_api_backend, self.bbox_model_qwen)
            else:
                require_api_key()
    ```

    (This re-raises the descriptive `RuntimeError` for the missing key, matching the existing comment's intent — `_has_provider_key` swallows the exception, so this second call is only reached when the primary truly lacks its key, exactly as today.)

  - Every place `self.bbox_model_qwen` is read as the qwen model string (inside `_model_for`) works unchanged since sentinel resolution now happens inside `request_seat_bbox`/`request_seat` via `resolve_qwen_target`, not at the node layer — the node still just passes through whatever string is in the param (empty or not).
  - Thread `qwen_api_backend=self.qwen_api_backend` into both the `request_seat_bbox_chain(...)` call (bbox_select path) and the `request_seat_chain(...)` call (legacy point path).
  - This node's existing fail-fast-at-`__init__`-for-`bbox_select` behavior (no non-VLM seat detector — Qwen primary with no fallback if `vlm_fallback_provider=''`) is preserved: `_resolve_provider_chain` still raises the same way, just now checking the correct backend's key.

- [ ] **Step 6: Run to verify pass.** `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/kimi_api/test/test_seat_vlm.py src/kimi_api/test/test_seat_bbox_vlm.py -v`

- [ ] **Step 7: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/kimi_api/kimi_api/_seat_vlm.py src/kimi_api/kimi_api/_seat_bbox_vlm.py src/kimi_api/kimi_api/seat_recommend_bbox.py src/kimi_api/test/test_seat_vlm.py src/kimi_api/test/test_seat_bbox_vlm.py
git commit -m "feat(kimi_api): wire qwen_api_backend through seat_recommend_bbox"
```

---

### Task 6: `object_match_server.py` + `qwen_match_vlm.py`

**Files:**
- Modify: `src/tk_vision_specialized/tk_vision_specialized/qwen_match_vlm.py`
- Modify: `src/tk_vision_specialized/tk_vision_specialized/object_match_server.py`
- Test: no existing `test_qwen_match_vlm.py` — this task creates one.

**Interfaces:**
- Consumes: `resolve_qwen_target` (Task 1) — **first cross-package import**: `tk_vision_specialized` importing from `kimi_api._env`. `tk_vision_specialized/package.xml` already has `<exec_depend>kimi_api</exec_depend>` (verified present, line 21) so no manifest change is needed.
- Produces: `request_match_bboxes(..., qwen_api_backend='dashscope')`.

`qwen_match_vlm.py` is Qwen-only (no provider chain) and is called unconditionally, un-guarded, from `object_match_server.py`'s request-time callback (not at `__init__` — this node's existing behavior is to catch `QwenMatchError` in the callback and return `status=1`, never crashing at construction). **Do not change when this check happens — only make it backend-aware.**

This node also has its own `vlm_base_url` ROS param (line 92, default the concrete DashScope URL — **not already `''`**, unlike every other `*_model_qwen`-style param in this plan). Left untouched, this param would always win over the backend selection, silently defeating the toggle for this one node. It must become sentinel-shaped too, as part of this task.

- [ ] **Step 1: Write the failing tests**

Create `src/tk_vision_specialized/test/test_qwen_match_vlm.py`:

```python
"""Unit tests for tk_vision_specialized.qwen_match_vlm's backend routing."""
import numpy as np
import pytest

from tk_vision_specialized.qwen_match_vlm import QwenMatchError, request_match_bboxes


def _img():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class _FakeOpenAI:
    last_init = None

    def __init__(self, **kw):
        type(self).last_init = kw

    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                class _Msg:
                    content = '{"detections": []}'
                class _Choice:
                    message = _Msg()
                class _Resp:
                    choices = [_Choice()]
                return _Resp()


def test_request_match_bboxes_dashscope_default(monkeypatch):
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    monkeypatch.setattr('tk_vision_specialized.qwen_match_vlm.OpenAI', _FakeOpenAI)

    request_match_bboxes(_img(), 'data:url', item_name='mug', qwen_api_backend='dashscope')

    assert _FakeOpenAI.last_init['base_url'] == \
        'https://dashscope.aliyuncs.com/compatible-mode/v1'
    assert _FakeOpenAI.last_init['api_key'] == 'ds-key'


def test_request_match_bboxes_openrouter_backend(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    monkeypatch.setattr('tk_vision_specialized.qwen_match_vlm.OpenAI', _FakeOpenAI)

    request_match_bboxes(_img(), 'data:url', item_name='mug', qwen_api_backend='openrouter')

    assert _FakeOpenAI.last_init['base_url'] == 'https://openrouter.ai/api/v1'
    assert _FakeOpenAI.last_init['api_key'] == 'or-key'


def test_request_match_bboxes_openrouter_missing_key_raises(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with pytest.raises(QwenMatchError, match='OPENROUTER_API_KEY'):
        request_match_bboxes(_img(), 'data:url', item_name='mug', qwen_api_backend='openrouter')


def test_request_match_bboxes_explicit_base_url_override_wins(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    monkeypatch.setattr('tk_vision_specialized.qwen_match_vlm.OpenAI', _FakeOpenAI)

    request_match_bboxes(
        _img(), 'data:url', item_name='mug',
        base_url='https://self-hosted.example/v1', qwen_api_backend='dashscope')

    assert _FakeOpenAI.last_init['base_url'] == 'https://self-hosted.example/v1'
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/tk_vision_specialized/test/test_qwen_match_vlm.py -v`
Expected: FAIL (`request_match_bboxes` doesn't yet accept `qwen_api_backend`; `OpenAI` isn't imported at module scope yet — check the exact import site your patch target needs to match, adjust the `monkeypatch.setattr` path if `qwen_match_vlm.py`'s `from openai import OpenAI` is inside the function rather than at module scope, in which case patch `openai.OpenAI` globally instead, matching the pattern other test files in this plan use).

- [ ] **Step 3: Implement in `qwen_match_vlm.py`**

Add `from kimi_api._env import resolve_qwen_target` to the imports. Remove `_KEY_NAMES`, `_resolve_api_key`, and `_DEFAULT_BASE_URL` (dead after this change — grep the file first to confirm nothing else references them). Change `request_match_bboxes`'s signature defaults:

```python
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
```

and replace the current:

```python
    api_key = _resolve_api_key()  # raises QwenMatchError if missing

    from openai import OpenAI

    top_k = max(1, min(int(top_k), 10))
    client = OpenAI(api_key=api_key, base_url=base_url)
```

with:

```python
    try:
        base_url, api_key, model = resolve_qwen_target(
            qwen_api_backend, model, base_url)
    except RuntimeError as exc:
        raise QwenMatchError(str(exc)) from exc

    from openai import OpenAI

    top_k = max(1, min(int(top_k), 10))
    client = OpenAI(api_key=api_key, base_url=base_url)
```

- [ ] **Step 4: Wire `object_match_server.py`**

In `_declare_parameters` (lines 89-102):

```python
self.declare_parameter('vlm_model', 'qwen3-vl-plus')
self.declare_parameter(
    'vlm_base_url',
    'https://dashscope.aliyuncs.com/compatible-mode/v1',
)
```

becomes:

```python
self.declare_parameter('vlm_model', '')
self.declare_parameter('vlm_base_url', '')
self.declare_parameter('qwen_api_backend', 'dashscope')
```

Read `self.qwen_api_backend = self.get_parameter('qwen_api_backend').value` alongside the existing `self.vlm_model`/`self.vlm_base_url` reads. In `_object_match_callback`, add `qwen_api_backend=self.qwen_api_backend` to the existing `request_match_bboxes(...)` call kwargs (leave `model=self.vlm_model, base_url=self.vlm_base_url` as-is — they now flow through as the sentinel/override values `resolve_qwen_target` expects).

- [ ] **Step 5: Run to verify pass**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/tk_vision_specialized/test/test_qwen_match_vlm.py -v`

- [ ] **Step 6: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/qwen_match_vlm.py src/tk_vision_specialized/tk_vision_specialized/object_match_server.py src/tk_vision_specialized/test/test_qwen_match_vlm.py
git commit -m "feat(tk_vision_specialized): wire qwen_api_backend through object_match_server"
```

---

### Task 7: `object_match_all_server.py` + `vlm_match_client.py` + `vlm_judge_client.py`

**Files:**
- Modify: `src/tk_vision_specialized/tk_vision_specialized/vlm_match_client.py`
- Modify: `src/tk_vision_specialized/tk_vision_specialized/vlm_judge_client.py`
- Modify: `src/tk_vision_specialized/tk_vision_specialized/object_match_all_server.py`
- Modify: `src/tk_vision_specialized/test/test_vlm_match_client.py`
- Modify: `src/tk_vision_specialized/test/test_vlm_judge_client.py`

**Interfaces:**
- Consumes: `resolve_qwen_target` (Task 1).
- Produces: `QwenMatchClient.__init__(self, model='', base_url='', qwen_api_backend='dashscope')`, `QwenJudgeClient.__init__(self, model='', base_url='', qwen_api_backend='dashscope')`, `build_match_client(provider, qwen_api_backend='dashscope', **opts)`, `build_judge_client(provider, qwen_api_backend='dashscope', **opts)`.

This node already fails fast at `__init__` today with **no non-VLM fallback** (confirmed: `build_match_client`/`build_judge_client` are called unguarded in `ObjectMatchAllServer.__init__`, and `QwenMatchClient`/`QwenJudgeClient` already raise `RuntimeError` synchronously on a missing key) — this task preserves that behavior exactly, just makes the key check backend-aware.

**Known side effect (see Global Constraints):** `QwenMatchClient`'s current key resolution (`_QWEN_KEY_NAMES = ('DASHCOPE_API_KEY', 'DASHSCOPE_API_KEY')`, typo-first) is replaced by `resolve_qwen_target`'s canonical-first order (via `kimi_api._env.require_dashscope_api_key()`). This breaks `test_qwen_client_resolves_dashcope_typo_first`'s current assertion — fix it in Step 1, not as an afterthought.

- [ ] **Step 1: Update tests first**

In `src/tk_vision_specialized/test/test_vlm_match_client.py`, replace the test at lines 143-149:

```python
def test_qwen_client_resolves_dashcope_typo_first(monkeypatch):
    """The workspace .env historically carries DASHCOPE_API_KEY (typo);
    that should resolve first for backward compatibility."""
    monkeypatch.setenv('DASHCOPE_API_KEY', 'typo-key')
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'official-key')
    client = QwenMatchClient(model='qwen3-vl-plus')
    assert client._api_key == 'typo-key'
```

with:

```python
def test_qwen_client_resolves_canonical_key_first(monkeypatch):
    """QwenMatchClient now shares kimi_api._env.require_dashscope_api_key(),
    which checks the canonical DASHSCOPE_API_KEY before the legacy typo'd
    DASHCOPE_API_KEY — the opposite order this client used before switching
    to the shared resolve_qwen_target (see docs/superpowers/specs/2026-07-03-
    qwen-openrouter-dashscope-toggle-design.md, "Components" section, on why
    centralizing onto one resolver was chosen over duplicating this logic).
    """
    monkeypatch.setenv('DASHCOPE_API_KEY', 'typo-key')
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'official-key')
    client = QwenMatchClient(model='qwen3-vl-plus')
    assert client._api_key == 'official-key'


def test_qwen_client_still_accepts_typo_alone(monkeypatch):
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.setenv('DASHCOPE_API_KEY', 'typo-key')
    client = QwenMatchClient(model='qwen3-vl-plus')
    assert client._api_key == 'typo-key'
```

Add a new test for the OpenRouter path, matching the file's existing `patch('tk_vision_specialized.vlm_match_client.OpenAI', FakeOpenAI)` construction pattern (see line 190-191):

```python
def test_qwen_client_openrouter_backend(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    with patch('tk_vision_specialized.vlm_match_client.OpenAI', FakeOpenAI):
        client = QwenMatchClient(model='', qwen_api_backend='openrouter')
    assert client._api_key == 'or-key'
    assert client._base_url == 'https://openrouter.ai/api/v1'
```

Apply the same two changes (order-flip test correction + new openrouter test) to `test_vlm_judge_client.py`, adapted to `QwenJudgeClient`'s equivalent assertions (`test_qwen_judge_client_init` at lines 52-58 doesn't test precedence, so no correction needed there — just add the new OpenRouter test).

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/tk_vision_specialized/test/test_vlm_match_client.py src/tk_vision_specialized/test/test_vlm_judge_client.py -v`

- [ ] **Step 3: Implement in `vlm_match_client.py`**

Add `from kimi_api._env import resolve_qwen_target` to the imports. Remove `_QWEN_DEFAULT_BASE_URL`, `_QWEN_KEY_NAMES`, `_QWEN_DEFAULT_MODEL` and `QwenMatchClient._resolve_api_key` (dead after this change). Replace `QwenMatchClient.__init__` (lines matching the quoted current version):

```python
    def __init__(
        self,
        model: str = '',
        base_url: str = '',
    ):
        self._api_key: str | None = self._resolve_api_key()
        if not self._api_key:
            raise RuntimeError(
                'DashScope API key not found in env (looked for '
                f'{_QWEN_KEY_NAMES})'
            )
        self._model = model or _QWEN_DEFAULT_MODEL
        self._base_url = base_url or _QWEN_DEFAULT_BASE_URL
```

with:

```python
    def __init__(
        self,
        model: str = '',
        base_url: str = '',
        qwen_api_backend: str = 'dashscope',
    ):
        self._base_url, self._api_key, self._model = resolve_qwen_target(
            qwen_api_backend, model, base_url)
```

`resolve_qwen_target` raises `RuntimeError` on a missing key, matching this constructor's existing raise-at-construction contract exactly — no try/except needed here (the caller, `build_match_client`, doesn't catch it today either).

- [ ] **Step 4: Implement in `vlm_judge_client.py`**

Same pattern for `QwenJudgeClient.__init__`: replace the manual `for name in _QWEN_KEY_NAMES: ...` loop + `if not self._api_key: raise RuntimeError(...)` block with a `resolve_qwen_target(qwen_api_backend, model, base_url)` call, add the import, remove the now-dead `_QWEN_KEY_NAMES`/`_QWEN_DEFAULT_BASE_URL`/`_QWEN_DEFAULT_MODEL` constants (the Gemini-only constants `_GEMINI_DEFAULT_BASE_URL`/`_GEMINI_DEFAULT_MODEL` stay — this task doesn't touch the Gemini leg).

Add `qwen_api_backend: str = 'dashscope'` to `build_match_client` and `build_judge_client`'s signatures, threading it into the `QwenMatchClient(**opts)`/`QwenJudgeClient(**opts)` constructor calls when `provider == 'qwen'` (the Gemini branches are unaffected — do not pass `qwen_api_backend` into `GeminiMatchClient`/`GeminiJudgeClient`, which don't accept it).

- [ ] **Step 5: Wire `object_match_all_server.py`**

Add `self.declare_parameter('qwen_api_backend', 'dashscope')`. In `__init__` (around lines 175-188), read it and thread it into both client-builder calls:

```python
qwen_api_backend = self.get_parameter('qwen_api_backend').value
provider = self.get_parameter('vlm_provider').value
judge_provider = (
    self.get_parameter('judge_provider').value or provider
)
model = self.get_parameter('vlm_model').value or ''
judge_model = self.get_parameter('judge_model').value or model
base_url = self.get_parameter('vlm_base_url').value or ''

self.match_client = build_match_client(
    provider, model=model, base_url=base_url,
    qwen_api_backend=qwen_api_backend,
)
self.judge_client = build_judge_client(
    judge_provider, model=judge_model,
    qwen_api_backend=qwen_api_backend,
)
```

Since both `match_client` and `judge_client` are fed the same `qwen_api_backend`, they can't diverge — matches the spec's "match/judge consistency holds by construction" note.

- [ ] **Step 6: Run to verify pass**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/tk_vision_specialized/test/test_vlm_match_client.py src/tk_vision_specialized/test/test_vlm_judge_client.py -v`

- [ ] **Step 7: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/vlm_match_client.py src/tk_vision_specialized/tk_vision_specialized/vlm_judge_client.py src/tk_vision_specialized/tk_vision_specialized/object_match_all_server.py src/tk_vision_specialized/test/test_vlm_match_client.py src/tk_vision_specialized/test/test_vlm_judge_client.py
git commit -m "feat(tk_vision_specialized): wire qwen_api_backend through object_match_all_server"
```

---

### Task 8: `waving_person_server.py` + `_waving_vlm.py`

**Files:**
- Modify: `src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py`
- Modify: `src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py`
- Modify: `src/tk_vision_specialized/test/test_waving_vlm.py`

**Interfaces:**
- Consumes: `resolve_qwen_target` (Task 1).
- Produces: `request_waving_persons(..., qwen_api_backend='dashscope')`, `request_waving_persons_chain(..., qwen_api_backend='dashscope')`, `has_provider_key(provider, qwen_api_backend='dashscope')`.

**This node MUST keep its graceful-disable behavior — do not make it fail fast.** `waving_person_server.py` defaults `vlm_provider='qwen'` (nominally primary within its VLM-only chain), but has a genuine non-VLM fallback: MediaPipe pose detection still serves waving detection with zero VLM calls. `_resolve_provider_chain` never raises today (a missing key just makes `has_provider_key` return `False`, producing an empty `chain`, logged as a warning); this task must preserve that exactly — it only changes *which* key `has_provider_key('qwen', ...)` checks.

- [ ] **Step 1: Update/add tests**

In `test_waving_vlm.py`, the existing test at lines 157-176 (`test_request_waving_persons_returns_boxes`) asserts `base_url`/`api_key` for the qwen leg with `model='qwen3-vl-plus'` passed explicitly — leave it as-is (it's testing explicit-model-honored-verbatim, still valid), but add `qwen_api_backend='dashscope'` to its `request_waving_persons(...)` call for explicitness. Add:

```python
def test_request_waving_persons_openrouter_backend(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    payload = json.dumps({'persons': []})
    fake = _make_fake_openai(lambda kw: payload)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_waving_persons(
        _img(), provider='qwen', model='', qwen_api_backend='openrouter')

    assert res.error is None
    assert fake.last_init['base_url'] == 'https://openrouter.ai/api/v1'
    assert fake.last_init['api_key'] == 'or-key'


def test_has_provider_key_qwen_respects_backend(monkeypatch):
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')

    assert has_provider_key('qwen', qwen_api_backend='dashscope') is False
    assert has_provider_key('qwen', qwen_api_backend='openrouter') is True
```

(Import `has_provider_key` at the top of the test file alongside the existing imports if not already imported.)

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/tk_vision_specialized/test/test_waving_vlm.py -v`

- [ ] **Step 3: Implement in `_waving_vlm.py`**

Add `from kimi_api._env import resolve_qwen_target` to imports. In `request_waving_persons` (lines 176-196), replace:

```python
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
```

with (add `qwen_api_backend: str = 'dashscope',` to the function signature first):

```python
    if provider == 'qwen':
        try:
            b_url, key, model = resolve_qwen_target(qwen_api_backend, model)
        except RuntimeError as exc:
            raise WavingVlmError(str(exc)) from exc
        reasoning = False
    elif provider == 'gemini':
        key = os.environ.get('OPENROUTER_API_KEY')
        if not key:
            raise WavingVlmError('gemini API key not set (OPENROUTER_API_KEY).')
        b_url, reasoning = _GEMINI_DEFAULT_BASE_URL, True
    else:
        raise WavingVlmError(f'unknown provider {provider!r} (expected qwen|gemini)')

    client = openai.OpenAI(api_key=key, base_url=b_url)
```

Add `qwen_api_backend: str = 'dashscope',` to `request_waving_persons_chain` too, threading it into its `request_waving_persons(...)` call.

Replace `_resolve_key`'s qwen branch — actually, `_resolve_key` becomes dead for the `'qwen'` case after the above change (only used by the old raise path); check whether it's still called anywhere else (e.g. by `has_provider_key`) before removing it. If `has_provider_key` currently calls `_resolve_key(provider) is not None`, rewrite `has_provider_key` instead of removing `_resolve_key` outright:

```python
def has_provider_key(provider: str, qwen_api_backend: str = 'dashscope') -> bool:
    if provider == 'qwen':
        try:
            resolve_qwen_target(qwen_api_backend, '')
            return True
        except RuntimeError:
            return False
    if provider == 'gemini':
        return bool(os.environ.get('OPENROUTER_API_KEY'))
    return False
```

If `_resolve_key` has no remaining callers after this, remove it along with `_QWEN_KEY_NAMES` (verify with a grep across the file before deleting).

- [ ] **Step 4: Wire `waving_person_server.py`**

`self.declare_parameter('vlm_model_qwen', 'qwen3-vl-plus')` (line 161) → `''`. Add `self.declare_parameter('qwen_api_backend', 'dashscope')` and `self.qwen_api_backend = self.get_parameter('qwen_api_backend').value`.

In `_resolve_provider_chain` (lines 407-432), the `build_provider_models(self.vlm_provider, self.vlm_fallback_provider, has_key=has_provider_key, model_for=model_for, logger=...)` call must pass a backend-aware key-check callable — since `has_provider_key` now takes `qwen_api_backend` as a second argument, wrap it:

```python
    chain = build_provider_models(
        self.vlm_provider, self.vlm_fallback_provider,
        has_key=lambda p: has_provider_key(p, self.qwen_api_backend),
        model_for=model_for,
        logger=self.get_logger())
```

Find the call site inside `_vlm_augment` that invokes `request_waving_persons_chain(...)` and add `qwen_api_backend=self.qwen_api_backend` to its kwargs.

This preserves the exact existing control flow: `_resolve_provider_chain` still never raises, `self._vlm_chain` still degrades to `[]` on a missing key (now the *correct* backend's key), and the per-request arming check (`if self._vlm_chain and request.min_waving_persons > 0 and ...`) is untouched.

- [ ] **Step 5: Run to verify pass**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/tk_vision_specialized/test/test_waving_vlm.py -v`

- [ ] **Step 6: Update the module docstring's "kimi_api-free" claim**

`_waving_vlm.py`'s docstring (lines 1-11) currently states it "stays kimi_api-free... the same decoupled convention `vlm_match_client.py` / `qwen_match_vlm.py` use." That's no longer true after this task (and Tasks 6-7). Update it to:

```python
"""Waving-person VLM client for the detect_waving fallback.

Mirrors the control flow of kimi_api/_seat_bbox_vlm.py (single call -> provider
chain, strict json_schema -> json_object fallback, errors-only fallthrough).
Imports kimi_api._env.resolve_qwen_target for the DashScope/OpenRouter
qwen_api_backend toggle (docs/superpowers/specs/2026-07-03-qwen-openrouter-
dashscope-toggle-design.md) — this deliberately overrides the module's prior
kimi_api-free convention to avoid a second copy of the backend-resolution
constant drifting out of sync with kimi_api's.

The VLM is asked for the whole-person box of every visibly-waving person so the
boxes overlap YOLO person masks; the server turns each box into a 3D centroid.
"""
```

- [ ] **Step 7: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/_waving_vlm.py src/tk_vision_specialized/tk_vision_specialized/waving_person_server.py src/tk_vision_specialized/test/test_waving_vlm.py
git commit -m "feat(tk_vision_specialized): wire qwen_api_backend through waving_person_server, preserve graceful-disable"
```

---

### Task 9: `placing_location_server.py` + `placing_vlm.py`

**Files:**
- Modify: `src/tk_vision_specialized/tk_vision_specialized/placing_vlm.py`
- Modify: `src/tk_vision_specialized/tk_vision_specialized/placing_location_server.py`
- Modify: `src/tk_vision_specialized/test/test_placing_vlm.py`

**Interfaces:**
- Consumes: `resolve_qwen_target` (Task 1).
- Produces: `request_placing_bboxes(..., qwen_api_backend='dashscope')`, `request_placing_bboxes_chain(..., qwen_api_backend='dashscope')`.

`placing_vlm.py` already imports `kimi_api._env` directly (it's the one file of the five `tk_vision_specialized` VLM modules that was never "kimi_api-free") — this task's import change is additive, not a convention override. This node never fails fast at `__init__` for either key (both Gemini-primary and Qwen-fallback checks are deferred to request time or chain-build time) — preserve that.

- [ ] **Step 1: Update/add tests**

In `test_placing_vlm.py`, the existing test at lines 79-90 (`test_request_placing_bboxes_qwen_uses_dashscope`) stays valid as-is (still exercises the dashscope path with an explicit model) — add `qwen_api_backend='dashscope'` to its call for explicitness. Add:

```python
def test_request_placing_bboxes_openrouter_backend(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    fake = _make_fake_openai(lambda kw: _ONE_REGION_PAYLOAD)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    boxes, _ranks, _elapsed = request_placing_bboxes(
        _img(), item_description='mug', model='', provider='qwen',
        qwen_api_backend='openrouter')

    assert fake.last_init['base_url'] == 'https://openrouter.ai/api/v1'
    assert fake.last_init['api_key'] == 'or-key'
```

The existing negative test `test_request_placing_bboxes_missing_key_raises` (lines 131-139) already stubs `load_env` and clears both DashScope key spellings — it remains valid unchanged (it's testing the `dashscope` backend, the default).

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/tk_vision_specialized/test/test_placing_vlm.py -v`

- [ ] **Step 3: Implement in `placing_vlm.py`**

Add `resolve_qwen_target` to the existing `from kimi_api._env import (...)` block. In `request_placing_bboxes` (lines 135-154 region), replace:

```python
    if provider == 'qwen':
        try:
            api_key = require_dashscope_api_key()
        except RuntimeError as exc:
            raise VlmPlacingError(str(exc)) from exc
        b_url = dashscope_base_url()
```

with (add `qwen_api_backend: str = 'dashscope',` to the function signature first):

```python
    if provider == 'qwen':
        try:
            b_url, api_key, model = resolve_qwen_target(qwen_api_backend, model)
        except RuntimeError as exc:
            raise VlmPlacingError(str(exc)) from exc
```

Check whether `dashscope_base_url`/`require_dashscope_api_key` are still referenced anywhere else in this file; if not, remove them from the import list (keep `base_url`, `require_api_key`, `load_env` — still used by the unchanged gemini branch). Add `qwen_api_backend: str = 'dashscope',` to `request_placing_bboxes_chain` too, threading it into its `request_placing_bboxes(...)` call.

- [ ] **Step 4: Wire `placing_location_server.py`**

`self.declare_parameter('placing_model_qwen', 'qwen3-vl-plus')` (line 77) → `''`. Add `self.declare_parameter('qwen_api_backend', 'dashscope')` and read it.

`_resolve_placing_provider_chain` (lines 93-120) currently:

```python
            try:
                require_dashscope_api_key()
                chain.append(('qwen', self.placing_model_qwen))
            except RuntimeError:
                self.get_logger().warn(
                    f'Fallback provider {fb!r} key missing; fallback disabled.'
                )
```

becomes:

```python
            try:
                _, _, resolved_model = resolve_qwen_target(
                    self.qwen_api_backend, self.placing_model_qwen)
                chain.append(('qwen', resolved_model))
            except RuntimeError:
                self.get_logger().warn(
                    f'Fallback provider {fb!r} key missing; fallback disabled.'
                )
```

(Add `resolve_qwen_target` to this file's `from kimi_api._env import (...)` too — `require_dashscope_api_key` may become unused here; check before removing.) Thread `qwen_api_backend=self.qwen_api_backend` into the `request_placing_bboxes_chain(...)` call site.

- [ ] **Step 5: Run to verify pass**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/tk_vision_specialized/test/test_placing_vlm.py -v`

- [ ] **Step 6: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/tk_vision_specialized/tk_vision_specialized/placing_vlm.py src/tk_vision_specialized/tk_vision_specialized/placing_location_server.py src/tk_vision_specialized/test/test_placing_vlm.py
git commit -m "feat(tk_vision_specialized): wire qwen_api_backend through placing_location_server"
```

---

### Task 10: `generalist_node.py` — fail-fast validation (different mechanism)

**Files:**
- Modify: `src/object_detection_generalist/object_detection_generalist/vlm_bbox.py`
- Modify: `src/object_detection_generalist/object_detection_generalist/generalist_node.py`
- Modify: `src/object_detection_generalist/test/test_vlm_bbox_fallback.py`

**Interfaces:**
- Produces: `validate_qwen_backend_models(qwen_api_backend: str, vlm_model: str, vlm_fallback_models: list[str]) -> None` in `vlm_bbox.py` — pure function, raises `ValueError`. `GeneralistDetectionNode.qwen_api_backend` (str, from the new param).

This node does **not** use `resolve_qwen_target` — it keeps its existing `dashscope/model`-prefix routing (`_split_provider`, already dual-host, already lazily degrades per-attempt on a missing key inside `request_bboxes`/`_client_for` — that graceful per-model-attempt fallback is unaffected by this task). The only new behavior is an **init-time fail-fast** if `qwen_api_backend='openrouter'` but a param still explicitly points at DashScope — per the spec, this is a validation gate, not an auto-rewrite, and it must cover all three DashScope-pointing surfaces: the primary `vlm_model`, `vlm_fallback_models`, and (via `prefer_dashscope_qwen`) `dashscope_qwen_model` — the last of which is already absorbed into `self.vlm_model` by the existing code before this check runs, so checking `vlm_model` + `vlm_fallback_models` covers all three without a third explicit check.

- [ ] **Step 1: Write the failing tests**

In `src/object_detection_generalist/test/test_vlm_bbox_fallback.py`, this file loads `vlm_bbox.py` standalone via `importlib` (see the existing `_SPEC`/`_SRC` block at the top) — add tests using that same `vlm_bbox` module object:

```python
def test_validate_qwen_backend_models_dashscope_always_passes():
    # dashscope backend never rejects any mix of prefixed/unprefixed entries —
    # _split_provider already routes per-entry regardless.
    vlm_bbox.validate_qwen_backend_models(
        'dashscope', 'dashscope/qwen3-vl-plus', ['openai/gpt-4o'])


def test_validate_qwen_backend_models_openrouter_passes_when_clean():
    vlm_bbox.validate_qwen_backend_models(
        'openrouter', 'google/gemini-2.5-flash', ['qwen/qwen3-vl-32b-instruct'])


def test_validate_qwen_backend_models_openrouter_rejects_dashscope_primary():
    with pytest.raises(ValueError, match='dashscope/qwen3-vl-plus'):
        vlm_bbox.validate_qwen_backend_models(
            'openrouter', 'dashscope/qwen3-vl-plus', [])


def test_validate_qwen_backend_models_openrouter_rejects_dashscope_fallback():
    with pytest.raises(ValueError, match='dashscope/qwen3-vl-plus'):
        vlm_bbox.validate_qwen_backend_models(
            'openrouter', 'google/gemini-2.5-flash', ['dashscope/qwen3-vl-plus'])


def test_validate_qwen_backend_models_invalid_backend_raises():
    with pytest.raises(ValueError, match='qwen_api_backend'):
        vlm_bbox.validate_qwen_backend_models('bogus', 'google/gemini-2.5-flash', [])
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/object_detection_generalist/test/test_vlm_bbox_fallback.py -v`
Expected: FAIL with `AttributeError: module 'vlm_bbox_source' has no attribute 'validate_qwen_backend_models'`

- [ ] **Step 3: Implement `validate_qwen_backend_models` in `vlm_bbox.py`**

Add near `_split_provider` (after its definition, around line 305):

```python
_VALID_QWEN_API_BACKENDS = ('dashscope', 'openrouter')


def validate_qwen_backend_models(
    qwen_api_backend: str, vlm_model: str, vlm_fallback_models: list,
) -> None:
    """Raise ValueError if qwen_api_backend='openrouter' but vlm_model or any
    entry of vlm_fallback_models still points at DashScope.

    Called once at node __init__ (fail-fast, not auto-rewrite — see
    docs/superpowers/specs/2026-07-03-qwen-openrouter-dashscope-toggle-design.md
    §Components on why an earlier draft's auto-flip was rejected). Only
    validates the openrouter direction: qwen_api_backend='dashscope' always
    passes, since _split_provider already routes any mix of dashscope/- and
    non-prefixed entries per-entry under that backend.
    """
    if qwen_api_backend not in _VALID_QWEN_API_BACKENDS:
        raise ValueError(
            f'Invalid qwen_api_backend {qwen_api_backend!r}; expected one '
            f'of {_VALID_QWEN_API_BACKENDS}.'
        )
    if qwen_api_backend != 'openrouter':
        return
    offending = [
        m for m in [vlm_model, *vlm_fallback_models]
        if m.startswith(_DASHSCOPE_PREFIX)
    ]
    if offending:
        raise ValueError(
            "qwen_api_backend='openrouter' but these params still point at "
            f"DashScope: {offending}. Pass an explicit OpenRouter-pointing "
            "value for vlm_model / dashscope_qwen_model / vlm_fallback_models."
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/object_detection_generalist/test/test_vlm_bbox_fallback.py -v`

- [ ] **Step 5: Wire `generalist_node.py`**

Add `validate_qwen_backend_models` to the existing `from .vlm_bbox import VlmBboxError, request_bboxes` line (line 58) → `from .vlm_bbox import VlmBboxError, request_bboxes, validate_qwen_backend_models`.

In `_declare_parameters` (after line 191, the `dashscope_qwen_model` declaration):

```python
self.declare_parameter('qwen_api_backend', 'dashscope')
```

In `_load_parameters` (after line 251, the end of the `prefer_dashscope_qwen` block — i.e. after `self.vlm_model`/`self.vlm_fallback_models` have their final resolved values):

```python
        self.qwen_api_backend = self.get_parameter('qwen_api_backend').value
        validate_qwen_backend_models(
            self.qwen_api_backend, self.vlm_model, self.vlm_fallback_models,
        )
```

Since `_load_parameters` runs during `__init__` (via the parent class's construction sequence, confirmed by `super()._load_parameters()` at line 233 and this being an override), an uncaught `ValueError` here fails node construction — the intended fail-fast behavior.

- [ ] **Step 6: Manually verify the fail-fast fires**

Run: `cd /home/tinker/tk25_ws && source install/setup.bash 2>/dev/null; python3 -c "
import rclpy
rclpy.init()
from object_detection_generalist.generalist_node import GeneralistDetectionNode
try:
    node = GeneralistDetectionNode()
    print('FAIL: node constructed without error')
except ValueError as e:
    print(f'OK: raised as expected: {e}')
" --ros-args -p qwen_api_backend:=openrouter`

Expected output: `OK: raised as expected: qwen_api_backend='openrouter' but these params still point at DashScope: ['dashscope/qwen3-vl-plus']` (the default `vlm_fallback_models` is still `['dashscope/qwen3-vl-plus']` at this point, since Task 10 doesn't change that default — only Task 12's docs update discusses it; the operator must pass an explicit override to actually use `openrouter`, matching the "no auto-flip" design). If this command fails for unrelated reasons (missing camera topics, CUDA, etc. at full node construction), it's acceptable to instead write a small standalone script that imports just `_declare_parameters`/`_load_parameters` in isolation, or note in the commit message that full-node manual verification requires hardware/GPU access not available in this environment and was verified via the unit test in Steps 1-4 instead.

- [ ] **Step 7: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/object_detection_generalist/object_detection_generalist/vlm_bbox.py src/object_detection_generalist/object_detection_generalist/generalist_node.py src/object_detection_generalist/test/test_vlm_bbox_fallback.py
git commit -m "feat(object_detection_generalist): add qwen_api_backend fail-fast validation to generalist_node"
```

---

### Task 11: `vision_bringup.launch.py` — launch-arg wiring

**Files:**
- Modify: `src/vision_bringup/launch/vision_bringup.launch.py`

**Interfaces:**
- Consumes: nothing new (all 5 nodes already declare `qwen_api_backend` after Tasks 2, 3, 10 — `feature_recognition`, `feature_matching`, `waving_person_server`, `generalist_node` — plus `seat_recommend_bbox` from Task 5).

- [ ] **Step 1: Add the launch argument**

In `generate_launch_description`'s `args` list (lines 97-108), add after the `enable_pick_place` declaration:

```python
        DeclareLaunchArgument('qwen_api_backend', default_value='dashscope',
                              description="'dashscope' or 'openrouter' — "
                              "routes all Qwen VLM calls on this launch."),
```

- [ ] **Step 2: Wire it into the 5 managed nodes**

The `_node()` helper (lines 83-87) already forwards `**kwargs` to `Node(...)`, and `Node(...)` accepts a `parameters=[...]` kwarg. Add `parameters=[{'qwen_api_backend': LaunchConfiguration('qwen_api_backend')}]` to each of these 5 `_node(...)` calls (lines 117-135):

```python
        _node('object_detection_generalist', 'generalist_node',
              _if('enable_generalist'),
              parameters=[{'qwen_api_backend': LaunchConfiguration('qwen_api_backend')}]),
        _node('vision_util', 'door_detection',
              _if('enable_door')),
        # --- shared across tasks (OR-gated, spawn once) ---
        _node('object_detection_new', 'yolo_seg_node',
              _if_any('enable_hri', 'enable_gpsr')),
        _node('vision_track', 'person_track_server',
              _if_any('enable_hri', 'enable_gpsr')),
        _node('tk_vision_specialized', 'waving_person_server',
              _if_any('enable_hri', 'enable_gpsr', 'enable_restaurant'),
              parameters=[{'qwen_api_backend': LaunchConfiguration('qwen_api_backend')}]),
        _node('kimi_api', 'feature_recognition',
              _if_any('enable_hri', 'enable_gpsr'),
              parameters=[{'qwen_api_backend': LaunchConfiguration('qwen_api_backend')}]),
        _node('pan_tilt', 'follow_head',
              _if_any('enable_hri', 'enable_restaurant'),
              parameters=[pan_tilt_cfg]),
        # --- HRI-only ---
        _node('kimi_api', 'feature_matching', _if('enable_hri'),
              parameters=[{'qwen_api_backend': LaunchConfiguration('qwen_api_backend')}]),
        _node('kimi_api', 'seat_recommend_bbox', _if('enable_hri'),
              parameters=[{'qwen_api_backend': LaunchConfiguration('qwen_api_backend')}]),
        # --- GPSR-only ---
        _node('vision_util', 'get_image', _if('enable_gpsr')),
```

Leave `door_detection`, `yolo_seg_node`, `person_track_server`, `follow_head`, `get_image` untouched — none of them call a Qwen VLM.

- [ ] **Step 3: Update the module docstring's "Always-on core" / node list to mention the new arg**

In the docstring's launch-invocation examples (lines 13-16), add one line showing the new argument:

```python
    ros2 launch vision_bringup vision_bringup.launch.py enable_hri:=true qwen_api_backend:=openrouter
```

- [ ] **Step 4: Verify the launch file parses**

Run: `cd /home/tinker/tk25_ws && source install/setup.bash 2>/dev/null; python3 -c "
from launch import LaunchDescription
import importlib.util
spec = importlib.util.spec_from_file_location(
    'vision_bringup_launch',
    'src/tk26_vision/src/vision_bringup/launch/vision_bringup.launch.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ld = mod.generate_launch_description()
assert isinstance(ld, LaunchDescription)
print('OK: launch description built without error,', len(ld.entities), 'entities')
"`
Expected: `OK: launch description built without error, N entities` (no traceback). This only checks the Python parses and `generate_launch_description()` doesn't raise — it does not actually launch nodes.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/vision_bringup/launch/vision_bringup.launch.py
git commit -m "feat(vision_bringup): add qwen_api_backend launch argument"
```

---

### Task 12: Documentation — `CLAUDE.md` param table + convention notes

**Files:**
- Modify: `CLAUDE.md` (in `src/tk26_vision/`)

**Interfaces:** none — documentation only.

- [ ] **Step 1: Update the `kimi_api/*` param line**

Find the line (currently around line 236) reading:

> `... plus a per-node Qwen model param (`feature_model_qwen` / `match_model_qwen` / `categorize_model_qwen`, all default `'qwen3-vl-plus'`) ...`

Change `all default 'qwen3-vl-plus'` to `all default '' — sentinel for "use the backend's own default model", see qwen_api_backend below`. Add one sentence after that paragraph:

> Every kimi_api and tk_vision_specialized Qwen call site additionally takes a `qwen_api_backend` param (`'dashscope'` default, or `'openrouter'`) that selects which host serves the Qwen leg — see `docs/superpowers/specs/2026-07-03-qwen-openrouter-dashscope-toggle-design.md`.

- [ ] **Step 2: Update the `seat_recommend_bbox` line**

Find the line (currently around line 237) with `bbox_model_qwen (default 'qwen3-vl-plus')` — change to `bbox_model_qwen (default '' — sentinel, see qwen_api_backend)`.

- [ ] **Step 3: Update the `waving_person_server` line**

Find the line (currently around line 239) with:

> `vlm_model_qwen (qwen3-vl-plus) / vlm_model_gemini (google/gemini-2.5-pro) ... resolved via _waving_vlm.py (no kimi_api import; same decoupled convention as vlm_match_client.py)`

Change `vlm_model_qwen (qwen3-vl-plus)` to `vlm_model_qwen (default '' — sentinel, see qwen_api_backend)`. Remove the `(no kimi_api import; same decoupled convention as vlm_match_client.py)` parenthetical entirely — it's no longer accurate after Task 8 (and Tasks 6-7 for the two files it names).

- [ ] **Step 4: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for the qwen_api_backend sentinel defaults and dropped kimi_api-free convention"
```

---

### Task 13: Benchmark and lock in the default OpenRouter Qwen model

**Files:**
- Create: `src/kimi_api/scripts/benchmark_qwen_backend.py`
- Modify: `src/kimi_api/kimi_api/_env.py` (update `_OPENROUTER_QWEN_DEFAULT_MODEL` with the winning result)

**Interfaces:** none new — this task only changes the value of the constant Task 1 introduced.

This is a pre-competition-deployment gate, not a code-correctness gate — Tasks 1-12 do not depend on its outcome (they ship with the provisional `qwen/qwen3-vl-32b-instruct` default). Do not skip it before actually relying on `qwen_api_backend='openrouter'` on a robot.

- [ ] **Step 1: Write the benchmark script**

Create `src/kimi_api/scripts/benchmark_qwen_backend.py`:

```python
#!/usr/bin/env python3
"""Benchmark candidate OpenRouter Qwen models against the three axes the
design spec requires before qwen_api_backend='openrouter' can be trusted:
image modality, bbox coordinate format (vs the known-position-target
methodology object_detection_generalist/vlm_bbox.py already uses), and
round-trip latency.

Requires OPENROUTER_API_KEY and DASHSCOPE_API_KEY in the environment (or a
workspace-root .env). Run manually — this is not part of the pytest suite,
it makes real paid API calls.

Usage:
    python3 src/kimi_api/scripts/benchmark_qwen_backend.py \\
        --image /path/to/a/frame/with/a/known-position/object.jpg \\
        --known-box 520 240 760 480 \\
        --candidates qwen/qwen3-vl-32b-instruct qwen/qwen3.6-plus qwen/qwen3.7-plus
"""
import argparse
import base64
import json
import time

from openai import OpenAI

from kimi_api._env import base_url, dashscope_base_url, load_env, require_api_key


def _encode(path: str) -> str:
    with open(path, 'rb') as f:
        return 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode()


def _probe_model(client: OpenAI, model: str, data_url: str, known_box) -> dict:
    """One round-trip: ask for a bbox, time it, check modality + format."""
    prompt = (
        'You are a precise visual grounding model. Return only JSON: '
        '{"box_2d": [x1, y1, x2, y2]} for the single most prominent object '
        'in the image, in pixel coordinates.'
    )
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': data_url}},
                ],
            }],
            response_format={'type': 'json_object'},
            timeout=30.0,
        )
        elapsed = time.perf_counter() - t0
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        box = parsed.get('box_2d')
        return {
            'model': model, 'ok': True, 'elapsed_s': elapsed,
            'raw_box': box, 'known_box': known_box,
        }
    except Exception as exc:  # noqa: BLE001 - benchmark script, report everything
        return {
            'model': model, 'ok': False,
            'elapsed_s': time.perf_counter() - t0, 'error': str(exc),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--known-box', type=int, nargs=4, required=True,
                        metavar=('X1', 'Y1', 'X2', 'Y2'))
    parser.add_argument('--candidates', nargs='+', required=True)
    args = parser.parse_args()

    load_env()
    data_url = _encode(args.image)
    client = OpenAI(api_key=require_api_key(), base_url=base_url())

    results = [
        _probe_model(client, model, data_url, tuple(args.known_box))
        for model in args.candidates
    ]

    print(json.dumps(results, indent=2))
    print(
        '\nManually compare each result\'s raw_box against known_box to '
        'determine coordinate format (pixel xyxy vs 0-1000-normalized xyxy — '
        'see object_detection_generalist/vlm_bbox.py:70-76 for the reference '
        'known-position-target check), and elapsed_s for latency from this '
        'machine. Run once per deployment region if possible. Update '
        '_OPENROUTER_QWEN_DEFAULT_MODEL in kimi_api/_env.py with the winner.'
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it manually (requires live API keys — operator step, not automated)**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
source .venv-vision-main/bin/activate 2>/dev/null || true
python3 src/kimi_api/scripts/benchmark_qwen_backend.py \
    --image /path/to/a/calibration/frame.jpg \
    --known-box 520 240 760 480 \
    --candidates qwen/qwen3-vl-32b-instruct qwen/qwen3.6-plus qwen/qwen3.7-plus
```

If this environment has no live `OPENROUTER_API_KEY`/camera calibration image available, this step cannot be executed here — leave `_OPENROUTER_QWEN_DEFAULT_MODEL` at its Task 1 provisional value (`qwen/qwen3-vl-32b-instruct`) and flag in the commit message that this step is pending an operator run with real credentials before competition use.

- [ ] **Step 3: Update the constant with the winning model**

In `src/kimi_api/kimi_api/_env.py`, change `_OPENROUTER_QWEN_DEFAULT_MODEL`'s value to the benchmark winner, and update its comment to record the result (image modality confirmed, bbox format confirmed/decoder-extended if needed, latency numbers per region) instead of "provisional... not yet verified."

- [ ] **Step 4: Re-run the full test suite to confirm nothing hardcoded the provisional value**

Run: `cd /home/tinker/tk25_ws/src/tk26_vision && python3 -m pytest src/kimi_api/test/ src/tk_vision_specialized/test/ src/object_detection_generalist/test/ -q`
Expected: all pass (Task 1's tests assert behavior via the constant, not its literal string value, so this should be unaffected — if any test does hardcode `qwen/qwen3-vl-32b-instruct` literally, update it here).

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add src/kimi_api/scripts/benchmark_qwen_backend.py src/kimi_api/kimi_api/_env.py
git commit -m "feat(kimi_api): add OpenRouter Qwen backend benchmark script + lock in default model"
```

---

### Task 14: Pre-deployment smoke checklist

**Files:**
- Create: `src/tk26_vision/docs/qwen-openrouter-smoke-checklist.md`

**Interfaces:** none — documentation only, operator-run checklist.

Per the spec's Testing section, a single representative-node smoke check is insufficient — a wrong bbox coordinate format fails silently (wrong geometry, not an error) on 5 of the 6 bbox-producing consumers, since only `vlm_bbox.py` has model-family-aware decoding (Task 10 confirms this, unchanged by this plan). This task documents the manual verification an operator must run before trusting `qwen_api_backend='openrouter'` in competition.

- [ ] **Step 1: Write the checklist**

Create `src/tk26_vision/docs/qwen-openrouter-smoke-checklist.md`:

```markdown
# Pre-deployment smoke checklist: qwen_api_backend=openrouter

Run this after Task 13's benchmark has locked in a real default model, before
relying on `qwen_api_backend=openrouter` in a competition run. Automated
tests (Tasks 1-10 above) cover code correctness; this covers whether the
*chosen model* actually produces correct results on real hardware.

For each row: launch the node with `-p qwen_api_backend:=openrouter`, trigger
the relevant service/action against a scene with a known object at a known
pixel position (reuse `object_detection_generalist/vlm_bbox.py:70-76`'s
known-position-target method — place an object, measure its true bbox by
hand, compare against what comes back), and confirm the returned coordinates
land within a few pixels of the true position (not offset by a 0-1000 vs
pixel scale-factor error, which is the specific silent-failure mode this
checklist exists to catch).

| Node | Service/action | Verified? | Notes |
|---|---|---|---|
| `seat_recommend_bbox` (`bbox_select` strategy) | `/seat_recommend_bbox` | ☐ | |
| `object_match_server` | `/object_match` | ☐ | |
| `object_match_all_server` | `/object_match_all` | ☐ | |
| `waving_person_server` | `/detect_waving_persons` (force VLM path via `min_waving_persons`) | ☐ | |
| `placing_location_server` | `/placing_location` | ☐ | |
| `generalist_node` (`vlm_bbox` path) | `/object_detection_generalist` with `use_vlm_sam_fallback:=true` | ☐ | Already has model-family-aware decoding — lower risk, verify anyway |

Also confirm text-only paths still work correctly (lower risk, but cheap to
check): `feature_recognition`, `feature_matching`, `grocery_categorize` —
each returns a sensible non-garbled answer, not just "no crash."

Record the date, robot, and model string tested. Re-run this checklist any
time `_OPENROUTER_QWEN_DEFAULT_MODEL` changes.
```

- [ ] **Step 2: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git add docs/qwen-openrouter-smoke-checklist.md
git commit -m "docs: add pre-deployment smoke checklist for qwen_api_backend=openrouter"
```

---

## Self-Review Notes

**Spec coverage:** Motivation/Scope (Global Constraints + all 14 tasks cover the 11 call sites + generalist + launch + docs) — ✅. Mechanism/no-env-var (Global Constraints, every task's `declare_parameter('qwen_api_backend', 'dashscope')` is a literal default) — ✅. No credential changes (no task adds a new env var or key) — ✅. Components/shared resolver + tk_vision_specialized override (Task 1, Tasks 6-9) — ✅. Model substitution/sentinel (Tasks 2-9 change 7 params' defaults to `''`; Task 6 additionally sentinels `vlm_base_url`) — ✅. Default model deferred (Task 13) — ✅. Error handling: fail-fast where no non-VLM fallback (Task 7's `object_match_all_server`, already-existing), graceful-disable preserved (Task 8's `waving_person_server`, Task 9's `placing_location_server`), backend-aware arming (Tasks 2-5, 8-9's `_resolve_*_provider_chain`/`has_provider_key` rewrites) — ✅. Generalist fail-fast on 3 DashScope-pointing surfaces via `vlm_model`+`vlm_fallback_models` (Task 10) — ✅. Launch wiring for the 5 managed nodes (Task 11) — ✅. Docs/convention updates (Task 8 Step 6, Task 12) — ✅. Testing: resolver unit tests (Task 1), per-file test updates (Tasks 2-10), smoke checklist (Task 14) — ✅. `object_match_all_server`'s `vlm_base_url` precedence noted as "explicit wins" (Task 7 Step 5 — `base_url or ''` pattern, unchanged from today, still explicit-wins) — ✅.

**Placeholder scan:** no TBD/TODO; Task 13/14 explicitly note they require live credentials/hardware not available in a sandboxed environment and specify the fallback (keep provisional constant, flag for operator) rather than leaving a gap unaddressed.

**Type consistency:** `resolve_qwen_target(backend, model_param_value, base_url_override='') -> tuple[str, str, str]` used identically in Tasks 2-9. `validate_qwen_backend_models(qwen_api_backend, vlm_model, vlm_fallback_models) -> None` used identically in Task 10. `qwen_api_backend` is the parameter name everywhere (never `backend` or `qwen_backend` at the ROS-param layer) — verified consistent across all 11 node-wiring steps.
