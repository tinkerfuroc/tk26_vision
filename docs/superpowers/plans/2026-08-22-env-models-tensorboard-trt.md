# VLM model IDs from `.env`, tensorboard upgrade, FoundationStereo TensorRT revival — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every vision VLM model default resolve from `.env` (`VISION_*` keys with a fallback chain), move `.venv-vision-main` to tensorboard 2.20 / protobuf 6.33, and bring FoundationStereo's TensorRT path to life on this box (tensorrt-cu12 10.16 in `.venv-fs`, weights fetched, FP16 engines built locally).

**Architecture:** A dependency-free resolver module in `vision_util` is the single source of model defaults; every `declare_parameter` / module constant calls it at declaration time (after `.env` is loaded). The two venv changes are `--no-deps` installs gated by freeze diffs. Two new scripts under `src/foundation_stereo/scripts/` fetch the `23-36-37` checkpoint and build the two-stage engines with the TensorRT Python API into a `weights_root` that exists.

**Tech Stack:** Python 3.10, ROS 2 Humble (rclpy params), python-dotenv, pytest; pip/uv venvs; TensorRT 10.16 Python API, ONNX export via the vendored Fast-FoundationStereo scripts, gdown.

**Spec:** `docs/superpowers/specs/2026-08-22-env-models-tensorboard-trt-design.md`

## Global Constraints

- **Venv freeze rule (user-mandated):** every venv install is `pip install --no-deps <named packages>`; `pip freeze | sort` before/after may differ **only** in the named lines; `pip check` must be clean (or byte-identical to before, where pre-existing complaints exist); any other movement → STOP and report, never "fix" by installing more.
- `.venv-vision-main` / `.venv-fs` live only in the main checkout `/home/tinker/tk25_ws/src/tk26_vision/` (git-ignored). Code is edited in the worktree.
- `tk_vision_specialized` VLM clients must not import `kimi_api` — the resolver lives in `vision_util.vlm_models`.
- Resolver fallback chains (exact): `vision_vlm_model()`: `VISION_VLM_MODEL` → `LLM_MODEL` → `'google/gemini-2.5-pro'`; `vision_flash_model()`: `VISION_VLM_FLASH_MODEL` → `FLASH_MODEL` → `'google/gemini-2.5-flash'`; `vision_qwen_model()`: `VISION_QWEN_MODEL` → `'qwen3-vl-plus'`. Empty/whitespace values count as unset.
- With no `VISION_*` / `LLM_MODEL` / `FLASH_MODEL` set, every declared default must equal today's literal (parity).
- `.env` additions (workspace root `/home/tinker/tk25_ws/.env`, outside this repo): `VISION_VLM_MODEL=google/gemini-3.1-pro-preview`, `VISION_VLM_FLASH_MODEL=google/gemini-3.7-flash`, `VISION_QWEN_MODEL=qwen3-vl-plus`. Never print, paste, or commit the file's secret values.
- tensorboard target `2.20.0`, `tensorboard-data-server 0.7.2`, `protobuf 6.33.6`. TensorRT target `tensorrt-cu12==10.16.1.11` (+ `-bindings`, `-libs` same version) from `https://pypi.nvidia.com`.
- FoundationStereo: checkpoint `23-36-37` (`model_best_bp2_serialize.pth`), export `--height 576 --width 960 --valid_iters 4 --max_disp 192`, engines FP16, variant dir `output_two_stage`, `weights_root=~/.cache/tk26_vision/weights/foundation_stereo`. GPU work runs under `nice -n 10`; the GPU is shared with a running simulator.
- Out of scope: monocular_depth removal, opencv, openai 3.x, torch, numpy 2, `.venv-calib`, D405 variant.
- Shell setup for every task (run once per shell):
  ```bash
  export WT=/home/tinker/tk25_ws/src/tk26_vision/.claude/worktrees/env-models-tensorboard-trt
  export MAIN=/home/tinker/tk25_ws/src/tk26_vision
  export VENV=$MAIN/.venv-vision-main
  export VENV_FS=$MAIN/.venv-fs
  source /opt/ros/humble/setup.bash
  source /home/tinker/tk25_ws/install/setup.bash
  export ROS2_PTH_WARNED=1
  export PYTHONPATH=$WT/src/vision_util:$WT/src/kimi_api:$WT/src/tk_vision_specialized:$WT/src/object_detection_generalist:$PYTHONPATH
  cd $WT
  ```
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01QtfVtdCLKh4HoFMVBd2TvR
  ```

---

## File structure

| Path | Responsibility |
|---|---|
| `src/vision_util/vision_util/vlm_models.py` (create) | the three resolvers; `os` only |
| `src/vision_util/test/test_vlm_models.py` (create) | precedence / empty-string tests |
| `src/kimi_api/kimi_api/_env.py` (modify) | re-export; aliases; `_DASHSCOPE_DEFAULT_MODEL` → resolver |
| `src/kimi_api/kimi_api/seat_recommend_bbox.py`, `src/kimi_api/seat_bench/providers.py` (modify) | defaults → resolver |
| `src/object_detection_generalist/object_detection_generalist/generalist_node.py` (modify) | defaults → resolver |
| `src/tk_vision_specialized/tk_vision_specialized/{placing_location_server,waving_person_server,vlm_match_client,vlm_judge_client,vlm_match_client_gemini}.py` (modify) | defaults → resolver |
| `src/tk_vision_specialized/scripts/produce_match_ground_truth.py`, `scripts/object_scan_webui/scan_core.py`, `scripts/compare_feature_matching_models.py`, `scripts/tests/manual/{gemini_bbox_decode,seat_recommend_vlm_bench,vlm_timeout_isolation,vlm_provider_bench,web_image_vlm_smoke}.py` (modify) | argparse/constant defaults → resolver |
| `src/tk_vision_specialized/test/test_no_hardcoded_vlm_models.py` (create) | grep-guard: no legacy model literals outside the resolver |
| `src/kimi_api/.env.example` (modify); `/home/tinker/tk25_ws/.env` (append, outside repo) | keys |
| `.venv-vision-main.uv-project/pyproject.toml`, `requirements.txt` (modify) | tensorboard/protobuf pins |
| `src/foundation_stereo/foundation_stereo/stereo_runner.py` (modify) | `expanduser(weights_root)` |
| `src/foundation_stereo/config/foundation_stereo.yaml`, `foundation_stereo_node.py` (modify) | new `weights_root` default |
| `src/foundation_stereo/scripts/fetch_fast_fs_weights.py` (create) | gdown checkpoint, SHA256SUMS |
| `src/foundation_stereo/scripts/build_trt_engines.py` (create) | ONNX export + TRT Python-API engine build |
| `src/foundation_stereo/test/test_build_trt_engines.py` (create) | unit tests for the builder's pure helpers |
| `.venv-fs.uv-project/unlockable.txt`, `src/foundation_stereo/README.md` (modify) | provisioning docs |
| `CLAUDE.md`, `docs/ENVIRONMENT.md`, `DEV_NOTES.md` (modify) | docs |

---

### Task 1: `vision_util.vlm_models` resolver

**Files:**
- Create: `src/vision_util/vision_util/vlm_models.py`
- Test: `src/vision_util/test/test_vlm_models.py`

**Interfaces:**
- Produces: `vision_vlm_model() -> str`, `vision_flash_model() -> str`, `vision_qwen_model() -> str`, plus the constants `LEGACY_VLM_MODEL = 'google/gemini-2.5-pro'`, `LEGACY_FLASH_MODEL = 'google/gemini-2.5-flash'`, `LEGACY_QWEN_MODEL = 'qwen3-vl-plus'` and `ENV_KEYS = ('VISION_VLM_MODEL', 'VISION_VLM_FLASH_MODEL', 'VISION_QWEN_MODEL', 'LLM_MODEL', 'FLASH_MODEL')`.

- [ ] **Step 1: Write the failing tests** — `src/vision_util/test/test_vlm_models.py`:

```python
"""Precedence tests for vision_util.vlm_models (pure os.environ, no dotenv)."""
import pytest

from vision_util import vlm_models as vm


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in vm.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_are_the_legacy_literals():
    assert vm.vision_vlm_model() == 'google/gemini-2.5-pro'
    assert vm.vision_flash_model() == 'google/gemini-2.5-flash'
    assert vm.vision_qwen_model() == 'qwen3-vl-plus'


def test_vision_keys_win(monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'openai/gpt-5.6-luna')
    monkeypatch.setenv('FLASH_MODEL', 'google/gemini-3.5-flash-lite')
    monkeypatch.setenv('VISION_VLM_MODEL', 'google/gemini-3.1-pro-preview')
    monkeypatch.setenv('VISION_VLM_FLASH_MODEL', 'google/gemini-3.7-flash')
    monkeypatch.setenv('VISION_QWEN_MODEL', 'qwen3.7-plus')
    assert vm.vision_vlm_model() == 'google/gemini-3.1-pro-preview'
    assert vm.vision_flash_model() == 'google/gemini-3.7-flash'
    assert vm.vision_qwen_model() == 'qwen3.7-plus'


def test_gpsr_keys_are_the_middle_fallback(monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'openai/gpt-5.6-luna')
    monkeypatch.setenv('FLASH_MODEL', 'google/gemini-3.5-flash-lite')
    assert vm.vision_vlm_model() == 'openai/gpt-5.6-luna'
    assert vm.vision_flash_model() == 'google/gemini-3.5-flash-lite'
    assert vm.vision_qwen_model() == 'qwen3-vl-plus'   # no GPSR key for qwen


def test_empty_and_whitespace_count_as_unset(monkeypatch):
    monkeypatch.setenv('VISION_VLM_MODEL', '   ')
    monkeypatch.setenv('LLM_MODEL', '')
    assert vm.vision_vlm_model() == 'google/gemini-2.5-pro'
    monkeypatch.setenv('VISION_QWEN_MODEL', ' qwen3-vl-flash ')
    assert vm.vision_qwen_model() == 'qwen3-vl-flash'   # stripped
```

- [ ] **Step 2: Run to verify failure**

Run: `$VENV/bin/python -m pytest src/vision_util/test/test_vlm_models.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'vision_util.vlm_models'`.

- [ ] **Step 3: Implement** — `src/vision_util/vision_util/vlm_models.py`:

```python
"""Single source of VLM model-ID defaults for every vision node.

Resolution (first non-empty wins):
  vision_vlm_model()   : VISION_VLM_MODEL       -> LLM_MODEL   -> 'google/gemini-2.5-pro'
  vision_flash_model() : VISION_VLM_FLASH_MODEL -> FLASH_MODEL -> 'google/gemini-2.5-flash'
  vision_qwen_model()  : VISION_QWEN_MODEL                     -> 'qwen3-vl-plus'

Reads ``os.environ`` only. Callers are responsible for having loaded the
workspace ``.env`` (python-dotenv) *before* declaring ROS parameters; every
VLM-using entry point in tk26_vision already does. Keep this module free of
any non-stdlib import — ``tk_vision_specialized`` deliberately does not
depend on ``kimi_api``, and both depend on this.
"""
from __future__ import annotations

import os

LEGACY_VLM_MODEL = 'google/gemini-2.5-pro'
LEGACY_FLASH_MODEL = 'google/gemini-2.5-flash'
LEGACY_QWEN_MODEL = 'qwen3-vl-plus'

ENV_KEYS = (
    'VISION_VLM_MODEL',
    'VISION_VLM_FLASH_MODEL',
    'VISION_QWEN_MODEL',
    'LLM_MODEL',
    'FLASH_MODEL',
)


def _first_set(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key, '')
        value = value.strip() if value else ''
        if value:
            return value
    return None


def vision_vlm_model() -> str:
    """Primary ("pro"-tier) OpenRouter model for vision VLM calls."""
    return _first_set('VISION_VLM_MODEL', 'LLM_MODEL') or LEGACY_VLM_MODEL


def vision_flash_model() -> str:
    """Fast/cheap OpenRouter model for latency-sensitive vision calls."""
    return _first_set('VISION_VLM_FLASH_MODEL', 'FLASH_MODEL') or LEGACY_FLASH_MODEL


def vision_qwen_model() -> str:
    """DashScope Qwen-VL model id (no provider prefix)."""
    return _first_set('VISION_QWEN_MODEL') or LEGACY_QWEN_MODEL
```

- [ ] **Step 4: Run tests**

Run: `$VENV/bin/python -m pytest src/vision_util/test/test_vlm_models.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vision_util/vision_util/vlm_models.py src/vision_util/test/test_vlm_models.py
git commit -m "feat(vision_util): vlm_models — VLM model-id defaults resolved from .env keys"
```

---

### Task 2: Wire every production default to the resolver, guard against regressions

**Files:**
- Modify: `src/kimi_api/kimi_api/_env.py:38-44` (`default_model`/`default_flash_model`), `:99-101` (`_DASHSCOPE_DEFAULT_MODEL`), `:134` (its use)
- Modify: `src/kimi_api/kimi_api/seat_recommend_bbox.py:68,129`; `src/kimi_api/seat_bench/providers.py:26-27`
- Modify: `src/object_detection_generalist/object_detection_generalist/generalist_node.py:158,164-166,175`
- Modify: `src/tk_vision_specialized/tk_vision_specialized/placing_location_server.py:72,77`; `waving_person_server.py:173-174`; `vlm_match_client.py:27`; `vlm_judge_client.py:24,27`; `vlm_match_client_gemini.py:24`
- Modify: `src/tk_vision_specialized/scripts/produce_match_ground_truth.py:42`; `scripts/object_scan_webui/scan_core.py:29-30`; `scripts/compare_feature_matching_models.py:44-45`; `scripts/tests/manual/gemini_bbox_decode.py:37`, `seat_recommend_vlm_bench.py:158`, `vlm_timeout_isolation.py:373`, `vlm_provider_bench.py:380`, `web_image_vlm_smoke.py:959`
- Test: `src/tk_vision_specialized/test/test_no_hardcoded_vlm_models.py` (create); existing `src/kimi_api/test/test_env_resolve_qwen_target.py` must stay green

**Interfaces:**
- Consumes: Task 1's `vision_vlm_model`, `vision_flash_model`, `vision_qwen_model`, `LEGACY_*`.
- Produces: `kimi_api._env.default_model()` / `default_flash_model()` now delegate to the resolver (signatures unchanged).

- [ ] **Step 1: Write the failing guard test** — `src/tk_vision_specialized/test/test_no_hardcoded_vlm_models.py`:

```python
"""Guard: no production code may hardcode a VLM model id — defaults come from
vision_util.vlm_models so .env controls them. Scans the vision source tree."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCAN_DIRS = [REPO / 'src', REPO / 'scripts']
ALLOWED = {
    REPO / 'src' / 'vision_util' / 'vision_util' / 'vlm_models.py',
}
SKIP_PARTS = {'test', 'tests', 'fixtures', 'thirdparty', 'seat_bench/report.md'}
LITERALS = re.compile(
    r"['\"](google/gemini-2\.5-(pro|flash)|qwen3-vl-plus|gemini-2\.5-flash)['\"]"
)


def _candidates():
    for root in SCAN_DIRS:
        for path in root.rglob('*.py'):
            if path in ALLOWED:
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            yield path


def test_no_literal_model_ids_outside_resolver():
    offenders = []
    for path in _candidates():
        for lineno, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
            if LITERALS.search(line) and not line.lstrip().startswith('#'):
                offenders.append(f'{path.relative_to(REPO)}:{lineno}: {line.strip()}')
    assert not offenders, 'hardcoded VLM model ids (use vision_util.vlm_models):\n' + '\n'.join(offenders)
```

- [ ] **Step 2: Run to verify failure**

Run: `$VENV/bin/python -m pytest src/tk_vision_specialized/test/test_no_hardcoded_vlm_models.py -v`
Expected: FAIL listing every site in the Files list above (≈20 lines). Keep that list — it is your checklist. (Docstring examples that mention a model id in prose without quotes are fine; quoted ones in docstrings must be reworded, e.g. "the VISION_VLM_FLASH_MODEL default".)

- [ ] **Step 3: `_env.py`** — replace lines 38–44 with

```python
from vision_util.vlm_models import (  # noqa: E402  (re-exported for kimi_api callers)
    vision_vlm_model,
    vision_flash_model,
    vision_qwen_model,
)


def default_model() -> str:
    """Primary vision VLM (VISION_VLM_MODEL -> LLM_MODEL -> legacy literal)."""
    return vision_vlm_model()


def default_flash_model() -> str:
    """Fast vision VLM (VISION_VLM_FLASH_MODEL -> FLASH_MODEL -> legacy literal)."""
    return vision_flash_model()
```
(move the import to the top-level import block under `import os`). Replace `_DASHSCOPE_DEFAULT_MODEL = 'qwen3-vl-plus'` with a function-time lookup: delete the constant and change line 134 to `resolved_model = model or vision_qwen_model()`. Keep `_OPENROUTER_QWEN_DEFAULT_MODEL` (OpenRouter slug differs by design).

- [ ] **Step 4: Nodes** — apply, with the import `from vision_util.vlm_models import vision_vlm_model, vision_flash_model, vision_qwen_model` (only the names each file uses) added to each file's imports:
  - `seat_recommend_bbox.py:68` → `self.declare_parameter('llm_model', vision_vlm_model())`; `:129` → `self.declare_parameter('bbox_model_gemini', vision_vlm_model())`
  - `providers.py:26-27` → `GEMINI_MODEL = _env.vision_vlm_model()` / `QWEN_MODEL = _env.vision_qwen_model()` (module already imports `_env`)
  - `generalist_node.py:158` → `vision_flash_model()`; `:164-166` → `self.declare_parameter('vlm_fallback_models', [f'dashscope/{vision_qwen_model()}'])`; `:175` → `self.declare_parameter('dashscope_qwen_model', f'dashscope/{vision_qwen_model()}')`
  - `placing_location_server.py:72` → `vision_vlm_model()`; `:77` → `vision_qwen_model()`
  - `waving_person_server.py:173` → `vision_qwen_model()`; `:174` → `vision_vlm_model()`
  - `vlm_match_client.py:27`, `vlm_judge_client.py:24`: replace the constant with `def _qwen_default_model() -> str: return vision_qwen_model()` and change each use of `_QWEN_DEFAULT_MODEL` to `_qwen_default_model()` (call time, so a later `load_dotenv` still counts); same for `_GEMINI_DEFAULT_MODEL` in `vlm_judge_client.py:27` / `vlm_match_client_gemini.py:24` via `_gemini_default_model()` → `vision_vlm_model()`. `grep -n "_QWEN_DEFAULT_MODEL\|_GEMINI_DEFAULT_MODEL"` in those three files must show only the new function definitions.
  - After each node's `declare_parameter` block, add one INFO log naming the resolved defaults, e.g. in `waving_person_server.py` right after line 174: `self.get_logger().info(f'VLM model defaults: gemini={vision_vlm_model()} qwen={vision_qwen_model()} (from .env VISION_*)')`. Same one-liner in `placing_location_server.py`, `seat_recommend_bbox.py`, `generalist_node.py` (flash + qwen there).

- [ ] **Step 5: Scripts** — `produce_match_ground_truth.py:42` → `default=vision_qwen_model()`; `scan_core.py:29-30` → `GEMINI_MODEL = vision_flash_model()` / `QWEN_MODEL = vision_qwen_model()` (after its dotenv load at lines 67-70 — move the two assignments below it or make them functions); `compare_feature_matching_models.py:44-45` → `PRO_MODEL = vision_vlm_model()` / `FLASH_MODEL = vision_flash_model()` (dotenv already loaded at 31-35); manual scripts: `gemini_bbox_decode.py:37` and `seat_recommend_vlm_bench.py:158` → `default=vision_vlm_model()`; `vlm_timeout_isolation.py:373`, `vlm_provider_bench.py:380` → `default=vision_flash_model()`; `vlm_provider_bench.py:381` (`--gemini-model`, native Google API id without `google/`) → `default=vision_flash_model().split('/', 1)[-1]`; `web_image_vlm_smoke.py:959` → `args.model = vision_flash_model()`. Scripts that run from the repo without colcon must splice `src/vision_util` onto `sys.path` the way `scripts/download_models.py` does (`sys.path.insert(0, str(VISION_DIR / "src" / "vision_util"))`) before the import.

- [ ] **Step 6: Run the guard + existing suites**

```bash
$VENV/bin/python -m pytest src/tk_vision_specialized/test/test_no_hardcoded_vlm_models.py src/kimi_api/test/test_env_resolve_qwen_target.py -v
$VENV/bin/python -m pytest src/kimi_api/test src/tk_vision_specialized/test src/object_detection_generalist/test -q -k "not test_flake8 and not test_pep257 and not test_copyright"
$VENV/bin/python -c "import tk_vision_specialized.waving_person_server, tk_vision_specialized.placing_location_server, kimi_api.seat_recommend_bbox, object_detection_generalist.generalist_node; print('imports ok')"
for s in scripts/object_scan_webui/scan_core.py scripts/compare_feature_matching_models.py scripts/tests/manual/gemini_bbox_decode.py scripts/tests/manual/vlm_provider_bench.py; do $VENV/bin/python $s --help >/dev/null && echo "ok $s"; done
```
Expected: guard PASS; suites green (pre-existing skips fine); imports ok; each `--help` ok.

- [ ] **Step 7: Commit**

```bash
git add -A src/kimi_api src/object_detection_generalist src/tk_vision_specialized scripts
git commit -m "refactor: VLM model defaults come from vision_util.vlm_models (.env VISION_* keys)"
```

---

### Task 3: `.env` keys and `.env.example`

**Files:**
- Modify: `src/kimi_api/.env.example`
- Append (outside repo, not committed): `/home/tinker/tk25_ws/.env`

- [ ] **Step 1: `.env.example`** — append:

```
# --- vision VLM model ids (read by vision_util.vlm_models) -------------------
# Resolution: VISION_VLM_MODEL -> LLM_MODEL -> google/gemini-2.5-pro
#             VISION_VLM_FLASH_MODEL -> FLASH_MODEL -> google/gemini-2.5-flash
#             VISION_QWEN_MODEL -> qwen3-vl-plus   (DashScope id, no provider prefix)
# Explicit ROS params (-p llm_model:=...) still override these.
VISION_VLM_MODEL=google/gemini-3.1-pro-preview
VISION_VLM_FLASH_MODEL=google/gemini-3.7-flash
VISION_QWEN_MODEL=qwen3-vl-plus
```

- [ ] **Step 2: Workspace `.env`** — append the same three `KEY=value` lines (no comments needed) to `/home/tinker/tk25_ws/.env` **only if** `grep -c '^VISION_VLM_MODEL=' /home/tinker/tk25_ws/.env` is 0. Use `printf '...\n' >> /home/tinker/tk25_ws/.env`; never `cat` the file. Verify with `grep -n '^VISION_' /home/tinker/tk25_ws/.env` (these lines contain no secrets).

- [ ] **Step 3: End-to-end check** — from `/home/tinker/tk25_ws` (so `load_dotenv` finds `.env`):

```bash
cd /home/tinker/tk25_ws && $VENV/bin/python -c "from dotenv import load_dotenv; load_dotenv(); from vision_util.vlm_models import *; print(vision_vlm_model(), vision_flash_model(), vision_qwen_model())"; cd $WT
```
Expected: `google/gemini-3.1-pro-preview google/gemini-3.7-flash qwen3-vl-plus`.

- [ ] **Step 4: Commit**

```bash
git add src/kimi_api/.env.example
git commit -m "docs(env): document VISION_VLM_MODEL / VISION_VLM_FLASH_MODEL / VISION_QWEN_MODEL"
```

---

### Task 4: tensorboard 2.20 / protobuf 6.33 in `.venv-vision-main`

**Files:**
- Modify: `.venv-vision-main.uv-project/pyproject.toml:101` (`protobuf==3.20.3` → `6.33.6`) and add `tensorboard==2.20.0`, `tensorboard-data-server==0.7.2` are **not** in this pyproject (unlockables) — leave them out; `requirements.txt` line `protobuf==3.20.3` → `6.33.6`, `tensorboard==2.11.2` → `2.20.0` if present (`grep -n "^tensorboard" requirements.txt`), `tensorboard-data-server` likewise.
- Refresh: `$VENV/freeze.lock.txt`

- [ ] **Step 1: Snapshot**

```bash
$VENV/bin/pip freeze | sort > /home/tinker/.claude/jobs/df430f11/tmp/tb_before.txt
$VENV/bin/pip check > /home/tinker/.claude/jobs/df430f11/tmp/tb_check_before.txt; echo "exit=$?"
```

- [ ] **Step 2: Install exactly three packages**

```bash
$VENV/bin/pip install --no-deps tensorboard==2.20.0 tensorboard-data-server==0.7.2 protobuf==6.33.6
$VENV/bin/pip check > /home/tinker/.claude/jobs/df430f11/tmp/tb_check_after.txt; echo "exit=$?"
diff /home/tinker/.claude/jobs/df430f11/tmp/tb_check_before.txt /home/tinker/.claude/jobs/df430f11/tmp/tb_check_after.txt && echo "pip check unchanged"
$VENV/bin/pip freeze | sort | diff /home/tinker/.claude/jobs/df430f11/tmp/tb_before.txt -
```
Expected: freeze diff shows exactly `protobuf`, `tensorboard`, `tensorboard-data-server` lines (old → new); `pip check` output identical to before. Anything else → STOP, report.

- [ ] **Step 3: Gate — torchreid still imports and builds**

```bash
$VENV/bin/python -c "import google.protobuf as p, tensorboard as tb; print(p.__version__, tb.__version__)"
$VENV/bin/python -c "import torchreid; from vision_track.reid import reid_backbone as rb; m = rb.build_osnet('osnet_ain_x1_0') if hasattr(rb,'build_osnet') else None; print('torchreid import ok')"
$VENV/bin/python -m pytest src/vision_track/test -q -k "not test_flake8 and not test_pep257 and not test_copyright" 2>&1 | tail -3
```
(If `reid_backbone` has no `build_osnet`, use whatever factory `grep -n "^def " src/vision_track/vision_track/reid/reid_backbone.py` shows that constructs the backbone, e.g. `ReIDBackbone(...)`; the point is one OSNet construction through torchreid.) Expected: `6.33.6 2.20.0`; import ok; vision_track suite green.

- [ ] **Step 4: Pins + lock**

```bash
sed -i 's/^    "protobuf==3.20.3",/    "protobuf==6.33.6",/' .venv-vision-main.uv-project/pyproject.toml
sed -i 's/^protobuf==3.20.3$/protobuf==6.33.6/; s/^tensorboard==2.11.2$/tensorboard==2.20.0/; s/^tensorboard-data-server==0.6.1$/tensorboard-data-server==0.7.2/' requirements.txt
grep -n "protobuf==\|^tensorboard" .venv-vision-main.uv-project/pyproject.toml requirements.txt
(cd .venv-vision-main.uv-project && uv lock) ; git diff --stat .venv-vision-main.uv-project/uv.lock; git diff .venv-vision-main.uv-project/uv.lock | grep '^[-+]name' | sort -u
$VENV/bin/pip freeze > $VENV/freeze.lock.txt
```
Expected: only `protobuf` (and nothing else) appears in the lock's changed `name` lines; if other packages re-resolve, `git checkout -- .venv-vision-main.uv-project/uv.lock` and note it.

- [ ] **Step 5: Commit**

```bash
git add .venv-vision-main.uv-project/pyproject.toml .venv-vision-main.uv-project/uv.lock requirements.txt
git commit -m "deps(vision-main): tensorboard 2.11.2 -> 2.20.0, protobuf 3.20.3 -> 6.33.6"
```

---

### Task 5: TensorRT into `.venv-fs` + `weights_root` expansion

**Files:**
- Modify: `src/foundation_stereo/foundation_stereo/stereo_runner.py:140-141` (`expanduser`)
- Modify: `src/foundation_stereo/config/foundation_stereo.yaml:4`; `src/foundation_stereo/foundation_stereo/foundation_stereo_node.py:289-290`
- Modify: `.venv-fs.uv-project/unlockable.txt`; create `$VENV_FS/freeze.lock.txt`
- Test: `src/foundation_stereo/test/test_stereo_runner_imports.py` (existing) + new test in the same file

**Interfaces:**
- Produces: `StereoRunner(weights_root)` accepts `~`-prefixed paths; default `weights_root` = `~/.cache/tk26_vision/weights/foundation_stereo`.

- [ ] **Step 1: Failing test** — append to `test_stereo_runner_imports.py`:

```python
def test_weights_root_is_user_expanded(tmp_path, monkeypatch):
    """`~` in weights_root must be expanded so the yaml default works."""
    from foundation_stereo import stereo_runner
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = stereo_runner.StereoRunner(weights_root="~/wr")
    assert runner._weights_root == os.path.join(str(tmp_path), "wr")
    assert runner._fast_pickle.startswith(str(tmp_path))
```

- [ ] **Step 2: Run to verify failure**

Run: `$VENV_FS/bin/python -m pytest src/foundation_stereo/test/test_stereo_runner_imports.py -v -k expanded` (use the fs venv; set `PYTHONPATH=$WT/src/foundation_stereo:$PYTHONPATH`)
Expected: FAIL — `_weights_root == '~/wr'`.

- [ ] **Step 3: Implement** — in `StereoRunner.__init__` replace `self._weights_root = weights_root` with:

```python
        weights_root = os.path.realpath(os.path.expanduser(weights_root))
        self._weights_root = weights_root
```
Then change the defaults: `foundation_stereo.yaml:4` → `weights_root: "~/.cache/tk26_vision/weights/foundation_stereo"`; `foundation_stereo_node.py:289-290` → `self.declare_parameter("weights_root", "~/.cache/tk26_vision/weights/foundation_stereo")`.

- [ ] **Step 4: Run tests** — same command without `-k`: all PASS (the import/namespace tests need torch in `.venv-fs`, which it has).

- [ ] **Step 5: Install TensorRT (freeze rule)**

```bash
$VENV_FS/bin/pip freeze | sort > /home/tinker/.claude/jobs/df430f11/tmp/fs_before.txt
$VENV_FS/bin/pip check > /home/tinker/.claude/jobs/df430f11/tmp/fs_check_before.txt; echo "exit=$?"
$VENV_FS/bin/pip install --no-deps --extra-index-url https://pypi.nvidia.com tensorrt-cu12==10.16.1.11 tensorrt-cu12-bindings==10.16.1.11 tensorrt-cu12-libs==10.16.1.11
$VENV_FS/bin/pip check > /home/tinker/.claude/jobs/df430f11/tmp/fs_check_after.txt; echo "exit=$?"; diff /home/tinker/.claude/jobs/df430f11/tmp/fs_check_before.txt /home/tinker/.claude/jobs/df430f11/tmp/fs_check_after.txt && echo "pip check unchanged"
$VENV_FS/bin/pip freeze | sort | diff /home/tinker/.claude/jobs/df430f11/tmp/fs_before.txt -
$VENV_FS/bin/python -c "import tensorrt as trt; print(trt.__version__); b = trt.Builder(trt.Logger(trt.Logger.WARNING)); print('builder ok, platform_has_fast_fp16 =', b.platform_has_fast_fp16)"
$VENV_FS/bin/pip freeze > $VENV_FS/freeze.lock.txt
```
Expected: freeze diff = exactly three added `tensorrt-cu12*==10.16.1.11` lines (pip may also list the meta-package as `tensorrt_cu12`; that counts as one of the three); `pip check` unchanged; prints `10.16.1.11` and `builder ok, platform_has_fast_fp16 = True`. The `-libs` wheel is ~1 GB — allow time.

- [ ] **Step 6: `unlockable.txt`** — under the `tensorrt_cu12==10.16.1.11` entry add the line `#   INSTALLED 2026-08-22 on tinker (RTX 2080 Ti); see src/foundation_stereo/README.md § Provisioning.`

- [ ] **Step 7: Commit**

```bash
git add src/foundation_stereo/foundation_stereo/stereo_runner.py src/foundation_stereo/config/foundation_stereo.yaml src/foundation_stereo/foundation_stereo/foundation_stereo_node.py src/foundation_stereo/test/test_stereo_runner_imports.py .venv-fs.uv-project/unlockable.txt
git commit -m "feat(foundation_stereo): weights_root under ~/.cache (expanduser); tensorrt-cu12 10.16.1.11 recorded"
```

---

### Task 6: `fetch_fast_fs_weights.py`

**Files:**
- Create: `src/foundation_stereo/scripts/fetch_fast_fs_weights.py`
- Test: `src/foundation_stereo/test/test_fetch_fast_fs_weights.py`

**Interfaces:**
- Produces: `checkpoint_path(weights_root: str) -> Path` = `<root>/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth`; `verify_or_write_sums(ckpt_dir: Path) -> str` (returns sha256; writes `SHA256SUMS` on first run, verifies on later runs, raises `RuntimeError` on mismatch); CLI `--weights-root` (default `~/.cache/tk26_vision/weights/foundation_stereo`), `--folder-id` (default `1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap`).

- [ ] **Step 1: Failing tests** — `src/foundation_stereo/test/test_fetch_fast_fs_weights.py`:

```python
import hashlib
from pathlib import Path

import pytest

from foundation_stereo.scripts import fetch_fast_fs_weights as f


def test_checkpoint_path_layout(tmp_path):
    p = f.checkpoint_path(str(tmp_path))
    assert p == tmp_path / "Fast-FoundationStereo" / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"


def test_sums_written_then_verified(tmp_path):
    ckpt_dir = tmp_path / "23-36-37"; ckpt_dir.mkdir()
    (ckpt_dir / "model_best_bp2_serialize.pth").write_bytes(b"abc")
    digest = f.verify_or_write_sums(ckpt_dir)
    assert digest == hashlib.sha256(b"abc").hexdigest()
    assert (ckpt_dir / "SHA256SUMS").read_text().split()[0] == digest
    assert f.verify_or_write_sums(ckpt_dir) == digest          # second run verifies


def test_sums_mismatch_raises(tmp_path):
    ckpt_dir = tmp_path / "23-36-37"; ckpt_dir.mkdir()
    (ckpt_dir / "model_best_bp2_serialize.pth").write_bytes(b"abc")
    f.verify_or_write_sums(ckpt_dir)
    (ckpt_dir / "model_best_bp2_serialize.pth").write_bytes(b"xyz")
    with pytest.raises(RuntimeError):
        f.verify_or_write_sums(ckpt_dir)
```
Location: both new scripts live **inside the package** at `src/foundation_stereo/foundation_stereo/scripts/` (create an empty `src/foundation_stereo/foundation_stereo/scripts/__init__.py`). `setup.py` already uses `packages=find_packages(exclude=['test'])` (line 8), so the subpackage is installed automatically; add the console entry point `'fetch_fast_fs_weights = foundation_stereo.scripts.fetch_fast_fs_weights:main'` to the `console_scripts` list (line 26-27). Wherever this plan says `src/foundation_stereo/scripts/`, it means this package-internal directory.

- [ ] **Step 2: Run to verify failure** — `$VENV_FS/bin/python -m pytest src/foundation_stereo/test/test_fetch_fast_fs_weights.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/foundation_stereo/foundation_stereo/scripts/fetch_fast_fs_weights.py`:

```python
#!/usr/bin/env python3
"""Fetch the Fast-FoundationStereo `23-36-37` checkpoint into the weights cache.

    python -m foundation_stereo.scripts.fetch_fast_fs_weights [--weights-root DIR]

Downloads the upstream Google-Drive folder with gdown (only the 23-36-37
subfolder is kept), records SHA256SUMS after the first successful download
and verifies against it on later runs. Idempotent. Fails loudly if Drive is
unreachable or the folder layout changed — no workaround is attempted.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_WEIGHTS_ROOT = "~/.cache/tk26_vision/weights/foundation_stereo"
DRIVE_FOLDER_ID = "1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap"   # readme "Weights and Trade-off"
CKPT_NAME = "23-36-37"
PICKLE_NAME = "model_best_bp2_serialize.pth"


def checkpoint_path(weights_root: str) -> Path:
    root = Path(os.path.expanduser(weights_root))
    return root / "Fast-FoundationStereo" / "weights" / CKPT_NAME / PICKLE_NAME


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_or_write_sums(ckpt_dir: Path) -> str:
    pickle = ckpt_dir / PICKLE_NAME
    digest = _sha256(pickle)
    sums = ckpt_dir / "SHA256SUMS"
    if sums.exists():
        recorded = sums.read_text().split()[0]
        if recorded != digest:
            raise RuntimeError(
                f"{pickle} sha256 {digest} != recorded {recorded} in {sums}; "
                "delete the directory to re-download")
    else:
        sums.write_text(f"{digest}  {PICKLE_NAME}\n")
    return digest


def download(folder_id: str, dest_ckpt_dir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="fast_fs_") as tmp:
        cmd = [sys.executable, "-m", "gdown", "--folder", "--remaining-ok",
               f"https://drive.google.com/drive/folders/{folder_id}", "-O", tmp]
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)
        hits = list(Path(tmp).rglob(f"{CKPT_NAME}/{PICKLE_NAME}"))
        if not hits:
            found = sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*"))[:40]
            raise RuntimeError(
                f"{CKPT_NAME}/{PICKLE_NAME} not found in Drive folder {folder_id}; "
                f"layout seen: {found}")
        src_dir = hits[0].parent
        dest_ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, dest_ckpt_dir, dirs_exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights-root", default=DEFAULT_WEIGHTS_ROOT)
    ap.add_argument("--folder-id", default=DRIVE_FOLDER_ID)
    args = ap.parse_args()
    pickle = checkpoint_path(args.weights_root)
    if not pickle.exists():
        download(args.folder_id, pickle.parent)
    if not pickle.exists():
        print(f"ERROR: {pickle} still missing after download", file=sys.stderr)
        return 1
    digest = verify_or_write_sums(pickle.parent)
    print(f"ok {pickle} ({pickle.stat().st_size / 1e6:.1f} MB) sha256 {digest[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Unit tests pass** — `$VENV_FS/bin/python -m pytest src/foundation_stereo/test/test_fetch_fast_fs_weights.py -v` → 3 PASS.

- [ ] **Step 5: Real download** (network; `gdown 6.0.0` is in `.venv-fs`)

```bash
cd $WT/src/foundation_stereo && $VENV_FS/bin/python -m foundation_stereo.scripts.fetch_fast_fs_weights; cd $WT
ls -la ~/.cache/tk26_vision/weights/foundation_stereo/Fast-FoundationStereo/weights/23-36-37/
```
Expected: `ok … sha256 …`; the directory holds `model_best_bp2_serialize.pth` (+ any `cfg.yaml` upstream ships) and `SHA256SUMS`. If gdown fails (quota / permission / layout), paste the full error and report **BLOCKED** — do not try mirrors.

- [ ] **Step 6: Commit**

```bash
git add src/foundation_stereo/foundation_stereo/scripts src/foundation_stereo/test/test_fetch_fast_fs_weights.py src/foundation_stereo/setup.py
git commit -m "feat(foundation_stereo): fetch_fast_fs_weights — gdown the 23-36-37 checkpoint with SHA256SUMS"
```

---

### Task 7: `build_trt_engines.py` — ONNX export + TensorRT engines

**Files:**
- Create: `src/foundation_stereo/foundation_stereo/scripts/build_trt_engines.py`
- Test: `src/foundation_stereo/test/test_build_trt_engines.py`
- Modify: `src/foundation_stereo/setup.py` (entry point `build_trt_engines = foundation_stereo.scripts.build_trt_engines:main`)

**Interfaces:**
- Consumes: Task 6's `checkpoint_path`; the vendored `Fast-FoundationStereo/scripts/make_onnx.py`; `stereo_runner._FAST_DIR` / `_swap_namespace` conventions (the vendor root is resolved the same way `stereo_runner.py:40-60` does — reuse `stereo_runner._vendor_root()` if such a helper exists, else copy its three-anchor logic).
- Produces: `variant_dir(weights_root, name='output_two_stage') -> Path`; `build_engine(onnx_path: Path, engine_path: Path, fp16: bool = True, workspace_gib: int = 4) -> float` (returns build seconds); CLI `--weights-root`, `--variant`, `--height 576 --width 960 --valid-iters 4 --max-disp 192`, `--force`, `--no-fp16`.

- [ ] **Step 1: Failing tests** — `src/foundation_stereo/test/test_build_trt_engines.py`:

```python
from pathlib import Path

import pytest

from foundation_stereo.scripts import build_trt_engines as b


def test_variant_dir_layout(tmp_path):
    assert b.variant_dir(str(tmp_path)) == tmp_path / "Fast-FoundationStereo" / "output_two_stage"
    assert b.variant_dir(str(tmp_path), "x") == tmp_path / "Fast-FoundationStereo" / "x"


def test_make_onnx_command_shape(tmp_path):
    cmd = b.make_onnx_command(Path("/v/Fast-FoundationStereo"), Path("/ck.pth"), tmp_path, 576, 960, 4, 192)
    assert cmd[1].endswith("scripts/make_onnx.py")
    for flag, val in (("--model_dir", "/ck.pth"), ("--save_path", str(tmp_path)), ("--height", "576"),
                      ("--width", "960"), ("--valid_iters", "4"), ("--max_disp", "192")):
        assert val == cmd[cmd.index(flag) + 1]


def test_refuses_overwrite_without_force(tmp_path):
    out = tmp_path / "Fast-FoundationStereo" / "output_two_stage"; out.mkdir(parents=True)
    for n in ("feature_runner.engine", "post_runner.engine", "onnx.yaml"):
        (out / n).write_bytes(b"x")
    with pytest.raises(SystemExit):
        b.ensure_writable(out, force=False)
    b.ensure_writable(out, force=True)   # no raise


def test_build_engine_requires_tensorrt_and_onnx(tmp_path):
    trt = pytest.importorskip("tensorrt")
    with pytest.raises(FileNotFoundError):
        b.build_engine(tmp_path / "missing.onnx", tmp_path / "o.engine")
```

- [ ] **Step 2: Run to verify failure** — `$VENV_FS/bin/python -m pytest src/foundation_stereo/test/test_build_trt_engines.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/foundation_stereo/foundation_stereo/scripts/build_trt_engines.py`:

```python
#!/usr/bin/env python3
"""Export Fast-FoundationStereo two-stage ONNX and build FP16 TensorRT engines.

    python -m foundation_stereo.scripts.build_trt_engines [--weights-root DIR] [--force]

Steps: (1) run the vendored scripts/make_onnx.py on the 23-36-37 pickle at
576x960 / 4 iters / max_disp 192 into a temp dir; (2) build feature_runner
and post_runner engines with the TensorRT Python API (trtexec is not shipped
in the pip wheels); (3) install {feature_runner.engine, post_runner.engine,
onnx.yaml} as <weights_root>/Fast-FoundationStereo/<variant>/ — the layout
stereo_runner._discover_trt_variants expects. Engines are GPU/TRT-locked:
rebuild on every new box.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from foundation_stereo.scripts.fetch_fast_fs_weights import DEFAULT_WEIGHTS_ROOT, checkpoint_path

ENGINE_FILES = ("feature_runner.engine", "post_runner.engine", "onnx.yaml")


def variant_dir(weights_root: str, name: str = "output_two_stage") -> Path:
    return Path(os.path.expanduser(weights_root)) / "Fast-FoundationStereo" / name


def vendor_fast_dir() -> Path:
    """Locate thirdparty/foundation_stereo/Fast-FoundationStereo like stereo_runner does."""
    env = os.environ.get("FOUNDATION_STEREO_VENDOR_ROOT")
    if env:
        return Path(os.path.expanduser(env)) / "Fast-FoundationStereo"
    here = Path(__file__).resolve()
    for anc in here.parents:
        cand = anc / "thirdparty" / "foundation_stereo" / "Fast-FoundationStereo"
        if cand.is_dir():
            return cand
    raise FileNotFoundError("Fast-FoundationStereo vendor tree not found; set FOUNDATION_STEREO_VENDOR_ROOT")


def make_onnx_command(fast_dir: Path, ckpt: Path, save_dir: Path,
                      height: int, width: int, valid_iters: int, max_disp: int) -> list[str]:
    return [sys.executable, str(fast_dir / "scripts" / "make_onnx.py"),
            "--model_dir", str(ckpt), "--save_path", str(save_dir),
            "--height", str(height), "--width", str(width),
            "--valid_iters", str(valid_iters), "--max_disp", str(max_disp)]


def ensure_writable(out_dir: Path, force: bool) -> None:
    if out_dir.exists() and any((out_dir / n).exists() for n in ENGINE_FILES) and not force:
        sys.exit(f"{out_dir} already holds engines; pass --force to rebuild")


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool = True, workspace_gib: int = 4) -> float:
    import tensorrt as trt  # noqa: WPS433 — only available in .venv-fs
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as fp:
        if not parser.parse(fp.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    if fp16:
        if not builder.platform_has_fast_fp16:
            print("warning: platform reports no fast fp16; building fp16 anyway")
        config.set_flag(trt.BuilderFlag.FP16)
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as fp:
        fp.write(blob)
    return time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights-root", default=DEFAULT_WEIGHTS_ROOT)
    ap.add_argument("--variant", default="output_two_stage")
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--valid-iters", type=int, default=4)
    ap.add_argument("--max-disp", type=int, default=192)
    ap.add_argument("--workspace-gib", type=int, default=4)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ckpt = checkpoint_path(args.weights_root)
    if not ckpt.is_file():
        sys.exit(f"checkpoint missing: {ckpt} — run fetch_fast_fs_weights first")
    out_dir = variant_dir(args.weights_root, args.variant)
    ensure_writable(out_dir, args.force)
    fast_dir = vendor_fast_dir()

    with tempfile.TemporaryDirectory(prefix="fs_onnx_") as tmp:
        tmp = Path(tmp)
        cmd = make_onnx_command(fast_dir, ckpt, tmp, args.height, args.width, args.valid_iters, args.max_disp)
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(fast_dir))   # make_onnx imports `core` relative to its tree
        for stem in ("feature_runner", "post_runner"):
            secs = build_engine(tmp / f"{stem}.onnx", tmp / f"{stem}.engine",
                                fp16=not args.no_fp16, workspace_gib=args.workspace_gib)
            print(f"built {stem}.engine in {secs:.0f} s")
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ENGINE_FILES:
            shutil.copy2(tmp / name, out_dir / name)
    print(f"installed {out_dir}: {sorted(p.name for p in out_dir.iterdir())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Unit tests pass** — `$VENV_FS/bin/python -m pytest src/foundation_stereo/test/test_build_trt_engines.py -v` → 4 PASS (the last needs `tensorrt`, present after Task 5).

- [ ] **Step 5: Build for real** (GPU shared with the simulator; budget 20 min)

```bash
cd $WT/src/foundation_stereo && nice -n 10 $VENV_FS/bin/python -m foundation_stereo.scripts.build_trt_engines 2>&1 | tee /home/tinker/.claude/jobs/df430f11/tmp/trt_build.log | tail -20; cd $WT
ls -la ~/.cache/tk26_vision/weights/foundation_stereo/Fast-FoundationStereo/output_two_stage/
```
Expected: two `built … in N s` lines, `installed …: ['feature_runner.engine', 'onnx.yaml', 'post_runner.engine']`. If `make_onnx.py` fails (torch.onnx export / triton on cc 7.5), paste the last 40 log lines and report **BLOCKED** with the exact error; do not patch the vendored tree.

- [ ] **Step 6: Commit**

```bash
git add src/foundation_stereo/foundation_stereo/scripts/build_trt_engines.py src/foundation_stereo/test/test_build_trt_engines.py src/foundation_stereo/setup.py
git commit -m "feat(foundation_stereo): build_trt_engines — two-stage ONNX export + TensorRT FP16 engines (Python API)"
```

---

### Task 8: FoundationStereo end-to-end gate

**Files:** none modified; results feed Task 9.

- [ ] **Step 1: Build the package** — `./scripts/build_foundation_stereo.sh` from `$MAIN` after merging? No — from the worktree it will fail on venv paths (seen before). Instead run the node from source:

```bash
export PYTHONPATH=$WT/src/foundation_stereo:$WT/src/vision_util:$PYTHONPATH
cd /home/tinker/tk25_ws && timeout 120 nice -n 10 $VENV_FS/bin/python -m foundation_stereo.foundation_stereo_node --ros-args -p warmup_on_launch:=true -p weights_root:="~/.cache/tk26_vision/weights/foundation_stereo" 2>&1 | grep -v "^I0000\|^W0000" | grep -i "warmup\|variant\|engine\|error\|Traceback\|loaded\|ms" | head -30; cd $WT
```
Expected: log lines showing `loading fast_trt variant=output_two_stage from …/output_two_stage`, a warmup-complete line with a forward time, no Traceback; exit 124 (timeout while spinning) is fine. Record the forward time.

- [ ] **Step 2: Live call if the RealSense is up** — `timeout 8 ros2 topic hz /camera/infra1/image_rect_raw --window 20` (or the IR topic the launch file uses — check `grep -n "infra\|ir" src/foundation_stereo/launch/foundation_stereo.launch.py`). If publishing: run the node (as above, 120 s) in the background and call `ros2 service call /foundation_stereo/get_depth …` with the request shape from `src/foundation_stereo/README.md` § API. If not publishing, state so — the warmup forward is the accepted gate.

- [ ] **Step 3: T0 + package suites**

```bash
$VENV/bin/python -m pytest src/vision_util/test src/kimi_api/test src/tk_vision_specialized/test src/object_detection_generalist/test src/vision_track/test -q -k "not test_flake8 and not test_pep257 and not test_copyright" 2>&1 | tail -3
$VENV_FS/bin/python -m pytest src/foundation_stereo/test -q -k "not test_flake8 and not test_pep257 and not test_copyright" 2>&1 | tail -3
cd $MAIN && bash scripts/tests/t0_static.sh 2>&1 | tail -25; cd $WT
```
(Run each `$VENV` package dir as its own pytest invocation if the combined run hits the duplicate `test_copyright` module clash seen before.) Expected: all green except pre-existing environment rows documented in DEV_NOTES (T0.1 unbuilt tree, T0.fs rows now should PASS since tensorrt is installed — report them).

- [ ] **Step 4:** Record every command's output verbatim for Task 9. Nothing to commit.

---

### Task 9: Documentation

**Files:**
- Modify: `CLAUDE.md` (OpenRouter section ~"### OpenRouter API key (kimi_api)"; `waving_person_server`, `kimi_api/*`, `seat_recommend_bbox`, `object_match_all_server`, generalist bullets; the `.venv-fs`/tensorboard/protobuf sentences), `docs/ENVIRONMENT.md`, `src/foundation_stereo/README.md` § Provisioning + weights, `src/kimi_api/README.md` if it documents `LLM_MODEL`, `DEV_NOTES.md` (new dated entry)

- [ ] **Step 1: CLAUDE.md** — in the OpenRouter section add: "Model ids: every vision VLM default resolves via `vision_util.vlm_models` — `VISION_VLM_MODEL` → `LLM_MODEL` → `google/gemini-2.5-pro`; `VISION_VLM_FLASH_MODEL` → `FLASH_MODEL` → `google/gemini-2.5-flash`; `VISION_QWEN_MODEL` → `qwen3-vl-plus`. Set them in the workspace `.env`; explicit `-p …model:=` params still override. `test_no_hardcoded_vlm_models.py` fails if a literal id reappears in production code." In each node bullet replace "(default `'google/gemini-2.5-pro'`)"-style text with "(default from `VISION_VLM_MODEL`)", etc. Replace the `.venv-fs` sentence to say tensorrt-cu12 10.16.1.11 is installed and engines are built locally by `build_trt_engines`. Fix the tensorboard/protobuf sentences: "tensorboard 2.20.0 / protobuf 6.33.6 (2026-08-22)".
- [ ] **Step 2: foundation_stereo README** — rewrite § Provisioning: venv recipe unchanged + `pip install --no-deps --extra-index-url https://pypi.nvidia.com tensorrt-cu12==10.16.1.11 tensorrt-cu12-bindings==10.16.1.11 tensorrt-cu12-libs==10.16.1.11`; new § Weights & engines: `python -m foundation_stereo.scripts.fetch_fast_fs_weights` then `python -m foundation_stereo.scripts.build_trt_engines`, the resulting layout under `~/.cache/tk26_vision/weights/foundation_stereo`, the "engines are GPU/TRT-locked — rebuild per box" note, and the measured warmup forward time from Task 8.
- [ ] **Step 3: ENVIRONMENT.md** — update the pin table rows for protobuf/tensorboard and the `.venv-fs` tensorrt row; note `freeze.lock.txt` now exists for `.venv-fs`.
- [ ] **Step 4: DEV_NOTES.md** — dated entry `## 2026-08-22 — .env-driven VLM model ids, tensorboard 2.20, FoundationStereo TensorRT revival` containing: the dependency-scan summary that motivated it (Python 3.10 / driver 560 / cv_bridge ceilings; items deferred: opencv, openai 3, numpy 2, torch; monocular_depth removal explicitly out of scope), the resolver chain, the `.env` values set, both freeze diffs verbatim, the engine build log tail (build times), the warmup forward time, Task 8's T0 rows, and open follow-ups (benchmark `gemini-3.1-pro-preview` / `gemini-3.7-flash` through `seat_bench` before competition use; D405 variant; drop the 2.11-era orphans).
- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/ENVIRONMENT.md src/foundation_stereo/README.md src/kimi_api/README.md DEV_NOTES.md
git commit -m "docs: .env VISION_* model keys, tensorboard/protobuf pins, FoundationStereo provisioning via fetch/build scripts"
```

---

## Self-review against the spec

- §1 resolver + call sites + logging + `.env`/`.env.example` + tests → Tasks 1–3 (guard test replaces the per-node "declared default equals resolver" tests: it is stronger — it forbids any literal — and avoids instantiating rclpy nodes). ✔
- §2 tensorboard/protobuf with freeze diff, torchreid gate, pins → Task 4. ✔
- §3.1 install + import + `test_stereo_runner_imports` → Task 5; §3.2 weights_root + expanduser → Task 5; §3.3 fetch script with SHA256SUMS, fail-loud → Task 6; §3.4 builder (make_onnx at 576×960/4/192, Python-API FP16, `--force`) → Task 7; §3.5 warmup gate + live call → Task 8. ✔
- §4 verification and docs → Tasks 8–9. ✔
- Placeholder scan: none. Type consistency: `checkpoint_path(weights_root) -> Path` (T6) used by T7; `variant_dir`, `make_onnx_command`, `ensure_writable`, `build_engine` signatures match between T7's tests and code; resolver names identical across T1/T2/T3. ✔
- Note on script location: the spec says `src/foundation_stereo/scripts/`; the plan places them inside the package (`src/foundation_stereo/foundation_stereo/scripts/`) so they are importable/testable and installable as entry points — same intent, recorded here as the one deliberate deviation.
