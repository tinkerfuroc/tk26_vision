"""Regression tests: `placing_location_server` and `waving_person_server`
must both have `.env` (VISION_* model-id overrides) loaded before/at the
point their VLM model-id parameters are declared, so an operator's override
in the workspace `.env` isn't silently ignored in favor of the legacy
literal default.

- `PlacingLocationServer.__init__` was the actual bug: it called
  `super().__init__()` (which declares `vlm_model` / `placing_model_qwen`,
  defaulting from `vision_util.vlm_models()`) before anything had loaded
  `.env`. Fixed to call `load_env()` first.
- `waving_person_server` was already correct: `_waving_vlm.py` calls
  `load_dotenv(override=False)` at *module import time* (see
  `_waving_vlm.py`), which always finishes running before any node can be
  constructed — Python fully executes a module's top-level imports before
  a class defined later in that module can be instantiated — well before
  `DetectWavingPersonsNode.__init__` declares `vlm_model_qwen` /
  `vlm_model_gemini`. No fix needed there; this test locks the behavior in
  as part of the final-review env-loading parity sweep.

Both nodes subclass/pull in `YOLOSegmentationNode` (torch/ultralytics at
module import time), so per the final-fix brief this uses a static source
check plus a functional loader check via subprocess rather than
constructing the real node.

The functional check runs the loader in a subprocess invoked as
`python -c <code>` with `cwd=tmp_path` and a `.env` planted in that
`tmp_path`. python-dotenv's `find_dotenv()` treats a `-c` invocation as
"interactive" (the synthesized `__main__` module has no `__file__`), so it
resolves relative to the process's cwd rather than walking up from any
source file's location — `tmp_path` shares no ancestor with the real
workspace-root `.env` (`/home/tinker/tk25_ws/.env`, which carries live
secrets this test must never read or print), so that file can never be
reached this way.
"""
from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

from tk_vision_specialized import placing_location_server

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
WORKTREE_SRC = WORKTREE_ROOT / 'src'

_ENV_KEYS = ('VISION_VLM_MODEL', 'VISION_VLM_FLASH_MODEL',
             'VISION_QWEN_MODEL', 'LLM_MODEL', 'FLASH_MODEL')


def test_placing_init_calls_load_env_before_declaring_parameters():
    src = inspect.getsource(
        placing_location_server.PlacingLocationServer.__init__)
    load_idx = src.find('load_env(')
    super_idx = src.find('super().__init__(')
    assert load_idx != -1, '__init__ must call load_env()'
    assert super_idx != -1, '__init__ must call super().__init__()'
    assert load_idx < super_idx, (
        'load_env() must run before super().__init__(), which is what '
        'triggers self._declare_parameters() (vlm_model, '
        'placing_model_qwen default from .env-derived '
        'vision_util.vlm_models() at declare time)'
    )


def test_waving_module_imports_waving_vlm_before_class_definition():
    # Read the source text directly rather than importing the module: the
    # real waving_person_server.py needs tinker_vision_msgs_26.action's
    # DetectWaving, which may not be present in every test environment
    # (e.g. a stale colcon install) — irrelevant to the ordering property
    # under test, which is purely textual.
    module_path = (WORKTREE_SRC / 'tk_vision_specialized'
                   / 'tk_vision_specialized' / 'waving_person_server.py')
    src = module_path.read_text()
    import_idx = src.find('from ._waving_vlm import')
    class_idx = src.find('class DetectWavingPersonsNode')
    assert import_idx != -1, (
        'waving_person_server.py must import from ._waving_vlm at module '
        'level so its load_dotenv(override=False) side effect runs before '
        'the node can be constructed'
    )
    assert class_idx != -1
    assert import_idx < class_idx


def _run_probe(code, tmp_path):
    (tmp_path / '.env').write_text(
        'VISION_VLM_FLASH_MODEL=probe/flash\n'
        'VISION_VLM_MODEL=probe/pro\n'
        'VISION_QWEN_MODEL=probe/qwen\n'
    )
    env = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}
    env['PYTHONPATH'] = os.pathsep.join([
        str(WORKTREE_SRC / 'vision_util'),
        str(WORKTREE_SRC / 'kimi_api'),
        str(WORKTREE_SRC / 'tk_vision_specialized'),
        env.get('PYTHONPATH', ''),
    ])
    return subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
    )


def test_placing_loader_makes_dotenv_reach_vlm_models(tmp_path):
    code = (
        'from kimi_api._env import load_env\n'
        'load_env()\n'
        'from vision_util.vlm_models import vision_flash_model\n'
        'print(vision_flash_model())\n'
    )
    result = _run_probe(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'probe/flash'


def test_waving_import_makes_dotenv_reach_vlm_models(tmp_path):
    code = (
        'import tk_vision_specialized._waving_vlm\n'
        'from vision_util.vlm_models import vision_flash_model\n'
        'print(vision_flash_model())\n'
    )
    result = _run_probe(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'probe/flash'
