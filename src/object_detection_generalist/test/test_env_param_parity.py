"""Regression test: `GeneralistDetectionNode` must load `.env` (VISION_*
model-id overrides) before its base class's `__init__` declares parameters.

`vlm_model`, `vlm_fallback_models`, and `dashscope_qwen_model` default from
`vision_util.vlm_models()`, which reads `os.environ` at declare time. Before
the fix, `GeneralistDetectionNode.__init__` called `super().__init__()`
(which runs `self._declare_parameters()`) before anything had loaded
`.env`, so a fresh process (no other module having already imported dotenv
first) silently got the legacy literal default forever, ignoring an
operator's `VISION_*` override.

Constructing the real node pulls in torch/ultralytics and loads a YOLO
model, so per the final-fix brief this uses the two-part check for heavy
nodes instead:
  1. Static: `__init__` calls `load_env()` textually before
     `super().__init__()`.
  2. Functional: the loader it calls (`vlm_bbox.load_env`, i.e.
     `kimi_api._env.load_env`) really does make an `.env` override reach
     `vision_util.vlm_models`, exercised in a subprocess.

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

from object_detection_generalist import generalist_node

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
WORKTREE_SRC = WORKTREE_ROOT / 'src'

_ENV_KEYS = ('VISION_VLM_MODEL', 'VISION_VLM_FLASH_MODEL',
             'VISION_QWEN_MODEL', 'LLM_MODEL', 'FLASH_MODEL')


def test_init_calls_load_env_before_declaring_parameters():
    src = inspect.getsource(generalist_node.GeneralistDetectionNode.__init__)
    load_idx = src.find('load_env(')
    super_idx = src.find('super().__init__(')
    assert load_idx != -1, '__init__ must call load_env()'
    assert super_idx != -1, '__init__ must call super().__init__()'
    assert load_idx < super_idx, (
        'load_env() must run before super().__init__(), which is what '
        'triggers self._declare_parameters() (vlm_model, '
        'vlm_fallback_models, dashscope_qwen_model default from '
        '.env-derived vision_util.vlm_models() at declare time)'
    )


def test_loader_makes_dotenv_reach_vlm_models(tmp_path):
    (tmp_path / '.env').write_text(
        'VISION_VLM_FLASH_MODEL=probe/flash\n'
        'VISION_VLM_MODEL=probe/pro\n'
        'VISION_QWEN_MODEL=probe/qwen\n'
    )
    env = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}
    env['PYTHONPATH'] = os.pathsep.join([
        str(WORKTREE_SRC / 'vision_util'),
        str(WORKTREE_SRC / 'kimi_api'),
        str(WORKTREE_SRC / 'object_detection_generalist'),
        env.get('PYTHONPATH', ''),
    ])
    code = (
        'from object_detection_generalist import vlm_bbox\n'
        'vlm_bbox.load_env()\n'
        'from vision_util.vlm_models import vision_flash_model\n'
        'print(vision_flash_model())\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'probe/flash'
