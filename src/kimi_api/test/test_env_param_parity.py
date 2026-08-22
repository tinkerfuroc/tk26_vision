"""Regression test: `seat_recommend_bbox`'s `main()` must load `.env`
before constructing the node, so `bbox_model_qwen` / `bbox_model_gemini`
(declared in `SeatRecommendBboxService._declare_parameters()`, defaulting
from `vision_util.vlm_models()`) correctly pick up an operator's `VISION_*`
override instead of silently falling back to the legacy literal.

Already-correct behavior (`main()` calls `load_env()` at
`kimi_api/seat_recommend_bbox.py`, before `SeatRecommendBboxService()`) —
this test locks it in as part of the final-review fix wave's env-loading
parity sweep across all four VLM-using entry points (the other three live
in `object_detection_generalist` and `tk_vision_specialized`).

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

import os
import subprocess
import sys
from pathlib import Path

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
WORKTREE_SRC = WORKTREE_ROOT / 'src'

_ENV_KEYS = ('VISION_VLM_MODEL', 'VISION_VLM_FLASH_MODEL',
             'VISION_QWEN_MODEL', 'LLM_MODEL', 'FLASH_MODEL')


def test_main_calls_load_env_before_constructing_node():
    # Read the source text directly rather than importing the module: the
    # real seat_recommend_bbox.py needs tinker_vision_msgs_26.action's
    # SeatRecommendBbox, which may not be present in every test environment
    # (e.g. a stale colcon install) — irrelevant to the ordering property
    # under test, which is purely textual.
    module_path = WORKTREE_SRC / 'kimi_api' / 'kimi_api' / 'seat_recommend_bbox.py'
    full_src = module_path.read_text()
    def_idx = full_src.find('\ndef main(')
    assert def_idx != -1, 'seat_recommend_bbox.py must define main()'
    next_def_idx = full_src.find('\ndef ', def_idx + 1)
    src = full_src[def_idx:next_def_idx if next_def_idx != -1 else None]
    load_idx = src.find('load_env(')
    construct_idx = src.find('SeatRecommendBboxService(')
    assert load_idx != -1, 'main() must call load_env()'
    assert construct_idx != -1, (
        'main() must construct SeatRecommendBboxService'
    )
    assert load_idx < construct_idx, (
        'load_env() must run before constructing the node, which is what '
        'triggers self._declare_parameters() (bbox_model_qwen, '
        'bbox_model_gemini default from .env-derived '
        'vision_util.vlm_models() at declare time)'
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
        env.get('PYTHONPATH', ''),
    ])
    code = (
        'from kimi_api._env import load_env\n'
        'load_env()\n'
        'from vision_util.vlm_models import vision_flash_model\n'
        'print(vision_flash_model())\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'probe/flash'
