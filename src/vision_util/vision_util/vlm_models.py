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
