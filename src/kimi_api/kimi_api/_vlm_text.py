"""Shared text sanitizers for VLM response parsing.

Some models (qwen3-vl-plus on DashScope in particular, and Gemini routes
without a response_format) wrap JSON payloads in markdown code fences
despite instructions. Mirrors _seat_bbox_vlm's private _strip_fences so the
plain-parse clients (_match_vlm, _categorize_vlm) don't burn their retry
budgets on a deterministic formatting quirk.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.MULTILINE)


def strip_fences(text: str) -> str:
    """Drop ```json ... ``` fences some models emit despite instructions."""
    return _FENCE_RE.sub('', text).strip() if '```' in text else text
