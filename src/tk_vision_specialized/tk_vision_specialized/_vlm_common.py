"""Shared utilities for the VLM match + judge clients.

Pure functions only (no ROS, no network) so they're trivially testable. The
clients build on top of these for prompt encoding and response decoding."""

from __future__ import annotations

import base64
import re

import cv2
import numpy as np


_FENCE_RE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.MULTILINE)


def strip_fences(text: str) -> str:
    """Drop ```json ... ``` or ``` ... ``` fences that some VLM revisions
    emit despite explicit "no markdown" instructions in the system prompt."""
    if '```' not in text:
        return text
    return _FENCE_RE.sub('', text).strip()


def encode_data_url(rgb_bgr: np.ndarray) -> str:
    """Encode a BGR image (HxWx3 uint8) as a base64 JPEG data URL suitable
    for the OpenAI-compatible chat completions API."""
    ok, buf = cv2.imencode('.jpg', rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError('cv2.imencode failed encoding scene image')
    return (
        'data:image/jpeg;base64,'
        + base64.b64encode(buf.tobytes()).decode('utf-8')
    )
