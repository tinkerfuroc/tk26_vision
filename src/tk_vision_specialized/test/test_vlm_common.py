"""Unit tests for _vlm_common.py."""

from __future__ import annotations

import base64

import numpy as np

from tk_vision_specialized._vlm_common import (
    strip_fences,
    encode_data_url,
)


def test_strip_fences_passthrough_when_no_fence():
    raw = '{"a": 1}'
    assert strip_fences(raw) == '{"a": 1}'


def test_strip_fences_removes_json_fence():
    raw = '```json\n{"a": 1}\n```'
    out = strip_fences(raw)
    assert out.strip() == '{"a": 1}'


def test_strip_fences_removes_bare_fence():
    raw = '```\n{"a": 1}\n```'
    out = strip_fences(raw)
    assert out.strip() == '{"a": 1}'


def test_encode_data_url_round_trips_jpeg_bgr():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :, 2] = 128  # red channel in BGR
    url = encode_data_url(img)
    assert url.startswith('data:image/jpeg;base64,')
    payload = url.split(',', 1)[1]
    decoded = base64.b64decode(payload)
    # JPEG SOI marker
    assert decoded[:2] == b'\xff\xd8'
