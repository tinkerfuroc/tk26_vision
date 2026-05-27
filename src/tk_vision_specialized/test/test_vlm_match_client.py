"""Unit tests for vlm_match_client.py — decoder logic and provider adapters.

Tests do not hit the network. The OpenAI client is monkeypatched per-test."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tk_vision_specialized.vlm_match_client import (
    MatchRow,
    QwenMatchClient,
    decode_qwen_response,
    build_match_client,
)

# Re-export so `MatchRow` is reachable from the test module — also satisfies
# ament_flake8 which would otherwise flag the import as unused.
__all__ = ['MatchRow']


def _canned_completion(content: str):
    """Return a SimpleNamespace shaped like openai's completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
        )],
    )


def test_decode_qwen_normalized_box_scales_to_pixels():
    body = json.dumps({
        'detections': [
            {'label': 'milk',
             'box_2d': [100, 200, 500, 800],   # 0..1000 normalized
             'confidence': 0.92},
        ],
    })
    rows = decode_qwen_response(
        body, scene_w=200, scene_h=400, allowed_labels={'milk'},
    )
    assert len(rows) == 1
    r = rows[0]
    assert r.label == 'milk'
    # x1 = 100 * 200/1000 = 20; y1 = 200 * 400/1000 = 80
    # x2 = 500 * 200/1000 = 100; y2 = 800 * 400/1000 = 320
    assert r.bbox == (20, 80, 100, 320)
    assert r.conf == pytest.approx(0.92)


def test_decode_qwen_drops_hallucinated_label():
    body = json.dumps({
        'detections': [
            {'label': 'banana',
             'box_2d': [0, 0, 100, 100], 'confidence': 0.99},
            {'label': 'milk',
             'box_2d': [100, 100, 500, 500], 'confidence': 0.5},
        ],
    })
    rows = decode_qwen_response(
        body, scene_w=1000, scene_h=1000,
        allowed_labels={'milk', 'cola'},
    )
    assert len(rows) == 1
    assert rows[0].label == 'milk'


def test_decode_qwen_clamps_degenerate_box():
    body = json.dumps({
        'detections': [
            {'label': 'milk',
             'box_2d': [500, 500, 500, 500], 'confidence': 0.9},
        ],
    })
    rows = decode_qwen_response(
        body, scene_w=100, scene_h=100, allowed_labels={'milk'},
    )
    assert rows == []


def test_decode_qwen_clamps_out_of_bounds():
    body = json.dumps({
        'detections': [
            {'label': 'milk',
             'box_2d': [-100, -100, 2000, 2000],
             'confidence': 0.9},
        ],
    })
    rows = decode_qwen_response(
        body, scene_w=100, scene_h=100, allowed_labels={'milk'},
    )
    assert len(rows) == 1
    assert rows[0].bbox == (0, 0, 99, 99)


def test_decode_qwen_handles_fenced_response():
    body = '```json\n{"detections": []}\n```'
    rows = decode_qwen_response(
        body, scene_w=100, scene_h=100, allowed_labels={'milk'},
    )
    assert rows == []


def test_decode_qwen_swaps_inverted_coords():
    body = json.dumps({
        'detections': [
            {'label': 'milk',
             'box_2d': [500, 800, 100, 200],   # x2 < x1, y2 < y1
             'confidence': 0.9},
        ],
    })
    rows = decode_qwen_response(
        body, scene_w=1000, scene_h=1000, allowed_labels={'milk'},
    )
    assert len(rows) == 1
    x1, y1, x2, y2 = rows[0].bbox
    assert x1 < x2 and y1 < y2


def test_decode_qwen_clamps_confidence_to_unit_range():
    body = json.dumps({
        'detections': [
            {'label': 'milk',
             'box_2d': [0, 0, 100, 100], 'confidence': 1.5},
            {'label': 'milk',
             'box_2d': [200, 200, 300, 300], 'confidence': -0.2},
        ],
    })
    rows = decode_qwen_response(
        body, scene_w=1000, scene_h=1000, allowed_labels={'milk'},
    )
    assert {r.conf for r in rows} == {1.0, 0.0}


def test_qwen_client_resolves_dashcope_typo_first(monkeypatch):
    """The workspace .env historically carries DASHCOPE_API_KEY (typo);
    that should resolve first for backward compatibility."""
    monkeypatch.setenv('DASHCOPE_API_KEY', 'typo-key')
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'official-key')
    client = QwenMatchClient(model='qwen3-vl-plus')
    assert client._api_key == 'typo-key'


def test_qwen_client_falls_back_to_dashscope_key(monkeypatch):
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'official-key')
    client = QwenMatchClient(model='qwen3-vl-plus')
    assert client._api_key == 'official-key'


def test_qwen_client_raises_when_key_missing(monkeypatch):
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='DashScope API key'):
        QwenMatchClient(model='qwen3-vl-plus')


def test_qwen_match_batch_end_to_end(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'fake')
    scene = np.zeros((400, 200, 3), dtype=np.uint8)
    body = json.dumps({
        'detections': [
            {'label': 'milk',
             'box_2d': [0, 0, 500, 500], 'confidence': 0.9},
        ],
    })

    class FakeOpenAI:
        def __init__(self, *a, **kw):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: _canned_completion(body),
                ),
            )

        def with_options(self, **kw):
            return self

        def close(self):
            pass

    with patch('tk_vision_specialized.vlm_match_client.OpenAI', FakeOpenAI):
        client = QwenMatchClient(model='qwen3-vl-plus')
        rows = client.match_batch(
            scene_bgr=scene,
            refs=[('milk', 'data:image/jpeg;base64,XXX')],
            timeout_s=5.0,
            max_retries=1,
        )

    assert len(rows) == 1
    assert rows[0].label == 'milk'


def test_build_match_client_unknown_provider_raises():
    with pytest.raises(ValueError, match='Unknown provider'):
        build_match_client('llama')
