"""Unit tests for _waving_vlm.py — pure decoders, key resolution, and the
provider chain. No network: the OpenAI client is monkeypatched per-test."""

from __future__ import annotations

from tk_vision_specialized._waving_vlm import (
    WavingVlmResult,
    decode_box_xyxy,
    select_boxes,
)


def test_decode_box_xyxy_scales_and_clamps():
    # 0-1000 normalized -> pixels on a 1000x500 frame; x by width, y by height.
    assert decode_box_xyxy([100, 200, 300, 400], 1000, 500) == (100, 100, 300, 200)
    # out-of-range clamps to image bounds (w-1, h-1).
    assert decode_box_xyxy([0, 0, 1000, 1000], 1000, 500) == (0, 0, 999, 499)


def test_decode_box_xyxy_swaps_inverted_corners():
    assert decode_box_xyxy([300, 400, 100, 200], 1000, 500) == (100, 100, 300, 200)


def test_decode_box_xyxy_rejects_degenerate_and_malformed():
    assert decode_box_xyxy([500, 500, 500, 500], 1000, 500) is None  # zero area
    assert decode_box_xyxy([10], 1000, 500) is None                  # wrong length
    assert decode_box_xyxy('nope', 1000, 500) is None                # wrong type


def test_select_boxes_keeps_only_waving_with_decodable_box():
    parsed = {'persons': [
        {'box_2d': [100, 200, 300, 400], 'waving': True},
        {'box_2d': [0, 0, 100, 100], 'waving': False},     # dropped: not waving
        {'box_2d': [500, 500, 500, 500], 'waving': True},  # dropped: zero area
    ]}
    res = select_boxes(parsed, 1000, 500)
    assert isinstance(res, WavingVlmResult)
    assert res.boxes == [(100, 100, 300, 200)]
    assert res.error is None


def test_select_boxes_clean_empty_when_no_wavers():
    res = select_boxes({'persons': []}, 1000, 500)
    assert res.boxes == []
    assert res.error is None  # clean empty is terminal, not an error


from tk_vision_specialized._waving_vlm import (  # noqa: E402
    has_provider_key,
    build_provider_models,
)


def test_has_provider_key_qwen_accepts_either_spelling(monkeypatch):
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    assert has_provider_key('qwen') is False
    monkeypatch.setenv('DASHCOPE_API_KEY', 'legacy')   # typo'd spelling
    assert has_provider_key('qwen') is True


def test_has_provider_key_gemini_uses_openrouter(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    assert has_provider_key('gemini') is False
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    assert has_provider_key('gemini') is True


def test_build_provider_models_primary_plus_fallback():
    chain = build_provider_models(
        'qwen', 'gemini',
        has_key=lambda p: True,
        model_for=lambda p: f'model-{p}',
    )
    assert chain == [('qwen', 'model-qwen'), ('gemini', 'model-gemini')]


def test_build_provider_models_drops_keyless_providers():
    # Primary key missing -> primary dropped; fallback present -> kept.
    chain = build_provider_models(
        'qwen', 'gemini',
        has_key=lambda p: p == 'gemini',
        model_for=lambda p: f'model-{p}',
    )
    assert chain == [('gemini', 'model-gemini')]


def test_build_provider_models_empty_when_no_keys():
    chain = build_provider_models(
        'qwen', 'gemini',
        has_key=lambda p: False,
        model_for=lambda p: f'model-{p}',
    )
    assert chain == []


def test_build_provider_models_blank_fallback_disabled():
    chain = build_provider_models(
        'qwen', '',
        has_key=lambda p: True,
        model_for=lambda p: f'model-{p}',
    )
    assert chain == [('qwen', 'model-qwen')]


import json  # noqa: E402
import numpy as np  # noqa: E402
import openai  # noqa: E402
from types import SimpleNamespace  # noqa: E402
import pytest  # noqa: E402
from tk_vision_specialized._waving_vlm import request_waving_persons, WavingVlmError  # noqa: E402


def _completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _make_fake_openai(script):
    """Build a fake openai.OpenAI whose .create runs `script(kwargs)`.

    `script` returns the response content string, or raises to simulate an API
    error. Records constructor kwargs and create kwargs on the class for asserts.
    """
    class _Fake:
        last_init = None
        calls = []

        def __init__(self, **kw):
            _Fake.last_init = kw

        def with_options(self, **_kw):
            return self

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kw):
            _Fake.calls.append(kw)
            return _completion(script(kw))

        def close(self):
            pass

    return _Fake


def _img():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_request_waving_persons_returns_boxes(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'k')
    payload = json.dumps({'persons': [{'box_2d': [100, 100, 200, 300],
                                       'waving': True}]})
    fake = _make_fake_openai(lambda kw: payload)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_waving_persons(_img(), provider='qwen', model='qwen3-vl-plus')

    # box_2d [100,100,200,300] on 640x480: x*640/1000, y*480/1000.
    assert res.boxes == [(64, 48, 128, 144)]
    assert res.provider == 'qwen'
    assert res.error is None
    assert fake.last_init['base_url'] == \
        'https://dashscope.aliyuncs.com/compatible-mode/v1'
    assert fake.last_init['api_key'] == 'k'


def test_request_waving_persons_missing_key_raises(monkeypatch):
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    with pytest.raises(WavingVlmError, match='key'):
        request_waving_persons(_img(), provider='qwen', model='m')


def test_request_waving_persons_falls_back_to_json_object_on_schema_reject(
        monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    payload = json.dumps({'persons': []})

    def script(kw):
        rf = kw.get('response_format') or {}
        if rf.get('type') == 'json_schema':
            raise RuntimeError('response_format json_schema not supported')
        return payload

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_waving_persons(_img(), provider='gemini', model='g',
                                 max_retries=3)
    assert res.boxes == []
    assert res.error is None
    # First attempt strict, retried attempt loose.
    assert fake.calls[0]['response_format']['type'] == 'json_schema'
    assert fake.calls[-1]['response_format']['type'] == 'json_object'
