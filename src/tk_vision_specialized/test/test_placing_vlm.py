"""Unit tests for placing_vlm.py's provider support and
request_placing_bboxes_chain -- the Qwen fallback for placing_location_server.
No network: the OpenAI client is monkeypatched per-test."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import openai
import pytest

from tk_vision_specialized.placing_vlm import (
    VlmPlacingError,
    request_placing_bboxes,
    request_placing_bboxes_chain,
)


def _completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _make_fake_openai(script):
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


_ONE_REGION_PAYLOAD = json.dumps({
    'detections': [{'label': 'rank1', 'box_2d': [100, 100, 300, 300]}]})
_EMPTY_PAYLOAD = json.dumps({'detections': []})


# --- request_placing_bboxes provider support ---

def test_request_placing_bboxes_defaults_to_gemini(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: _ONE_REGION_PAYLOAD)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    boxes, ranks, _elapsed = request_placing_bboxes(
        _img(), item_description='mug', model='g')

    assert boxes == [(64, 48, 192, 144)]
    assert ranks == ['rank1']
    assert fake.last_init['base_url'] == 'https://openrouter.ai/api/v1'


def test_request_placing_bboxes_qwen_uses_dashscope(monkeypatch):
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: _ONE_REGION_PAYLOAD)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    boxes, _ranks, _elapsed = request_placing_bboxes(
        _img(), item_description='mug', model='qwen3-vl-plus', provider='qwen')

    assert boxes == [(64, 48, 192, 144)]
    assert fake.last_init['base_url'] == \
        'https://dashscope.aliyuncs.com/compatible-mode/v1'


def test_request_placing_bboxes_legit_empty_returns_cleanly(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: _EMPTY_PAYLOAD)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    boxes, ranks, _elapsed = request_placing_bboxes(
        _img(), item_description='mug', model='g')

    assert boxes == [] and ranks == []


def test_request_placing_bboxes_all_failed_decode_is_retried(monkeypatch):
    # A response with detections whose boxes ALL fail to decode is unusable
    # payload (wrong coordinate convention, malformed boxes), not a
    # legitimate "no suitable region" — it must consume a retry so the
    # chain's fallback provider can fire instead of reading as a full desk.
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('tk_vision_specialized.placing_vlm.time.sleep',
                        lambda s: None, raising=False)
    attempts = {'n': 0}

    def script(kw):
        attempts['n'] += 1
        if attempts['n'] < 2:
            return json.dumps({'detections': [
                {'label': 'rank1', 'box_2d': [500, 500, 500, 500]}]})
        return _ONE_REGION_PAYLOAD

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    boxes, _ranks, _elapsed = request_placing_bboxes(
        _img(), item_description='mug', model='g', max_retries=2)

    assert boxes == [(64, 48, 192, 144)]
    assert attempts['n'] == 2


def test_request_placing_bboxes_missing_key_raises(monkeypatch):
    # request_placing_bboxes calls load_env() (load_dotenv, override=False)
    # on every invocation, which would silently repopulate the key from the
    # workspace .env after delenv -- stub it out for a hermetic negative test.
    monkeypatch.setattr('tk_vision_specialized.placing_vlm.load_env', lambda: None)
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    with pytest.raises(VlmPlacingError, match='DASHSCOPE_API_KEY'):
        request_placing_bboxes(_img(), item_description='mug', model='m', provider='qwen')


def test_request_placing_bboxes_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    with pytest.raises(VlmPlacingError, match='unknown provider'):
        request_placing_bboxes(_img(), item_description='mug', model='m', provider='nope')


def test_request_placing_bboxes_exhaustion_raises(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: 'not json at all')
    monkeypatch.setattr(openai, 'OpenAI', fake)

    with pytest.raises(VlmPlacingError, match='exhausted 2 retries'):
        request_placing_bboxes(
            _img(), item_description='mug', model='g', max_retries=2)
    assert len(fake.calls) == 2


# --- request_placing_bboxes_chain ---

def _fake_chain(monkeypatch, by_provider):
    def fake(rgb, *, item_description, max_candidates, provider, model, **kw):
        v = by_provider[provider]
        if isinstance(v, Exception):
            raise v
        return v
    monkeypatch.setattr('tk_vision_specialized.placing_vlm.request_placing_bboxes', fake)


def test_chain_first_success_short_circuits(monkeypatch):
    good = ([(1, 2, 3, 4)], ['rank1'], 1.0)
    _fake_chain(monkeypatch, {
        'gemini': good, 'qwen': RuntimeError('should not call')})

    boxes, ranks, _elapsed, provider = request_placing_bboxes_chain(
        _img(), item_description='mug',
        provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert boxes == [(1, 2, 3, 4)] and provider == 'gemini'


def test_chain_falls_back_on_missing_key(monkeypatch):
    good = ([(5, 6, 7, 8)], ['rank1'], 0.5)
    _fake_chain(monkeypatch, {
        'gemini': VlmPlacingError('no key'), 'qwen': good})

    boxes, _ranks, _elapsed, provider = request_placing_bboxes_chain(
        _img(), item_description='mug',
        provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert boxes == [(5, 6, 7, 8)] and provider == 'qwen'


def test_chain_falls_back_on_exhaustion(monkeypatch):
    good = ([(9, 9, 9, 9)], ['rank1'], 0.3)
    _fake_chain(monkeypatch, {
        'gemini': VlmPlacingError('exhausted 1 retries'), 'qwen': good})

    boxes, _ranks, _elapsed, provider = request_placing_bboxes_chain(
        _img(), item_description='mug',
        provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert boxes == [(9, 9, 9, 9)] and provider == 'qwen'


def test_chain_legit_empty_does_not_fall_back(monkeypatch):
    legit_empty = ([], [], 0.4)
    _fake_chain(monkeypatch, {
        'gemini': legit_empty, 'qwen': RuntimeError('should not call')})

    boxes, ranks, _elapsed, provider = request_placing_bboxes_chain(
        _img(), item_description='mug',
        provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert boxes == [] and ranks == [] and provider == 'gemini'


def test_chain_all_fail_raises(monkeypatch):
    # Total failure must surface as VlmPlacingError so the caller reports
    # 'VLM unavailable' instead of 'no usable placing regions'.
    _fake_chain(monkeypatch, {
        'gemini': VlmPlacingError('a'), 'qwen': VlmPlacingError('b')})

    with pytest.raises(VlmPlacingError, match='all providers failed'):
        request_placing_bboxes_chain(
            _img(), item_description='mug',
            provider_models=[('gemini', 'g'), ('qwen', 'q')])


def test_chain_empty_provider_models_raises():
    with pytest.raises(VlmPlacingError, match='all providers failed'):
        request_placing_bboxes_chain(
            _img(), item_description='mug', provider_models=[])
