"""Unit tests for _seat_vlm.py's provider support and request_seat_chain --
the Qwen fallback for seat_recommend_bbox.py's legacy 'point' vlm_strategy.
No network: the OpenAI client is monkeypatched per-test."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import openai
import pytest

from kimi_api._seat_vlm import VlmSeatError, request_seat, request_seat_chain


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


_VALID_PAYLOAD = json.dumps({
    'visible_seats': [{'label': 'left chair', 'occupied': False, 'reason': 'empty'}],
    'label': 'left chair',
    'point': [500, 500],
})

_NONE_PAYLOAD = json.dumps({'visible_seats': [], 'label': 'none', 'point': [0, 0]})


# --- request_seat provider support ---

def test_request_seat_defaults_to_gemini(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: _VALID_PAYLOAD)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    label, point_xy, visible_seats, _elapsed = request_seat(_img(), [], [], model='g')

    assert label == 'left chair'
    assert fake.last_init['base_url'] == 'https://openrouter.ai/api/v1'


def test_request_seat_qwen_uses_dashscope(monkeypatch):
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: _VALID_PAYLOAD)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    label, point_xy, visible_seats, _elapsed = request_seat(
        _img(), [], [], model='qwen3-vl-plus', provider='qwen')

    assert label == 'left chair'
    assert fake.last_init['base_url'] == \
        'https://dashscope.aliyuncs.com/compatible-mode/v1'


def test_request_seat_qwen_openrouter_backend(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    fake = _make_fake_openai(lambda kw: _VALID_PAYLOAD)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    label, point_xy, visible_seats, _elapsed = request_seat(
        _img(), [], [], model='', provider='qwen', qwen_api_backend='openrouter')

    assert label == 'left chair'
    assert fake.last_init['base_url'] == 'https://openrouter.ai/api/v1'
    assert fake.last_init['api_key'] == 'or-key'


def test_request_seat_qwen_openrouter_missing_key_raises(monkeypatch):
    # request_seat calls load_env() (load_dotenv, override=False) on every
    # invocation, which would silently repopulate the key from the
    # workspace .env after delenv -- stub it out for a hermetic negative test.
    monkeypatch.setattr('kimi_api._seat_vlm.load_env', lambda: None)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with pytest.raises(VlmSeatError, match='OPENROUTER_API_KEY'):
        request_seat(
            _img(), [], [], model='', provider='qwen', qwen_api_backend='openrouter')


def test_request_seat_missing_key_raises(monkeypatch):
    # request_seat calls load_env() (load_dotenv, override=False) on every
    # invocation, which would silently repopulate the key from the
    # workspace .env after delenv -- stub it out for a hermetic negative test.
    monkeypatch.setattr('kimi_api._seat_vlm.load_env', lambda: None)
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    with pytest.raises(VlmSeatError, match='DASHSCOPE_API_KEY'):
        request_seat(_img(), [], [], model='m', provider='qwen')


def test_request_seat_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    with pytest.raises(VlmSeatError, match='unknown provider'):
        request_seat(_img(), [], [], model='m', provider='nope')


# --- request_seat_chain ---

def _fake_chain(monkeypatch, by_provider):
    """by_provider: dict provider -> tuple result or Exception."""
    def fake(rgb, names, features, *, provider, model, **kw):
        v = by_provider[provider]
        if isinstance(v, Exception):
            raise v
        return v
    monkeypatch.setattr('kimi_api._seat_vlm.request_seat', fake)


def test_chain_first_success_short_circuits(monkeypatch):
    good = ('left chair', (10, 20), [{'label': 'left chair'}], 1.5)
    _fake_chain(monkeypatch, {
        'gemini': good, 'qwen': RuntimeError('should not call')})

    label, point_xy, visible_seats, elapsed, provider = request_seat_chain(
        _img(), [], [], provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert label == 'left chair' and point_xy == (10, 20) and provider == 'gemini'


def test_chain_falls_back_on_missing_key(monkeypatch):
    good = ('right chair', (5, 5), [], 0.5)
    _fake_chain(monkeypatch, {
        'gemini': VlmSeatError('no key'), 'qwen': good})

    label, point_xy, _seats, _elapsed, provider = request_seat_chain(
        _img(), [], [], provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert label == 'right chair' and provider == 'qwen'


def test_chain_falls_back_on_exhaustion(monkeypatch):
    # request_seat's own convention: exhausted retries -> label == ''.
    exhausted = ('', None, [], 2.0)
    good = ('qwen chair', (1, 1), [], 0.3)
    _fake_chain(monkeypatch, {'gemini': exhausted, 'qwen': good})

    label, _point, _seats, _elapsed, provider = request_seat_chain(
        _img(), [], [], provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert label == 'qwen chair' and provider == 'qwen'


def test_chain_legit_none_does_not_fall_back(monkeypatch):
    none_result = ('none', None, [], 0.4)
    _fake_chain(monkeypatch, {
        'gemini': none_result, 'qwen': RuntimeError('should not call')})

    label, point_xy, _seats, _elapsed, provider = request_seat_chain(
        _img(), [], [], provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert label == 'none' and point_xy is None and provider == 'gemini'


def test_chain_all_fail_raises(monkeypatch):
    # Total failure (both providers exhausted) must surface as an error the
    # caller can report as 'VLM unavailable', NOT as a semantic 'no seat'.
    _fake_chain(monkeypatch, {
        'gemini': ('', None, [], 1.0), 'qwen': ('', None, [], 1.0)})

    with pytest.raises(VlmSeatError, match='all providers failed'):
        request_seat_chain(
            _img(), [], [], provider_models=[('gemini', 'g'), ('qwen', 'q')])


def test_chain_all_config_errors_raises(monkeypatch):
    _fake_chain(monkeypatch, {
        'gemini': VlmSeatError('no key'), 'qwen': VlmSeatError('no key')})

    with pytest.raises(VlmSeatError, match='all providers failed'):
        request_seat_chain(
            _img(), [], [], provider_models=[('gemini', 'g'), ('qwen', 'q')])


def test_chain_empty_provider_models_raises():
    with pytest.raises(VlmSeatError, match='all providers failed'):
        request_seat_chain(_img(), [], [], provider_models=[])


def test_request_seat_undecodable_point_with_real_label_is_retried(monkeypatch):
    # A non-"none" label whose point fails to decode ([0,0] sentinel or
    # malformed) is an unusable localization, not an answer — it must consume
    # a retry so the chain's fallback can fire, instead of surfacing as a
    # spurious "no empty seat".
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._seat_vlm.time.sleep', lambda s: None)
    attempts = {'n': 0}

    def script(kw):
        attempts['n'] += 1
        if attempts['n'] < 2:
            return json.dumps({
                'visible_seats': [
                    {'label': 'left couch', 'occupied': False, 'reason': 'x'}],
                'label': 'left couch',
                'point': [0, 0],
            })
        return _VALID_PAYLOAD

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    label, point_xy, _seats, _elapsed = request_seat(_img(), [], [], model='g')

    assert label == 'left chair'
    assert point_xy is not None
    assert attempts['n'] == 2


def test_request_seat_none_label_still_returns_cleanly(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: _NONE_PAYLOAD)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    label, point_xy, _seats, _elapsed = request_seat(_img(), [], [], model='g')

    assert label == 'none' and point_xy is None


def test_request_seat_empty_label_is_retried(monkeypatch):
    # A schema-valid response with label='' is useless to the caller and must
    # not be confused with the chain's exhaustion signal — retry it instead.
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._seat_vlm.time.sleep', lambda s: None)
    attempts = {'n': 0}

    def script(kw):
        attempts['n'] += 1
        if attempts['n'] < 2:
            return json.dumps(
                {'visible_seats': [], 'label': '', 'point': [0, 0]})
        return _VALID_PAYLOAD

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    label, _point, _seats, _elapsed = request_seat(_img(), [], [], model='g')

    assert label == 'left chair'
    assert attempts['n'] == 2
