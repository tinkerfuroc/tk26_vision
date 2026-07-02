"""Unit tests for _categorize_vlm.py -- the Gemini/Qwen shelf-layer provider
chain backing grocery_categorize's Categorize action. No network: the
OpenAI client is monkeypatched per-test, mirroring test_feature_vlm.py."""

from __future__ import annotations

import json
from types import SimpleNamespace

import openai
import pytest

from kimi_api._categorize_vlm import (
    ShelfVlmError,
    ShelfVlmResult,
    request_shelf_layer,
    request_shelf_layer_chain,
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


_VALID_PAYLOAD = json.dumps({
    'object_description': 'a can of soup',
    'shelf_description': ['drinks', 'food'],
    'reason': 'goes with food',
    'layer': 1,
})


# --- request_shelf_layer ---

def test_request_shelf_layer_returns_response_on_success(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: _VALID_PAYLOAD)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_shelf_layer(
        'sys', 'shelf-url', 'obj-url', provider='gemini', model='g')

    assert isinstance(res, ShelfVlmResult)
    assert res.response['layer'] == 1
    assert res.provider == 'gemini'
    assert fake.last_init['base_url'] == 'https://openrouter.ai/api/v1'


def test_request_shelf_layer_qwen_uses_dashscope(monkeypatch):
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: _VALID_PAYLOAD)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_shelf_layer(
        'sys', 'shelf-url', 'obj-url', provider='qwen', model='qwen3-vl-plus')

    assert res.provider == 'qwen'
    assert fake.last_init['base_url'] == \
        'https://dashscope.aliyuncs.com/compatible-mode/v1'


def test_request_shelf_layer_missing_key_raises(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with pytest.raises(ShelfVlmError, match='OPENROUTER_API_KEY'):
        request_shelf_layer('sys', 'shelf-url', 'obj-url', provider='gemini', model='g')


def test_request_shelf_layer_retries_on_missing_field(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._categorize_vlm.time.sleep', lambda s: None)
    attempts = {'n': 0}

    def script(kw):
        attempts['n'] += 1
        if attempts['n'] < 2:
            return json.dumps({'reason': 'no layer field here'})
        return _VALID_PAYLOAD

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_shelf_layer(
        'sys', 'shelf-url', 'obj-url', provider='gemini', model='g', max_retries=3)

    assert res.response['layer'] == 1
    assert attempts['n'] == 2


def test_request_shelf_layer_retries_on_missing_reason(monkeypatch):
    # The caller reads response['reason'] unconditionally (place_reason), so
    # a response missing it must consume a retry, not be accepted.
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._categorize_vlm.time.sleep', lambda s: None)
    attempts = {'n': 0}

    def script(kw):
        attempts['n'] += 1
        if attempts['n'] < 2:
            return json.dumps({'layer': 1, 'shelf_description': ['food']})
        return _VALID_PAYLOAD

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_shelf_layer(
        'sys', 'shelf-url', 'obj-url', provider='gemini', model='g', max_retries=3)

    assert res.response['reason'] == 'goes with food'
    assert attempts['n'] == 2


def test_request_shelf_layer_retries_on_unparseable_json(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._categorize_vlm.time.sleep', lambda s: None)
    attempts = {'n': 0}

    def script(kw):
        attempts['n'] += 1
        return 'not json at all' if attempts['n'] < 2 else _VALID_PAYLOAD

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_shelf_layer(
        'sys', 'shelf-url', 'obj-url', provider='gemini', model='g', max_retries=3)

    assert res.response['layer'] == 1


def test_request_shelf_layer_exhausts_retries_raises(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._categorize_vlm.time.sleep', lambda s: None)
    fake = _make_fake_openai(lambda kw: 'not json at all')
    monkeypatch.setattr(openai, 'OpenAI', fake)

    with pytest.raises(ShelfVlmError, match='exhausted 3 attempts'):
        request_shelf_layer(
            'sys', 'shelf-url', 'obj-url', provider='gemini', model='g', max_retries=3)
    assert len(fake.calls) == 3


def test_request_shelf_layer_strips_markdown_fences(monkeypatch):
    # qwen3-vl-plus (the fallback leg) routinely wraps JSON in ```json fences
    # when no response_format is sent — must parse, not burn retries.
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: f'```json\n{_VALID_PAYLOAD}\n```')
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_shelf_layer(
        'sys', 'shelf-url', 'obj-url', provider='gemini', model='g')

    assert res.response['layer'] == 1
    assert len(fake.calls) == 1


def test_request_shelf_layer_none_content_is_retried(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._categorize_vlm.time.sleep', lambda s: None)
    attempts = {'n': 0}

    def script(kw):
        attempts['n'] += 1
        return None if attempts['n'] < 2 else _VALID_PAYLOAD

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_shelf_layer(
        'sys', 'shelf-url', 'obj-url', provider='gemini', model='g')

    assert res.response['layer'] == 1
    assert attempts['n'] == 2


def test_request_shelf_layer_non_int_layer_is_retried(monkeypatch):
    # The caller does clusters[int(layer)] — a non-integer layer must consume
    # a retry, not crash the action callback after acceptance.
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._categorize_vlm.time.sleep', lambda s: None)
    attempts = {'n': 0}

    def script(kw):
        attempts['n'] += 1
        if attempts['n'] < 2:
            return json.dumps({
                'layer': 'top shelf', 'shelf_description': ['x'],
                'reason': 'r'})
        return _VALID_PAYLOAD

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_shelf_layer(
        'sys', 'shelf-url', 'obj-url', provider='gemini', model='g')

    assert res.response['layer'] == 1
    assert attempts['n'] == 2


def test_request_shelf_layer_coerces_numeric_string_layer(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    payload = json.dumps({
        'layer': '2', 'shelf_description': ['x'], 'reason': 'r'})
    fake = _make_fake_openai(lambda kw: payload)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_shelf_layer(
        'sys', 'shelf-url', 'obj-url', provider='gemini', model='g')

    assert res.response['layer'] == 2


# --- request_shelf_layer_chain ---

def _fake_chain(monkeypatch, by_provider):
    def fake(sys_prompt, shelf_url, obj_url, *, provider, model, **kw):
        v = by_provider[provider]
        if isinstance(v, Exception):
            raise v
        return v
    monkeypatch.setattr('kimi_api._categorize_vlm.request_shelf_layer', fake)


def test_chain_first_success_short_circuits(monkeypatch):
    good = ShelfVlmResult(response={'layer': 0}, provider='gemini')
    _fake_chain(monkeypatch, {
        'gemini': good, 'qwen': RuntimeError('should not call')})

    res = request_shelf_layer_chain(
        'sys', 'shelf-url', 'obj-url',
        provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert res.response == {'layer': 0} and res.provider == 'gemini'


def test_chain_falls_back_on_gemini_exhaustion(monkeypatch):
    good = ShelfVlmResult(response={'layer': 2}, provider='qwen')
    _fake_chain(monkeypatch, {
        'gemini': ShelfVlmError('exhausted 3 attempts'), 'qwen': good})

    res = request_shelf_layer_chain(
        'sys', 'shelf-url', 'obj-url',
        provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert res.provider == 'qwen' and res.response == {'layer': 2}


def test_chain_all_fail_raises(monkeypatch):
    _fake_chain(monkeypatch, {
        'gemini': ShelfVlmError('a'), 'qwen': ShelfVlmError('b')})

    with pytest.raises(ShelfVlmError, match='all providers failed'):
        request_shelf_layer_chain(
            'sys', 'shelf-url', 'obj-url',
            provider_models=[('gemini', 'g'), ('qwen', 'q')])


def test_chain_empty_provider_models_raises():
    with pytest.raises(ShelfVlmError, match='no providers configured'):
        request_shelf_layer_chain(
            'sys', 'shelf-url', 'obj-url', provider_models=[])
