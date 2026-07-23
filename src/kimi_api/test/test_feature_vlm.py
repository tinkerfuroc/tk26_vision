"""Unit tests for _feature_vlm.py -- the plain-text Gemini/Qwen provider
chain backing feature_extraction_service. No network: the OpenAI client is
monkeypatched per-test, mirroring test_waving_vlm.py's approach."""

from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest

from kimi_api._feature_vlm import (
    FeatureVlmError,
    FeatureVlmResult,
    request_feature_description,
    request_feature_description_chain,
)


def _completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _make_fake_openai(script):
    """Build a fake openai.OpenAI whose .create runs `script(kwargs)`.

    `script` returns the response content string, or raises to simulate an
    API error. Records constructor kwargs and create kwargs for asserts.
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


def _img_url():
    return 'data:image/jpeg;base64,fake'


# --- request_feature_description ---

def test_request_feature_description_returns_text_on_success(monkeypatch):
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: 'a description')
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_feature_description(
        _img_url(), 'sys prompt', 'user prompt',
        provider='qwen', model='qwen3-vl-plus', qwen_api_backend='dashscope')

    assert isinstance(res, FeatureVlmResult)
    assert res.text == 'a description'
    assert res.provider == 'qwen'
    assert fake.last_init['base_url'] == \
        'https://dashscope.aliyuncs.com/compatible-mode/v1'
    assert fake.last_init['api_key'] == 'k'
    assert fake.last_init['max_retries'] == 0


def test_request_feature_description_qwen_openrouter_backend(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    fake = _make_fake_openai(lambda kw: 'a description')
    monkeypatch.setattr(openai, 'OpenAI', fake)

    request_feature_description(
        'data:url', 'sys', 'user',
        provider='qwen', model='', qwen_api_backend='openrouter')

    assert fake.last_init['base_url'] == 'https://openrouter.ai/api/v1'
    assert fake.last_init['api_key'] == 'or-key'


def test_request_feature_description_qwen_openrouter_missing_key_raises(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with pytest.raises(FeatureVlmError, match='OPENROUTER_API_KEY'):
        request_feature_description(
            'data:url', 'sys', 'user',
            provider='qwen', model='', qwen_api_backend='openrouter')


def test_request_feature_description_gemini_uses_openrouter(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'g-key')
    fake = _make_fake_openai(lambda kw: 'a description')
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_feature_description(
        _img_url(), 'sys prompt', 'user prompt',
        provider='gemini', model='google/gemini-2.5-flash')

    assert res.provider == 'gemini'
    assert fake.last_init['base_url'] == 'https://openrouter.ai/api/v1'
    assert fake.last_init['api_key'] == 'g-key'


def test_request_feature_description_missing_key_raises(monkeypatch):
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    with pytest.raises(FeatureVlmError, match='DASHSCOPE_API_KEY'):
        request_feature_description(
            _img_url(), 'sys', 'user', provider='qwen', model='m')


def test_request_feature_description_unknown_provider_raises(monkeypatch):
    with pytest.raises(FeatureVlmError, match='unknown provider'):
        request_feature_description(
            _img_url(), 'sys', 'user', provider='nope', model='m')


def test_request_feature_description_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._feature_vlm.time.sleep', lambda s: None)
    attempts = {'n': 0}

    def script(kw):
        attempts['n'] += 1
        if attempts['n'] < 2:
            raise RuntimeError('transient failure')
        return 'ok after retry'

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_feature_description(
        _img_url(), 'sys', 'user', provider='gemini', model='g',
        max_retries=3)

    assert res.text == 'ok after retry'
    assert attempts['n'] == 2


def test_request_feature_description_empty_content_is_retried(monkeypatch):
    # None/empty content (OpenRouter safety block / empty candidate) must not
    # count as success — retry, so the chain's Qwen fallback can fire.
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._feature_vlm.time.sleep', lambda s: None)
    attempts = {'n': 0}

    def script(kw):
        attempts['n'] += 1
        return None if attempts['n'] < 2 else 'a real description'

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_feature_description(
        _img_url(), 'sys', 'user', provider='gemini', model='g',
        max_retries=3)

    assert res.text == 'a real description'
    assert attempts['n'] == 2


def test_request_feature_description_all_empty_content_raises(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._feature_vlm.time.sleep', lambda s: None)
    fake = _make_fake_openai(lambda kw: '')
    monkeypatch.setattr(openai, 'OpenAI', fake)

    with pytest.raises(FeatureVlmError, match='exhausted 3 retries'):
        request_feature_description(
            _img_url(), 'sys', 'user', provider='gemini', model='g',
            max_retries=3)


def test_request_feature_description_exhausts_retries_raises(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._feature_vlm.time.sleep', lambda s: None)

    def script(kw):
        raise RuntimeError('always fails')

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    with pytest.raises(FeatureVlmError, match='exhausted 3 retries'):
        request_feature_description(
            _img_url(), 'sys', 'user', provider='gemini', model='g',
            max_retries=3)
    assert len(fake.calls) == 3


def test_request_feature_description_abort_stops_retries_and_sleep(
    monkeypatch,
):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    aborted = {'value': False}
    sleeps = []

    def script(kw):
        aborted['value'] = True
        raise RuntimeError('transient failure')

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)
    monkeypatch.setattr(
        'kimi_api._feature_vlm.time.sleep',
        lambda seconds: sleeps.append(seconds))

    with pytest.raises(FeatureVlmError, match='aborted'):
        request_feature_description(
            _img_url(), 'sys', 'user', provider='gemini', model='g',
            max_retries=3, should_abort=lambda: aborted['value'])

    assert len(fake.calls) == 1
    assert sleeps == []


# --- request_feature_description_chain (monkeypatch the per-provider call) ---

def _fake_chain(monkeypatch, by_provider):
    """by_provider: dict provider -> FeatureVlmResult or Exception."""
    def fake_request(
            image_url, sys_prompt, user_text, *, provider, model, **kw):
        v = by_provider[provider]
        if isinstance(v, Exception):
            raise v
        return v
    monkeypatch.setattr(
        'kimi_api._feature_vlm.request_feature_description', fake_request)


def test_chain_first_success_short_circuits(monkeypatch):
    good = FeatureVlmResult(text='desc', provider='gemini')
    _fake_chain(monkeypatch, {
        'gemini': good, 'qwen': RuntimeError('should not call')})

    res = request_feature_description_chain(
        _img_url(), 'sys', 'user',
        provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert res.text == 'desc' and res.provider == 'gemini'


def test_chain_falls_back_on_gemini_exhaustion(monkeypatch):
    good = FeatureVlmResult(text='qwen desc', provider='qwen')
    _fake_chain(monkeypatch, {
        'gemini': FeatureVlmError('exhausted 3 retries'), 'qwen': good})

    res = request_feature_description_chain(
        _img_url(), 'sys', 'user',
        provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert res.provider == 'qwen' and res.text == 'qwen desc'


def test_chain_all_fail_raises(monkeypatch):
    _fake_chain(monkeypatch, {
        'gemini': FeatureVlmError('a'), 'qwen': FeatureVlmError('b')})

    with pytest.raises(FeatureVlmError, match='all providers failed'):
        request_feature_description_chain(
            _img_url(), 'sys', 'user',
            provider_models=[('gemini', 'g'), ('qwen', 'q')])


def test_chain_abort_stops_provider_fallthrough(monkeypatch):
    aborted = {'value': False}
    providers = []

    def fake_request(
            image_url, sys_prompt, user_text, *, provider, model, **kw):
        providers.append(provider)
        aborted['value'] = True
        raise FeatureVlmError('provider failed')

    monkeypatch.setattr(
        'kimi_api._feature_vlm.request_feature_description', fake_request)

    with pytest.raises(FeatureVlmError, match='aborted'):
        request_feature_description_chain(
            _img_url(), 'sys', 'user',
            provider_models=[('gemini', 'g'), ('qwen', 'q')],
            should_abort=lambda: aborted['value'])

    assert providers == ['gemini']


def test_chain_empty_provider_models_raises(monkeypatch):
    with pytest.raises(FeatureVlmError, match='all providers failed'):
        request_feature_description_chain(
            _img_url(), 'sys', 'user', provider_models=[])
