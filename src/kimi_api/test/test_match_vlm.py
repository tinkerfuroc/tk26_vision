"""Unit tests for _match_vlm.py -- the Gemini/Qwen JSON-list provider chain
backing the feature-matching action. No network: the OpenAI client is
monkeypatched per-test, mirroring test_feature_vlm.py's approach."""

from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest

from kimi_api._match_vlm import (
    MatchVlmError,
    MatchVlmResult,
    patch_result,
    request_match_indices,
    request_match_indices_chain,
)


# --- patch_result ---

def test_patch_result_valid_passthrough():
    patched, msg = patch_result([0, 2, 1], 3, 3)
    assert patched == [0, 2, 1] and msg == ''


def test_patch_result_not_a_list_is_unsalvageable():
    patched, msg = patch_result('nope', 3, 3)
    assert patched is None and 'not a list' in msg


def test_patch_result_empty_list_with_targets_is_unsalvageable():
    patched, msg = patch_result([], 2, 3)
    assert patched is None and 'empty list' in msg


def test_patch_result_missing_entries_use_cyclic_fallback():
    patched, _ = patch_result([0], 3, 3)
    assert patched == [0, 1, 2]


def test_patch_result_out_of_range_uses_cyclic_fallback():
    patched, _ = patch_result([-1, 99, 1], 3, 3)
    assert patched == [0, 1, 1]


def test_patch_result_numeric_string_coerces():
    patched, _ = patch_result(['1', 2], 2, 3)
    assert patched == [1, 2]


def test_patch_result_none_and_bool_use_cyclic_fallback():
    # One real cell keeps the result salvageable; the bad cells are patched.
    patched, _ = patch_result([None, True, 2], 3, 3)
    assert patched == [0, 1, 2]


def test_patch_result_all_cells_fabricated_is_unsalvageable():
    # When EVERY cell required the cyclic i%n_cand fallback the VLM produced
    # zero usable signal — with a fallback provider available, fabricating a
    # full assignment must not count as success.
    patched, msg = patch_result([-1, -1], 2, 3)
    assert patched is None and 'no usable signal' in msg


def test_patch_result_all_fabricated_single_candidate_still_accepted():
    # With one candidate every assignment is [0, ...] regardless — nothing to
    # gain from retrying, keep the tk23-legacy acceptance.
    patched, _ = patch_result([None, None], 2, 1)
    assert patched == [0, 0]


def test_request_match_indices_all_fabricated_is_retried(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._match_vlm.time.sleep', lambda s: None)
    attempts = {'n': 0}

    def script(kw):
        attempts['n'] += 1
        return '[-1, -1]' if attempts['n'] < 2 else '[0, 1]'

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_match_indices(
        'sys', [], n_feats=2, n_cand=2, provider='gemini', model='g',
        max_retries=3)

    assert res.indices == [0, 1]
    assert attempts['n'] == 2


# --- request_match_indices ---

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


def test_request_match_indices_returns_patched_list(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: '[0, 1]')
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_match_indices(
        'sys', [{'type': 'text', 'text': 'x'}],
        n_feats=2, n_cand=2, provider='gemini', model='g')

    assert isinstance(res, MatchVlmResult)
    assert res.indices == [0, 1]
    assert res.provider == 'gemini'
    assert fake.last_init['base_url'] == 'https://openrouter.ai/api/v1'
    assert fake.last_init['max_retries'] == 0


def test_request_match_indices_qwen_uses_dashscope(monkeypatch):
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: '[0]')
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_match_indices(
        'sys', [], n_feats=1, n_cand=1, provider='qwen', model='qwen3-vl-plus')

    assert res.provider == 'qwen'
    assert fake.last_init['base_url'] == \
        'https://dashscope.aliyuncs.com/compatible-mode/v1'


def test_request_match_indices_qwen_openrouter_backend(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    fake = _make_fake_openai(lambda kw: '[0]')
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_match_indices(
        'sys', [], n_feats=1, n_cand=1, provider='qwen', model='',
        qwen_api_backend='openrouter')

    assert res.provider == 'qwen'
    assert fake.last_init['base_url'] == 'https://openrouter.ai/api/v1'
    assert fake.last_init['api_key'] == 'or-key'


def test_request_match_indices_qwen_openrouter_missing_key_raises(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with pytest.raises(MatchVlmError, match='OPENROUTER_API_KEY'):
        request_match_indices(
            'sys', [], n_feats=1, n_cand=1, provider='qwen', model='',
            qwen_api_backend='openrouter')


def test_request_match_indices_missing_key_raises(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with pytest.raises(MatchVlmError, match='OPENROUTER_API_KEY'):
        request_match_indices(
            'sys', [], n_feats=1, n_cand=1, provider='gemini', model='g')


def test_request_match_indices_retries_on_unsalvageable_parse(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._match_vlm.time.sleep', lambda s: None)
    attempts = {'n': 0}

    def script(kw):
        attempts['n'] += 1
        return 'not a list at all' if attempts['n'] < 2 else '[0, 1]'

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_match_indices(
        'sys', [], n_feats=2, n_cand=2, provider='gemini', model='g',
        max_retries=3)

    assert res.indices == [0, 1]
    assert attempts['n'] == 2


def test_request_match_indices_exhausts_retries_raises(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._match_vlm.time.sleep', lambda s: None)
    fake = _make_fake_openai(lambda kw: '[]')  # empty list w/ targets -> unsalvageable
    monkeypatch.setattr(openai, 'OpenAI', fake)

    with pytest.raises(MatchVlmError, match='exhausted 3 attempts'):
        request_match_indices(
            'sys', [], n_feats=2, n_cand=2, provider='gemini', model='g',
            max_retries=3)
    assert len(fake.calls) == 3


def test_request_match_indices_strips_markdown_fences(monkeypatch):
    # qwen3-vl-plus (the fallback leg) routinely wraps answers in ```json
    # fences when no response_format is sent — must parse, not burn retries.
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    fake = _make_fake_openai(lambda kw: '```json\n[0, 1]\n```')
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_match_indices(
        'sys', [], n_feats=2, n_cand=2, provider='gemini', model='g')

    assert res.indices == [0, 1]
    assert len(fake.calls) == 1


def test_request_match_indices_none_content_is_retried(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    monkeypatch.setattr('kimi_api._match_vlm.time.sleep', lambda s: None)
    attempts = {'n': 0}

    def script(kw):
        attempts['n'] += 1
        return None if attempts['n'] < 2 else '[0]'

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)

    res = request_match_indices(
        'sys', [], n_feats=1, n_cand=2, provider='gemini', model='g')

    assert res.indices == [0]
    assert attempts['n'] == 2


def test_request_match_indices_abort_stops_retries_and_sleep(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'k')
    aborted = {'value': False}
    sleeps = []

    def script(kw):
        aborted['value'] = True
        raise RuntimeError('transient failure')

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)
    monkeypatch.setattr(
        'kimi_api._match_vlm.time.sleep', lambda seconds: sleeps.append(seconds))

    with pytest.raises(MatchVlmError, match='aborted'):
        request_match_indices(
            'sys', [], n_feats=1, n_cand=1,
            provider='gemini', model='g', max_retries=3,
            should_abort=lambda: aborted['value'])

    assert len(fake.calls) == 1
    assert sleeps == []


# --- request_match_indices_chain ---

def _fake_chain(monkeypatch, by_provider):
    def fake(sys_prompt, user_content, *, provider, model, **kw):
        v = by_provider[provider]
        if isinstance(v, Exception):
            raise v
        return v
    monkeypatch.setattr('kimi_api._match_vlm.request_match_indices', fake)


def test_chain_first_success_short_circuits(monkeypatch):
    good = MatchVlmResult(indices=[0, 1], provider='gemini')
    _fake_chain(monkeypatch, {
        'gemini': good, 'qwen': RuntimeError('should not call')})

    res = request_match_indices_chain(
        'sys', [], n_feats=2, n_cand=2,
        provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert res.indices == [0, 1] and res.provider == 'gemini'


def test_chain_falls_back_on_gemini_exhaustion(monkeypatch):
    good = MatchVlmResult(indices=[1, 0], provider='qwen')
    _fake_chain(monkeypatch, {
        'gemini': MatchVlmError('exhausted 3 attempts'), 'qwen': good})

    res = request_match_indices_chain(
        'sys', [], n_feats=2, n_cand=2,
        provider_models=[('gemini', 'g'), ('qwen', 'q')])

    assert res.provider == 'qwen' and res.indices == [1, 0]


def test_chain_all_fail_raises(monkeypatch):
    _fake_chain(monkeypatch, {
        'gemini': MatchVlmError('a'), 'qwen': MatchVlmError('b')})

    with pytest.raises(MatchVlmError, match='all providers failed'):
        request_match_indices_chain(
            'sys', [], n_feats=2, n_cand=2,
            provider_models=[('gemini', 'g'), ('qwen', 'q')])


def test_chain_abort_stops_provider_fallthrough(monkeypatch):
    aborted = {'value': False}
    providers = []

    def fake(sys_prompt, user_content, *, provider, model, **kw):
        providers.append(provider)
        aborted['value'] = True
        raise MatchVlmError('provider failed')

    monkeypatch.setattr('kimi_api._match_vlm.request_match_indices', fake)

    with pytest.raises(MatchVlmError, match='aborted'):
        request_match_indices_chain(
            'sys', [], n_feats=2, n_cand=2,
            provider_models=[('gemini', 'g'), ('qwen', 'q')],
            should_abort=lambda: aborted['value'])

    assert providers == ['gemini']


def test_chain_empty_provider_models_raises():
    with pytest.raises(MatchVlmError, match='no providers configured'):
        request_match_indices_chain(
            'sys', [], n_feats=2, n_cand=2, provider_models=[])
