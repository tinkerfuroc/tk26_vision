"""Unit tests for kimi_api._env.resolve_qwen_target."""
import pytest

from kimi_api._env import resolve_qwen_target


def test_dashscope_sentinel_model_uses_dashscope_default(monkeypatch):
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    base_url, api_key, model = resolve_qwen_target('dashscope', '')
    assert base_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    assert api_key == 'ds-key'
    assert model == 'qwen3-vl-plus'


def test_dashscope_explicit_model_honored_verbatim(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    _, _, model = resolve_qwen_target('dashscope', 'qwen-vl-max')
    assert model == 'qwen-vl-max'


def test_dashscope_missing_key_raises(monkeypatch):
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='DASHSCOPE_API_KEY'):
        resolve_qwen_target('dashscope', '')


def test_dashscope_rejects_openrouter_shaped_model(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    with pytest.raises(RuntimeError, match='dashscope'):
        resolve_qwen_target('dashscope', 'qwen/qwen3-vl-32b-instruct')


def test_openrouter_sentinel_model_uses_openrouter_default(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    base_url, api_key, model = resolve_qwen_target('openrouter', '')
    assert base_url == 'https://openrouter.ai/api/v1'
    assert api_key == 'or-key'
    assert model == 'qwen/qwen3-vl-32b-instruct'


def test_openrouter_explicit_model_honored_verbatim(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    _, _, model = resolve_qwen_target('openrouter', 'qwen/qwen3.7-plus')
    assert model == 'qwen/qwen3.7-plus'


def test_openrouter_missing_key_raises(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='OPENROUTER_API_KEY'):
        resolve_qwen_target('openrouter', '')


def test_openrouter_rejects_dashscope_shaped_model(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    with pytest.raises(RuntimeError, match='openrouter'):
        resolve_qwen_target('openrouter', 'qwen3-vl-plus')


def test_invalid_backend_raises(monkeypatch):
    with pytest.raises(RuntimeError, match='qwen_api_backend'):
        resolve_qwen_target('bogus', '')


def test_base_url_override_wins_regardless_of_backend(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    base_url, _, _ = resolve_qwen_target(
        'dashscope', '', base_url_override='https://self-hosted.example/v1')
    assert base_url == 'https://self-hosted.example/v1'


def test_openrouter_base_url_override_still_uses_dashscope_key_when_backend_dashscope(monkeypatch):
    # base_url_override does not change which key is required — only which
    # host is called. Confirms the two concerns (key selection vs base URL)
    # are independent.
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    _, api_key, _ = resolve_qwen_target(
        'dashscope', '', base_url_override='https://gateway.example/v1')
    assert api_key == 'ds-key'
