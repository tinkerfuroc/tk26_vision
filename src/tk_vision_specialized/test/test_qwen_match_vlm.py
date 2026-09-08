"""Unit tests for tk_vision_specialized.qwen_match_vlm's backend routing."""
import numpy as np
import openai
import pytest

from tk_vision_specialized.qwen_match_vlm import QwenMatchError, request_match_bboxes


def _img():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class _FakeOpenAI:
    last_init = None

    def __init__(self, **kw):
        type(self).last_init = kw

    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                class _Msg:
                    content = '{"detections": []}'
                class _Choice:
                    message = _Msg()
                class _Resp:
                    choices = [_Choice()]
                return _Resp()


def test_request_match_bboxes_dashscope_default(monkeypatch):
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    monkeypatch.setattr(openai, 'OpenAI', _FakeOpenAI)

    request_match_bboxes(_img(), 'data:url', item_name='mug', qwen_api_backend='dashscope')

    assert _FakeOpenAI.last_init['base_url'] == \
        'https://dashscope.aliyuncs.com/compatible-mode/v1'
    assert _FakeOpenAI.last_init['api_key'] == 'ds-key'


def test_request_match_bboxes_openrouter_backend(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    monkeypatch.setattr(openai, 'OpenAI', _FakeOpenAI)

    request_match_bboxes(_img(), 'data:url', item_name='mug', qwen_api_backend='openrouter')

    assert _FakeOpenAI.last_init['base_url'] == 'https://openrouter.ai/api/v1'
    assert _FakeOpenAI.last_init['api_key'] == 'or-key'


def test_request_match_bboxes_openrouter_missing_key_raises(monkeypatch):
    # request_match_bboxes calls load_env() (load_dotenv, default
    # override=False) before resolving the key so a real .env at the
    # workspace root would silently repopulate OPENROUTER_API_KEY after
    # delenv below -- stub it out for a hermetic negative test, mirroring
    # kimi_api/test/test_seat_bbox_vlm.py's
    # test_request_seat_bbox_qwen_openrouter_missing_key_raises.
    monkeypatch.setattr(
        'tk_vision_specialized.qwen_match_vlm.load_env', lambda: None)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with pytest.raises(QwenMatchError, match='OPENROUTER_API_KEY'):
        request_match_bboxes(_img(), 'data:url', item_name='mug', qwen_api_backend='openrouter')


def test_request_match_bboxes_explicit_base_url_override_wins(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    monkeypatch.setattr(openai, 'OpenAI', _FakeOpenAI)

    request_match_bboxes(
        _img(), 'data:url', item_name='mug',
        base_url='https://self-hosted.example/v1', qwen_api_backend='dashscope')

    assert _FakeOpenAI.last_init['base_url'] == 'https://self-hosted.example/v1'
