"""Unit tests for vlm_judge_client.py."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tk_vision_specialized.vlm_judge_client import (
    JudgeChoice,
    decode_judge_response,
    QwenJudgeClient,
    GeminiJudgeClient,
    build_judge_client,
)


def test_decode_judge_winner():
    body = json.dumps({'label': 'milk', 'confidence': 0.95})
    choice = decode_judge_response(body, competing_labels={'milk', 'cola'})
    assert choice == JudgeChoice(label='milk', conf=0.95)


def test_decode_judge_abstain_via_null():
    body = json.dumps({'label': None, 'confidence': 0.0})
    choice = decode_judge_response(body, competing_labels={'milk', 'cola'})
    assert choice == JudgeChoice(label='', conf=0.0)


def test_decode_judge_abstain_via_empty_string():
    body = json.dumps({'label': '', 'confidence': 0.0})
    choice = decode_judge_response(body, competing_labels={'milk', 'cola'})
    assert choice == JudgeChoice(label='', conf=0.0)


def test_decode_judge_rejects_out_of_set_label():
    body = json.dumps({'label': 'banana', 'confidence': 0.9})
    choice = decode_judge_response(body, competing_labels={'milk', 'cola'})
    # Hallucinated label -> abstain.
    assert choice == JudgeChoice(label='', conf=0.0)


def test_decode_judge_bad_json_returns_none():
    body = 'not json'
    choice = decode_judge_response(body, competing_labels={'milk'})
    assert choice is None


def test_qwen_judge_client_init(monkeypatch):
    # The workspace .env carries DASHCOPE_API_KEY (typo); drop it so the
    # canonical name resolves cleanly. Matches Task 5's match-client test.
    monkeypatch.delenv('DASHCOPE_API_KEY', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'fake')
    client = QwenJudgeClient(model='qwen3-vl-plus')
    assert client._api_key == 'fake'


def test_gemini_judge_client_init(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fake')
    client = GeminiJudgeClient(model='google/gemini-2.5-pro')
    assert client._api_key == 'fake'


def test_qwen_judge_end_to_end(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'fake')
    crop = np.zeros((40, 60, 3), dtype=np.uint8)
    body = json.dumps({'label': 'cola', 'confidence': 0.88})

    class FakeOpenAI:
        def __init__(self, *a, **kw):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[SimpleNamespace(
                            message=SimpleNamespace(content=body),
                        )],
                    ),
                ),
            )

        def with_options(self, **kw):
            return self

        def close(self):
            pass

    with patch('tk_vision_specialized.vlm_judge_client.OpenAI', FakeOpenAI):
        client = QwenJudgeClient(model='qwen3-vl-plus')
        choice = client.choose(
            crop_bgr=crop,
            competing=[('milk', 'data:image/jpeg;base64,M'),
                       ('cola', 'data:image/jpeg;base64,C')],
            timeout_s=5.0, max_retries=1,
        )

    assert choice is not None
    assert choice.label == 'cola'


def test_build_judge_client_unknown_provider_raises():
    with pytest.raises(ValueError, match='Unknown provider'):
        build_judge_client('llama')
