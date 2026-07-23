# Copyright 2026 Tinker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the object-scan VLM provider chain."""

from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest

from kimi_api._scan_vlm import (
    ScanVlmError,
    request_scan_labels,
    request_scan_labels_chain,
    validate_labels,
)


def _completion(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _make_fake_openai(script):
    class _Fake:
        calls = []
        last_init = None

        def __init__(self, **kwargs):
            _Fake.last_init = kwargs

        def with_options(self, **_kwargs):
            return self

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            _Fake.calls.append(kwargs)
            return _completion(script(kwargs))

        def close(self):
            pass

    return _Fake


def test_validate_labels_filters_and_preserves_response_order():
    labels = validate_labels(
        '["CUP", "hallucination", "apple", "cup"]',
        ['apple', 'cup'],
    )

    assert labels == ['cup', 'apple']


def test_request_scan_labels_disables_sdk_retries(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'key')
    fake = _make_fake_openai(lambda _kwargs: '["cup"]')
    monkeypatch.setattr(openai, 'OpenAI', fake)

    result = request_scan_labels(
        'data:image',
        ['cup'],
        provider='gemini',
        model='gemini',
    )

    assert result.labels == ['cup']
    assert result.provider == 'gemini'
    assert fake.last_init['max_retries'] == 0


def test_request_scan_labels_abort_skips_retry_and_backoff(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'key')
    aborted = {'value': False}
    sleeps = []

    def script(_kwargs):
        aborted['value'] = True
        raise RuntimeError('transient failure')

    fake = _make_fake_openai(script)
    monkeypatch.setattr(openai, 'OpenAI', fake)
    monkeypatch.setattr(
        'kimi_api._scan_vlm.time.sleep',
        lambda seconds: sleeps.append(seconds),
    )

    with pytest.raises(ScanVlmError, match='aborted'):
        request_scan_labels(
            'data:image',
            ['cup'],
            provider='gemini',
            model='gemini',
            max_retries=3,
            should_abort=lambda: aborted['value'],
        )

    assert len(fake.calls) == 1
    assert sleeps == []


def test_request_scan_labels_chain_abort_skips_fallback(monkeypatch):
    aborted = {'value': False}
    providers = []

    def fake_request(
        image_url,
        candidates,
        *,
        provider,
        model,
        **kwargs,
    ):
        providers.append(provider)
        assert kwargs['should_abort'] is should_abort
        aborted['value'] = True
        raise ScanVlmError('provider failed')

    def should_abort():
        return aborted['value']

    monkeypatch.setattr(
        'kimi_api._scan_vlm.request_scan_labels',
        fake_request,
    )

    with pytest.raises(ScanVlmError, match='aborted'):
        request_scan_labels_chain(
            'data:image',
            ['cup'],
            provider_models=[('gemini', 'g'), ('qwen', 'q')],
            should_abort=should_abort,
        )

    assert providers == ['gemini']
