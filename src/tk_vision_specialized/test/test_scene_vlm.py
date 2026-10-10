"""ROS-free tests; HTTP and JPEG encoding are mocked, no API requests."""
import json
import sys
from types import SimpleNamespace

import pytest

from tk_vision_specialized.scene_vlm import (
    BEHAVIOURS, PROMPTS, infer_scene, parse_observations,
)


def payload(labels=None, bbox=None):
    return json.dumps({'observations': [{
        'labels': labels or ['sitting_on_chair', 'sitting_waving'],
        'description': 'A seated person raises a hand.',
        'bbox': bbox if bbox is not None else [100, 200, 600, 900],
    }]})


def test_multiple_behaviours_and_pixel_bounds():
    row, = parse_observations(payload(), 'behaviour', 640, 480)
    assert row.labels == ('sitting_on_chair', 'sitting_waving')
    assert row.bbox == (64, 96, 384, 432)
    row, = parse_observations(payload(bbox=[0, 0, 1000, 1000]), 'behaviour', 640, 480)
    assert row.bbox == (0, 0, 640, 480)


@pytest.mark.parametrize('label', BEHAVIOURS)
def test_supported_behaviours(label):
    assert parse_observations(payload([label]), 'behaviour', 20, 20)[0].labels == (label,)


def test_litter_and_empty_results():
    assert parse_observations(payload(['floor_litter']), 'litter', 20, 20)
    assert parse_observations('```json\n{"observations": []}\n```', 'litter', 20, 20) == []
    with pytest.raises(ValueError):
        parse_observations(payload(['sitting_on_chair']), 'litter', 20, 20)


@pytest.mark.parametrize('box', [[], [0, 0, 1], [5, 0, 1, 5], [0, 0, 0, 1],
                               [-1, 0, 1, 5], [0, 0, 1001, 5],
                               [False, 0, 5, 5], [0, 0, float('nan'), 5]])
def test_malformed_boxes_are_errors_not_negative(box):
    with pytest.raises(ValueError):
        parse_observations(payload(bbox=box), 'behaviour', 640, 480)


@pytest.mark.parametrize('content', ['{}', '{"observations": null}',
                                     '{"observations": {}}', 'not json',
                                     '{"observations": [null]}'])
def test_invalid_schema(content):
    with pytest.raises(ValueError):
        parse_observations(content, 'behaviour', 640, 480)


def fake_factory(monkeypatch, answers):
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(imencode=lambda *a: (True, b'jpeg')))
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-qwen-key')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-gemini-key')
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create(self, **kwargs):
            calls[-1]['request'] = kwargs
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])

    return Client, calls


def test_error_fallback_and_payload(monkeypatch):
    factory, calls = fake_factory(monkeypatch, [ValueError('bad'), payload(['floor_litter'])])
    rows, info = infer_scene(SimpleNamespace(shape=(480, 640, 3)), 'litter',
                             [('qwen', 'q'), ('gemini', 'g')], attempts=1,
                             client_factory=factory)
    assert info == {'provider': 'gemini', 'model': 'g'}
    assert rows[0].bbox == (64, 96, 384, 432)
    assert all(c['max_retries'] == 0 for c in calls)
    assert calls[0]['request']['messages'][0]['content'] == PROMPTS['litter']


def test_clean_negative_does_not_fall_back(monkeypatch):
    factory, calls = fake_factory(monkeypatch, ['{"observations": []}'])
    rows, _ = infer_scene(SimpleNamespace(shape=(20, 20, 3)), 'behaviour',
                          [('qwen', 'q'), ('gemini', 'g')], client_factory=factory)
    assert rows == [] and len(calls) == 1


def test_malformed_json_retried(monkeypatch):
    factory, calls = fake_factory(monkeypatch, ['{}', payload()])
    rows, _ = infer_scene(SimpleNamespace(shape=(20, 20, 3)), 'behaviour',
                          [('qwen', 'q')], client_factory=factory)
    assert rows and len(calls) == 2


def test_no_key_is_failure(monkeypatch):
    factory, calls = fake_factory(monkeypatch, [])
    for key in ('DASHSCOPE_API_KEY', 'DASHCOPE_API_KEY', 'OPENROUTER_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match='missing API key'):
        infer_scene(SimpleNamespace(shape=(20, 20, 3)), 'litter',
                    [('qwen', 'q')], client_factory=factory)
    assert calls == []


def test_timeout_budget_prevents_request(monkeypatch):
    factory, calls = fake_factory(monkeypatch, [])
    clock = iter([0, 31])
    monkeypatch.setattr('tk_vision_specialized.scene_vlm.time.monotonic', lambda: next(clock))
    with pytest.raises(RuntimeError, match='budget'):
        infer_scene(SimpleNamespace(shape=(20, 20, 3)), 'litter',
                    [('qwen', 'q')], client_factory=factory)
    assert calls == []
