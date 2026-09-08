"""Unit tests for seat-pointing few-shot loader + request_seat wiring.

Run:
    pytest src/tk26_vision/src/kimi_api/test/test_seat_fewshot.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PKG_DIR))

from kimi_api import _seat_fewshot, _seat_vlm  # noqa: E402


def _good_answer(label='left cushion of gray sofa'):
    return {
        'visible_seats': [
            {'label': label, 'occupied': False, 'reason': 'cushion clear'},
            {'label': 'armchair under window', 'occupied': True, 'reason': 'person sitting'},
        ],
        'label': label,
        'point': [620, 300],
    }


def _write_example(root: Path, slug: str, answer: dict, image=None):
    d = root / slug
    d.mkdir(parents=True, exist_ok=False)
    if image is None:
        image = np.full((48, 64, 3), 200, dtype=np.uint8)
    import cv2
    cv2.imwrite(str(d / 'image.jpg'), image)
    (d / 'answer.json').write_text(json.dumps(answer))
    (d / 'meta.json').write_text(json.dumps({'created': 'test'}))
    return d


def test_validate_answer_accepts_good_payload():
    assert _seat_fewshot._validate_answer(_good_answer())


def test_validate_answer_rejects_label_not_in_seats():
    bad = _good_answer()
    bad['label'] = 'a seat we never enumerated'
    assert not _seat_fewshot._validate_answer(bad)


def test_validate_answer_rejects_point_out_of_range():
    bad = _good_answer()
    bad['point'] = [1500, 0]
    assert not _seat_fewshot._validate_answer(bad)


def test_validate_answer_none_label_requires_zero_point():
    payload = {
        'visible_seats': [
            {'label': 'a', 'occupied': True, 'reason': 'occupied'},
        ],
        'label': 'none',
        'point': [0, 0],
    }
    assert _seat_fewshot._validate_answer(payload)
    payload['point'] = [10, 10]
    assert not _seat_fewshot._validate_answer(payload)


def test_load_fewshots_round_trip(tmp_path, monkeypatch):
    _write_example(tmp_path, 'a_2026', _good_answer('seat A'))
    _write_example(tmp_path, 'b_2026', _good_answer('seat B'))
    _seat_fewshot._CACHE.clear()
    monkeypatch.setattr(_seat_fewshot, '_resolve_fewshot_dir', lambda: str(tmp_path))

    out = _seat_fewshot.load_fewshots(max_n=10)
    assert [e.slug for e in out] == ['a_2026', 'b_2026']
    assert all(e.image_bgr.ndim == 3 for e in out)
    assert out[0].answer['label'] == 'seat A'


def test_load_fewshots_max_n_truncates(tmp_path, monkeypatch):
    for i in range(5):
        _write_example(tmp_path, f's{i}', _good_answer(f'seat {i}'))
    _seat_fewshot._CACHE.clear()
    monkeypatch.setattr(_seat_fewshot, '_resolve_fewshot_dir', lambda: str(tmp_path))
    assert len(_seat_fewshot.load_fewshots(max_n=3)) == 3


def test_load_fewshots_skips_bad_slug(tmp_path, monkeypatch, capsys):
    _write_example(tmp_path, 'good', _good_answer('seat A'))
    bad_dir = tmp_path / 'bad'
    bad_dir.mkdir()
    (bad_dir / 'answer.json').write_text('not json')
    _seat_fewshot._CACHE.clear()
    monkeypatch.setattr(_seat_fewshot, '_resolve_fewshot_dir', lambda: str(tmp_path))

    logger = MagicMock()
    out = _seat_fewshot.load_fewshots(max_n=10, logger=logger)
    assert [e.slug for e in out] == ['good']
    assert logger.warn.called


def test_load_fewshots_returns_empty_when_dir_missing(monkeypatch):
    _seat_fewshot._CACHE.clear()
    monkeypatch.setattr(_seat_fewshot, '_resolve_fewshot_dir', lambda: None)
    assert _seat_fewshot.load_fewshots(max_n=3) == []


def test_request_seat_no_fewshots_message_count(monkeypatch):
    """fewshots=None ⇒ [system, user] (the original 2-message list)."""
    captured = {}

    def fake_create(**kwargs):
        captured['messages'] = kwargs['messages']
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = json.dumps({
            'visible_seats': [],
            'label': 'none',
            'point': [0, 0],
        })
        return completion

    fake_client = MagicMock()
    fake_client.with_options.return_value.chat.completions.create = fake_create

    monkeypatch.setattr(
        _seat_vlm, 'load_env', lambda: None, raising=True,
    )
    monkeypatch.setattr(
        _seat_vlm, 'require_api_key', lambda: 'sk-test', raising=True,
    )
    monkeypatch.setattr(
        _seat_vlm, 'base_url', lambda: 'https://example/api/v1', raising=True,
    )

    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setitem(sys.modules, 'openai', fake_openai)

    rgb = np.full((48, 64, 3), 100, dtype=np.uint8)
    _seat_vlm.request_seat(rgb, [], [], model='test/model', max_retries=1)

    msgs = captured['messages']
    roles = [m['role'] for m in msgs]
    assert roles == ['system', 'user']


def test_request_seat_with_fewshots_message_layout(monkeypatch):
    """fewshots=[a, b] ⇒ system + (user, assistant) * 2 + user_live."""
    captured = {}

    def fake_create(**kwargs):
        captured['messages'] = kwargs['messages']
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = json.dumps({
            'visible_seats': [],
            'label': 'none',
            'point': [0, 0],
        })
        return completion

    fake_client = MagicMock()
    fake_client.with_options.return_value.chat.completions.create = fake_create

    monkeypatch.setattr(_seat_vlm, 'load_env', lambda: None)
    monkeypatch.setattr(_seat_vlm, 'require_api_key', lambda: 'sk-test')
    monkeypatch.setattr(_seat_vlm, 'base_url', lambda: 'https://example/api/v1')

    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setitem(sys.modules, 'openai', fake_openai)

    rgb = np.full((48, 64, 3), 100, dtype=np.uint8)
    fewshots = [
        _seat_fewshot.FewshotExample(
            slug='a',
            image_bgr=np.full((24, 32, 3), 50, dtype=np.uint8),
            answer=_good_answer('a-seat'),
        ),
        _seat_fewshot.FewshotExample(
            slug='b',
            image_bgr=np.full((24, 32, 3), 150, dtype=np.uint8),
            answer=_good_answer('b-seat'),
        ),
    ]
    _seat_vlm.request_seat(
        rgb, ['Alice'], ['red shirt'],
        model='test/model', max_retries=1, fewshots=fewshots,
    )

    msgs = captured['messages']
    roles = [m['role'] for m in msgs]
    assert roles == [
        'system',
        'user', 'assistant',
        'user', 'assistant',
        'user',
    ]
    # Few-shot user prompts must NOT carry the live names/features —
    # they teach the form, not the content.
    assert msgs[1]['content'][1]['text'] == 'Recommend a seat for a new guest.'
    assert msgs[3]['content'][1]['text'] == 'Recommend a seat for a new guest.'
    # The live (final) user message DOES carry the prompt-built names.
    assert 'Alice' in msgs[5]['content'][1]['text']
    # Each assistant turn is the JSON-serialized answer dict.
    a0 = json.loads(msgs[2]['content'])
    a1 = json.loads(msgs[4]['content'])
    assert a0['label'] == 'a-seat'
    assert a1['label'] == 'b-seat'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
