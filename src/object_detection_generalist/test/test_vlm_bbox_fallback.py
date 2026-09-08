"""Unit tests for VLM bbox fallback orchestration and provider decoders."""

import importlib.util
from pathlib import Path
import sys
import threading
import time

import pytest


_SRC = (
    Path(__file__).resolve().parents[1]
    / 'object_detection_generalist'
    / 'vlm_bbox.py'
)
_SPEC = importlib.util.spec_from_file_location('vlm_bbox_source', _SRC)
vlm_bbox = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = vlm_bbox
_SPEC.loader.exec_module(vlm_bbox)


MODEL_GEMINI = 'google/gemini-2.5-flash'
MODEL_QWEN = 'qwen/qwen3-vl-8b-instruct'


def _attempt(model, status='success', error=None):
    record = {'model': model, 'status': status}
    if error:
        record['error'] = error
    return record


def test_primary_timeout_falls_back_to_qwen():
    calls = []

    def single_model(model):
        calls.append(model)
        if model == MODEL_GEMINI:
            raise vlm_bbox._VlmModelFailure(
                'timeout',
                [_attempt(model, 'error', 'timeout')],
            )
        return [(10, 20, 30, 40)], ['mug'], [_attempt(model)]

    boxes, labels, _, meta = vlm_bbox._run_model_chain(
        model_chain=[MODEL_GEMINI, MODEL_QWEN],
        fallback_on_empty=False,
        single_model_fn=single_model,
    )

    assert calls == [MODEL_GEMINI, MODEL_QWEN]
    assert boxes == [(10, 20, 30, 40)]
    assert labels == ['mug']
    assert meta['model_used'] == MODEL_QWEN
    assert meta['error'] is None


def test_primary_parse_error_falls_back_to_qwen():
    calls = []

    def single_model(model):
        calls.append(model)
        if model == MODEL_GEMINI:
            raise vlm_bbox._VlmModelFailure(
                'bad json',
                [_attempt(model, 'error', 'JSONDecodeError')],
            )
        return [(1, 2, 3, 4)], ['cup'], [_attempt(model)]

    boxes, labels, _, meta = vlm_bbox._run_model_chain(
        model_chain=[MODEL_GEMINI, MODEL_QWEN],
        fallback_on_empty=False,
        single_model_fn=single_model,
    )

    assert calls == [MODEL_GEMINI, MODEL_QWEN]
    assert boxes == [(1, 2, 3, 4)]
    assert labels == ['cup']
    assert meta['model_used'] == MODEL_QWEN


def test_primary_clean_empty_does_not_fallback_by_default():
    calls = []

    def single_model(model):
        calls.append(model)
        return [], [], [_attempt(model, 'empty')]

    boxes, labels, _, meta = vlm_bbox._run_model_chain(
        model_chain=[MODEL_GEMINI, MODEL_QWEN],
        fallback_on_empty=False,
        single_model_fn=single_model,
    )

    assert calls == [MODEL_GEMINI]
    assert boxes == []
    assert labels == []
    assert meta['model_used'] == MODEL_GEMINI
    assert meta['error'] is None


def test_abandon_before_fallback_prevents_qwen_call():
    calls = []
    abandon_event = threading.Event()

    def single_model(model):
        calls.append(model)
        abandon_event.set()
        raise vlm_bbox._VlmModelFailure(
            'timeout',
            [_attempt(model, 'error', 'timeout')],
        )

    boxes, labels, _, meta = vlm_bbox._run_model_chain(
        model_chain=[MODEL_GEMINI, MODEL_QWEN],
        fallback_on_empty=False,
        single_model_fn=single_model,
        abandon_event=abandon_event,
    )

    assert calls == [MODEL_GEMINI]
    assert boxes == []
    assert labels == []
    assert meta['model_used'] is None
    assert meta['error'] == 'abandoned'
    assert meta['abandoned'] is True


def test_hard_timeout_before_fallback_prevents_qwen_call():
    calls = []
    started_at = time.perf_counter()

    def single_model(model):
        calls.append(model)
        time.sleep(0.002)
        raise vlm_bbox._VlmModelFailure(
            'timeout',
            [_attempt(model, 'error', 'timeout')],
        )

    boxes, labels, _, meta = vlm_bbox._run_model_chain(
        model_chain=[MODEL_GEMINI, MODEL_QWEN],
        fallback_on_empty=False,
        single_model_fn=single_model,
        started_at=started_at,
        hard_deadline=started_at + 0.001,
        timeout_s=0.001,
    )

    assert calls == [MODEL_GEMINI]
    assert boxes == []
    assert labels == []
    assert meta['model_used'] is None
    assert meta['error'].startswith('timeout')


def test_gemini_normalized_yxyx_decoder():
    assert vlm_bbox._decode_gemini_bbox(
        [100, 200, 300, 400], w=1000, h=500,
    ) == (200, 50, 400, 150)


def test_qwen_pixel_xyxy_decoder():
    assert vlm_bbox._decode_qwen_bbox(
        [10, 20, 30, 40], w=1000, h=500,
    ) == (10, 20, 30, 40)


def test_qwen3_normalized_xyxy_decoder():
    # Qwen3-VL emits [x1, y1, x2, y2] normalized 0-1000. On a 1280x720 frame a
    # box at pixel (520,240,760,480) round-trips through 0-1000 as
    # [406,333,594,667] (verified against the live model).
    assert vlm_bbox._decode_qwen3_bbox(
        [406, 333, 594, 667], w=1280, h=720,
    ) == (520, 240, 760, 480)
    # x-axis scales by width, y-axis by height (not interchangeable).
    # The bottom edge clips to h-1 (499), the standard _clip_pixel_bbox cap.
    assert vlm_bbox._decode_qwen3_bbox(
        [0, 0, 500, 1000], w=1000, h=500,
    ) == (0, 0, 500, 499)


def test_qwen3_profile_selected_for_qwen3_models():
    # qwen3 (normalized) must win over the generic qwen (pixel) match.
    assert vlm_bbox._provider_profile_for_model(
        'dashscope/qwen3-vl-plus'
    ) is vlm_bbox._QWEN3_PROFILE
    assert vlm_bbox._provider_profile_for_model(
        'qwen/qwen3-vl-8b-instruct'
    ) is vlm_bbox._QWEN3_PROFILE
    # older qwen2.5 family stays on the pixel decoder
    assert vlm_bbox._provider_profile_for_model(
        'qwen/qwen2.5-vl-7b-instruct'
    ) is vlm_bbox._QWEN_PROFILE


def test_degenerate_boxes_are_clean_empty_not_error():
    # Well-formed numbers that clip to zero area => no-detection, no retry.
    boxes, labels, raw_count = vlm_bbox._parse_detections(
        '{"detections": [{"label": "x", "box_2d": [500, 500, 500, 500]}]}',
        w=640, h=480, profile=vlm_bbox._QWEN3_PROFILE,
    )
    assert boxes == []
    assert raw_count == 1


def test_malformed_boxes_still_raise_for_retry():
    # Wrong-length payload is a model glitch => error so the retry loop fires.
    import pytest as _pytest
    with _pytest.raises(ValueError, match='invalid box_2d'):
        vlm_bbox._parse_detections(
            '{"detections": [{"label": "x", "box_2d": [10]}]}',
            w=640, h=480, profile=vlm_bbox._QWEN3_PROFILE,
        )


def test_malformed_boxes_are_model_failure_not_clean_empty():
    raw = '{"detections": [{"label": "cup", "box_2d": [10]}]}'
    with pytest.raises(ValueError, match='invalid box_2d'):
        vlm_bbox._parse_detections(
            raw, w=640, h=480, profile=vlm_bbox._QWEN_PROFILE,
        )


def test_clean_empty_response_is_not_decode_failure():
    boxes, labels, raw_count = vlm_bbox._parse_detections(
        '{"detections": []}',
        w=640,
        h=480,
        profile=vlm_bbox._QWEN_PROFILE,
    )

    assert boxes == []
    assert labels == []
    assert raw_count == 0


# --- provider routing (OpenRouter vs DashScope) ----------------------------


def test_split_provider_routes_dashscope_prefix():
    # Default provider is OpenRouter; the id passes through untouched.
    assert vlm_bbox._split_provider('google/gemini-2.5-flash') == (
        'openrouter', 'google/gemini-2.5-flash'
    )
    # A plain qwen/* id still goes to OpenRouter — only the explicit
    # 'dashscope/' prefix reroutes.
    assert vlm_bbox._split_provider('qwen/qwen3-vl-8b-instruct') == (
        'openrouter', 'qwen/qwen3-vl-8b-instruct'
    )
    # 'dashscope/' prefix reroutes and is stripped from the API model id.
    assert vlm_bbox._split_provider('dashscope/qwen3-vl-plus') == (
        'dashscope', 'qwen3-vl-plus'
    )


def test_dashscope_model_builds_client_against_dashscope_host(monkeypatch):
    """A 'dashscope/' model must build its client with the DashScope key +
    base URL and call the API with the prefix stripped."""
    import numpy as np
    import openai

    constructed = []

    class FakeClient:
        def __init__(self, *, api_key, base_url, max_retries):
            constructed.append({'api_key': api_key, 'base_url': base_url})

        def close(self):
            pass

    monkeypatch.setattr(openai, 'OpenAI', FakeClient)
    monkeypatch.setattr(vlm_bbox, 'load_env', lambda: None)
    monkeypatch.setattr(vlm_bbox, 'require_dashscope_api_key', lambda: 'ds-key')
    monkeypatch.setattr(
        vlm_bbox, 'dashscope_base_url', lambda: 'https://dashscope.test/v1'
    )
    monkeypatch.setattr(
        vlm_bbox, 'encode_to_data_url',
        lambda img: 'data:image/jpeg;base64,XXXX',
    )

    seen = {}

    def fake_single(*, client, model, **kwargs):
        seen['client'] = client
        seen['model'] = model
        return [(1, 2, 3, 4)], ['tennis'], [{'model': model, 'status': 'success'}]

    monkeypatch.setattr(vlm_bbox, '_request_bboxes_single_model', fake_single)

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    boxes, labels, _elapsed, meta = vlm_bbox.request_bboxes(
        img, 'tennis', model='dashscope/qwen3-vl-plus',
        fallback_models=[], stream=False,
    )

    assert boxes == [(1, 2, 3, 4)]
    assert seen['model'] == 'qwen3-vl-plus'             # prefix stripped
    assert isinstance(seen['client'], FakeClient)
    assert len(constructed) == 1
    assert constructed[0]['api_key'] == 'ds-key'
    assert constructed[0]['base_url'] == 'https://dashscope.test/v1'
    assert meta['model_used'] == 'dashscope/qwen3-vl-plus'  # full id in meta


def test_missing_dashscope_key_fails_only_that_model(monkeypatch):
    """A missing DashScope key must not raise out of request_bboxes — it is
    recorded as a failed attempt so the chain can fall through."""
    import numpy as np
    import openai

    def boom():
        raise RuntimeError('DASHSCOPE_API_KEY is not set')

    def no_construct(**_kwargs):
        raise AssertionError('client must not be built without a key')

    monkeypatch.setattr(openai, 'OpenAI', no_construct)
    monkeypatch.setattr(vlm_bbox, 'load_env', lambda: None)
    monkeypatch.setattr(vlm_bbox, 'require_dashscope_api_key', boom)
    monkeypatch.setattr(
        vlm_bbox, 'encode_to_data_url',
        lambda img: 'data:image/jpeg;base64,XXXX',
    )

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    boxes, labels, _elapsed, meta = vlm_bbox.request_bboxes(
        img, 'tennis', model='dashscope/qwen3-vl-plus',
        fallback_models=[], stream=False,
    )

    assert boxes == []
    assert any(
        'DASHSCOPE' in (a.get('error') or '') for a in meta['attempts']
    )
