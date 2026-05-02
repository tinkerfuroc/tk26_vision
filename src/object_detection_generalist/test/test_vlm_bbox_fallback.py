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
